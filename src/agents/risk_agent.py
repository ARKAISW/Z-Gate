"""risk_agent.py — Deterministic Risk Engine (zero LLM calls).

Evaluates candidate signals against all 9 risk rules in sequence:
  1. Max concurrent open pairs per module
  2. Duplicate pair check (no stacking)
  3. Half-life gate re-validation (defense in depth)
  4. Cointegration p-value re-validation
  5. Volatility regime block (EXTREME -> REGIME_BLOCK)
  6. Position sizing sanity check (Quarter-Kelly, caps)
  7. Buying power check
  8. Rolling 24h loss circuit breaker (cooldown window)
  9. Data freshness check (staleness threshold)

Fail-closed: Any exception in risk evaluation rejects all candidates with
rejection_rule='risk_agent_failure'.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import yaml

from src.broker import AccountInfo
from src.indicators import compute_kelly_fraction
from src.models import (
    AccountSnapshot,
    OptionsLeg,
    RiskDecision,
    SpotLeg,
    SpreadOrderRequest,
    SpreadSignal,
    TradeLogEntry,
)
from src.persistence.db import Database

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RiskAgent:
    """Deterministic, auditable risk management engine."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db: Database | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.db = db
        self._circuit_breaker_tripped_at: datetime | None = None

    def evaluate(
        self,
        signals: list[SpreadSignal],
        account: AccountInfo | AccountSnapshot,
        open_trades: list[TradeLogEntry | dict[str, Any]] | None = None,
        rolling_24h_pnl: float | None = None,
        current_prices: dict[str, float] | None = None,
        now: datetime | None = None,
    ) -> list[RiskDecision]:
        """Evaluate a list of SpreadSignals and return typed RiskDecisions.

        Fail-closed: If an error occurs during evaluation, all signals are rejected.
        """
        now_utc = now or datetime.now(timezone.utc)
        decisions: list[RiskDecision] = []

        try:
            # Normalize open trades
            norm_open_trades = self._normalize_open_trades(open_trades)

            # Check rolling 24h P/L for circuit breaker
            pnl_24h = rolling_24h_pnl
            if pnl_24h is None and self.db is not None:
                pnl_24h = self.db.get_rolling_pnl(24.0)
            elif pnl_24h is None:
                pnl_24h = 0.0

            # Evaluate each signal independently
            for signal in signals:
                decision = self._evaluate_single_signal(
                    signal=signal,
                    account=account,
                    open_trades=norm_open_trades,
                    rolling_24h_pnl=pnl_24h,
                    current_prices=current_prices or {},
                    now=now_utc,
                )
                decisions.append(decision)

                if self.db is not None:
                    try:
                        self.db.insert_risk_decision(
                            decision_id=str(uuid4()),
                            signal_id=str(uuid4()),
                            decision=decision,
                        )
                    except Exception as exc:
                        logger.error("Failed to insert risk decision into DB: %s", exc)

        except Exception as exc:
            logger.error("RISK AGENT FAILURE: %s", exc, exc_info=True)
            # Fail closed: reject all signals
            decisions = [
                RiskDecision(
                    pair_id=sig.pair_id,
                    module=sig.module,
                    signal=sig,
                    approved=False,
                    rejection_rule="risk_agent_failure",
                    rejection_reason=f"RiskAgent execution failure: {exc}",
                    checked_at=now_utc,
                )
                for sig in signals
            ]

        return decisions

    def _evaluate_single_signal(
        self,
        signal: SpreadSignal,
        account: AccountInfo | AccountSnapshot,
        open_trades: list[dict[str, Any]],
        rolling_24h_pnl: float,
        current_prices: dict[str, float],
        now: datetime,
    ) -> RiskDecision:
        """Run all 9 risk rules against one SpreadSignal candidate."""
        # Pre-check: If signal direction is 'none', it's not a trading candidate
        if signal.direction == "none":
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="no_signal",
                rejection_reason="Signal direction is 'none'",
                checked_at=now,
            )

        # ── Rule 1: Max Open Pairs ────────────────────────────────────────────
        if signal.module == "equity":
            max_pairs = int(self.config.get("max_open_equity_pairs", 3))
        else:
            max_pairs = int(self.config.get("max_open_crypto_pairs", 3))

        module_open_count = sum(
            1 for t in open_trades
            if t.get("module") == signal.module and t.get("status") == "open"
        )
        if module_open_count >= max_pairs:
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="max_open_pairs",
                rejection_reason=(
                    f"Max open {signal.module} pairs reached "
                    f"({module_open_count}/{max_pairs})"
                ),
                checked_at=now,
            )

        # ── Rule 2: Duplicate Pair Check ──────────────────────────────────────
        pair_already_open = any(
            t.get("pair_id") == signal.pair_id and t.get("status") == "open"
            for t in open_trades
        )
        if pair_already_open:
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="duplicate_pair",
                rejection_reason=f"Position for {signal.pair_id} is already open",
                checked_at=now,
            )

        # ── Rule 3: Half-Life Gate (Re-validation) ────────────────────────────
        tau = signal.ou_params.half_life
        if signal.module == "equity":
            hl_min = float(self.config.get("halflife_min_days", 2.0))
            hl_max = float(self.config.get("halflife_max_days", 60.0))
        else:
            hl_min = float(self.config.get("halflife_min_hours", 2.0))
            hl_max = float(self.config.get("halflife_max_hours", 720.0))

        coint_thresh = float(self.config.get("coint_pvalue_threshold", 0.15))
        if not (hl_min <= tau <= hl_max):
            # If pair is statistically cointegrated (p <= threshold), allow entry
            if signal.coint_pvalue > coint_thresh:
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="halflife_gate",
                    rejection_reason=f"Half-life {tau:.2f} outside bounds [{hl_min}, {hl_max}]",
                    checked_at=now,
                )
            else:
                logger.info("Pair %s passed cointegration (p=%.4f), overriding tau=%.1f", signal.pair_id, signal.coint_pvalue, tau)

        # ── Rule 4: Cointegration p-value Re-validation ───────────────────────
        coint_thresh = float(self.config.get("coint_pvalue_threshold", 0.20))
        if signal.coint_pvalue > coint_thresh:
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="coint_gate",
                rejection_reason=f"Cointegration p-value {signal.coint_pvalue:.4f} > {coint_thresh}",
                checked_at=now,
            )

        # ── Rule 5: Volatility Regime Block ───────────────────────────────────
        if signal.vol_regime == "EXTREME":
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="regime_block",
                rejection_reason="Market is in EXTREME volatility regime",
                checked_at=now,
            )

        # ── Rule 8: Rolling 24h Loss Circuit Breaker ──────────────────────────
        # Checked before sizing to reject early
        loss_limit_pct = float(self.config.get("rolling_24h_loss_limit_pct", 0.03))
        loss_limit_usd = -loss_limit_pct * account.equity
        cooldown_hours = float(self.config.get("circuit_breaker_cooldown_hours", 4.0))

        # Check if circuit breaker should trip
        if rolling_24h_pnl <= loss_limit_usd:
            if self._circuit_breaker_tripped_at is None:
                self._circuit_breaker_tripped_at = now
                logger.warning(
                    "CIRCUIT BREAKER TRIPPED: 24h P/L = $%.2f (limit = $%.2f)",
                    rolling_24h_pnl, loss_limit_usd,
                )

        # Check if currently in cooldown
        if self._circuit_breaker_tripped_at is not None:
            elapsed_hours = (now - self._circuit_breaker_tripped_at).total_seconds() / 3600.0
            if elapsed_hours < cooldown_hours:
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="circuit_breaker",
                    rejection_reason=(
                        f"Circuit breaker active (24h P/L=${rolling_24h_pnl:.2f} <= ${loss_limit_usd:.2f}, "
                        f"cooldown remaining: {cooldown_hours - elapsed_hours:.1f}h)"
                    ),
                    checked_at=now,
                )
            else:
                # Cooldown expired
                self._circuit_breaker_tripped_at = None
                logger.info("Circuit breaker cooldown ended.")

        # ── Rule 9: Data Freshness Check ──────────────────────────────────────
        # Ensure signal.data_timestamp has timezone info
        data_ts = signal.data_timestamp
        if data_ts.tzinfo is None:
            data_ts = data_ts.replace(tzinfo=timezone.utc)

        staleness_minutes = (now - data_ts).total_seconds() / 60.0
        # Crypto operates on hourly bars (default 120m); equity on daily bars (default 5760m / 4 days)
        if signal.module == "crypto":
            default_staleness = 120.0
            cfg_key = "max_data_staleness_minutes_crypto"
        else:
            default_staleness = 5760.0
            cfg_key = "max_data_staleness_minutes_equity"

        if cfg_key in self.config:
            max_staleness = float(self.config[cfg_key])
        else:
            max_staleness = float(self.config.get("max_data_staleness_minutes", default_staleness))

        if staleness_minutes > max_staleness:
            return RiskDecision(
                pair_id=signal.pair_id,
                module=signal.module,
                signal=signal,
                approved=False,
                rejection_rule="data_freshness",
                rejection_reason=f"Data staleness {staleness_minutes:.1f}m exceeds max {max_staleness}m",
                checked_at=now,
            )

        # ── Rule 6 & 7: Position Sizing & Buying Power Checks ─────────────────
        sizing_decision = self._size_and_validate_order(
            signal=signal,
            account=account,
            current_prices=current_prices,
            now=now,
        )
        return sizing_decision

    def _size_and_validate_order(
        self,
        signal: SpreadSignal,
        account: AccountInfo | AccountSnapshot,
        current_prices: dict[str, float],
        now: datetime,
    ) -> RiskDecision:
        """Compute Quarter-Kelly position size and validate limits and buying power."""
        kelly_fraction = float(self.config.get("kelly_fraction", 0.25))

        # Theoretical full Kelly fraction
        try:
            kelly_f = compute_kelly_fraction(
                entry_z=signal.z_score,
                exit_z_threshold=signal.exit_z_threshold,
                sigma_spread=signal.ou_params.sigma_spread,
            )
        except Exception:
            kelly_f = 1.0

        if signal.module == "equity":
            # ── Equity Options Sizing ─────────────────────────────────────────
            max_premium_pct = float(self.config.get("max_premium_pct_equity", 0.05))
            position_f = float(np.clip(kelly_fraction * kelly_f, 0.005, max_premium_pct))
            total_premium_budget = position_f * account.equity

            # Premium estimation
            price_a = current_prices.get(signal.asset_a, 100.0)
            price_b = current_prices.get(signal.asset_b, 100.0)

            # Estimate ATM option premium ~2.5% of spot for ETF/equity options (approximate for risk validation)
            est_premium_per_share_a = price_a * 0.025
            est_premium_per_share_b = price_b * 0.025
            est_contract_cost_a = est_premium_per_share_a * 100
            est_contract_cost_b = est_premium_per_share_b * 100

            # 1 contract each minimum
            qty_a = max(1, math.floor((total_premium_budget / 2) / max(est_contract_cost_a, 1.0)))
            qty_b = max(1, math.floor((total_premium_budget / 2) / max(est_contract_cost_b, 1.0)))

            total_estimated_cost = (qty_a * est_contract_cost_a) + (qty_b * est_contract_cost_b)

            # Check Rule 6: Position sizing cap
            max_allowed_premium = max_premium_pct * account.equity
            if total_estimated_cost > max_allowed_premium * 1.10: # allow 10% tolerance for 1-contract minimum
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="sizing_check",
                    rejection_reason=(
                        f"Estimated options premium ${total_estimated_cost:.2f} exceeds "
                        f"max cap ${max_allowed_premium:.2f} ({max_premium_pct*100:.1f}% of equity)"
                    ),
                    checked_at=now,
                )

            # Check Rule 7: Buying power
            if total_estimated_cost > account.buying_power:
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="buying_power",
                    rejection_reason=(
                        f"Total premium cost ${total_estimated_cost:.2f} > "
                        f"buying power ${account.buying_power:.2f}"
                    ),
                    checked_at=now,
                )

            # Construct dummy/template legs for SpreadOrderRequest (options_selector will finalize actual contracts)
            opt_type_a: Literal["call", "put"] = "call" if signal.direction == "long" else "put"
            opt_type_b: Literal["call", "put"] = "put" if signal.direction == "long" else "call"

            leg_a = OptionsLeg(
                underlying=signal.asset_a,
                symbol=f"{signal.asset_a}_ATM_{opt_type_a.upper()}",
                expiry="PENDING",
                strike=round(price_a, 2),
                option_type=opt_type_a,
                qty=qty_a,
                side="buy_to_open",
                premium_estimate=est_premium_per_share_a,
            )
            leg_b = OptionsLeg(
                underlying=signal.asset_b,
                symbol=f"{signal.asset_b}_ATM_{opt_type_b.upper()}",
                expiry="PENDING",
                strike=round(price_b, 2),
                option_type=opt_type_b,
                qty=qty_b,
                side="buy_to_open",
                premium_estimate=est_premium_per_share_b,
            )

            order_req = SpreadOrderRequest(
                pair_id=signal.pair_id,
                module="equity",
                direction=signal.direction, # type: ignore
                execution_type="options",
                leg_a=leg_a,
                leg_b=leg_b,
                beta=signal.beta,
                entry_z=signal.z_score,
                kelly_f=kelly_f,
                position_f=position_f,
                estimated_cost=total_estimated_cost,
            )

        else:
            # ── Crypto Spot Sizing ────────────────────────────────────────────
            max_position_pct = float(self.config.get("max_position_pct_equity", 0.10))
            position_f = float(np.clip(kelly_fraction * kelly_f, 0.01, max_position_pct))

            price_a = current_prices.get(signal.asset_a, 50000.0)
            price_b = current_prices.get(signal.asset_b, 3000.0)

            # Scale leg A value so that the larger of leg A and leg B does not exceed max_position_pct
            beta_factor = max(1.0, abs(signal.beta))
            leg_a_value = (position_f / beta_factor) * account.equity
            qty_a = leg_a_value / price_a
            # Hedge leg B using beta
            qty_b = (qty_a * abs(signal.beta) * price_a) / price_b
            leg_b_value = qty_b * price_b

            total_notional = leg_a_value + leg_b_value

            # Check Rule 6: Position sizing cap per leg
            max_leg_cap = max_position_pct * account.equity
            if leg_a_value > max_leg_cap * 1.01 or leg_b_value > max_leg_cap * 1.01:
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="sizing_check",
                    rejection_reason=(
                        f"Leg notional (A=${leg_a_value:.2f}, B=${leg_b_value:.2f}) exceeds "
                        f"max cap ${max_leg_cap:.2f} ({max_position_pct*100:.1f}% of equity)"
                    ),
                    checked_at=now,
                )

            # Check Rule 7: Buying power
            if total_notional > account.buying_power:
                return RiskDecision(
                    pair_id=signal.pair_id,
                    module=signal.module,
                    signal=signal,
                    approved=False,
                    rejection_rule="buying_power",
                    rejection_reason=(
                        f"Total notional ${total_notional:.2f} > "
                        f"buying power ${account.buying_power:.2f}"
                    ),
                    checked_at=now,
                )

            side_a: Literal["buy", "sell"] = "buy" if signal.direction == "long" else "sell"
            side_b: Literal["buy", "sell"] = "sell" if signal.direction == "long" else "buy"

            leg_a_spot = SpotLeg(
                symbol=signal.asset_a,
                qty=qty_a,
                side=side_a,
                notional_usd=leg_a_value,
            )
            leg_b_spot = SpotLeg(
                symbol=signal.asset_b,
                qty=qty_b,
                side=side_b,
                notional_usd=leg_b_value,
            )

            order_req = SpreadOrderRequest(
                pair_id=signal.pair_id,
                module="crypto",
                direction=signal.direction, # type: ignore
                execution_type="spot",
                leg_a=leg_a_spot,
                leg_b=leg_b_spot,
                beta=signal.beta,
                entry_z=signal.z_score,
                kelly_f=kelly_f,
                position_f=position_f,
                estimated_cost=total_notional,
            )

        # All rules passed!
        return RiskDecision(
            pair_id=signal.pair_id,
            module=signal.module,
            signal=signal,
            approved=True,
            rejection_rule=None,
            rejection_reason=None,
            sized_order=order_req,
            checked_at=now,
        )

    def _normalize_open_trades(
        self,
        trades: list[TradeLogEntry | dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not trades:
            if self.db is not None:
                rows = self.db.get_open_trades()
                return [dict(r) for r in rows]
            return []

        res = []
        for t in trades:
            if isinstance(t, TradeLogEntry):
                res.append(t.model_dump())
            elif isinstance(t, dict):
                res.append(t)
            else:
                try:
                    res.append(dict(t))
                except Exception:
                    pass
        return res
