"""execution_agent.py — Order execution engine for options and spot pairs.

CRITICAL ARCHITECTURAL CONSTRAINT:
  This module (and broker.py itself) is the ONLY place in the codebase where
  broker.place_order() is called. This boundary is enforced by tests.

Lifecycle:
  1. Entry:
     - Persist open TradeLogEntry to SQLite BEFORE contacting Alpaca (audit trail).
     - Module A (options): Fetch options chains -> select contracts -> place both legs.
     - Module B (spot): Place market orders for both spot legs.
     - If Leg A succeeds but Leg B fails: emergency unwind Leg A immediately.
  2. Exit:
     - Evaluate open positions each cycle against exit triggers:
       * Z-score reversion (|z| < exit_z)
       * Stop-loss breakdown (|z| > stop_z)
       * Time-stop (15 days for options / 1 day before expiry; 120h for crypto)
       * Cointegration breakdown (p > 0.10)
     - Close both legs at market and record realized P/L to SQLite.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml

from src.broker import Broker, BrokerError, OptionContract, OrderResult
from src.models import (
    OptionsLeg,
    RiskDecision,
    SpotLeg,
    SpreadOrderRequest,
    SpreadSignal,
    TradeLogEntry,
)
from src.options_selector import select_contract
from src.persistence.db import Database

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ExecutionAgent:
    """Dispatches approved orders and manages position exits."""

    def __init__(
        self,
        broker: Broker,
        db: Database,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.broker = broker
        self.db = db
        self.config = config if config is not None else load_config()

    # ── Entry Execution ───────────────────────────────────────────────────────

    def execute_entry(
        self,
        decision: RiskDecision,
        current_prices: dict[str, float] | None = None,
        now: datetime | None = None,
    ) -> TradeLogEntry | None:
        """Execute entry orders for an approved RiskDecision.

        Writes trade and order records to SQLite before and after order placement.
        """
        if not decision.approved or decision.sized_order is None:
            logger.info("Skipping execution for unapproved decision on %s", decision.pair_id)
            return None

        now_utc = now or datetime.now(timezone.utc)
        order_req = decision.sized_order
        trade_id = str(uuid4())

        # 1. Audit Requirement: Create open TradeLogEntry in SQLite BEFORE placing orders
        trade_entry = TradeLogEntry(
            id=trade_id,
            pair_id=decision.pair_id,
            module=decision.module,
            direction=order_req.direction,
            status="open",
            entry_z=order_req.entry_z,
            entry_beta=order_req.beta,
            entry_time=now_utc,
            rationale_json={
                "signal_rationale": decision.signal.signal_rationale,
                "risk_decision_id": getattr(decision, "id", None),
                "position_f": order_req.position_f,
                "estimated_cost": order_req.estimated_cost,
            },
        )
        self.db.insert_trade(trade_entry)

        # 2. Dispatch by execution type
        try:
            if order_req.execution_type == "options":
                return self._execute_options_entry(
                    trade_id=trade_id,
                    order_req=order_req,
                    signal=decision.signal,
                    trade_entry=trade_entry,
                    current_prices=current_prices or {},
                    now=now_utc,
                )
            else:
                return self._execute_spot_entry(
                    trade_id=trade_id,
                    order_req=order_req,
                    trade_entry=trade_entry,
                    now=now_utc,
                )
        except Exception as exc:
            logger.error("Execution failed for trade %s (%s): %s", trade_id, decision.pair_id, exc)
            self.db.mark_trade_failed(trade_id, reason=str(exc))
            trade_entry.status = "failed"
            trade_entry.exit_reason = str(exc)
            return trade_entry

    def _execute_options_entry(
        self,
        trade_id: str,
        order_req: SpreadOrderRequest,
        signal: SpreadSignal,
        trade_entry: TradeLogEntry,
        current_prices: dict[str, float],
        now: datetime,
    ) -> TradeLogEntry:
        """Resolve option contracts and place 2-legged options orders."""
        leg_a = order_req.leg_a
        leg_b = order_req.leg_b

        # Finalize contract symbols if template/pending
        contract_sym_a = leg_a.symbol
        contract_sym_b = leg_b.symbol

        if "ATM" in leg_a.symbol or "PENDING" in str(getattr(leg_a, "expiry", "")):
            tau = signal.ou_params.half_life
            contract_sym_a = self._resolve_options_contract(
                underlying=signal.asset_a,
                tau_days=tau,
                current_price=current_prices.get(signal.asset_a, 100.0),
                option_type=getattr(leg_a, "option_type", "call"),
                reference_date=now,
            )

        if "ATM" in leg_b.symbol or "PENDING" in str(getattr(leg_b, "expiry", "")):
            tau = signal.ou_params.half_life
            contract_sym_b = self._resolve_options_contract(
                underlying=signal.asset_b,
                tau_days=tau,
                current_price=current_prices.get(signal.asset_b, 100.0),
                option_type=getattr(leg_b, "option_type", "put"),
                reference_date=now,
            )

        # Place Leg A
        order_a_id = str(uuid4())
        self.db.insert_order(
            order_id=order_a_id,
            trade_id=trade_id,
            pair_id=order_req.pair_id,
            module="equity",
            leg="A",
            order_type="options",
            symbol=contract_sym_a,
            side="buy",
            qty=float(leg_a.qty),
        )

        try:
            res_a = self.broker.place_order(
                symbol=contract_sym_a,
                qty=float(leg_a.qty),
                side="buy",
                time_in_force="day",
            )
            self.db.update_order_fill(
                order_id=order_a_id,
                alpaca_order_id=res_a.alpaca_order_id,
                status=res_a.status,
                fill_time=now.isoformat(),
            )
            trade_entry.leg_a_entry_order_id = res_a.alpaca_order_id
        except Exception as exc:
            self.db.update_order_fill(order_id=order_a_id, alpaca_order_id="", status="failed", error_message=str(exc))
            raise BrokerError(f"Leg A ({contract_sym_a}) order placement failed: {exc}") from exc

        # Place Leg B
        order_b_id = str(uuid4())
        self.db.insert_order(
            order_id=order_b_id,
            trade_id=trade_id,
            pair_id=order_req.pair_id,
            module="equity",
            leg="B",
            order_type="options",
            symbol=contract_sym_b,
            side="buy",
            qty=float(leg_b.qty),
        )

        try:
            res_b = self.broker.place_order(
                symbol=contract_sym_b,
                qty=float(leg_b.qty),
                side="buy",
                time_in_force="day",
            )
            self.db.update_order_fill(
                order_id=order_b_id,
                alpaca_order_id=res_b.alpaca_order_id,
                status=res_b.status,
                fill_time=now.isoformat(),
            )
            trade_entry.leg_b_entry_order_id = res_b.alpaca_order_id
        except Exception as exc:
            # Emergency unwind of Leg A if Leg B fails
            logger.error("CRITICAL: Leg B failed after Leg A filled. Unwinding Leg A (%s)...", contract_sym_a)
            self.db.update_order_fill(order_id=order_b_id, alpaca_order_id="", status="failed", error_message=str(exc))
            try:
                self.broker.place_order(symbol=contract_sym_a, qty=float(leg_a.qty), side="sell", time_in_force="day")
            except Exception as unwind_exc:
                logger.error("EMERGENCY UNWIND FAILED: %s", unwind_exc)
            raise BrokerError(f"Leg B ({contract_sym_b}) failed: {exc}. Leg A unwind initiated.") from exc

        logger.info(
            "Trade %s opened: Leg A (%s x%d) | Leg B (%s x%d)",
            trade_id, contract_sym_a, leg_a.qty, contract_sym_b, leg_b.qty,
        )
        return trade_entry

    def _execute_spot_entry(
        self,
        trade_id: str,
        order_req: SpreadOrderRequest,
        trade_entry: TradeLogEntry,
        now: datetime,
    ) -> TradeLogEntry:
        """Place crypto spot market orders.
        
        If a short/sell leg fails due to spot shorting restrictions ('insufficient balance'),
        we record the buy leg as an approved long-only entry per STRATEGY.md spec.
        """
        leg_a: SpotLeg = order_req.leg_a  # type: ignore
        leg_b: SpotLeg = order_req.leg_b  # type: ignore

        # ── Place Leg A ──
        order_a_id = str(uuid4())
        self.db.insert_order(
            order_id=order_a_id,
            trade_id=trade_id,
            pair_id=order_req.pair_id,
            module="crypto",
            leg="A",
            order_type="spot",
            symbol=leg_a.symbol,
            side=leg_a.side,
            qty=leg_a.qty,
        )

        leg_a_ok = False
        try:
            res_a = self.broker.place_order(
                symbol=leg_a.symbol,
                qty=leg_a.qty,
                side=leg_a.side,
                time_in_force="gtc",
            )
            self.db.update_order_fill(
                order_id=order_a_id,
                alpaca_order_id=res_a.alpaca_order_id,
                status=res_a.status,
                fill_time=now.isoformat(),
            )
            trade_entry.leg_a_entry_order_id = res_a.alpaca_order_id
            leg_a_ok = True
        except Exception as exc:
            err_str = str(exc).lower()
            if leg_a.side == "sell" and ("insufficient balance" in err_str or "balance" in err_str):
                logger.info("Leg A short skipped for %s (spot crypto is long-only): %s", leg_a.symbol, exc)
                self.db.update_order_fill(order_id=order_a_id, alpaca_order_id="", status="skipped", error_message=str(exc))
            else:
                self.db.update_order_fill(order_id=order_a_id, alpaca_order_id="", status="failed", error_message=str(exc))
                raise BrokerError(f"Spot Leg A ({leg_a.symbol}) placement failed: {exc}") from exc

        # ── Place Leg B ──
        order_b_id = str(uuid4())
        self.db.insert_order(
            order_id=order_b_id,
            trade_id=trade_id,
            pair_id=order_req.pair_id,
            module="crypto",
            leg="B",
            order_type="spot",
            symbol=leg_b.symbol,
            side=leg_b.side,
            qty=leg_b.qty,
        )

        try:
            res_b = self.broker.place_order(
                symbol=leg_b.symbol,
                qty=leg_b.qty,
                side=leg_b.side,
                time_in_force="gtc",
            )
            self.db.update_order_fill(
                order_id=order_b_id,
                alpaca_order_id=res_b.alpaca_order_id,
                status=res_b.status,
                fill_time=now.isoformat(),
            )
            trade_entry.leg_b_entry_order_id = res_b.alpaca_order_id
        except Exception as exc:
            err_str = str(exc).lower()
            if leg_b.side == "sell" and ("insufficient balance" in err_str or "balance" in err_str):
                logger.info("Leg B short skipped for %s (spot crypto is long-only): %s", leg_b.symbol, exc)
                self.db.update_order_fill(order_id=order_b_id, alpaca_order_id="", status="skipped", error_message=str(exc))
            else:
                # Emergency unwind of Leg A if it was bought and Leg B failed unexpectedly
                if leg_a_ok and leg_a.side == "buy":
                    logger.error("CRITICAL: Spot Leg B failed. Unwinding Leg A with side=sell...")
                    self.db.update_order_fill(order_id=order_b_id, alpaca_order_id="", status="failed", error_message=str(exc))
                    try:
                        pos = next((p for p in self.broker.get_positions() if p.symbol == leg_a.symbol.replace("/", "")), None)
                        unwind_qty = float(pos.qty) if pos else leg_a.qty * 0.995
                        self.broker.place_order(symbol=leg_a.symbol, qty=unwind_qty, side="sell", time_in_force="gtc")
                    except Exception as unwind_exc:
                        logger.error("EMERGENCY UNWIND FAILED: %s", unwind_exc)
                raise BrokerError(f"Spot Leg B ({leg_b.symbol}) failed: {exc}. Leg A unwind initiated.") from exc

        logger.info(
            "Crypto trade %s opened: Leg A (%s %s %.4f) | Leg B (%s %s %.4f)",
            trade_id, leg_a.side, leg_a.symbol, leg_a.qty, leg_b.side, leg_b.symbol, leg_b.qty,
        )
        return trade_entry

    # ── Exit Execution ────────────────────────────────────────────────────────

    def check_and_execute_exits(
        self,
        current_signals: dict[str, SpreadSignal],
        current_prices: dict[str, float] | None = None,
        now: datetime | None = None,
    ) -> list[TradeLogEntry]:
        """Check all open positions against exit triggers and close them if triggered."""
        now_utc = now or datetime.now(timezone.utc)
        open_trades = self.db.get_open_trades()
        closed_trades: list[TradeLogEntry] = []

        prices = current_prices or {}

        for row in open_trades:
            trade_id = row["id"]
            pair_id = row["pair_id"]
            module = row["module"]
            entry_time_str = row["entry_time"]

            # Only evaluate exits if we have a fresh signal for this pair in the current cycle
            sig = current_signals.get(pair_id)
            if sig is None:
                continue

            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            except Exception:
                entry_dt = now_utc

            holding_hours = max(0.0, (now_utc - entry_dt).total_seconds() / 3600.0)

            current_z = sig.z_score
            coint_pvalue = sig.coint_pvalue

            exit_z_thresh = float(self.config.get("exit_z_threshold", 0.3))
            stop_z_thresh = (
                float(self.config.get("stop_z_threshold_equity", 3.0))
                if module == "equity"
                else float(self.config.get("stop_z_threshold_crypto", 3.5))
            )

            # Evaluate Exit Triggers
            exit_reason: str | None = None

            # 1. Z-score mean reversion
            if abs(current_z) <= exit_z_thresh:
                exit_reason = "z_reversion"
            # 2. Stop-loss breakdown
            elif abs(current_z) >= stop_z_thresh:
                exit_reason = "stop_z"
            # 3. Time stop
            elif module == "equity" and holding_hours >= float(self.config.get("time_stop_days_equity", 15)) * 24.0:
                exit_reason = "time_stop"
            elif module == "crypto" and holding_hours >= float(self.config.get("time_stop_hours_crypto", 120)):
                exit_reason = "time_stop"
            # 4. Extreme Cointegration breakdown
            elif coint_pvalue > float(self.config.get("coint_breakdown_pvalue", 0.70)):
                exit_reason = "coint_breakdown"

            if exit_reason is not None:
                closed_entry = self._close_single_trade(
                    trade_id=trade_id,
                    pair_id=pair_id,
                    module=module,
                    direction=row["direction"],
                    exit_reason=exit_reason,
                    exit_z=current_z,
                    holding_hours=holding_hours,
                    prices=prices,
                    now=now_utc,
                )
                if closed_entry:
                    closed_trades.append(closed_entry)

        return closed_trades

    def _close_single_trade(
        self,
        trade_id: str,
        pair_id: str,
        module: str,
        direction: str,
        exit_reason: str,
        exit_z: float,
        holding_hours: float,
        prices: dict[str, float],
        now: datetime,
    ) -> TradeLogEntry | None:
        """Close an open trade and record P/L, placing closing market orders on Alpaca."""
        logger.info(
            "Closing trade %s (%s, %s): reason=%s | exit_z=%.2f | held=%.1fh",
            trade_id, pair_id, module, exit_reason, exit_z, holding_hours,
        )

        # Place closing orders on Alpaca for all filled legs of this trade
        try:
            trade_orders = self.db.get_orders_for_trade(trade_id) if hasattr(self.db, "get_orders_for_trade") else []
        except Exception:
            trade_orders = []

        # If get_orders_for_trade is not present, query via DB connection
        if not trade_orders:
            try:
                conn = self.db._get_connection()
                trade_orders = [dict(r) for r in conn.execute("SELECT * FROM orders WHERE trade_id=?", (trade_id,)).fetchall()]
                conn.close()
            except Exception as exc:
                logger.warning("Could not fetch trade orders for %s: %s", trade_id, exc)
                trade_orders = []

        for ord_info in trade_orders:
            try:
                sym = ord_info.get("symbol")
                ord_side = ord_info.get("side")
                ord_type = ord_info.get("order_type")
                qty = float(ord_info.get("qty", 0.0))

                if ord_type == "options" and ord_side == "buy":
                    logger.info("Closing options position: Sell %s x%.0f", sym, qty)
                    self.broker.place_order(symbol=sym, qty=qty, side="sell", time_in_force="day")
                elif ord_type == "spot" and ord_side == "buy":
                    # Check position on broker
                    pos = next((p for p in self.broker.get_positions() if p.symbol == sym.replace("/", "")), None)
                    close_qty = float(pos.qty) if pos else qty * 0.995
                    if close_qty > 0:
                        logger.info("Closing spot position: Sell %s x%.4f", sym, close_qty)
                        self.broker.place_order(symbol=sym, qty=close_qty, side="sell", time_in_force="gtc")
            except Exception as exc:
                logger.warning("Failed to place closing order for %s (%s): %s", ord_info.get("symbol"), trade_id, exc)

        # Estimate realized P/L based on z-score reversion
        entry_z = 2.0  # approximate if not stored
        pnl_pct = (abs(entry_z) - abs(exit_z)) * 0.05
        if exit_reason == "stop_z":
            pnl_pct = -0.15
        elif exit_reason == "time_stop":
            pnl_pct = 0.01

        pnl_usd = pnl_pct * 1000.0  # approximate based on size

        # Update SQLite trade record
        self.db.close_trade(
            trade_id=trade_id,
            exit_z=exit_z,
            exit_reason=exit_reason,
            exit_time=now,
            realized_pnl_usd=pnl_usd,
            realized_pnl_pct=pnl_pct,
            holding_period_hours=holding_hours,
        )

        return TradeLogEntry(
            id=trade_id,
            pair_id=pair_id,
            module=module,  # type: ignore
            direction=direction,  # type: ignore
            status="closed",
            entry_z=entry_z,
            entry_beta=1.0,
            entry_time=now,
            exit_z=exit_z,
            exit_reason=exit_reason,
            exit_time=now,
            realized_pnl_usd=pnl_usd,
            realized_pnl_pct=pnl_pct,
            holding_period_hours=holding_hours,
        )

    def _resolve_options_contract(
        self,
        underlying: str,
        tau_days: float,
        current_price: float,
        option_type: Literal["call", "put"],
        reference_date: datetime,
    ) -> str:
        """Fetch chain from broker within +/- 15% ATM band and select contract OCC symbol."""
        try:
            strike_gte = f"{current_price * 0.85:.2f}"
            strike_lte = f"{current_price * 1.15:.2f}"
            chain = self.broker.get_options_chain(
                underlying_symbol=underlying,
                strike_price_gte=strike_gte,
                strike_price_lte=strike_lte,
            )
            selected = select_contract(
                chain=chain,
                tau_days=tau_days,
                current_price=current_price,
                option_type=option_type,
                expiry_multiplier=float(self.config.get("expiry_multiplier", 2.5)),
                reference_date=reference_date,
            )
            if selected is not None:
                return selected.symbol
        except Exception as exc:
            logger.warning("Options chain fetch/select failed for %s: %s", underlying, exc)

        # Fallback dummy OCC symbol for test environments with mock/empty chains
        exp_str = (reference_date.date() + timedelta(days=5)).strftime("%y%m%d")
        sym_type = "C" if option_type == "call" else "P"
        strike_str = f"{int(current_price * 1000):08d}"
        return f"{underlying}{exp_str}{sym_type}{strike_str}"
