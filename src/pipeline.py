"""pipeline.py — Master orchestrator for the hybrid statistical arbitrage system.

Runs the trading loop for both modules:
  - Module A (Equity Options): runs during US equity market hours (9:30 - 16:00 ET).
  - Module B (Crypto Spot): runs 24/7.

For each cycle:
  1. Market data ingestion (via Broker)
  2. Exit evaluations on open positions (via ExecutionAgent)
  3. Signal generation (via SignalAgent)
  4. Deterministic risk checks & sizing (via RiskAgent)
  5. Order execution for approved candidates (via ExecutionAgent)
  6. Audit logging to SQLite (via Database)
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import yaml

from src.agents.execution_agent import ExecutionAgent
from src.agents.risk_agent import RiskAgent
from src.agents.signal_agent import SignalAgent
from src.broker import Broker, create_broker
from src.models import (
    RiskDecision,
    SpreadSignal,
    TradeLogEntry,
)
from src.persistence.db import Database, create_database

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class StatArbPipeline:
    """Master orchestrator running both equity options and crypto spot modules."""

    def __init__(
        self,
        broker: Broker | None = None,
        db: Database | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.db = db if db is not None else create_database(self.config.get("db_path", "data/trades.db"))
        self.broker = broker if broker is not None else create_broker()

        self.signal_agent = SignalAgent(config=self.config, db=self.db)
        self.risk_agent = RiskAgent(config=self.config, db=self.db)
        self.execution_agent = ExecutionAgent(broker=self.broker, db=self.db, config=self.config)

        self._last_coint_recheck_date: str | None = None

    # ── Orchestration Loop ────────────────────────────────────────────────────

    def run_cycle(
        self,
        module: Literal["equity", "crypto"] | None = None,
        now: datetime | None = None,
        force_market_open: bool = False,
    ) -> dict[str, Any]:
        """Execute one complete evaluation and trading cycle.

        Args:
            module: Specific module to run ('equity' | 'crypto'), or None to run both.
            now: Current UTC timestamp (defaults to actual current time).
            force_market_open: If True, bypasses equity market hours check (for testing).

        Returns:
            Dict summary of the cycle results.
        """
        now_utc = now or datetime.now(timezone.utc)
        results: dict[str, Any] = {
            "timestamp": now_utc.isoformat(),
            "signals": [],
            "decisions": [],
            "executed_entries": [],
            "executed_exits": [],
            "skipped_reasons": [],
        }

        # Check daily midnight UTC jobs
        if now_utc.hour == int(self.config.get("coint_recheck_hour_utc", 0)):
            today_str = now_utc.strftime("%Y-%m-%d")
            if self._last_coint_recheck_date != today_str:
                self.recheck_cointegration(now=now_utc)
                self._last_coint_recheck_date = today_str

        # ── Module A: Equity Options ──────────────────────────────────────────
        if module in ("equity", None):
            if force_market_open or self.is_equity_market_open(now=now_utc):
                eq_res = self._run_module_cycle(module="equity", now=now_utc)
                self._merge_results(results, eq_res)
            else:
                logger.info("Equity market is closed. Skipping Module A.")
                results["skipped_reasons"].append("equity_market_closed")

        # ── Module B: Crypto Spot (24/7) ──────────────────────────────────────
        if module in ("crypto", None):
            cr_res = self._run_module_cycle(module="crypto", now=now_utc)
            self._merge_results(results, cr_res)

        return results

    def _run_module_cycle(
        self,
        module: Literal["equity", "crypto"],
        now: datetime,
    ) -> dict[str, Any]:
        """Run the end-to-end pipeline for a single module."""
        cycle_signals: dict[str, SpreadSignal] = {}
        cycle_decisions: list[RiskDecision] = []
        executed_entries: list[TradeLogEntry] = []
        current_prices: dict[str, float] = {}

        pairs_key = "equity_pairs" if module == "equity" else "crypto_pairs"
        pairs: list[list[str]] = self.config.get(pairs_key, [])

        # 1. Ingest Market Data for all pairs in module
        for pair_list in pairs:
            if len(pair_list) < 2:
                continue
            asset_a, asset_b = pair_list[0], pair_list[1]
            pair_id = f"{asset_a}-{asset_b}"

            try:
                if module == "equity":
                    limit = int(self.config.get("coint_lookback_days_equity", 90)) + 15
                    bars_dict = self.broker.get_equity_bars([asset_a, asset_b], limit=limit)
                else:
                    limit = int(self.config.get("coint_lookback_hours_crypto", 720)) + 24
                    bars_dict = self.broker.get_crypto_bars([asset_a, asset_b], limit=limit)

                if asset_a not in bars_dict or asset_b not in bars_dict:
                    logger.warning("Missing bars for pair %s", pair_id)
                    continue

                bars_a = bars_dict[asset_a]
                bars_b = bars_dict[asset_b]

                current_prices[asset_a] = bars_a.latest_close
                current_prices[asset_b] = bars_b.latest_close

                # 2. Signal Generation
                signal = self.signal_agent.evaluate_pair(
                    asset_a=asset_a,
                    asset_b=asset_b,
                    module=module,
                    prices_a=bars_a,
                    prices_b=bars_b,
                    data_timestamp=bars_a.latest_timestamp,
                )
                cycle_signals[pair_id] = signal

            except Exception as exc:
                logger.error("Error processing pair %s: %s", pair_id, exc, exc_info=True)

        # 3. Position Exit Evaluation
        executed_exits = self.execution_agent.check_and_execute_exits(
            current_signals=cycle_signals,
            current_prices=current_prices,
            now=now,
        )

        # 4. Filter Candidate Entry Signals
        candidate_signals = [s for s in cycle_signals.values() if s.direction != "none"]

        if candidate_signals:
            try:
                account = self.broker.get_account()
                # 5. Risk Assessment & Position Sizing
                cycle_decisions = self.risk_agent.evaluate(
                    signals=candidate_signals,
                    account=account,
                    current_prices=current_prices,
                    now=now,
                )

                # 6. Order Placement for Approved Decisions
                for decision in cycle_decisions:
                    if decision.approved:
                        trade = self.execution_agent.execute_entry(
                            decision=decision,
                            current_prices=current_prices,
                            now=now,
                        )
                        if trade:
                            executed_entries.append(trade)

            except Exception as exc:
                logger.error("Risk/Execution stage failed for %s: %s", module, exc, exc_info=True)

        return {
            "signals": list(cycle_signals.values()),
            "decisions": cycle_decisions,
            "executed_entries": executed_entries,
            "executed_exits": executed_exits,
            "skipped_reasons": [],
        }

    # ── Daily Cointegration Recheck ───────────────────────────────────────────

    def recheck_cointegration(self, now: datetime | None = None) -> dict[str, float]:
        """Daily midnight UTC job: re-tests cointegration across all watchlist pairs."""
        logger.info("Starting midnight UTC cointegration health recheck...")
        now_utc = now or datetime.now(timezone.utc)
        results: dict[str, float] = {}

        for module in ("equity", "crypto"):
            pairs_key = "equity_pairs" if module == "equity" else "crypto_pairs"
            pairs: list[list[str]] = self.config.get(pairs_key, [])

            for pair_list in pairs:
                if len(pair_list) < 2:
                    continue
                asset_a, asset_b = pair_list[0], pair_list[1]
                pair_id = f"{asset_a}-{asset_b}"

                try:
                    if module == "equity":
                        bars_dict = self.broker.get_equity_bars([asset_a, asset_b], limit=100)
                    else:
                        bars_dict = self.broker.get_crypto_bars([asset_a, asset_b], limit=750)

                    if asset_a in bars_dict and asset_b in bars_dict:
                        sig = self.signal_agent.evaluate_pair(
                            asset_a=asset_a,
                            asset_b=asset_b,
                            module=module,  # type: ignore
                            prices_a=bars_dict[asset_a],
                            prices_b=bars_dict[asset_b],
                            data_timestamp=now_utc,
                        )
                        results[pair_id] = sig.coint_pvalue
                        logger.info("Pair %s coint p-value: %.4f", pair_id, sig.coint_pvalue)
                except Exception as exc:
                    logger.error("Coint recheck failed for %s: %s", pair_id, exc)

        return results

    # ── Utility Helpers ───────────────────────────────────────────────────────

    def is_equity_market_open(self, now: datetime | None = None) -> bool:
        """Check if US equity markets are currently open (9:30 - 16:00 ET, Mon-Fri)."""
        now_utc = now or datetime.now(timezone.utc)
        try:
            ny_tz = ZoneInfo("America/New_York")
            now_ny = now_utc.astimezone(ny_tz)
        except Exception:
            # Fallback if zoneinfo tz database is unavailable
            return True

        # Check weekday: 0 = Monday, 4 = Friday
        if now_ny.weekday() >= 5:
            return False

        open_time = time(9, 30)
        close_time = time(16, 0)
        return open_time <= now_ny.time() <= close_time

    def _merge_results(self, dest: dict[str, Any], src: dict[str, Any]) -> None:
        dest["signals"].extend(src.get("signals", []))
        dest["decisions"].extend(src.get("decisions", []))
        dest["executed_entries"].extend(src.get("executed_entries", []))
        dest["executed_exits"].extend(src.get("executed_exits", []))
        dest["skipped_reasons"].extend(src.get("skipped_reasons", []))
