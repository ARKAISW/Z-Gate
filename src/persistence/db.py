"""persistence/db.py — SQLite connection management and CRUD operations.

Uses plain sqlite3 with WAL mode for read concurrency. The pipeline writes
signals and decisions; the dashboard reads them. Both hit the same file.

Design constraints:
  - Every write goes through a context manager — no unclosed transactions.
  - All trade log entries are written BEFORE orders are placed (audit requirement).
  - JSON blobs are serialized/deserialized at this layer, not in the agents.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from src.persistence.schema import ALL_CREATE_STATEMENTS

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> str:
    """Serialize a dict or pydantic model to a JSON string."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str)
    return json.dumps(obj, default=str)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class Database:
    """Thin SQLite wrapper providing typed write and read operations."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Create all tables and indexes if they don't exist."""
        with self._get_connection() as conn:
            for stmt in ALL_CREATE_STATEMENTS:
                conn.execute(stmt)
            conn.commit()
        logger.info("Database initialized at %s", self.db_path)

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a single write transaction."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Signals ──────────────────────────────────────────────────────────────

    def insert_signal(self, signal_id: str, signal: Any) -> None:
        """Write a SpreadSignal to the signals table.

        Called by signal_agent.py before any downstream processing.
        """
        ou = signal.ou_params
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    id, pair_id, module, asset_a, asset_b, direction,
                    z_score, beta, half_life, kappa, sigma_spread,
                    vol_regime, vol_a, vol_b, coint_pvalue,
                    sentiment_modifier, sentiment_raw,
                    entry_z_threshold_used, signal_rationale,
                    generated_at, data_timestamp
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,
                (
                    signal_id, signal.pair_id, signal.module,
                    signal.asset_a, signal.asset_b, signal.direction,
                    signal.z_score, signal.beta, ou.half_life, ou.kappa, ou.sigma_spread,
                    signal.vol_regime, signal.vol_a, signal.vol_b, signal.coint_pvalue,
                    signal.sentiment_modifier,
                    signal.sentiment.raw_response if signal.sentiment else None,
                    signal.entry_z_threshold_used,
                    _serialize(signal.signal_rationale),
                    signal.generated_at.isoformat(),
                    signal.data_timestamp.isoformat(),
                ),
            )
        logger.debug("Inserted signal %s for %s (%s)", signal_id, signal.pair_id, signal.direction)

    # ── Risk decisions ────────────────────────────────────────────────────────

    def insert_risk_decision(self, decision_id: str, signal_id: str, decision: Any) -> None:
        """Write a RiskDecision to the risk_decisions table."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO risk_decisions (
                    id, signal_id, pair_id, module,
                    approved, rejection_rule, rejection_reason,
                    sized_order, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id, signal_id,
                    decision.pair_id, decision.module,
                    1 if decision.approved else 0,
                    decision.rejection_rule,
                    decision.rejection_reason,
                    _serialize(decision.sized_order) if decision.sized_order else None,
                    decision.checked_at.isoformat(),
                ),
            )

    # ── Orders ────────────────────────────────────────────────────────────────

    def insert_order(
        self,
        order_id: str,
        trade_id: str | None,
        pair_id: str,
        module: str,
        leg: str,
        order_type: str,
        symbol: str,
        side: str,
        qty: float,
    ) -> None:
        """Create an order record BEFORE placement (attempt logging)."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    id, trade_id, pair_id, module, leg, order_type,
                    symbol, side, qty, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (order_id, trade_id, pair_id, module, leg, order_type,
                 symbol, side, qty, _now_iso()),
            )

    def update_order_fill(
        self,
        order_id: str,
        alpaca_order_id: str,
        status: str,
        fill_price: float | None = None,
        fill_time: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update an order record after Alpaca responds."""
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE orders SET
                    alpaca_order_id = ?,
                    status = ?,
                    fill_price = ?,
                    fill_time = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (alpaca_order_id, status, fill_price, fill_time, error_message, order_id),
            )

    # ── Trades ────────────────────────────────────────────────────────────────

    def insert_trade(self, trade: Any) -> None:
        """Create an open trade record BEFORE order placement."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    id, pair_id, module, direction, status,
                    entry_z, entry_beta, entry_time,
                    leg_a_entry_order_id, leg_b_entry_order_id,
                    rationale_json
                ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id, trade.pair_id, trade.module, trade.direction,
                    trade.entry_z, trade.entry_beta, trade.entry_time.isoformat(),
                    trade.leg_a_entry_order_id, trade.leg_b_entry_order_id,
                    _serialize(trade.rationale_json),
                ),
            )
        logger.info(
            "Trade OPEN: %s | %s | %s | entry_z=%.2f",
            trade.id, trade.pair_id, trade.direction, trade.entry_z,
        )

    def close_trade(
        self,
        trade_id: str,
        exit_z: float,
        exit_reason: str,
        exit_time: datetime,
        realized_pnl_usd: float,
        realized_pnl_pct: float,
        holding_period_hours: float,
        leg_a_exit_order_id: str | None = None,
        leg_b_exit_order_id: str | None = None,
    ) -> None:
        """Mark a trade as closed with exit details."""
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    status = 'closed',
                    exit_z = ?, exit_reason = ?, exit_time = ?,
                    realized_pnl_usd = ?, realized_pnl_pct = ?,
                    holding_period_hours = ?,
                    leg_a_exit_order_id = ?, leg_b_exit_order_id = ?
                WHERE id = ?
                """,
                (
                    exit_z, exit_reason, exit_time.isoformat(),
                    realized_pnl_usd, realized_pnl_pct, holding_period_hours,
                    leg_a_exit_order_id, leg_b_exit_order_id,
                    trade_id,
                ),
            )
        logger.info(
            "Trade CLOSED: %s | reason=%s | pnl=%.2f (%.2f%%)",
            trade_id, exit_reason, realized_pnl_usd, realized_pnl_pct * 100,
        )

    def mark_trade_failed(self, trade_id: str, reason: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE trades SET status = 'failed', exit_reason = ? WHERE id = ?",
                (reason, trade_id),
            )

    # ── Reads (for pipeline logic and dashboard) ──────────────────────────────

    def get_open_trades(self, module: str | None = None) -> list[sqlite3.Row]:
        conn = self._get_connection()
        try:
            if module:
                return conn.execute(
                    "SELECT * FROM trades WHERE status = 'open' AND module = ?", (module,)
                ).fetchall()
            return conn.execute(
                "SELECT * FROM trades WHERE status = 'open'"
            ).fetchall()
        finally:
            conn.close()

    def get_rolling_pnl(self, hours: float = 24.0) -> float:
        """Sum of realized + unrealized P/L for trades closed in the last N hours."""
        since = datetime.utcnow().timestamp() - hours * 3600
        since_iso = datetime.utcfromtimestamp(since).isoformat()
        conn = self._get_connection()
        try:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(realized_pnl_usd), 0.0)
                FROM trades
                WHERE status = 'closed' AND exit_time >= ?
                """,
                (since_iso,),
            ).fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def get_closed_trades_today(self, utc_date: str) -> list[sqlite3.Row]:
        conn = self._get_connection()
        try:
            return conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' AND exit_time LIKE ?",
                (f"{utc_date}%",),
            ).fetchall()
        finally:
            conn.close()

    # ── Reflections ───────────────────────────────────────────────────────────

    def insert_reflection(self, reflection_id: str, data: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO reflections (
                    id, trade_id, utc_date, module, pair_id,
                    reflection_type,
                    outcome_summary, rule_alignment, ou_observation, notable_observation,
                    day_summary, risk_rejection_pattern, cointegration_health,
                    provider_used, generated_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    reflection_id,
                    data.get("trade_id"),
                    data["utc_date"],
                    data.get("module"),
                    data.get("pair_id"),
                    data["reflection_type"],
                    data.get("outcome_summary"),
                    data.get("rule_alignment"),
                    data.get("ou_observation"),
                    data.get("notable_observation"),
                    data.get("day_summary"),
                    data.get("risk_rejection_pattern"),
                    data.get("cointegration_health"),
                    data["provider_used"],
                    _now_iso(),
                ),
            )


def create_database(db_path: str | Path) -> Database:
    """Factory function — use this everywhere instead of the constructor directly."""
    return Database(db_path)
