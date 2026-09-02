"""reflection_agent.py — Nightly LLM review of closed trades and daily performance.

Runs once daily at midnight UTC:
  1. Reviews each trade closed during the preceding 24 hours.
  2. Generates an aggregate day summary across both modules.
  3. Writes reviews to the SQLite `reflections` table.

Safety constraint:
  A failure in the reflection agent NEVER blocks, alters, or delays the trading pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from src.llm_provider import LLMProvider, _clean_json_text
from src.models import DayReflection, TradeLogEntry, TradeReflection
from src.persistence.db import Database

logger = logging.getLogger(__name__)

TRADE_REFLECTION_SYSTEM_PROMPT = """You are a quantitative trading journal assistant reviewing a closed spread trade made
by a rules-based statistical arbitrage system. The system uses an Ornstein-Uhlenbeck
mean-reversion model with a Kalman-filtered hedge ratio.

Write a short, honest review. Assess whether the mathematical rules produced a sensible
outcome given what was knowable at entry time.

Respond with ONLY a JSON object, no other text:
{
  "outcome_summary": "<one sentence: what happened to the spread and how the position resolved>",
  "rule_alignment": "<one sentence: did the exit rule that fired make sense>",
  "ou_observation": "<one sentence: anything notable about the OU dynamics or 'nothing notable'>",
  "notable_observation": "<one sentence: anything else worth a human's attention or 'nothing notable'>"
}

Rules:
- Base your review only on the data provided. No hindsight bias.
- Do not suggest specific parameter changes. Keep it factual and concise.
"""

DAY_REFLECTION_SYSTEM_PROMPT = """You are a quantitative trading journal assistant summarizing a full trading day.

Respond with ONLY a JSON object:
{
  "day_summary": "<one sentence: summary of equity and crypto P/L and win counts>",
  "risk_rejection_pattern": "<one sentence: notable risk rejection patterns or 'no notable pattern'>",
  "cointegration_health": "<one sentence: overall health of cointegrated pairs>"
}
"""


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ReflectionAgent:
    """Nightly automated review engine for trade auditing."""

    def __init__(
        self,
        db: Database,
        llm_provider: LLMProvider | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.llm = llm_provider or LLMProvider()
        self.config = config if config is not None else load_config()

    def run_nightly_reflection(
        self,
        utc_date: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Execute nightly reflection over all trades closed on utc_date.

        Never raises — logs errors and returns summary dict.
        """
        now_utc = now or datetime.now(timezone.utc)
        date_str = utc_date or now_utc.strftime("%Y-%m-%d")

        logger.info("Starting nightly reflection for UTC date: %s", date_str)
        results: dict[str, Any] = {
            "utc_date": date_str,
            "trade_reflections": [],
            "day_reflection": None,
        }

        try:
            closed_trades_rows = self.db.get_closed_trades_today(date_str)
            logger.info("Found %d closed trades for %s", len(closed_trades_rows), date_str)

            # 1. Individual Trade Reflections
            for row in closed_trades_rows:
                trade_dict = dict(row)
                ref = self._reflect_on_trade(trade_dict, date_str)
                if ref:
                    results["trade_reflections"].append(ref)

            # 2. Aggregate Day Reflection (always generated)
            day_ref = self._reflect_on_day(closed_trades_rows, date_str)
            if day_ref:
                results["day_reflection"] = day_ref

        except Exception as exc:
            logger.error("Nightly reflection error: %s", exc, exc_info=True)

        return results

    def _reflect_on_trade(
        self,
        trade: dict[str, Any],
        utc_date: str,
    ) -> TradeReflection | None:
        trade_id = trade["id"]
        pair_id = trade["pair_id"]
        module = trade["module"]

        user_prompt = (
            f"Module: {module.title()}\n"
            f"Pair: {pair_id}\n"
            f"Direction: {trade['direction']}\n"
            f"Entry z-score: {trade.get('entry_z', 'N/A')}\n"
            f"Exit z-score: {trade.get('exit_z', 'N/A')}\n"
            f"Exit reason: {trade.get('exit_reason', 'N/A')}\n"
            f"Realized P/L: ${trade.get('realized_pnl_usd', 0.0):.2f} ({trade.get('realized_pnl_pct', 0.0)*100:.2f}%)\n"
            f"Holding period: {trade.get('holding_period_hours', 0.0):.1f} hours\n"
        )

        raw_text, provider = self.llm.generate_completion(
            system_prompt=TRADE_REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        if provider == "none" or not raw_text:
            return None

        try:
            cleaned = _clean_json_text(raw_text)
            data = json.loads(cleaned)

            ref_id = str(uuid4())
            db_payload = {
                "trade_id": trade_id,
                "utc_date": utc_date,
                "module": module,
                "pair_id": pair_id,
                "reflection_type": "trade",
                "outcome_summary": data.get("outcome_summary", "Trade completed."),
                "rule_alignment": data.get("rule_alignment", "Exit followed rules."),
                "ou_observation": data.get("ou_observation", "nothing notable"),
                "notable_observation": data.get("notable_observation", "nothing notable"),
                "provider_used": provider,
            }
            self.db.insert_reflection(ref_id, db_payload)

            return TradeReflection(
                trade_id=trade_id,
                pair_id=pair_id,
                module=module,
                outcome_summary=db_payload["outcome_summary"],
                rule_alignment=db_payload["rule_alignment"],
                ou_observation=db_payload["ou_observation"],
                notable_observation=db_payload["notable_observation"],
                provider_used=provider,
            )
        except Exception as exc:
            logger.warning("Failed to parse trade reflection for %s: %s", trade_id, exc)
            return None

    def _reflect_on_day(
        self,
        trades_rows: list[Any],
        utc_date: str,
    ) -> DayReflection | None:
        total_trades = len(trades_rows)
        total_pnl = sum(float(r["realized_pnl_usd"] or 0.0) for r in trades_rows)
        wins = sum(1 for r in trades_rows if float(r["realized_pnl_usd"] or 0.0) > 0)
        losses = total_trades - wins

        user_prompt = (
            f"UTC Date: {utc_date}\n"
            f"Total Trades Closed: {total_trades} (Wins: {wins}, Losses: {losses})\n"
            f"Total Realized P/L: ${total_pnl:.2f}\n"
        )

        raw_text, provider = self.llm.generate_completion(
            system_prompt=DAY_REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        if provider == "none" or not raw_text:
            return None

        try:
            cleaned = _clean_json_text(raw_text)
            data = json.loads(cleaned)

            ref_id = str(uuid4())
            db_payload = {
                "trade_id": None,
                "utc_date": utc_date,
                "module": None,
                "pair_id": None,
                "reflection_type": "day",
                "day_summary": data.get("day_summary", f"Total P/L: ${total_pnl:.2f} across {total_trades} trades."),
                "risk_rejection_pattern": data.get("risk_rejection_pattern", "no notable pattern"),
                "cointegration_health": data.get("cointegration_health", "pairs healthy"),
                "provider_used": provider,
            }
            self.db.insert_reflection(ref_id, db_payload)

            return DayReflection(
                utc_date=utc_date,
                day_summary=db_payload["day_summary"],
                risk_rejection_pattern=db_payload["risk_rejection_pattern"],
                cointegration_health=db_payload["cointegration_health"],
                provider_used=provider,
            )
        except Exception as exc:
            logger.warning("Failed to parse day reflection: %s", exc)
            return None
