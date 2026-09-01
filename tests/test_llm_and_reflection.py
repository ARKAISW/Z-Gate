"""tests/test_llm_and_reflection.py — Unit tests for LLMProvider and ReflectionAgent.

Tests:
  1. Fallback chain: Featherless -> Groq -> Ollama -> none.
  2. Graceful degradation when all providers fail (modifier = 0, no exception).
  3. Sentiment parsing: clean JSON, markdown fenced JSON, malformed text.
  4. ReflectionAgent: trade reviews and day summary written to SQLite.
  5. ReflectionAgent: zero closed trades handled cleanly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from src.agents.reflection_agent import ReflectionAgent
from src.llm_provider import LLMProvider, _clean_json_text, summarize_sentiment
from src.models import SentimentResult, TradeLogEntry
from src.persistence.db import Database


# ---------------------------------------------------------------------------
# Test LLMProvider & Fallback Chain
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_clean_json_text_markdown_fences(self):
        """Markdown fences (```json ... ```) are cleanly stripped."""
        raw = '```json\n{"sentiment": "positive", "confidence": 0.85, "reason": "strong earnings"}\n```'
        cleaned = _clean_json_text(raw)
        assert cleaned == '{"sentiment": "positive", "confidence": 0.85, "reason": "strong earnings"}'

    def test_fallback_chain_sequence(self):
        """Simulate Featherless failure -> Groq called -> Groq returns output."""
        provider = LLMProvider(primary_provider="featherless")

        def mock_call(p, sys_prompt, user_prompt, temp):
            if p == "featherless":
                raise ConnectionError("Featherless API unreachable")
            if p == "groq":
                return '{"sentiment": "positive", "confidence": 0.8, "reason": "good news"}'
            return ""

        with patch.object(provider, "_call_provider", side_effect=mock_call):
            res = provider.summarize_sentiment("GLD", ["Gold prices reach all-time high"])

        assert res.sentiment == "positive"
        assert res.confidence == 0.8
        assert res.provider_used == "groq"
        assert res.modifier == pytest.approx(0.12)

    def test_total_fallback_to_none_never_raises(self):
        """When all LLM backends fail, summarize_sentiment returns neutral with modifier=0.0."""
        provider = LLMProvider(primary_provider="featherless")

        with patch.object(provider, "_call_provider", side_effect=RuntimeError("All APIs down")):
            res = provider.summarize_sentiment("BTC", ["BTC drops 5%"])

        assert isinstance(res, SentimentResult)
        assert res.sentiment == "neutral"
        assert res.confidence == 0.0
        assert res.modifier == 0.0
        assert res.provider_used == "none"

    def test_malformed_json_fallback(self):
        """If model returns non-JSON text, provider returns neutral with modifier=0.0."""
        provider = LLMProvider()

        with patch.object(provider, "_call_provider", return_value="I am an AI, not JSON!"):
            res = provider.summarize_sentiment("ETH", ["ETH upgrade"])

        assert res.sentiment == "neutral"
        assert res.modifier == 0.0
        assert res.reason == "Sentiment parse error"


# ---------------------------------------------------------------------------
# Test ReflectionAgent
# ---------------------------------------------------------------------------


class TestReflectionAgent:
    def test_nightly_reflection_flow(self, tmp_path):
        """ReflectionAgent reviews closed trades today and inserts entries into reflections table."""
        db = Database(tmp_path / "test_reflection.db")
        mock_llm = MagicMock(spec=LLMProvider)

        mock_llm.generate_completion.return_value = (
            '{"outcome_summary": "Spread reverted to mean cleanly.", '
            '"rule_alignment": "Z-reversion exit fired at +0.15σ.", '
            '"ou_observation": "Half-life was 3.5 days as predicted.", '
            '"notable_observation": "Both options legs filled cleanly.", '
            '"day_summary": "Net +$450 on 1 equity trade.", '
            '"risk_rejection_pattern": "no notable pattern", '
            '"cointegration_health": "all pairs passed recheck"}',
            "featherless",
        )

        agent = ReflectionAgent(db=db, llm_provider=mock_llm)

        # Insert a closed trade
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        trade = TradeLogEntry(
            id="trade_123",
            pair_id="GLD-SLV",
            module="equity",
            direction="long",
            status="open",
            entry_z=-2.2,
            entry_beta=1.1,
            entry_time=now,
        )
        db.insert_trade(trade)
        db.close_trade(
            trade_id="trade_123",
            exit_z=0.15,
            exit_reason="z_reversion",
            exit_time=now,
            realized_pnl_usd=450.0,
            realized_pnl_pct=0.045,
            holding_period_hours=48.0,
        )

        results = agent.run_nightly_reflection(utc_date=today_str, now=now)

        assert len(results["trade_reflections"]) == 1
        assert results["day_reflection"] is not None
        assert results["trade_reflections"][0].trade_id == "trade_123"

        # Check SQLite table
        conn = db._get_connection()
        ref_rows = conn.execute("SELECT * FROM reflections").fetchall()
        conn.close()
        assert len(ref_rows) == 2  # 1 trade reflection + 1 day reflection

    def test_reflection_with_no_closed_trades(self, tmp_path):
        """Reflection runs cleanly when no trades closed today."""
        db = Database(tmp_path / "test_reflection_empty.db")
        agent = ReflectionAgent(db=db)

        results = agent.run_nightly_reflection(utc_date="2026-01-01")

        assert len(results["trade_reflections"]) == 0
        assert results["day_reflection"] is None
