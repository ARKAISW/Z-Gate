"""tests/test_signal_agent.py — Unit tests for the SignalAgent pipeline.

Tests:
  1. Cointegration gate rejection
  2. Long spread signal generation (z < -entry_z)
  3. Short spread signal generation (z > +entry_z)
  4. Half-life gate rejection
  5. Volatility regime adjustments (HIGH widens threshold, EXTREME blocks entry)
  6. Sentiment modifier adjustment
  7. SQLite database persistence
  8. Insufficient bar handling
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pytest

from src.agents.signal_agent import SignalAgent
from src.broker import BarData
from src.models import SentimentResult, SpreadSignal
from src.persistence.db import Database


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_clean_pair(
    n: int = 150,
    seed: int = 42,
    ar_phi: float = 0.70,
    noise_scale: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cointegrated prices A and B with stationary AR(1) spread."""
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, noise_scale, n)) + 4.5
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = ar_phi * spread[i - 1] + rng.normal(0, noise_scale)

    log_a = 1.0 * log_b + spread
    return np.exp(log_a), np.exp(log_b)


def make_random_walk_pair(n: int = 150, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent random walks (non-cointegrated)."""
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.normal(0, 0.02, n)) + 4.5
    log_b = np.cumsum(rng.normal(0, 0.02, n)) + 4.5
    return np.exp(log_a), np.exp(log_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> dict:
    return {
        "coint_pvalue_threshold": 0.05,
        "coint_lookback_days_equity": 90,
        "coint_lookback_hours_crypto": 720,
        "kalman_observation_noise": 0.001,
        "kalman_transition_noise_equity": 0.0001,
        "kalman_transition_noise_crypto": 0.001,
        "ou_lookback_days_equity": 30,
        "ou_lookback_hours_crypto": 168,
        "halflife_min_days": 1.0,
        "halflife_max_days": 30.0,
        "halflife_min_hours": 2.0,
        "halflife_max_hours": 100.0,
        "entry_z_threshold_equity": 1.5,
        "entry_z_threshold_crypto": 1.75,
        "entry_z_threshold_high_vol_equity": 2.0,
        "entry_z_threshold_high_vol_crypto": 2.25,
        "exit_z_threshold": 0.3,
        "stop_z_threshold_equity": 3.0,
        "stop_z_threshold_crypto": 3.5,
        "high_vol_threshold_equity": 0.30,
        "extreme_vol_threshold_equity": 0.60,
        "high_vol_threshold_crypto": 0.80,
        "extreme_vol_threshold_crypto": 1.20,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalAgent:
    def test_cointegration_gate_rejection(self, mock_config):
        """Non-cointegrated pair must fail the coint gate and return direction='none'."""
        agent = SignalAgent(config=mock_config)
        p_a, p_b = make_random_walk_pair(n=120, seed=999)

        signal = agent.evaluate_pair(
            asset_a="FAKE1",
            asset_b="FAKE2",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert isinstance(signal, SpreadSignal)
        # Should be rejected due to high coint p-value
        assert signal.direction == "none"
        assert signal.coint_pvalue > mock_config["coint_pvalue_threshold"]
        assert not signal.signal_rationale["coint_passed"]

    def test_long_spread_signal(self, mock_config):
        """When asset A is cheap relative to B, direction is 'long'."""
        agent = SignalAgent(config=mock_config)
        # Seed 264 generates z=-1.76 (< -1.5) with coint=0.0065 (< 0.05)
        p_a, p_b = make_clean_pair(n=200, seed=264, ar_phi=0.65, noise_scale=0.008)

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert signal.coint_pvalue < 0.05
        assert signal.z_score < -mock_config["entry_z_threshold_equity"]
        assert signal.direction == "long"
        assert signal.signal_rationale["final_direction"] == "long"

    def test_short_spread_signal(self, mock_config):
        """When asset A is rich relative to B, direction is 'short'."""
        agent = SignalAgent(config=mock_config)
        # Seed 45 generates z=1.71 (> 1.5) with coint=0.0005 (< 0.05)
        p_a, p_b = make_clean_pair(n=200, seed=45, ar_phi=0.70, noise_scale=0.008)

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert signal.coint_pvalue < 0.05
        assert signal.z_score > mock_config["entry_z_threshold_equity"]
        assert signal.direction == "short"
        assert signal.signal_rationale["final_direction"] == "short"

    def test_halflife_gate_rejection(self, mock_config):
        """Half-life outside bounds is flagged in rationale but signal still passes.

        The signal agent no longer hard-blocks on half-life — that responsibility
        moved to the risk agent. Signal agent just reports halflife_passed=False.
        """
        strict_config = dict(mock_config)
        # Force a minimum half life of 50 days (our clean pair has ~2-4 days)
        strict_config["halflife_min_days"] = 50.0

        agent = SignalAgent(config=strict_config)
        p_a, p_b = make_clean_pair(n=200, seed=264, ar_phi=0.65, noise_scale=0.008)

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        # Signal passes through (direction != none) because signal agent is permissive
        assert signal.direction != "none"
        # But halflife_passed is flagged false for risk agent to act on
        assert not signal.signal_rationale["halflife_passed"]

    def test_vol_regime_high_widens_threshold(self, mock_config):
        """In HIGH vol regime, entry threshold widens to entry_z_threshold_high_vol."""
        agent = SignalAgent(config=mock_config)
        p_a, p_b = make_clean_pair(n=200, seed=264, ar_phi=0.65, noise_scale=0.008)

        # Inject moderate volatility into asset A's recent prices (e.g. 40% annualized vol)
        p_a[-11:] = p_a[-11:] * np.array([1.0, 1.03, 0.97, 1.04, 0.96, 1.03, 0.97, 1.04, 0.96, 1.03, 0.96])

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert signal.vol_regime in ("HIGH", "EXTREME")
        if signal.vol_regime == "HIGH":
            assert signal.entry_z_threshold_used >= mock_config["entry_z_threshold_high_vol_equity"]

    def test_vol_regime_extreme_blocks_entry(self, mock_config):
        """In EXTREME vol regime (>60% for equity), entries are completely blocked."""
        agent = SignalAgent(config=mock_config)
        p_a, p_b = make_clean_pair(n=200, seed=264, ar_phi=0.65, noise_scale=0.008)

        # Inject extreme volatility (>100% annualized)
        p_a[-11:] = p_a[-11:] * np.array([1.0, 1.15, 0.85, 1.20, 0.80, 1.15, 0.85, 1.20, 0.80, 1.15, 0.80])

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert signal.vol_regime == "EXTREME"
        assert signal.direction == "none"
        assert any("extreme_vol_regime_block" in r for r in signal.signal_rationale["rejection_reasons"])

    def test_sentiment_modifier_nudge(self, mock_config):
        """Negative sentiment on base asset when spread is LONG nudges threshold higher."""
        agent = SignalAgent(config=mock_config)
        p_a, p_b = make_clean_pair(n=200, seed=264, ar_phi=0.65, noise_scale=0.008)

        sentiment_neg = SentimentResult(
            asset="GLD",
            sentiment="negative",
            confidence=0.8,
            reason="Gold ETF seeing outflows",
            modifier=0.12,
            provider_used="featherless",
        )

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
            sentiment=sentiment_neg,
        )

        expected_modifier = 0.15 * 0.8  # 0.12
        assert signal.sentiment_modifier == pytest.approx(expected_modifier)
        assert signal.entry_z_threshold_used == pytest.approx(1.5 + expected_modifier)

    def test_insufficient_bars(self, mock_config):
        """SignalAgent cleanly handles fewer than 20 bars."""
        agent = SignalAgent(config=mock_config)
        p_a = np.array([100.0, 101.0, 102.0])
        p_b = np.array([50.0, 51.0, 52.0])

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        assert signal.direction == "none"
        assert "Insufficient bars" in signal.signal_rationale["reason"]

    def test_bardata_input_compatibility(self, mock_config):
        """SignalAgent accepts BarData containers with timestamps."""
        agent = SignalAgent(config=mock_config)
        p_a, p_b = make_clean_pair(n=120, seed=42)

        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 120
        bd_a = BarData(
            symbol="GLD",
            timestamps=ts,
            opens=p_a.tolist(),
            highs=p_a.tolist(),
            lows=p_a.tolist(),
            closes=p_a.tolist(),
            volumes=[1000.0] * 120,
        )
        bd_b = BarData(
            symbol="SLV",
            timestamps=ts,
            opens=p_b.tolist(),
            highs=p_b.tolist(),
            lows=p_b.tolist(),
            closes=p_b.tolist(),
            volumes=[1000.0] * 120,
        )

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=bd_a,
            prices_b=bd_b,
        )

        assert isinstance(signal, SpreadSignal)
        assert signal.pair_id == "GLD-SLV"

    def test_database_logging(self, tmp_path, mock_config):
        """When Database is supplied, signals are persisted to SQLite."""
        db_file = tmp_path / "test_trades.db"
        db = Database(db_file)

        agent = SignalAgent(config=mock_config, db=db)
        p_a, p_b = make_clean_pair(n=120, seed=42)

        signal = agent.evaluate_pair(
            asset_a="GLD",
            asset_b="SLV",
            module="equity",
            prices_a=p_a,
            prices_b=p_b,
        )

        # Query database to confirm insertion
        conn = db._get_connection()
        rows = conn.execute("SELECT * FROM signals WHERE pair_id = 'GLD-SLV'").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["pair_id"] == "GLD-SLV"
        assert rows[0]["module"] == "equity"
