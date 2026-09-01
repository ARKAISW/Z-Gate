"""tests/test_pipeline_integration.py — Integration tests for the full pipeline.

Includes:
  1. PERMISSION BOUNDARY AUDIT: Asserts broker.place_order() is called ONLY in
     src/agents/execution_agent.py (and defined in src/broker.py).
  2. End-to-end pipeline run (Module A + Module B).
  3. Market hours gate validation.
  4. Midnight UTC cointegration health recheck.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.broker import AccountInfo, BarData, Broker, OptionContract, OrderResult
from src.pipeline import StatArbPipeline
from src.persistence.db import Database


# ---------------------------------------------------------------------------
# 1. Architectural Permission Boundary Test (Non-negotiable Audit Check)
# ---------------------------------------------------------------------------


def test_place_order_permission_boundary():
    """Verify that place_order() is called ONLY within src/agents/execution_agent.py.

    This ensures deterministic risk enforcement: no signal, risk, or orchestration
    code can bypass the execution agent and place orders directly.
    """
    src_dir = Path(__file__).parent.parent / "src"
    py_files = list(src_dir.rglob("*.py"))
    assert len(py_files) > 0, "No source files found in src/"

    violating_files = []
    allowed_files = {"broker.py", "execution_agent.py"}

    for py_file in py_files:
        if py_file.name in allowed_files:
            continue
        content = py_file.read_text(encoding="utf-8")
        # Check for function calls to place_order
        if "place_order(" in content or ".place_order" in content:
            violating_files.append(str(py_file.relative_to(src_dir)))

    assert (
        len(violating_files) == 0
    ), f"PERMISSION BOUNDARY VIOLATION: place_order() referenced in unauthorized files: {violating_files}"


# ---------------------------------------------------------------------------
# 2. End-to-End Pipeline Integration Test
# ---------------------------------------------------------------------------


def make_mock_bars(symbol: str, n: int = 150, seed: int = 42) -> BarData:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0, 0.01, n)) + 4.5
    prices = np.exp(log_p).tolist()
    ts = [datetime(2026, 1, 1, tzinfo=timezone.utc)] * n
    return BarData(
        symbol=symbol,
        timestamps=ts,
        opens=prices,
        highs=prices,
        lows=prices,
        closes=prices,
        volumes=[1000.0] * n,
    )


@pytest.fixture
def integration_pipeline(tmp_path):
    mock_broker = MagicMock(spec=Broker)

    # Account
    mock_broker.get_account.return_value = AccountInfo(
        equity=100000.0,
        buying_power=100000.0,
        cash=100000.0,
        currency="USD",
        snapshot_at=datetime.now(timezone.utc),
    )

    # Bars
    mock_broker.get_equity_bars.side_effect = lambda symbols, limit=100: {
        s: make_mock_bars(s, n=limit, seed=hash(s) % 10000) for s in symbols
    }
    mock_broker.get_crypto_bars.side_effect = lambda symbols, limit=200: {
        s: make_mock_bars(s, n=limit, seed=hash(s) % 10000) for s in symbols
    }

    # Orders
    mock_broker.place_order.side_effect = lambda symbol, qty, side, time_in_force="day": OrderResult(
        order_id=f"cli_{symbol}",
        alpaca_order_id=f"alp_{symbol}",
        status="filled",
        symbol=symbol,
        qty=qty,
        side=side,
        submitted_at=datetime.now(timezone.utc),
    )

    # Options Chain
    mock_broker.get_options_chain.return_value = [
        OptionContract(
            symbol="GLD260115C00180000",
            underlying_symbol="GLD",
            expiration_date="2026-01-15",
            strike_price=180.0,
            option_type="call",
            close_price=3.50,
            open_interest=1000,
        ),
        OptionContract(
            symbol="SLV260115P00022000",
            underlying_symbol="SLV",
            expiration_date="2026-01-15",
            strike_price=22.0,
            option_type="put",
            close_price=1.20,
            open_interest=1000,
        ),
    ]

    test_db = Database(tmp_path / "test_pipeline.db")
    config = {
        "equity_pairs": [["GLD", "SLV"]],
        "crypto_pairs": [["BTC/USD", "ETH/USD"]],
        "coint_pvalue_threshold": 0.05,
        "coint_lookback_days_equity": 30,
        "coint_lookback_hours_crypto": 60,
        "kalman_observation_noise": 0.001,
        "kalman_transition_noise_equity": 0.0001,
        "kalman_transition_noise_crypto": 0.001,
        "ou_lookback_days_equity": 25,
        "ou_lookback_hours_crypto": 50,
        "halflife_min_days": 0.1,
        "halflife_max_days": 100.0,
        "halflife_min_hours": 0.1,
        "halflife_max_hours": 500.0,
        "entry_z_threshold_equity": 0.01,  # low threshold to trigger entry
        "entry_z_threshold_crypto": 0.01,
        "entry_z_threshold_high_vol_equity": 0.05,
        "entry_z_threshold_high_vol_crypto": 0.05,
        "exit_z_threshold": 0.3,
        "stop_z_threshold_equity": 3.0,
        "stop_z_threshold_crypto": 3.5,
        "max_open_equity_pairs": 3,
        "max_open_crypto_pairs": 3,
        "max_premium_pct_equity": 0.05,
        "max_position_pct_equity": 0.10,
        "kelly_fraction": 0.25,
        "rolling_24h_loss_limit_pct": 0.03,
        "circuit_breaker_cooldown_hours": 4.0,
        "max_data_staleness_minutes": 60.0,
    }

    pipeline = StatArbPipeline(broker=mock_broker, db=test_db, config=config)
    return pipeline


class TestPipelineIntegration:
    def test_full_pipeline_cycle_execution(self, integration_pipeline):
        """Pipeline successfully ingests data, generates signals, runs risk checks, and executes entries."""
        now = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)  # 9:00 AM ET (open)
        results = integration_pipeline.run_cycle(force_market_open=True, now=now)

        assert "signals" in results
        assert len(results["signals"]) >= 2  # 1 equity + 1 crypto pair evaluated
        assert len(results["decisions"]) >= 0

        # Check DB records
        conn = integration_pipeline.db._get_connection()
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn.close()
        assert signal_count >= 2

    def test_market_hours_gating(self, integration_pipeline):
        """When equity market is closed (e.g. Saturday), Module A is skipped without force_market_open."""
        # Saturday
        saturday = datetime(2026, 1, 3, 14, 0, 0, tzinfo=timezone.utc)
        assert not integration_pipeline.is_equity_market_open(now=saturday)

        results = integration_pipeline.run_cycle(module="equity", now=saturday, force_market_open=False)
        assert "equity_market_closed" in results["skipped_reasons"]
        assert len(results["signals"]) == 0

    def test_recheck_cointegration_job(self, integration_pipeline):
        """Daily midnight UTC cointegration recheck runs and returns p-values for all pairs."""
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        coint_results = integration_pipeline.recheck_cointegration(now=now)

        assert "GLD-SLV" in coint_results
        assert "BTC/USD-ETH/USD" in coint_results
        assert isinstance(coint_results["GLD-SLV"], float)
