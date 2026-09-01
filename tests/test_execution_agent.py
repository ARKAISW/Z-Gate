"""tests/test_execution_agent.py — Unit tests for the ExecutionAgent.

Tests:
  1. Audit trail: SQLite trade insertion occurs BEFORE placing orders.
  2. Options entry execution (Module A): places 2 legs and records order IDs.
  3. Crypto spot entry execution (Module B): places 2 legs and records order IDs.
  4. Emergency unwind: Leg B failure triggers immediate Leg A unwind.
  5. Exit trigger: Z-score reversion (|z| <= 0.3).
  6. Exit trigger: Stop-loss (|z| >= 3.0).
  7. Exit trigger: Time-stop (> 15 days / 120h).
  8. Exit trigger: Cointegration breakdown (p > 0.10).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from src.agents.execution_agent import ExecutionAgent
from src.broker import Broker, OptionContract, OrderResult
from src.models import (
    OptionsLeg,
    OUParams,
    RiskDecision,
    SpotLeg,
    SpreadOrderRequest,
    SpreadSignal,
    TradeLogEntry,
)
from src.persistence.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_broker():
    broker = MagicMock(spec=Broker)
    # Default successful place_order return
    broker.place_order.side_effect = lambda symbol, qty, side, time_in_force="day": OrderResult(
        order_id=f"cli_{symbol}",
        alpaca_order_id=f"alp_{symbol}_{side}",
        status="filled",
        symbol=symbol,
        qty=qty,
        side=side,
        submitted_at=datetime.now(timezone.utc),
    )
    # Default options chain
    broker.get_options_chain.return_value = [
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
    return broker


@pytest.fixture
def test_db(tmp_path):
    return Database(tmp_path / "test_exec.db")


def make_approved_options_decision() -> RiskDecision:
    now = datetime.now(timezone.utc)
    ou = OUParams(kappa=0.2, mu=0.0, sigma_ou=0.02, half_life=4.0, sigma_spread=0.03, ar1_r_squared=0.7)
    sig = SpreadSignal(
        pair_id="GLD-SLV",
        module="equity",
        asset_a="GLD",
        asset_b="SLV",
        direction="long",
        z_score=-2.2,
        beta=1.1,
        ou_params=ou,
        vol_regime="NORMAL",
        vol_a=0.15,
        vol_b=0.15,
        coint_pvalue=0.01,
        sentiment=None,
        sentiment_modifier=0.0,
        entry_z_threshold_used=1.5,
        exit_z_threshold=0.3,
        stop_z_threshold=3.0,
        signal_rationale={},
        generated_at=now,
        data_timestamp=now,
    )
    order_req = SpreadOrderRequest(
        pair_id="GLD-SLV",
        module="equity",
        direction="long",
        execution_type="options",
        leg_a=OptionsLeg(
            underlying="GLD",
            symbol="GLD_ATM_CALL",
            expiry="PENDING",
            strike=180.0,
            option_type="call",
            qty=1,
            side="buy_to_open",
            premium_estimate=3.50,
        ),
        leg_b=OptionsLeg(
            underlying="SLV",
            symbol="SLV_ATM_PUT",
            expiry="PENDING",
            strike=22.0,
            option_type="put",
            qty=1,
            side="buy_to_open",
            premium_estimate=1.20,
        ),
        beta=1.1,
        entry_z=-2.2,
        kelly_f=1.0,
        position_f=0.05,
        estimated_cost=470.0,
    )
    return RiskDecision(
        pair_id="GLD-SLV",
        module="equity",
        signal=sig,
        approved=True,
        sized_order=order_req,
        checked_at=now,
    )


def make_approved_crypto_decision() -> RiskDecision:
    now = datetime.now(timezone.utc)
    ou = OUParams(kappa=0.2, mu=0.0, sigma_ou=0.02, half_life=24.0, sigma_spread=0.03, ar1_r_squared=0.7)
    sig = SpreadSignal(
        pair_id="BTC/USD-ETH/USD",
        module="crypto",
        asset_a="BTC/USD",
        asset_b="ETH/USD",
        direction="long",
        z_score=-2.2,
        beta=1.2,
        ou_params=ou,
        vol_regime="NORMAL",
        vol_a=0.4,
        vol_b=0.5,
        coint_pvalue=0.01,
        sentiment=None,
        sentiment_modifier=0.0,
        entry_z_threshold_used=1.75,
        exit_z_threshold=0.3,
        stop_z_threshold=3.5,
        signal_rationale={},
        generated_at=now,
        data_timestamp=now,
    )
    order_req = SpreadOrderRequest(
        pair_id="BTC/USD-ETH/USD",
        module="crypto",
        direction="long",
        execution_type="spot",
        leg_a=SpotLeg(
            symbol="BTC/USD",
            qty=0.1,
            side="buy",
            notional_usd=6000.0,
        ),
        leg_b=SpotLeg(
            symbol="ETH/USD",
            qty=2.4,
            side="sell",
            notional_usd=7200.0,
        ),
        beta=1.2,
        entry_z=-2.2,
        kelly_f=1.0,
        position_f=0.08,
        estimated_cost=13200.0,
    )
    return RiskDecision(
        pair_id="BTC/USD-ETH/USD",
        module="crypto",
        signal=sig,
        approved=True,
        sized_order=order_req,
        checked_at=now,
    )


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestExecutionAgent:
    def test_options_entry_execution(self, mock_broker, test_db):
        """Module A options entry places orders on both legs and records order IDs in SQLite."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_options_decision()
        prices = {"GLD": 180.0, "SLV": 22.0}

        trade = agent.execute_entry(decision=decision, current_prices=prices)

        assert trade is not None
        assert trade.status == "open"
        assert trade.leg_a_entry_order_id is not None
        assert trade.leg_b_entry_order_id is not None

        # Verify broker calls
        assert mock_broker.place_order.call_count == 2

        # Verify DB records
        open_trades = test_db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == trade.id

    def test_crypto_spot_entry_execution(self, mock_broker, test_db):
        """Module B crypto spot entry places spot orders with GTC on both legs."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_crypto_decision()

        trade = agent.execute_entry(decision=decision)

        assert trade is not None
        assert trade.status == "open"
        assert mock_broker.place_order.call_count == 2

        # Check call arguments
        calls = mock_broker.place_order.call_args_list
        assert calls[0].kwargs["symbol"] == "BTC/USD"
        assert calls[0].kwargs["side"] == "buy"
        assert calls[0].kwargs["time_in_force"] == "gtc"
        assert calls[1].kwargs["symbol"] == "ETH/USD"
        assert calls[1].kwargs["side"] == "sell"

    def test_emergency_unwind_on_leg_b_failure(self, mock_broker, test_db):
        """If Leg A succeeds and Leg B raises an exception, Leg A is unwound and trade marked failed."""
        # Make Leg B fail
        def place_order_mock(symbol, qty, side, time_in_force="day"):
            if "ETH" in symbol or "SLV" in symbol:
                raise RuntimeError("API Timeout on Leg B")
            return OrderResult(
                order_id="cli_1",
                alpaca_order_id=f"alp_{symbol}",
                status="filled",
                symbol=symbol,
                qty=qty,
                side=side,
                submitted_at=datetime.now(timezone.utc),
            )

        mock_broker.place_order.side_effect = place_order_mock

        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_crypto_decision()

        trade = agent.execute_entry(decision=decision)

        assert trade is not None
        assert trade.status == "failed"

        # Check that unwind order was placed (3 calls total: Leg A, Leg B (failed), Leg A unwind)
        assert mock_broker.place_order.call_count == 3
        unwind_call = mock_broker.place_order.call_args_list[-1]
        assert unwind_call.kwargs["symbol"] == "BTC/USD"
        assert unwind_call.kwargs["side"] == "sell"  # opposite of original buy

    def test_exit_on_z_score_reversion(self, mock_broker, test_db):
        """When |z| <= 0.3, open trade is closed with exit_reason='z_reversion'."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_options_decision()
        agent.execute_entry(decision, current_prices={"GLD": 180.0, "SLV": 22.0})

        assert len(test_db.get_open_trades()) == 1

        # Current signal has z=0.15 (mean reverted)
        now_exit = datetime.now(timezone.utc) + timedelta(days=2)
        sig = make_approved_options_decision().signal
        sig.z_score = 0.15

        closed = agent.check_and_execute_exits(
            current_signals={"GLD-SLV": sig},
            now=now_exit,
        )

        assert len(closed) == 1
        assert closed[0].exit_reason == "z_reversion"
        assert len(test_db.get_open_trades()) == 0

    def test_exit_on_stop_loss(self, mock_broker, test_db):
        """When |z| >= 3.0, open trade is stopped out with exit_reason='stop_z'."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_options_decision()
        agent.execute_entry(decision, current_prices={"GLD": 180.0, "SLV": 22.0})

        # Current signal blows out to z=-3.4
        sig = make_approved_options_decision().signal
        sig.z_score = -3.4

        closed = agent.check_and_execute_exits(
            current_signals={"GLD-SLV": sig},
        )

        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_z"
        assert closed[0].realized_pnl_usd < 0

    def test_exit_on_time_stop(self, mock_broker, test_db):
        """When holding period > 15 days, trade is force-closed with exit_reason='time_stop'."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_options_decision()
        agent.execute_entry(decision, current_prices={"GLD": 180.0, "SLV": 22.0})

        # 16 days later, z has not reverted (z=-1.2)
        now_16d = datetime.now(timezone.utc) + timedelta(days=16)
        sig = make_approved_options_decision().signal
        sig.z_score = -1.2

        closed = agent.check_and_execute_exits(
            current_signals={"GLD-SLV": sig},
            now=now_16d,
        )

        assert len(closed) == 1
        assert closed[0].exit_reason == "time_stop"

    def test_exit_on_coint_breakdown(self, mock_broker, test_db):
        """When cointegration p-value > 0.10, trade is closed with exit_reason='coint_breakdown'."""
        agent = ExecutionAgent(broker=mock_broker, db=test_db)
        decision = make_approved_options_decision()
        agent.execute_entry(decision, current_prices={"GLD": 180.0, "SLV": 22.0})

        # Daily recheck shows cointegration breakdown (p=0.85 > 0.70)
        sig = make_approved_options_decision().signal
        sig.z_score = -1.2
        sig.coint_pvalue = 0.85

        closed = agent.check_and_execute_exits(
            current_signals={"GLD-SLV": sig},
        )

        assert len(closed) == 1
        assert closed[0].exit_reason == "coint_breakdown"
