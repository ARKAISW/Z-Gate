"""tests/test_risk_agent.py — Unit tests for the deterministic RiskAgent.

Tests:
  1. Rule 1: Max open pairs per module
  2. Rule 2: Duplicate pair check
  3. Rule 3: Half-life gate re-validation
  4. Rule 4: Cointegration p-value re-validation
  5. Rule 5: Volatility regime block (EXTREME)
  6. Rule 6: Sizing cap check
  7. Rule 7: Buying power check
  8. Rule 8: 24h loss circuit breaker & cooldown window
  9. Rule 9: Data freshness check
  10. Valid approval & sizing (equity options module)
  11. Valid approval & sizing (crypto spot module)
  12. Fail-closed behavior on exception
  13. SQLite database audit logging
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal
import pytest

from src.agents.risk_agent import RiskAgent
from src.broker import AccountInfo
from src.models import (
    AccountSnapshot,
    OUParams,
    RiskDecision,
    SpreadOrderRequest,
    SpreadSignal,
    TradeLogEntry,
)
from src.persistence.db import Database


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_test_signal(
    pair_id: str = "GLD-SLV",
    module: Literal["equity", "crypto"] = "equity",
    direction: Literal["long", "short", "none"] = "long",
    z_score: float = -2.0,
    half_life: float = 4.0,
    coint_pvalue: float = 0.01,
    vol_regime: Literal["NORMAL", "HIGH", "EXTREME"] = "NORMAL",
    timestamp: datetime | None = None,
) -> SpreadSignal:
    now = datetime.now(timezone.utc)
    ts = timestamp or now
    ou = OUParams(
        kappa=0.2,
        mu=0.0,
        sigma_ou=0.02,
        half_life=half_life,
        sigma_spread=0.03,
        ar1_r_squared=0.7,
    )
    parts = pair_id.split("-")
    return SpreadSignal(
        pair_id=pair_id,
        module=module,
        asset_a=parts[0],
        asset_b=parts[1],
        direction=direction,
        z_score=z_score,
        beta=1.1,
        ou_params=ou,
        vol_regime=vol_regime,
        vol_a=0.15,
        vol_b=0.15,
        coint_pvalue=coint_pvalue,
        sentiment=None,
        sentiment_modifier=0.0,
        entry_z_threshold_used=1.5 if module == "equity" else 1.75,
        exit_z_threshold=0.3,
        stop_z_threshold=3.0 if module == "equity" else 3.5,
        signal_rationale={},
        generated_at=now,
        data_timestamp=ts,
    )


@pytest.fixture
def test_account() -> AccountInfo:
    return AccountInfo(
        equity=100000.0,
        buying_power=100000.0,
        cash=100000.0,
        currency="USD",
        snapshot_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def risk_config() -> dict:
    return {
        "max_open_equity_pairs": 3,
        "max_open_crypto_pairs": 3,
        "coint_pvalue_threshold": 0.05,
        "halflife_min_days": 2.0,
        "halflife_max_days": 20.0,
        "halflife_min_hours": 6.0,
        "halflife_max_hours": 96.0,
        "max_premium_pct_equity": 0.05,
        "max_position_pct_equity": 0.10,
        "kelly_fraction": 0.25,
        "rolling_24h_loss_limit_pct": 0.03,
        "circuit_breaker_cooldown_hours": 4.0,
        "max_data_staleness_minutes": 15.0,
    }


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestRiskAgent:
    def test_rule1_max_open_pairs(self, risk_config, test_account):
        """Reject if number of open positions for that module equals or exceeds max."""
        agent = RiskAgent(config=risk_config)
        open_trades = [
            {"pair_id": "XOM-CVX", "module": "equity", "status": "open"},
            {"pair_id": "KO-PEP", "module": "equity", "status": "open"},
            {"pair_id": "GLD-IAU", "module": "equity", "status": "open"},
        ]

        signal = make_test_signal(pair_id="GLD-SLV", module="equity")
        decisions = agent.evaluate([signal], account=test_account, open_trades=open_trades)

        assert len(decisions) == 1
        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "max_open_pairs"

    def test_rule2_duplicate_pair(self, risk_config, test_account):
        """Reject if position for the same pair_id is already open."""
        agent = RiskAgent(config=risk_config)
        open_trades = [
            {"pair_id": "GLD-SLV", "module": "equity", "status": "open"},
        ]

        signal = make_test_signal(pair_id="GLD-SLV", module="equity")
        decisions = agent.evaluate([signal], account=test_account, open_trades=open_trades)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "duplicate_pair"

    def test_rule3_halflife_gate(self, risk_config, test_account):
        """Reject if half-life is outside valid bounds."""
        agent = RiskAgent(config=risk_config)
        # 30 days is above max of 20 days for equity
        signal = make_test_signal(half_life=30.0, module="equity")
        decisions = agent.evaluate([signal], account=test_account)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "halflife_gate"

    def test_rule4_coint_gate(self, risk_config, test_account):
        """Reject if coint_pvalue > threshold."""
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal(coint_pvalue=0.08)
        decisions = agent.evaluate([signal], account=test_account)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "coint_gate"

    def test_rule5_regime_block(self, risk_config, test_account):
        """Reject if vol_regime is EXTREME."""
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal(vol_regime="EXTREME")
        decisions = agent.evaluate([signal], account=test_account)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "regime_block"

    def test_rule6_sizing_cap_rejection(self, risk_config, test_account):
        """Reject if sizing overshoots the maximum position cap."""
        # Set max position pct to a tiny 0.1% of equity ($100 on $100k equity)
        tiny_config = dict(risk_config)
        tiny_config["max_premium_pct_equity"] = 0.001
        agent = RiskAgent(config=tiny_config)
        # Sizing 1 contract of GLD (~$450 premium) exceeds $100 cap
        signal = make_test_signal(pair_id="GLD-SLV", module="equity")
        current_prices = {"GLD": 180.0, "SLV": 22.0}

        decisions = agent.evaluate([signal], account=test_account, current_prices=current_prices)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "sizing_check"

    def test_rule7_buying_power(self, risk_config):
        """Reject if total cost exceeds available buying power."""
        low_bp_account = AccountInfo(
            equity=100000.0,
            buying_power=50.0,  # only $50 buying power
            cash=50.0,
            currency="USD",
            snapshot_at=datetime.now(timezone.utc),
        )
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal()
        decisions = agent.evaluate([signal], account=low_bp_account)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "buying_power"

    def test_rule8_circuit_breaker_and_cooldown(self, risk_config, test_account):
        """Rolling 24h loss <= -3% triggers 4h cooldown blocking all entries."""
        agent = RiskAgent(config=risk_config)
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 24h loss of -$3500 on $100k equity = -3.5% (exceeds -3% threshold)
        signal = make_test_signal()
        decisions = agent.evaluate(
            [signal],
            account=test_account,
            rolling_24h_pnl=-3500.0,
            now=now,
        )

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "circuit_breaker"

        # Check 2 hours later (still within 4h cooldown)
        now_2h = now + timedelta(hours=2)
        decisions_2h = agent.evaluate(
            [signal],
            account=test_account,
            rolling_24h_pnl=0.0,  # even if PnL recovered, cooldown is active
            now=now_2h,
        )
        assert not decisions_2h[0].approved
        assert decisions_2h[0].rejection_rule == "circuit_breaker"

        # Check 5 hours later (cooldown expired)
        now_5h = now + timedelta(hours=5)
        decisions_5h = agent.evaluate(
            [signal],
            account=test_account,
            rolling_24h_pnl=0.0,
            now=now_5h,
        )
        assert decisions_5h[0].approved

    def test_rule9_data_freshness(self, risk_config, test_account):
        """Reject if bar timestamp is older than 15 minutes."""
        agent = RiskAgent(config=risk_config)
        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(minutes=25)

        signal = make_test_signal(timestamp=stale_ts)
        decisions = agent.evaluate([signal], account=test_account, now=now)

        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "data_freshness"

    def test_approved_equity_options_sizing(self, risk_config, test_account):
        """Valid equity signal is approved and produces SpreadOrderRequest with options legs."""
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal(
            pair_id="GLD-SLV",
            module="equity",
            direction="long",
            z_score=-2.2,
        )
        current_prices = {"GLD": 180.0, "SLV": 22.0}

        decisions = agent.evaluate(
            [signal],
            account=test_account,
            current_prices=current_prices,
        )

        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.approved
        assert decision.rejection_rule is None
        assert decision.sized_order is not None

        order = decision.sized_order
        assert order.module == "equity"
        assert order.direction == "long"
        assert order.execution_type == "options"
        assert order.leg_a.option_type == "call"  # Long spread: buy call on A
        assert order.leg_b.option_type == "put"   # Long spread: buy put on B
        assert order.estimated_cost <= risk_config["max_premium_pct_equity"] * test_account.equity

    def test_approved_crypto_spot_sizing(self, risk_config, test_account):
        """Valid crypto signal is approved and produces SpreadOrderRequest with spot legs."""
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal(
            pair_id="BTC/USD-ETH/USD",
            module="crypto",
            direction="short",
            z_score=2.2,
            half_life=24.0,  # 24 hours
        )
        current_prices = {"BTC/USD": 60000.0, "ETH/USD": 3000.0}

        decisions = agent.evaluate(
            [signal],
            account=test_account,
            current_prices=current_prices,
        )

        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.approved
        assert decision.sized_order is not None

        order = decision.sized_order
        assert order.module == "crypto"
        assert order.direction == "short"
        assert order.execution_type == "spot"
        assert order.leg_a.side == "sell"  # Short spread: sell A
        assert order.leg_b.side == "buy"   # Short spread: buy B
        assert order.leg_a.notional_usd <= risk_config["max_position_pct_equity"] * test_account.equity

    def test_fail_closed_on_exception(self, risk_config):
        """Simulate unexpected failure (e.g. malformed account) -> reject with risk_agent_failure."""
        agent = RiskAgent(config=risk_config)
        signal = make_test_signal()

        # Pass invalid account object to trigger error
        decisions = agent.evaluate([signal], account=None)  # type: ignore

        assert len(decisions) == 1
        assert not decisions[0].approved
        assert decisions[0].rejection_rule == "risk_agent_failure"

    def test_database_logging(self, tmp_path, risk_config, test_account):
        """Decisions are written to SQLite risk_decisions table."""
        db = Database(tmp_path / "test_risk.db")
        agent = RiskAgent(config=risk_config, db=db)
        signal = make_test_signal()

        decisions = agent.evaluate([signal], account=test_account)
        assert len(decisions) == 1

        conn = db._get_connection()
        rows = conn.execute("SELECT * FROM risk_decisions WHERE pair_id = 'GLD-SLV'").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["approved"] == 1
        assert rows[0]["module"] == "equity"
