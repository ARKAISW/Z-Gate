"""models.py — Pydantic data models for every structured object passed between agents.

No raw dicts cross agent boundaries. Every signal, decision, order, and log entry
has a typed schema defined here. This file has zero business logic — just schemas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ── OU parameter container ────────────────────────────────────────────────────


class OUParams(BaseModel):
    """Fitted Ornstein-Uhlenbeck parameters for a spread series."""

    kappa: float = Field(description="Mean-reversion speed (per bar unit).")
    mu: float = Field(description="Long-run mean of the spread.")
    sigma_ou: float = Field(description="OU diffusion coefficient.")
    half_life: float = Field(description="ln(2)/kappa — in bar units (hours or days).")
    sigma_spread: float = Field(
        description="Stationary std dev = sigma_ou / sqrt(2*kappa)."
    )
    ar1_r_squared: float = Field(
        le=1.0, description="AR(1) goodness of fit (R²). Can be negative for degraded fits."
    )


# ── Sentiment result ──────────────────────────────────────────────────────────


class SentimentResult(BaseModel):
    """Structured output from the LLM sentiment classifier."""

    asset: str
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)
    modifier: float = Field(
        description=(
            "Computed entry threshold nudge. Positive means harder to enter. "
            "Range: [0, 0.15]."
        )
    )
    provider_used: Literal["featherless", "groq", "ollama", "none"]
    raw_response: str | None = None


# ── Spread signal ─────────────────────────────────────────────────────────────


class SpreadSignal(BaseModel):
    """Output of the Signal Agent for one pair at one cycle.

    Produced for every pair regardless of whether an entry fires — gate failures
    are recorded with direction='none' and a rationale explaining which gate failed.
    """

    pair_id: str = Field(description="E.g. 'GLD-SLV' or 'BTC/USD-ETH/USD'.")
    module: Literal["equity", "crypto"]
    asset_a: str
    asset_b: str

    # Signal output
    direction: Literal["long", "short", "none"]
    """long = buy A / sell B (A cheap relative to B).
    short = sell A / buy B (A rich relative to B).
    none = no entry signal this cycle."""

    z_score: float
    beta: float = Field(description="Kalman-filtered hedge ratio at signal time.")
    ou_params: OUParams
    vol_regime: Literal["NORMAL", "HIGH", "EXTREME"]
    vol_a: float = Field(description="Annualized realized vol for asset A.")
    vol_b: float = Field(description="Annualized realized vol for asset B.")
    coint_pvalue: float = Field(
        ge=0.0, le=1.0, description="Latest cointegration test p-value."
    )

    # Sentiment (optional)
    sentiment: SentimentResult | None = None
    sentiment_modifier: float = 0.0

    # Thresholds actually used this cycle (depends on vol regime and module)
    entry_z_threshold_used: float
    exit_z_threshold: float
    stop_z_threshold: float

    # Full structured rationale — logged to SQLite before any order is placed
    signal_rationale: dict[str, Any] = Field(default_factory=dict)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    data_timestamp: datetime = Field(
        description="Timestamp of the most recent bar used to generate this signal."
    )


# ── Order leg models ──────────────────────────────────────────────────────────


class OptionsLeg(BaseModel):
    """One leg of an equity options spread trade."""

    underlying: str = Field(description="Underlying equity symbol, e.g. 'GLD'.")
    symbol: str = Field(description="OCC options contract symbol.")
    expiry: str = Field(description="Expiry date in YYYY-MM-DD format.")
    strike: float
    option_type: Literal["call", "put"]
    qty: int = Field(ge=1)
    side: Literal["buy_to_open", "sell_to_close"]
    premium_estimate: float | None = Field(
        default=None, description="Mid-price estimate at time of order construction."
    )


class SpotLeg(BaseModel):
    """One leg of a crypto spot spread trade."""

    symbol: str = Field(description="E.g. 'BTC/USD'.")
    qty: float = Field(gt=0)
    side: Literal["buy", "sell"]
    notional_usd: float = Field(description="Dollar value of this leg.")


class SpreadOrderRequest(BaseModel):
    """Fully-specified order for one spread position (both legs)."""

    pair_id: str
    module: Literal["equity", "crypto"]
    direction: Literal["long", "short"]
    execution_type: Literal["options", "spot"]
    leg_a: OptionsLeg | SpotLeg
    leg_b: OptionsLeg | SpotLeg

    # Sizing metadata (logged for auditability)
    beta: float = Field(description="Hedge ratio at time of order construction.")
    entry_z: float
    kelly_f: float = Field(description="Full Kelly fraction (before applying kelly_fraction).")
    position_f: float = Field(
        description="Actual fraction used = kelly_fraction * kelly_f, clamped."
    )
    estimated_cost: float = Field(
        description="Total premium (options) or total notional (spot) in USD."
    )


# ── Risk decision ─────────────────────────────────────────────────────────────


class RiskDecision(BaseModel):
    """Output of the Risk Agent for one pair candidate."""

    pair_id: str
    module: Literal["equity", "crypto"]
    signal: SpreadSignal
    approved: bool
    rejection_reason: str | None = None
    rejection_rule: str | None = None
    """Name of the first rule that rejected this candidate.
    One of: max_open_pairs, duplicate_pair, halflife_gate, coint_gate,
    regime_block, sizing_check, buying_power, circuit_breaker,
    data_freshness, risk_agent_failure."""
    sized_order: SpreadOrderRequest | None = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


# ── Trade log entry ───────────────────────────────────────────────────────────


class TradeLogEntry(BaseModel):
    """Persistent record of one spread position lifecycle (open → closed)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    pair_id: str
    module: Literal["equity", "crypto"]
    direction: Literal["long", "short"]
    status: Literal["open", "closed", "failed"]

    # Entry
    entry_z: float
    entry_beta: float
    entry_time: datetime

    # Exit
    exit_z: float | None = None
    exit_reason: str | None = None
    """z_reversion | stop_z | time_stop | coint_breakdown | execution_failure"""
    exit_time: datetime | None = None

    # P/L
    realized_pnl_usd: float | None = None
    realized_pnl_pct: float | None = None
    holding_period_hours: float | None = None

    # Alpaca order IDs
    leg_a_entry_order_id: str | None = None
    leg_b_entry_order_id: str | None = None
    leg_a_exit_order_id: str | None = None
    leg_b_exit_order_id: str | None = None

    # Full rationale chain (signal + risk decision), logged before order placement
    rationale_json: dict[str, Any] = Field(default_factory=dict)


# ── Account state snapshot ────────────────────────────────────────────────────


class AccountSnapshot(BaseModel):
    """Lightweight snapshot of account state for the Risk Agent."""

    equity: float
    buying_power: float
    cash: float
    daytrade_count: int = 0
    snapshot_at: datetime = Field(default_factory=datetime.utcnow)


class PositionSnapshot(BaseModel):
    """Snapshot of a single open position."""

    symbol: str
    qty: float
    side: Literal["long", "short"]
    market_value: float
    unrealized_pnl: float
    avg_entry_price: float


# ── Reflection ────────────────────────────────────────────────────────────────


class TradeReflection(BaseModel):
    """Nightly LLM reflection on a closed trade."""

    trade_id: str
    pair_id: str
    module: Literal["equity", "crypto"]
    outcome_summary: str
    rule_alignment: str
    ou_observation: str
    notable_observation: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    provider_used: str


class DayReflection(BaseModel):
    """Nightly LLM aggregate summary for one UTC trading day."""

    utc_date: str = Field(description="YYYY-MM-DD")
    day_summary: str
    risk_rejection_pattern: str
    cointegration_health: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    provider_used: str
