"""persistence/schema.py — SQLite table definitions for the trade log.

Plain sqlite3 — no ORM magic. Schema is defined here as SQL strings so it is
trivially readable and diffable. The `db.py` module uses these to create tables
and run queries.

All tables include a `module` column ('equity' | 'crypto') and a `pair_id` column
(e.g. 'GLD-SLV', 'BTC/USD-ETH/USD') for filtering in the dashboard.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CREATE TABLE statements
# ---------------------------------------------------------------------------

CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              TEXT PRIMARY KEY,
    pair_id         TEXT NOT NULL,
    module          TEXT NOT NULL CHECK(module IN ('equity', 'crypto')),
    asset_a         TEXT NOT NULL,
    asset_b         TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK(direction IN ('long', 'short', 'none')),
    z_score         REAL NOT NULL,
    beta            REAL NOT NULL,
    half_life       REAL NOT NULL,
    kappa           REAL NOT NULL,
    sigma_spread    REAL NOT NULL,
    vol_regime      TEXT NOT NULL CHECK(vol_regime IN ('NORMAL', 'HIGH', 'EXTREME')),
    vol_a           REAL NOT NULL,
    vol_b           REAL NOT NULL,
    coint_pvalue    REAL NOT NULL,
    sentiment_modifier REAL NOT NULL DEFAULT 0.0,
    sentiment_raw   TEXT,                    -- raw LLM response (nullable)
    entry_z_threshold_used REAL NOT NULL,
    signal_rationale TEXT NOT NULL,          -- JSON blob
    generated_at    TEXT NOT NULL,           -- ISO8601 UTC
    data_timestamp  TEXT NOT NULL            -- ISO8601 UTC of latest bar
);
"""

CREATE_RISK_DECISIONS = """
CREATE TABLE IF NOT EXISTS risk_decisions (
    id              TEXT PRIMARY KEY,
    signal_id       TEXT,
    pair_id         TEXT NOT NULL,
    module          TEXT NOT NULL CHECK(module IN ('equity', 'crypto')),
    approved        INTEGER NOT NULL CHECK(approved IN (0, 1)),
    rejection_rule  TEXT,
    rejection_reason TEXT,
    sized_order     TEXT,                    -- JSON blob (nullable if rejected)
    checked_at      TEXT NOT NULL            -- ISO8601 UTC
);
"""

CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,        -- UUID
    trade_id        TEXT,                    -- FK to trades (set after trade created)
    pair_id         TEXT NOT NULL,
    module          TEXT NOT NULL CHECK(module IN ('equity', 'crypto')),
    leg             TEXT NOT NULL CHECK(leg IN ('A', 'B')),
    order_type      TEXT NOT NULL CHECK(order_type IN ('options', 'spot')),
    symbol          TEXT NOT NULL,
    alpaca_order_id TEXT,                    -- set after placement
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    fill_price      REAL,
    fill_time       TEXT,                    -- ISO8601 UTC
    error_message   TEXT,
    created_at      TEXT NOT NULL            -- ISO8601 UTC
);
"""

CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,        -- UUID
    pair_id         TEXT NOT NULL,
    module          TEXT NOT NULL CHECK(module IN ('equity', 'crypto')),
    direction       TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    status          TEXT NOT NULL CHECK(status IN ('open', 'closed', 'failed')),
    entry_z         REAL NOT NULL,
    entry_beta      REAL NOT NULL,
    entry_time      TEXT NOT NULL,           -- ISO8601 UTC
    exit_z          REAL,
    exit_reason     TEXT,
    exit_time       TEXT,                    -- ISO8601 UTC
    realized_pnl_usd REAL,
    realized_pnl_pct REAL,
    holding_period_hours REAL,
    leg_a_entry_order_id TEXT,
    leg_b_entry_order_id TEXT,
    leg_a_exit_order_id  TEXT,
    leg_b_exit_order_id  TEXT,
    rationale_json  TEXT NOT NULL DEFAULT '{}'  -- full rationale chain
);
"""

CREATE_REFLECTIONS = """
CREATE TABLE IF NOT EXISTS reflections (
    id              TEXT PRIMARY KEY,        -- UUID
    trade_id        TEXT,                    -- NULL for day-level reflections
    utc_date        TEXT NOT NULL,           -- YYYY-MM-DD
    module          TEXT,                    -- NULL for day-level
    pair_id         TEXT,                    -- NULL for day-level
    reflection_type TEXT NOT NULL CHECK(reflection_type IN ('trade', 'day')),
    outcome_summary  TEXT,
    rule_alignment   TEXT,
    ou_observation   TEXT,
    notable_observation TEXT,
    day_summary      TEXT,
    risk_rejection_pattern TEXT,
    cointegration_health   TEXT,
    provider_used    TEXT NOT NULL,
    generated_at     TEXT NOT NULL           -- ISO8601 UTC
);
"""

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair_id, module, generated_at);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_pair ON risk_decisions(pair_id, module, checked_at);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status, module, pair_id);",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_time);",
    "CREATE INDEX IF NOT EXISTS idx_reflections_date ON reflections(utc_date);",
]

ALL_CREATE_STATEMENTS = [
    CREATE_SIGNALS,
    CREATE_RISK_DECISIONS,
    CREATE_ORDERS,
    CREATE_TRADES,
    CREATE_REFLECTIONS,
    *CREATE_INDEXES,
]
