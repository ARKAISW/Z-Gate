# ARCHITECTURE.md

## System overview

Two simultaneous strategy modules sharing one signal engine, one risk agent, and one
persistence layer. Only the execution layer differs.

```
                    ┌─────────────────────────────────────────────────┐
                    │            Shared Signal Engine                 │
  Market Data ─────▶│  Kalman Filter → OU Estimation → Z-Score       │
  (equity daily +   │  Vol Regime → Sentiment Modifier               │
   crypto hourly)   └────────────────┬────────────────────────────────┘
                                     │ SpreadSignal (same schema)
                    ┌────────────────▼────────────────┐
                    │         Risk Agent              │
                    │  (deterministic, zero LLM)      │
                    └──────┬─────────────────┬────────┘
               approved    │                 │  approved
               equity      │                 │  crypto
         ┌─────────────────▼──┐   ┌──────────▼──────────────┐
         │  Execution Agent A │   │   Execution Agent B     │
         │  (equity options)  │   │   (crypto spot)          │
         └─────────┬──────────┘   └──────────┬──────────────┘
                   │                          │
         ┌─────────▼──────────────────────────▼──────────┐
         │               Alpaca Paper Account            │
         │  (equity options + crypto spot, same account) │
         └─────────────────────────┬─────────────────────┘
                                   │
                    ┌──────────────▼──────────┐
                    │     SQLite Trade Log    │
                    │  signals, decisions,    │
                    │  orders, trades,        │
                    │  reflections            │
                    └──────────────┬──────────┘
                                   │
                    ┌──────────────▼──────────┐
                    │   Reflection Agent      │
                    │   (LLM, nightly UTC)    │
                    └─────────────────────────┘
```

---

## Directory layout

```
alpaca-stat-arb/
├── README.md
├── AGENTS.md
├── ARCHITECTURE.md
├── STRATEGY.md
├── RISK_RULES.md
├── ROADMAP.md
├── TESTING.md
├── RESULTS.md               # backtest results, limitations (created after Phase 5)
├── prompts/
│   ├── SIGNAL_AGENT_PROMPT.md
│   └── REFLECTION_AGENT_PROMPT.md
├── config.yaml
├── .env.example
├── requirements.txt
├── src/
│   ├── broker.py              # Alpaca wrapper: equity + crypto + options (paper only)
│   ├── llm_provider.py        # pluggable: Featherless → Groq → Ollama → none
│   ├── models.py              # pydantic: SpreadSignal, RiskDecision, SpreadOrderRequest,
│   │                          #   OptionsLeg, SpotLeg, TradeLogEntry, OUParams
│   ├── kalman.py              # KalmanFilter class — pure, no I/O, unit-tested
│   ├── indicators.py          # fit_ou_parameters, compute_z_score, compute_realized_vol,
│   │                          #   classify_vol_regime — pure functions, no I/O
│   ├── options_selector.py    # select_contract(chain, tau_days, price) → OptionsLeg
│   ├── agents/
│   │   ├── signal_agent.py    # shared pipeline: coint gate → Kalman → OU → z-score
│   │   │                      #   → vol regime → sentiment → SpreadSignal
│   │   ├── risk_agent.py      # deterministic rules, zero LLM, handles both modules
│   │   ├── execution_agent.py # ONLY module that calls broker.place_order()
│   │   │                      #   dispatches to options or spot execution based on module
│   │   └── reflection_agent.py
│   ├── persistence/
│   │   ├── db.py
│   │   └── schema.py          # unified schema for both modules; module_type field on each row
│   └── pipeline.py            # orchestrator: runs both modules each cycle, schedules daily jobs
├── scripts/
│   ├── run_backtest.py        # equity (daily bars + BS-approximated options) + crypto (hourly)
│   ├── run_live_paper.py      # main entry point: handles market-hours guard for equity module
│   └── dashboard.py           # Streamlit: unified view of both modules
└── tests/
    ├── test_kalman.py
    ├── test_indicators.py
    ├── test_options_selector.py
    ├── test_signal_agent.py
    ├── test_risk_agent.py
    ├── test_execution_agent.py
    └── test_pipeline_integration.py
```

---

## Data flow (one pipeline cycle)

### Equity module (runs only during market hours 9:30–16:00 ET)

1. `pipeline.py` fetches **daily bars** for GLD, SLV, XOM, CVX, KO, PEP from `broker.py`.
2. `signal_agent.py` processes each equity pair:
   - Cointegration gate (cached daily p-value).
   - `KalmanFilter.update()` with daily close prices.
   - `fit_ou_parameters()` on rolling daily spread.
   - Half-life gate (days).
   - Z-score, vol regime, sentiment modifier (base asset only).
   - Produces `SpreadSignal` with `module = "equity"`.
3. `risk_agent.py` applies rules. Approved signals include a `SpreadOrderRequest` with
   `execution_type = "options"`.
4. `execution_agent.py` calls `options_selector.py` to pick the contract (strike, expiry
   from tau_days × expiry_multiplier), then places two option buy orders via `broker.py`.
5. Exit monitoring: every cycle during market hours, check z-score for all open equity
   positions. Issue sell-to-close on both legs if exit conditions are met.

### Crypto module (runs 24/7)

1. `pipeline.py` fetches **hourly bars** for BTC/USD, ETH/USD, SOL/USD from `broker.py`.
2. `signal_agent.py` processes each crypto pair (same logic, hourly frequency).
   Produces `SpreadSignal` with `module = "crypto"`.
3. `risk_agent.py` applies rules (same engine, module-specific thresholds from config).
   Approved signals include `SpreadOrderRequest` with `execution_type = "spot"`.
4. `execution_agent.py` places market orders for both spot legs via `broker.py`.
5. Exit monitoring: every cycle, check z-score and time-stop for all open crypto positions.

### Daily jobs (triggered by UTC hour check in the polling loop)

- **Midnight UTC:** cointegration recheck for all 6 pairs (equity + crypto). Update
  cached p-values. If any pair fails: log, suspend new entries, close open positions
  if applicable.
- **Midnight UTC:** Reflection Agent run — reviews all trades closed in the last 24h.
- **Market open (9:30 ET):** equity cointegration recheck (in addition to midnight UTC).

---

## Scheduling model

`run_live_paper.py` uses a single polling loop:

```python
while True:
    now_utc = datetime.utcnow()
    now_et = convert_to_et(now_utc)

    # Daily jobs
    if is_new_utc_day(now_utc):
        pipeline.run_daily_jobs()   # coint recheck + reflection agent

    # Equity module (market hours only)
    if is_market_hours(now_et):
        pipeline.run_equity_cycle()

    # Crypto module (always)
    pipeline.run_crypto_cycle()

    time.sleep(config.poll_interval_seconds)
```

---

## LLM provider chain

`llm_provider.py` tries providers in order, falls back on any failure:

```
Featherless AI (primary, $25 credit, 7B model quality)
  → Groq (fallback, free tier, llama-3.1-8b-instant)
    → Ollama (local fallback, ≤2B model if installed)
      → none (technical-only, sentiment_modifier = 0)
```

Each backend implements the same interface:
`summarize_sentiment(asset: str, headlines: list[str]) → SentimentResult`

The fallback chain must be tested explicitly: simulate Featherless failure → verify Groq
is called; simulate both failures → verify Ollama is called; simulate all failures →
verify sentiment_modifier = 0 and system continues.

---

## Persistence (SQLite — unified schema for both modules)

All tables have a `module` column (`"equity"` or `"crypto"`) and `pair_id` (e.g.
`"GLD-SLV"` or `"BTC/USD-ETH/USD"`).

- **`signals`** — every SpreadSignal, including gate failures. Columns: module, pair_id,
  z_score, beta, half_life, vol_regime, sentiment_modifier, signal_direction, rationale_json,
  timestamp.
- **`risk_decisions`** — every RiskDecision. Columns: module, pair_id, approved, rejection_rule,
  rejection_reason, sized_order_json, checked_at.
- **`orders`** — every order placed. Columns: module, pair_id, order_type (equity_option / spot),
  leg (A/B), symbol, alpaca_order_id, status, fill_price, fill_time, quantity.
- **`trades`** — closed round-trips. Columns: module, pair_id, direction, entry_z, exit_z,
  exit_reason, realized_pnl_usd, realized_pnl_pct, holding_period, entry_time, exit_time.
- **`reflections`** — nightly LLM notes keyed to UTC date and optionally to trade IDs.

---

## Permission boundary

Only `execution_agent.py` imports `broker.place_order()`. All other modules use
`broker.get_bars()`, `broker.get_account()`, `broker.get_positions()`,
`broker.get_options_chain()` only. This boundary is enforced by a grep-based assertion
in `test_pipeline_integration.py`.
