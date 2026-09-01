# ROADMAP.md — Build plan (hackathon window: 28 Aug – 4 Sep 2026)

Work phase by phase. Do not start a phase until the previous one's tests pass. Check
items off as you go — this file doubles as the submission changelog.

The system has two strategy modules (equity options, crypto spot) sharing one Kalman/OU
signal core. Build the shared math first, then each module's execution layer, then wire
them together. The shared core is the most complex and the most valuable — do not rush it.

---

## Phase 0 — Setup (Day 1 morning)

- [ ] Create Alpaca paper trading account. Verify it is the **competition paper account**
      (dedicated $100K balance) per Alpaca's hackathon instructions.
- [ ] Verify that the paper account supports:
      - Equity options trading (check via API: `GET /v2/options/contracts`)
      - Crypto spot trading (BTC/USD, ETH/USD, SOL/USD)
      - Short positions on crypto (if not available, document and fall back to long-only)
- [ ] Scaffold repo per `ARCHITECTURE.md` directory layout.
- [ ] `.env.example` with:
      `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`,
      `FEATHERLESS_API_KEY`, `FEATHERLESS_MODEL` (default: `Qwen/Qwen2.5-7B-Instruct`),
      `GROQ_API_KEY` (backup), `OLLAMA_MODEL` (local backup),
      `LLM_PROVIDER` (`featherless` / `groq` / `ollama` / `none`)
- [ ] `broker.py`: wrap Alpaca client. Hard-fail on startup if `ALPACA_PAPER != true`.
      Implement and manually verify:
      - `get_bars(symbol, timeframe, limit)` for both equity and crypto
      - `get_options_chain(symbol, expiry_date)` → list of contracts
      - `get_account()`, `get_positions()`
      - `place_order(order_request)` — single method, dispatches to correct endpoint
- [ ] `models.py`: all pydantic models (SpreadSignal, RiskDecision, SpreadOrderRequest,
      OptionsLeg, SpotLeg, OUParams, SentimentResult, TradeLogEntry).
- [ ] `config.yaml`: all parameters from `STRATEGY.md` and `RISK_RULES.md`.

---

## Phase 1 — Kalman Filter + OU Estimation (Day 1 afternoon – Day 2)

The mathematical core. Get this right and well-tested before touching any agent code.

- [ ] `kalman.py`:
  - `KalmanFilter` class: `update(price_a, price_b) → (beta, mu, spread)`.
  - `initialize_filter(prices_a, prices_b) → KalmanFilter` (warm-start on historical data).
  - Pure functions only. No imports of broker, models, or I/O of any kind.
- [ ] `test_kalman.py`:
  - Convergence: known true β → filter within 5% by observation 50.
  - Drift tracking: β shifts mid-series → filter tracks within 20 observations.
    (Include a comment showing static OLS would fail this test.)
  - Spread stationarity: Kalman spread on a synthetic cointegrated series passes ADF test.
  - Numerical stability: 2000 observations, no NaN or inf output.
- [ ] `indicators.py`:
  - `fit_ou_parameters(spread: np.ndarray, dt: float) → OUParams` (AR(1) regression via statsmodels).
  - `compute_z_score(spread, mu_ou, sigma_spread) → float`.
  - `compute_realized_vol(log_returns, periods_per_year) → float`.
  - `classify_vol_regime(rv_a, rv_b, high_thresh, extreme_thresh) → str`.
  - All pure functions, no I/O.
- [ ] `test_indicators.py`:
  - `fit_ou_parameters` on synthetic OU (known κ, μ, σ) within 15% tolerance.
  - Half-life `ln(2)/κ` within 10% of true value.
  - `compute_z_score` verified by hand on known values.
  - `compute_realized_vol` verified against manual annualization formula.
  - `classify_vol_regime` tested at all three boundaries.

---

## Phase 2 — Signal Agent (Day 2)

- [ ] `signal_agent.py`:
  - Cointegration gate: `statsmodels.tsa.stattools.coint`; cache result per pair.
  - Per-pair `KalmanFilter` instance in memory, warm-started from historical bars on startup.
  - OU estimation on rolling window (module-specific lookback).
  - Half-life gate.
  - Z-score, vol regime, entry/exit/stop classification.
  - Sentiment stub (returns `sentiment_modifier = 0`, flagged in rationale).
  - Produces one `SpreadSignal` per pair with full rationale dict. `module` field set correctly.
- [ ] `test_signal_agent.py`:
  - Entry fires only when ALL conditions hold.
  - Entry blocked by each gate individually (coint fail, half-life out of bounds, EXTREME regime).
  - Correct z-direction → correct signal direction (LONG vs SHORT spread).
  - Exit signal fires at |z| < exit_z_threshold.
  - Stop signal fires at |z| > stop_z_threshold.
  - Sentiment fallback: LLM raises → `sentiment_modifier = 0`, rationale noted, no exception raised.

---

## Phase 3 — Risk Agent (Day 3 morning)

- [ ] `risk_agent.py`: all 9 rules from `RISK_RULES.md`, each as a named function.
- [ ] `test_risk_agent.py`:
  - One isolated test per rule.
  - Combined scenario: 4 candidates across both modules, varied failures.
  - Circuit breaker: triggers → blocks all subsequent candidates that cycle.
  - Fail-closed: `broker.get_account()` raises → zero approvals, `RISK_AGENT_FAILURE` logged.

---

## Phase 4 — Execution Agent + Options Selector + Persistence (Day 3 afternoon – Day 4)

- [ ] `src/persistence/schema.py` + `db.py`: unified SQLite schema from `ARCHITECTURE.md`.
- [ ] `options_selector.py`:
  - `select_contract(chain, target_dte, current_price, option_type) → OptionsLeg`
  - Selects ATM strike, nearest expiry ≥ target_dte.
  - Unit tests: correct expiry selection for various tau_days values; correct strike selection.
- [ ] `execution_agent.py`:
  - Equity path: calls `options_selector` for both legs, places two buy-to-open orders.
  - Crypto path: places spot market orders for both legs.
  - Partial fill handling: if either leg fails, cancel the other. Log both attempt and outcome.
  - `TradeLogEntry` written before orders placed (attempt), updated after (result).
- [ ] `pipeline.py`:
  - Market-hours guard for equity module.
  - Daily job trigger (coint recheck + reflection agent at midnight UTC).
  - Exit monitoring for both modules each cycle.
- [ ] `test_execution_agent.py`:
  - Correct options order construction (symbol, qty, side, expiry, strike).
  - Correct spot order construction.
  - Leg A success + Leg B failure → Leg A cancelled, failure logged.
  - Log persistence: attempt record exists even on total failure.
- [ ] `test_pipeline_integration.py`:
  - Full mocked cycle for both modules; verify correct SQLite write sequence.
  - Permission boundary: `place_order` only imported in `execution_agent.py`.
  - Exit monitoring: synthetic open position triggers close on next cycle.
- [ ] Manual smoke test: run one cycle against the live paper account, confirm orders appear
      in Alpaca paper dashboard (both an equity options order and a crypto spot order).

---

## Phase 5 — Backtest (Day 4–5)

- [ ] `scripts/run_backtest.py`:
  - **Equity module**: 1–2 years of daily bars for GLD, SLV, XOM, CVX, KO, PEP.
    Approximate options premiums using Black-Scholes (scipy) with historical volatility.
    Disclose this approximation in `RESULTS.md`.
  - **Crypto module**: 6+ months of hourly bars for BTC/USD, ETH/USD, SOL/USD.
  - Both modules use the same `kalman.py` and `indicators.py` — validate the signal engine.
- [ ] Output per module: equity curve, trade list, metrics from `TESTING.md`.
- [ ] Sanity-check all results against `TESTING.md` checklist before live-paper.
- [ ] Write `RESULTS.md` with numbers, equity curve images, and known limitations.

---

## Phase 6 — LLM Sentiment + Reflection Agent (Day 5)

- [ ] `llm_provider.py`:
  - Featherless AI backend (primary): OpenAI-compatible API, target `Qwen/Qwen2.5-7B-Instruct`.
  - Groq backend (fallback): `llama-3.1-8b-instant` on free tier.
  - Ollama backend (local fallback): configurable model, ≤2B.
  - `none` fallback: `sentiment_modifier = 0`, log `"LLM unavailable"`.
  - Test the fallback chain explicitly: simulate each backend failing in sequence.
- [ ] Wire sentiment into `signal_agent.py` per `STRATEGY.md` Step 6. Test graceful fallback.
- [ ] `reflection_agent.py`: midnight UTC job, uses `REFLECTION_AGENT_PROMPT.md`,
      writes to `reflections` table. Handles both equity and crypto trade types.

---

## Phase 7 — Dashboard + Polish (Day 6)

- [ ] `scripts/dashboard.py` (Streamlit), two tabs or sections:

  **Equity Options tab:**
  - Open equity pair positions: pair, direction, entry_z, current_z, tau_days, expiry,
    days-to-expiry, current premium value, unrealized P/L.
  - Pair health panel: cointegration p-value, vol regime, half-life for each equity pair.
  - Closed trades with exit reason and realized P/L.

  **Crypto Spot tab:**
  - Open crypto pair positions: pair, direction, entry_z, current_z, tau_hours, unrealized P/L.
  - Pair health panel: coint p-value, vol regime, half-life.
  - Closed trades.

  **Shared:**
  - Combined equity curve (both modules, cumulative P/L in dollars).
  - Risk event log: CIRCUIT_BREAKER and REGIME_BLOCK events highlighted.
  - Risk decision log: all rejections with rule name.
  - Nightly reflection notes.

- [ ] Let the pipeline run live for 24–48h to accumulate genuine paper trades and logs.
      Do not fabricate or cherry-pick results for the demo.

---

## Phase 8 — Submission Prep (Day 7)

- [ ] `RESULTS.md`: backtest metrics + equity curve images for both modules.
      Known limitations section (options premium approximation, survivorship bias,
      no cross-module correlation limits, no Greeks monitoring).
- [ ] README polish: architecture diagram, how to run, one-paragraph plain-English
      strategy description, disclaimer.
- [ ] Record demo: live cycle (show both modules firing), at least one risk rejection,
      equity curve, pair health panel, one reflection note. Show the z-score and half-life
      updating live in the dashboard.
- [ ] Verify: `git log -p | grep -i key` — no secrets in history.
- [ ] Submit.
