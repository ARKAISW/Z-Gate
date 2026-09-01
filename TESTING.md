# TESTING.md

## Unit tests (required — run before each phase is marked done)

All broker calls in automated tests must be mocked. Never let `pytest` hit the real
Alpaca paper API. Keep a clearly separate manual smoke-test script for that.

---

### `test_kalman.py`

- **Convergence**: generate synthetic log-prices with known true β = 1.3. Feed 200
  observations; assert estimated β is within 5% of true value by observation 50.
  Include a comment showing what static OLS gives for comparison.
- **Drift tracking**: β shifts from 1.3 to 0.9 at observation 100. Assert Kalman
  estimate tracks the shift within 20 observations (static OLS cannot do this).
- **Spread stationarity**: on a synthetic cointegrated pair (known true β), assert that
  the Kalman-derived spread series passes an ADF stationarity test (p < 0.05).
- **Numerical stability**: 2000 consecutive observations; assert no NaN or inf in output.

---

### `test_indicators.py`

- **`fit_ou_parameters`**: generate synthetic OU process (Euler-Maruyama, known κ, μ, σ).
  Assert estimated parameters within 15% of true values. Assert half-life `ln(2)/κ`
  within 10% of true value.
- **Half-life gate boundary**: assert τ < halflife_min correctly classifies as "too fast";
  τ > halflife_max as "too slow". Test at exact boundary values.
- **`compute_z_score`**: verify z = 0 when spread = μ, z = 1 when spread = μ + σ_spread.
  Hand-compute for a short series and assert exact match.
- **`compute_realized_vol`**: verify annualization is correct for both daily bars
  (×√252) and hourly bars (×√8760). Test against manual calculation on a 10-bar series.
- **`classify_vol_regime`**: test NORMAL, HIGH, and EXTREME branches with values at
  and near each threshold boundary. Both assets high → HIGH. One asset extreme → EXTREME.

---

### `test_options_selector.py`

- **Expiry selection**: given a mock options chain with weekly expiries, assert that for
  τ = 5 days and expiry_multiplier = 2.5, the selected expiry is 12–13 DTE (nearest
  weekly ≥ 12.5 DTE).
- **Strike selection**: assert ATM strike is the strike nearest to the current underlying
  price, not the strike with highest volume.
- **Chain edge cases**: no expiry ≥ target_dte available → raise a specific exception
  (`NoSuitableContractError`), do not silently return a wrong contract.
- **Option type**: LONG spread selects call for asset A, put for asset B; SHORT spread
  selects put for asset A, call for asset B.

---

### `test_signal_agent.py`

Tests for equity and crypto modules separately (use module parameter to distinguish).

- Entry fires only when ALL conditions hold for that module.
- Entry blocked when cointegration p-value fails (each module independently).
- Entry blocked when half-life is outside module-specific bounds.
- Entry blocked in EXTREME vol regime.
- LONG signal fires when z < -entry_z_threshold; SHORT when z > +entry_z_threshold.
- EXIT signal fires when |z| < exit_z_threshold on an open position.
- STOP signal fires when |z| > stop_z_threshold on an open position.
- **Sentiment fallback (primary → backup → none)**: simulate Featherless failure → Groq
  called; simulate both failures → `sentiment_modifier = 0` and rationale records
  "sentiment: unavailable"; no exception propagates from `signal_agent.run()`.
- **Module field**: assert `signal.module == "equity"` for equity pairs and `"crypto"`
  for crypto pairs.

---

### `test_risk_agent.py`

One isolated test per named rule from `RISK_RULES.md`:

- `max_open_pairs`: module-specific limit; equity full does not block crypto and vice versa.
- `duplicate_pair`: same pair already open in same module → rejected.
- `halflife_gate`: τ outside bounds → rejected (even if Signal Agent passed it).
- `coint_gate`: p-value on signal object exceeds threshold → rejected.
- `regime_block`: EXTREME regime → rejected with `rejection_rule == "regime_block"`.
- `sizing_check`: premium > max_premium_pct_equity (equity) or notional > max_position_pct_equity (crypto) → rejected with which leg overshot.
- `buying_power`: account buying power < order cost → rejected.
- `circuit_breaker`: rolling 24h loss > limit → ALL candidates both modules rejected;
  event logged as `CIRCUIT_BREAKER`. After cooldown period elapses → approvals resume.
- `data_freshness`: bar timestamp stale → rejected.

**Combined scenario**: 6 candidates (3 equity, 3 crypto), various failures — assert
correct approve/reject counts and that per-module limits are applied independently.

**Fail-closed**: `broker.get_account()` raises `ConnectionError` → zero approvals,
all decisions have `rejection_rule == "risk_agent_failure"`.

---

### `test_execution_agent.py`

- **Equity options order**: from an approved equity `RiskDecision`, assert both legs are
  buy-to-open, correct symbols (underlying for options), correct expiry, correct strike
  type (call for leg A on LONG spread, put for leg B).
- **Crypto spot order**: from an approved crypto `RiskDecision`, assert leg A is a buy
  and leg B is a sell (for LONG spread), correct quantities, correct crypto symbols.
- **Leg A success + Leg B failure**: Leg A cancel is issued; failure logged with both
  the Leg A order ID and the Leg B error.
- **Both legs fail**: no orphaned orders; failure logged for both.
- **Log persistence**: `TradeLogEntry` attempt record exists in SQLite even on total
  execution failure — the attempt must be persisted before the first order is placed.

---

### `test_pipeline_integration.py`

- **Full mocked equity cycle**: synthetic daily bars → mocked Kalman + OU → mocked options
  chain → mocked broker; assert correct SQLite write sequence (signal → risk_decision →
  orders → trade_entry).
- **Full mocked crypto cycle**: same with hourly bars and spot orders.
- **Market hours guard**: assert equity cycle does not run when `is_market_hours()` returns False.
- **Permission boundary**: grep the source tree for `place_order` (or exact broker method
  name); assert it is imported only in `execution_agent.py`.
- **Exit monitoring**: put a synthetic open equity position in DB with z-score now
  triggering exit; assert pipeline issues sell-to-close orders on both legs.
- **Coint breakdown close**: put a synthetic open position in DB and mark its pair as
  failing the daily coint recheck; assert pipeline closes the position and logs
  `COINT_BREAKDOWN_CLOSE`.
- **LLM fallback chain**: simulate Featherless timeout → assert Groq is called; simulate
  Groq timeout → assert `sentiment_modifier = 0`.

---

## Backtest validation (before trusting live-paper)

### Equity options module (daily bars, 1–2 years)

Run `scripts/run_backtest.py --module equity`.

Options premium approximation: use Black-Scholes (scipy) with 20-day historical
realized vol as the IV estimate. This is an approximation — disclose it in `RESULTS.md`.
Do not treat the backtest P/L as precise; it is indicative.

Compute per-pair and aggregate:
- Total return, CAGR
- Sharpe ratio (annualized), Sortino ratio
- Max drawdown (% and duration in days)
- Win rate, avg win, avg loss, profit factor
- Trade count by exit reason: z-reversion / stop-z / time-stop / coint-breakdown
- Half-life distribution at entry (was the gate doing useful work?)
- Cointegration gate firing rate (how often was each pair suspended?)

### Crypto spot module (hourly bars, 6+ months)

Run `scripts/run_backtest.py --module crypto`.

Same metrics as above, holding periods in hours, z-reversion rate particularly important
(it should be the dominant exit type if OU mean reversion is working).

### Shared sanity checks (both modules, do not skip)

- Max drawdown: a number you would read aloud without wincing.
- Results not driven by 1–2 outlier trades. Check: remove the single best trade —
  does the conclusion change? If yes, say so honestly in `RESULTS.md`.
- **No lookahead bias**: confirm indicators at time `t` only use data at or before `t`.
  The Kalman filter is naturally causal. The OU estimation window must not use future
  data. Write a specific test: truncate input to end at observation `t`; assert signal
  at `t` is identical.
- Trade counts look consistent with `time_stop` parameters and observed half-lives.
- Cointegration gate p-values from backtest are similar to live values — large
  discrepancy suggests a bug in the backtest's data handling.

---

## Known limitations — document in `RESULTS.md`

> **Options premium approximation**: Backtested options P/L is computed using Black-Scholes
> with historical realized vol as a proxy for IV. Real IV can differ significantly from RV,
> especially around earnings and macro events. This approximation overstates the accuracy
> of the backtest for the equity options module.

> **Survivorship bias**: All six equity underlyings (GLD, SLV, XOM, CVX, KO, PEP) and all
> three crypto assets survived the backtest period and remained liquid. Assets that failed
> or became illiquid during that window are not represented.

> **No transaction cost model**: Alpaca paper trading does not simulate bid/ask spread
> costs on options. Real-world options spreads (especially for lower-volume names) would
> meaningfully reduce the strategy's edge.

> **No cross-module correlation**: In risk-off events, equity and crypto often move
> together. The circuit breaker handles aggregate drawdown but does not model correlation
> between the two strategy modules.
