# STRATEGY.md — Hybrid Stat Arb Trading Rules (implement exactly as specified)

This is the spec for `kalman.py`, `indicators.py`, `signal_agent.py`, and
`options_selector.py`. Every number here is a named constant in `config.yaml`.

The system runs **two simultaneous strategy modules** sharing one Kalman/OU signal engine
and one deterministic Risk Agent:

| Module | Assets | Execution | Schedule |
|---|---|---|---|
| **A — Equity Options** | GLD/SLV, XOM/CVX, KO/PEP | Buy calls + puts (directional) | Market hours 9:30–16:00 ET |
| **B — Crypto Spot** | BTC/ETH, ETH/SOL, BTC/SOL | Spot long/short spread | 24/7 |

The signal math is **identical** for both modules. Only the execution layer differs.

---

## Universe

### Module A — Equity Options Pairs
```yaml
equity_pairs:
  - [GLD, SLV]    # Gold vs Silver ETFs — commodity cointegration
  - [XOM, CVX]    # ExxonMobil vs Chevron — oil major cointegration
  - [KO, PEP]     # Coca-Cola vs PepsiCo — consumer staples duopoly
```

### Module B — Crypto Spot Pairs
```yaml
crypto_pairs:
  - [BTC/USD, ETH/USD]
  - [ETH/USD, SOL/USD]
  - [BTC/USD, SOL/USD]
```

---

## Shared Signal Pipeline (Steps 1–6 apply to BOTH modules identically)

### Step 1 — Cointegration Gate (daily, gating condition)

Before generating any signal for a pair, verify cointegration using the
**Engle-Granger two-step test** (`statsmodels.tsa.stattools.coint`).

- **Equity pairs**: use last `coint_lookback_days_equity` (config, default 90) days of
  **daily** closing log-prices.
- **Crypto pairs**: use last `coint_lookback_hours_crypto` (config, default 720 = 30 days)
  of **hourly** log-prices.

A pair is tradable only if cointegration p-value < `coint_pvalue_threshold` (default 0.05).

Recheck: equity pairs daily at market open; crypto pairs daily at midnight UTC.

If a currently-open position's pair fails the recheck:
- **Equity**: close all options legs at market open the next trading day. Log as `COINT_BREAKDOWN_CLOSE`.
- **Crypto**: close both spot legs immediately. Log as `COINT_BREAKDOWN_CLOSE`.

Log the p-value for every pair on every daily check — part of the audit trail.

---

### Step 2 — Dynamic Hedge Ratio (Kalman Filter)

Use a **Kalman filter** to estimate the time-varying hedge ratio β. Implement in
`kalman.py` as pure functions with no I/O.

**State-space formulation:**
```
Observation:  log(price_A_t) = β_t · log(price_B_t) + μ_t + ε_t,  ε_t ~ N(0, R)
State:        [β_{t+1}, μ_{t+1}] = [β_t, μ_t] + η_t,              η_t ~ N(0, Q)
```

Config parameters:
- `kalman_observation_noise` (R, default 0.001)
- `kalman_transition_noise` (Q, default 0.0001)

**Spread:** `spread_t = log(price_A_t) - β_t · log(price_B_t)`

`kalman.py` must expose:
1. `KalmanFilter` class: `update(price_a, price_b) → (beta, mu, spread)`
2. `initialize_filter(hist_prices_a, hist_prices_b) → KalmanFilter` (warm-start)

---

### Step 3 — OU Parameter Estimation

On the rolling spread window, fit an Ornstein-Uhlenbeck process:
`dX_t = κ(μ - X_t)dt + σ dW_t`

Use the discrete-time AR(1) equivalence:
```
X_{t+1} = a + b·X_t + ε   (OLS)

κ = -ln(1 + b) / Δt        (mean-reversion speed)
μ = -a / b                 (long-run mean)
σ_ou = std(ε) / sqrt(Δt)  (diffusion)
```

Lookback windows (config):
- Equity: `ou_lookback_days_equity: 30` (daily bars)
- Crypto: `ou_lookback_hours_crypto: 168` (hourly bars, = 7 days)

**Half-life:** `τ = ln(2) / κ` (in the bar-frequency time unit; convert to days/hours
as needed for gating and options expiry selection).

**Half-life gate:** only generate entry signals if:
- Equity: `halflife_min_days` < τ < `halflife_max_days` (defaults: 2 days, 20 days)
- Crypto: `halflife_min_hours` < τ < `halflife_max_hours` (defaults: 6h, 96h)

Implement as `fit_ou_parameters(spread: np.ndarray, dt: float) → OUParams` in
`indicators.py`. Pure function, unit-tested on synthetic OU data.

---

### Step 4 — Z-Score Signal

```
σ_spread = σ_ou / sqrt(2κ)        # OU stationary std (analytical)
z_t = (spread_t - μ_ou) / σ_spread
```

**Entry thresholds** (adjusted per vol regime — see Step 5):

| z_t | Signal direction |
|---|---|
| `z_t < -entry_z_threshold` | **LONG the spread** (A cheap vs B) |
| `z_t > +entry_z_threshold` | **SHORT the spread** (A rich vs B) |

Defaults: `entry_z_threshold: 1.5`, high-vol override: `entry_z_threshold_high_vol: 2.0`

**Exit conditions** (checked every cycle on open positions):

| Condition | Action |
|---|---|
| `\|z_t\| < exit_z_threshold` | Close — spread reverted |
| `\|z_t\| > stop_z_threshold` | Stop out — cointegration breaking down |
| Position age > time stop | Force close (module-specific, see below) |

Defaults: `exit_z_threshold: 0.3`, `stop_z_threshold: 3.0`

---

### Step 5 — Volatility Regime Filter

Compute realized volatility for each asset in the pair:
- Equity: 10-day realized vol (daily log-returns, annualized: `RV = std × sqrt(252)`)
- Crypto: 24h realized vol (hourly log-returns, annualized: `RV = std × sqrt(8760)`)

Regime classification per pair:

| Both assets RV below `high_vol_threshold` | NORMAL — use `entry_z_threshold` |
|---|---|
| Either asset ≥ `high_vol_threshold` | HIGH — use `entry_z_threshold_high_vol` |
| Either asset ≥ `extreme_vol_threshold` | EXTREME — no new entries, log `REGIME_BLOCK` |

Defaults: `high_vol_threshold: 0.30` (equity), `0.80` (crypto);
`extreme_vol_threshold: 0.60` (equity), `1.20` (crypto).

Log the regime for every cycle — it is part of the signal rationale.

---

### Step 6 — Sentiment Signal (Optional, LLM-Assisted)

If an LLM provider is configured (`llm_provider.py`), fetch up to 5 recent headlines
for the **base asset** (asset A) of each pair and get a `sentiment_score` in [-1, 1].

Sentiment applies as a minor entry threshold modifier only:
```
if entering LONG spread AND base asset sentiment NEGATIVE:
    effective_z_threshold += 0.15 * confidence   # harder to enter

if entering SHORT spread AND base asset sentiment POSITIVE:
    effective_z_threshold += 0.15 * confidence   # harder to enter
```

Fallback: `sentiment_modifier = 0`, log `"sentiment: unavailable, OU signal unmodified"`.
System continues — this fallback is required, not optional.

---

## Module A — Equity Options Execution

This is what makes the system hackathon-compliant. The OU signal decides *direction*;
the half-life decides *which expiry to buy*; the execution buys directional options.

### Options contract selection (implement in `options_selector.py`)

**Strike selection:** nearest ATM strike to the current price of each leg.

**Expiry selection based on OU half-life:**
```python
target_dte = round(tau_days * expiry_multiplier)   # config: expiry_multiplier = 2.5
# Snap to the nearest weekly/monthly expiry >= target_dte available in the chain
```

Rationale: if the spread takes τ days to revert, you need at least τ DTE. The 2.5×
multiplier gives time for reversion without excessive theta decay on the first day.
This is a real and defensible methodology — be ready to explain it.

**Trade structure** (LONG spread signal as example):
```
Leg A: Buy 1 ATM call on asset A   (expect A to rise)
Leg B: Buy 1 ATM put  on asset B   (expect B to fall)
```

For SHORT spread: flip calls/puts. Quantity is 1 contract each for v1 (sizing below).

**Position sizing (premium-based, quarter-Kelly):**
```python
expected_return = (abs(entry_z) - exit_z_threshold) * sigma_spread
kelly_f = expected_return / sigma_spread**2
position_f = clamp(kelly_fraction * kelly_f, 0.005, max_premium_pct_equity)

total_premium_budget = position_f * account_equity
contracts_A = max(1, floor(total_premium_budget / 2 / premium_per_contract_A))
contracts_B = max(1, floor(total_premium_budget / 2 / premium_per_contract_B))
```

Config: `kelly_fraction: 0.25`, `max_premium_pct_equity: 0.05` (5% max of equity in
total premium per spread — options have defined max loss, this is the natural cap).

**Exit for options positions:**
- On z-score reversion or stop: sell to close both legs at market.
- On time stop: close at market 1 trading day before expiry (never hold to expiration
  in v1 — assignment/pin risk is out of scope).
- On cointegration breakdown: close immediately at market.

Time stop: `time_stop_days_equity: 15` (config).

**Important caveat to note in README and RESULTS.md:** options premiums in the paper
account may not reflect real bid/ask spreads accurately. Stated P/L on options in paper
trading should be treated as approximate.

---

## Module B — Crypto Spot Execution

**Trade structure** (LONG spread signal as example):
```
Leg A: Buy  shares_A of asset A   (long the cheap asset)
Leg B: Sell shares_B of asset B   (short the rich asset)
```

**Position sizing (notional-based, quarter-Kelly):**
```python
expected_return = (abs(entry_z) - exit_z_threshold) * sigma_spread
kelly_f = expected_return / sigma_spread**2
position_f = clamp(kelly_fraction * kelly_f, 0.01, max_position_pct_equity)

leg_A_value = position_f * account_equity
shares_A = floor(leg_A_value / price_A)
shares_B = floor(shares_A * beta_t * price_A / price_B)
```

Config: `kelly_fraction: 0.25`, `max_position_pct_equity: 0.10`.

**Exit:** market orders on both legs. If either leg fails, cancel the other immediately.

Time stop: `time_stop_hours_crypto: 120` (5 days, config).

**Note on crypto shorting:** Verify that Alpaca crypto paper supports short positions
on your account before Phase 0 is marked done. If not, implement long-only single-leg
entries informed by the OU signal (document this as a limitation in RESULTS.md).

---

## Why module-specific parameters

The core math is identical across both modules, but three parameter groups must be
tuned separately because the markets have structurally different properties:

| Parameter | Equity (lower) | Crypto (higher) | Reason |
|---|---|---|---|
| `kalman_transition_noise` (Q) | 0.0001 | 0.001 | Crypto β drifts 10× faster; a slow filter lags dangerously |
| `entry_z_threshold` | 1.5 | 1.75 | Crypto has fatter tails → more false z-score spikes that don't revert |
| `stop_z_threshold` | 3.0 | 3.5 | Wider stop for crypto because gap events can transiently blow through 3.0σ without being a true cointegration breakdown |

The z-score is already normalized by σ_spread, so it is partially self-calibrating across
asset classes. But fat tails in crypto mean the tails of the actual spread distribution
are heavier than the OU model assumes — the module-specific thresholds compensate for
that model mismatch explicitly.

---

## Parameters to expose in `config.yaml`

```yaml
# --- Universe ---
equity_pairs:
  - [GLD, SLV]
  - [XOM, CVX]
  - [KO, PEP]
crypto_pairs:
  - [BTC/USD, ETH/USD]
  - [ETH/USD, SOL/USD]
  - [BTC/USD, SOL/USD]

# --- Cointegration ---
coint_pvalue_threshold: 0.05
coint_lookback_days_equity: 90
coint_lookback_hours_crypto: 720

# --- Kalman ---
# Observation noise (R): how much we trust the price data. Same for both modules —
# both feed clean exchange prices with similar measurement error characteristics.
kalman_observation_noise: 0.001

# Transition noise (Q): how fast we think β can drift between bars.
# Crypto β shifts much faster (exchange arb, liquidity fragmentation) so Q is 10× higher.
# A too-low Q on crypto causes the filter to lag during regime shifts and trade stale β.
kalman_transition_noise_equity: 0.0001
kalman_transition_noise_crypto: 0.001

# --- OU ---
ou_lookback_days_equity: 30
ou_lookback_hours_crypto: 168
halflife_min_days: 2
halflife_max_days: 20
halflife_min_hours: 6
halflife_max_hours: 96

# --- Signal thresholds ---
# Equity: tighter entry (1.5σ) — pairs like GLD/SLV have stable, near-Gaussian spreads.
# Crypto: wider entry (1.75σ) — fat tails produce more false z-score spikes that
# don't revert; entering too early increases stop-out rate significantly.
entry_z_threshold_equity: 1.5
entry_z_threshold_crypto: 1.75

# High-vol override for each module.
entry_z_threshold_high_vol_equity: 2.0
entry_z_threshold_high_vol_crypto: 2.25

# Exit threshold: same for both — once the spread is within 0.3σ of mean, take the win.
exit_z_threshold: 0.3

# Stop threshold: equity 3.0σ, crypto 3.5σ.
# Rationale: crypto gap events (exchange outages, liquidation cascades) can transiently
# spike the spread past 3.0σ and snap back within the same hour — those are NOT
# genuine cointegration breakdowns. The wider stop reduces premature stop-outs
# while still protecting against real regime changes.
stop_z_threshold_equity: 3.0
stop_z_threshold_crypto: 3.5

# --- Vol regime ---
high_vol_threshold_equity: 0.30
extreme_vol_threshold_equity: 0.60
high_vol_threshold_crypto: 0.80
extreme_vol_threshold_crypto: 1.20

# --- Options (Module A) ---
expiry_multiplier: 2.5       # target_dte = tau_days * expiry_multiplier
time_stop_days_equity: 15
max_premium_pct_equity: 0.05

# --- Crypto spot (Module B) ---
time_stop_hours_crypto: 120
max_position_pct_equity: 0.10

# --- Shared sizing ---
kelly_fraction: 0.25
max_open_equity_pairs: 3
max_open_crypto_pairs: 3

# --- Risk ---
rolling_24h_loss_limit_pct: 0.03
circuit_breaker_cooldown_hours: 4
max_data_staleness_minutes: 15

# --- Scheduling ---
poll_interval_seconds: 900        # 15 min
market_open_et: "09:30"
market_close_et: "16:00"
coint_recheck_hour_utc: 0

# --- LLM ---
llm_provider: featherless          # featherless | groq | ollama | none
llm_model: "Qwen/Qwen2.5-7B-Instruct"
llm_fallback_chain: [groq, ollama, none]
```

---

## Backtesting Requirement

- **Module A (equity options):** backtest over 1–2 years of daily bars for all 3 equity pairs.
  Approximate options premiums using Black-Scholes on historical IV (from yfinance
  `options` or CBOE data). Disclose this approximation explicitly — it is a known
  limitation.
- **Module B (crypto):** backtest over 6+ months of hourly bars for all 3 crypto pairs.
- Both modules use the same `indicators.py` and `kalman.py` — the backtest validates
  the signal engine, not just the execution.

See `TESTING.md` for sanity checks, lookahead-bias tests, and required metrics.
