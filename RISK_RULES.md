# RISK_RULES.md — Risk Agent specification (deterministic, zero LLM)

This is the spec for `risk_agent.py`. Every rule is a separate, independently
unit-testable function returning a clear pass/fail + reason string. The Risk Agent's
output is a list of `RiskDecision` objects — never a bare boolean.

The Risk Agent handles **both modules** (equity options and crypto spot) using the same
rule set, with module-specific thresholds read from `config.yaml`. It does not need to
know the execution details — it only approves/rejects and returns a `SpreadOrderRequest`.

---

## Inputs

- All `SpreadSignal` objects generated this cycle (pre-filtered by Signal Agent gates).
- Current account state from `broker.py`: equity, buying power, open positions.
- Open trades from SQLite: current count per module, rolling 24h P/L.

---

## Rules (applied in order; first failing rule rejects the candidate)

### 1. Max concurrent open pairs (per module)
Reject if the number of currently open positions for that module equals or exceeds:
- `max_open_equity_pairs` (config, default 3) for equity options
- `max_open_crypto_pairs` (config, default 3) for crypto spot

Both limits are checked independently — a full equity book does not block new crypto
entries and vice versa.

### 2. Duplicate pair check
Reject if a position for the same `pair_id` is already open in that module. No stacking,
no averaging in — one position per pair per module at any time.

### 3. Half-life gate (re-validation)
Reject if the signal's `half_life` is outside the module-specific bounds:
- Equity: [halflife_min_days, halflife_max_days]
- Crypto: [halflife_min_hours, halflife_max_hours]

The Signal Agent pre-filters this, but the Risk Agent independently re-validates the
value from the signal object — defense in depth.

### 4. Cointegration p-value re-validation
Reject if `signal.coint_pvalue > coint_pvalue_threshold`. Same defense-in-depth logic.

### 5. Volatility regime block
Reject if `signal.vol_regime == "EXTREME"`. Log as `REGIME_BLOCK` — this event must
be distinctly visible in the dashboard, separate from ordinary rejections.

### 6. Position sizing sanity check
Compute the position size using the appropriate sizing formula from `STRATEGY.md`:
- **Equity options:** total premium for both legs must not exceed
  `max_premium_pct_equity × account_equity`.
- **Crypto spot:** each leg's notional must not exceed `max_position_pct_equity × account_equity`.

If sizing overshoots the cap, **reject** — do not silently downsize. Log which leg
overshot and by how much. Predictable, auditable behavior is more important than
always trading.

### 7. Buying power check
Reject if `buying_power < total_cost_of_order`:
- Equity options: total_cost = sum of both leg premiums.
- Crypto spot: total_cost = sum of both leg notionals.

Standard sanity check against stale account data.

### 8. Rolling 24h loss circuit breaker
If the account's realized + unrealized P/L over the last rolling 24 hours is ≤
`-rolling_24h_loss_limit_pct × account_equity` (config, default -3%):
- Reject **all** new entries across **both modules** for `circuit_breaker_cooldown_hours`
  (config, default 4 hours).
- Log as `CIRCUIT_BREAKER` — must be prominently flagged in the trade log and dashboard.
- The cooldown is based on the timestamp of the trigger event, not reset each cycle.
- Existing open positions are not force-closed — only new entries are blocked.

### 9. Data freshness check
Reject if the bar/quote data timestamp used to generate the signal is older than
`max_data_staleness_minutes` (config, default 15 minutes).
- Equity: during market hours, bars should be at most 15 minutes old.
- Crypto: 24/7 bars should always be fresh — staleness signals an API or connectivity issue.

---

## Output

```python
class RiskDecision(BaseModel):
    pair_id: str
    module: Literal["equity", "crypto"]
    signal: SpreadSignal
    approved: bool
    rejection_reason: str | None       # None if approved
    rejection_rule: str | None         # rule name, e.g. "circuit_breaker", "regime_block"
    sized_order: SpreadOrderRequest | None   # only present if approved
    checked_at: datetime
```

Every decision — approved or rejected — is written to `risk_decisions` table. Rejections
demonstrate risk discipline to any reviewer.

---

## Fail-closed behavior

If any rule's underlying data fetch (account state, buying power, P/L from SQLite)
throws an exception or times out:
- Reject **all** candidates for that cycle across both modules.
- Log the failure at ERROR level with full exception trace.
- Write a `RiskDecision` with `rejection_rule = "RISK_AGENT_FAILURE"` for each candidate.
- Never proceed with partial or assumed data.

This behavior must have a dedicated unit test: simulate `broker.get_account()` raising
`ConnectionError` and assert zero approvals and the failure is logged.

---

## Explicitly out of scope for v1

- Cross-pair correlation limits (e.g. holding BTC/ETH and BTC/SOL simultaneously
  concentrates BTC risk — v2 improvement).
- Cross-module correlation (equity and crypto can move together in risk-off events —
  not modeled here).
- Options Greeks monitoring (delta drift as underlying moves, gamma risk near expiry).
- Intraday drawdown on individual open legs (only spread P/L is monitored).

List these in `README.md` and `RESULTS.md` as known next steps. Acknowledging what is
missing is a positive signal for a quant/prop audience.
