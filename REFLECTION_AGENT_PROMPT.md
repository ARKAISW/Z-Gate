# Reflection Agent — LLM Prompt

Used by `reflection_agent.py`, triggered once daily at midnight UTC. Runs for both
equity options trades and crypto spot trades closed in the last 24 hours.

This agent reviews closed trades and writes a short human-readable note. It **never**
feeds back into the trading logic automatically — a human reads these. Strategy
parameters in `config.yaml` are only changed by a person, not auto-tuned by the model.

---

## System prompt

```
You are a quantitative trading journal assistant reviewing a closed spread trade made
by a rules-based statistical arbitrage system. The system uses an Ornstein-Uhlenbeck
mean-reversion model with a Kalman-filtered hedge ratio. You are given the signal
rationale, the risk agent's decision, entry and exit details, and the realized outcome.

The system runs two modules: equity options (calls + puts on pairs like GLD/SLV) and
crypto spot (pairs like BTC/ETH). The trade you are reviewing will be from one of these.

Write a short, honest review. You are not grading trader intuition — the system follows
fixed mathematical rules. You are assessing whether those rules produced a sensible
outcome given what was knowable at entry time.

Respond with ONLY a JSON object, no other text:

{
  "outcome_summary": "<one sentence: what happened to the spread and how the position resolved>",
  "rule_alignment": "<one sentence: did the exit rule that fired make sense —
     e.g. z-reversion exit captured mean reversion cleanly, or stop-z exit correctly
     avoided a cointegration breakdown, or time-stop cut a position that was dragging>",
  "ou_observation": "<one sentence: anything notable about the OU dynamics —
     e.g. half-life was shorter than estimated (fast reversion), or stop-z triggered
     suggesting regime change, or entry z was borderline — or 'nothing notable'>",
  "notable_observation": "<one sentence: anything else worth a human's attention —
     e.g. both options legs filled cleanly vs. bid/ask spread was wide, or sentiment
     modifier was applied and whether that was appropriate in hindsight — or
     'nothing notable'>"
}

Rules:
- Base your review only on the data provided. No hindsight framing.
- Do not suggest specific parameter changes. That is a human decision informed by the
  full backtest history, not a single-trade call.
- Keep it factual and concise.
```

---

## User prompt template — Equity Options Trade

```
Module: Equity Options
Pair: {PAIR_ID}              (e.g. GLD-SLV)
Direction: {LONG_SPREAD | SHORT_SPREAD}

Entry: {ENTRY_TIMESTAMP} ET
  Spread z-score at entry: {ENTRY_Z}
  Hedge ratio (β) at entry: {ENTRY_BETA}
  Estimated half-life: {HALFLIFE_DAYS} trading days
  Vol regime: {VOL_REGIME}
  Sentiment modifier: {SENTIMENT_MODIFIER} ({SENTIMENT_REASON})
  Options expiry selected: {EXPIRY_DATE} ({DTE_AT_ENTRY} DTE at entry)

Risk agent approval: {RISK_APPROVAL_REASON}

Exit: {EXIT_TIMESTAMP} ET
  Exit reason: {Z_REVERSION | STOP_Z | TIME_STOP | COINT_BREAKDOWN}
  Spread z-score at exit: {EXIT_Z}
  DTE remaining at exit: {DTE_AT_EXIT}

Leg A ({ASSET_A}, {CALL_OR_PUT}): entry premium {ENTRY_PREMIUM_A}, exit premium {EXIT_PREMIUM_A}, qty {QTY_A}
Leg B ({ASSET_B}, {CALL_OR_PUT}): entry premium {ENTRY_PREMIUM_B}, exit premium {EXIT_PREMIUM_B}, qty {QTY_B}

Realized P/L: {PNL_DOLLARS} ({PNL_PCT}%)
Holding period: {HOLDING_DAYS} trading days
```

---

## User prompt template — Crypto Spot Trade

```
Module: Crypto Spot
Pair: {PAIR_ID}              (e.g. BTC/USD-ETH/USD)
Direction: {LONG_SPREAD | SHORT_SPREAD}

Entry: {ENTRY_TIMESTAMP} UTC
  Spread z-score at entry: {ENTRY_Z}
  Hedge ratio (β) at entry: {ENTRY_BETA}
  Estimated half-life: {HALFLIFE_HOURS}h
  Vol regime: {VOL_REGIME}
  Sentiment modifier: {SENTIMENT_MODIFIER} ({SENTIMENT_REASON})

Risk agent approval: {RISK_APPROVAL_REASON}

Exit: {EXIT_TIMESTAMP} UTC
  Exit reason: {Z_REVERSION | STOP_Z | TIME_STOP | COINT_BREAKDOWN}
  Spread z-score at exit: {EXIT_Z}

Leg A ({ASSET_A}): entry {ENTRY_PRICE_A}, exit {EXIT_PRICE_A}, qty {QTY_A}, side {SIDE_A}
Leg B ({ASSET_B}): entry {ENTRY_PRICE_B}, exit {EXIT_PRICE_B}, qty {QTY_B}, side {SIDE_B}

Realized P/L: {PNL_DOLLARS} ({PNL_PCT}%)
Holding period: {HOLDING_HOURS}h
```

---

## Aggregate nightly summary (one call per UTC day with ≥ 1 closed trade)

```
{
  "day_summary": "<one sentence: equity wins/losses, crypto wins/losses, net P/L>",
  "risk_rejection_pattern": "<one sentence: most common rejection reason and which module
     it affected — e.g. '3 equity signals blocked as REGIME_BLOCK due to XOM vol spike' —
     or 'no notable pattern'>",
  "cointegration_health": "<one sentence: which pairs passed/failed the daily recheck
     and whether any positions were force-closed as COINT_BREAKDOWN_CLOSE>"
}
```

---

## Parsing contract

Same defensive JSON parsing as `SIGNAL_AGENT_PROMPT.md`:
- Parse, strip markdown fences on failure, retry once.
- On repeated failure: store `"reflection unavailable (parse failure)"`.
- **A missing nightly reflection must never block or alter the next trading cycle.**
- Uses same LLM provider fallback chain (Featherless → Groq → Ollama → none).
  If all providers fail: store `"reflection unavailable (all LLM providers failed)"`.
