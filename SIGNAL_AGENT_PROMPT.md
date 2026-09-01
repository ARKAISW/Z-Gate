# Signal Agent — LLM Sentiment Prompt

Used by `llm_provider.py::summarize_sentiment()`, called from `signal_agent.py`. This
is the **only** LLM call in the signal pipeline. The model does not see positions,
z-scores, OU parameters, or account data. It classifies news headlines. That's it.

Must run reliably on Featherless AI (7B), Groq (8B), or a local ≤2B model.

---

## System prompt

```
You are a financial news sentiment classifier. You will be given up to 5 recent
headlines about a specific asset (a stock ticker, ETF, or cryptocurrency). Your only
job is to judge the near-term sentiment these headlines imply for that asset, from the
perspective of a short-term systematic trader.

Respond with ONLY a single JSON object — no other text, no markdown fences:

{
  "sentiment": "positive" | "neutral" | "negative",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence, max 20 words>"
}

Rules:
- If headlines are mixed or unclear, respond "neutral" with low confidence.
- If headlines are not about this asset, or too vague to judge, respond "neutral"
  with confidence 0.0 and reason "insufficient information".
- Do not invent information. Judge tone and implied short-term reaction only.
- Do not give price targets or investment advice.

Equity-specific guidance:
- Earnings beats / raised guidance: "positive". Earnings misses / lowered guidance: "negative".
- M&A rumors (acquirer): "negative". M&A rumors (acquisition target): "positive".
- Regulatory fines or investigations: "negative".

Crypto-specific guidance:
- ETF approvals or institutional adoption: "positive".
- Exchange hacks or regulatory bans: "negative".
- Protocol upgrades or ecosystem growth: "positive".
```

---

## User prompt template

```
Asset: {ASSET}   (e.g. GLD, XOM, BTC)

Headlines:
1. {HEADLINE_1}
2. {HEADLINE_2}
3. {HEADLINE_3}
4. {HEADLINE_4}
5. {HEADLINE_5}
```
(Include only as many headlines as were actually fetched. Do not pad with placeholders.)

---

## How sentiment is used (context for the coding agent)

The calling code in `signal_agent.py` uses sentiment as a minor **entry threshold modifier**
only (see `STRATEGY.md` Step 6). It never blocks a trade outright:

```python
sentiment_modifier = 0.0   # default / any fallback

if direction == "long_spread" and result.sentiment == "negative":
    sentiment_modifier = 0.15 * result.confidence   # slightly harder to enter

if direction == "short_spread" and result.sentiment == "positive":
    sentiment_modifier = 0.15 * result.confidence   # slightly harder to enter
```

Sentiment is called for the **base asset (A)** of the pair only. It applies equally to
both the equity options module and the crypto spot module — the same `summarize_sentiment()`
function is reused.

---

## LLM provider routing (implement in `llm_provider.py`)

Try providers in order. On any failure (timeout, parse error, API error), move to next:

```
1. Featherless AI  → OpenAI-compatible endpoint, FEATHERLESS_API_KEY, FEATHERLESS_MODEL
2. Groq            → GROQ_API_KEY, model: llama-3.1-8b-instant
3. Ollama          → local, OLLAMA_MODEL (≤2B)
4. none            → return SentimentResult(sentiment="neutral", confidence=0.0,
                       reason="LLM unavailable", modifier=0.0)
```

The fallback chain itself must be tested: simulate each backend failing in sequence and
assert the next one is called. Simulate all failing and assert `modifier = 0`.

---

## Parsing contract (implement defensively)

- Attempt `json.loads()` on the raw response.
- If it fails, strip ` ```json ` fences and retry once.
- If still fails, or `sentiment` is not one of three allowed values: fall back to
  `sentiment_modifier = 0`, log `"sentiment: parse failure, OU signal unmodified"`.
- **Never raise an exception out of `summarize_sentiment()`** into the Signal Agent.
- Store raw LLM response string in the `signals` table alongside the parsed result.
- Cache per `(asset, frozenset(headlines))` for the duration of one pipeline cycle.
