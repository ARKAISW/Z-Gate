# AGENTS.md — Instructions for the coding agent building this project

You are building a **hybrid multi-agent statistical arbitrage system** on Alpaca for a
hackathon submission that will also be shown to quant/prop trading recruiters as a
portfolio piece. The system runs two simultaneous strategy modules:

- **Module A — Equity Options:** stat arb on GLD/SLV, XOM/CVX, KO/PEP using directional
  options (calls + puts). Hackathon-compliant: satisfies the options trading requirement.
- **Module B — Crypto Spot:** stat arb on BTC/ETH, ETH/SOL, BTC/SOL. 24/7 operation.

Both modules share one Kalman/OU signal engine and one deterministic risk agent.
Only the execution layer differs. Optimize for correctness, clarity, and auditability
over cleverness. A hiring manager reading this should think "this person understands
risk management" — not "this person can call an LLM API."

Read `ARCHITECTURE.md`, `STRATEGY.md`, and `RISK_RULES.md` before writing code — they are
the spec. Don't invent trading rules; implement exactly what those files say, and if
something is ambiguous, stop and ask rather than guessing.

## Non-negotiables

1. **Paper trading only.** Every Alpaca client must be constructed against the paper
   trading base URL. There should be no code path that can hit the live trading endpoint.
   Add a startup assertion that refuses to run if `ALPACA_PAPER` is not explicitly `true`.
2. **No secrets in the repo.** All API keys/tokens come from environment variables loaded
   via `.env` (gitignored). Commit a `.env.example` with placeholder values and comments.
3. **Risk Agent and Execution Agent contain zero LLM calls.** Their logic must be plain,
   testable Python — no prompting, no "ask the model if this trade is safe." The LLM is
   only allowed in the Signal Agent (optional sentiment summarization) and the Reflection
   Agent (post-trade review). This separation is a design decision to call out explicitly
   in the README and demo — deterministic risk logic is auditable and reproducible;
   LLM-decided risk is not, and that distinction matters a lot to a prop desk reader.
4. **Every trade decision is logged with its full rationale** (which indicators fired,
   what the risk agent checked, what was approved/rejected and why) before any order is
   placed. If it isn't logged, it didn't happen.
5. **Fail closed.** Any error, timeout, or unexpected API response in the Risk or
   Execution agent must result in "do not trade," never a silent default to "trade
   anyway." Log the failure loudly.
6. **Backtest before you trust it live-paper.** Don't wire the Execution Agent to real
   paper orders until the strategy has been run through the backtest module in
   `TESTING.md` and the results look sane (see the sanity checks listed there).

## Tech stack

- **Language:** Python 3.11+
- **Data models:** `pydantic` for every structured object passed between agents (signals,
  risk decisions, orders, trade log entries). No raw dicts crossing agent boundaries.
- **Alpaca access:** `alpaca-py` SDK (crypto-enabled), or the Alpaca MCP server if the
  coding environment supports MCP tool calls directly — either is fine, but keep the
  Alpaca client wrapped behind a single `broker.py` interface so the rest of the code
  doesn't care which one you used. Use the paper trading base URL only.
- **Quant math:** `numpy`, `scipy`, `statsmodels` — Kalman filter in `kalman.py` (pure
  Python/numpy), OU parameter estimation and cointegration test via `statsmodels`
  (`tsa.stattools.coint`, `tsa.ar_model.AutoReg`). No ML frameworks needed.
- **Persistence:** SQLite via `sqlmodel` or plain `sqlite3` — no external DB needed for a
  hackathon-scale project.
- **Scheduling:** a simple polling loop (every 15 min, 24/7 — crypto has no market hours)
  using a plain `while True` loop with `time.sleep`. The daily midnight UTC jobs
  (cointegration recheck, Reflection Agent) are triggered by checking the UTC hour
  inside the same loop. No distributed job queue — that's over-engineering.
- **Config:** one `config.yaml` for strategy parameters (thresholds, pair list, sizing,
  OU lookbacks) so a reviewer can see the exact rules without reading code, and one
  `.env` for secrets/infra.
- **Testing:** `pytest`. See `TESTING.md` for required coverage including Kalman
  convergence tests and OU parameter accuracy tests.
- **Formatting/linting:** `ruff` + `black`. Type hints everywhere; run `mypy` if time
  allows.
- **Logging:** Python `logging` module with structured (JSON) file output for the trade
  log, plus human-readable console output for the demo.

## LLM usage constraints — read carefully

- **Primary:** Featherless AI — OpenAI-compatible API, FEATHERLESS_API_KEY, target model
  `Qwen/Qwen2.5-7B-Instruct` (or similar 7B instruction model). $25 credit available.
- **Fallback 1:** Groq free tier — `llama-3.1-8b-instant`. Use if Featherless fails.
- **Fallback 2:** Local Ollama — ≤2B parameter model. Use if both above fail.
- **Fallback 3:** `none` — `sentiment_modifier = 0`, system continues on pure OU signals.

The LLM's job is narrow: given up to 5 recent headlines for an asset, return a
structured sentiment judgment (`positive` / `neutral` / `negative` + confidence + one
sentence reason). It does **not** decide trades, does **not** see account/position data
or z-scores, and its output only nudges the entry z-threshold by ±0.15 × confidence.

Build `llm_provider.py` with a single `summarize_sentiment(asset, headlines) → SentimentResult`
function and a pluggable backend. The fallback chain (Featherless → Groq → Ollama → none)
must be tested explicitly. If all providers fail, the system degrades to technical-only
signals — this path is required and must be tested.

## Build order

Follow `ROADMAP.md` phase by phase. Do not skip ahead to the Execution Agent before the
Risk Agent has its own passing unit tests — order placement is the one part of this
system where a bug has consequences (even on paper, bad logs = a useless portfolio piece).

## Definition of done for each phase

- Code has type hints and docstrings.
- Unit tests exist and pass (`pytest`) for any new deterministic logic.
- No hardcoded credentials, tickers-as-magic-strings-everywhere, or bare `except:`.
- New config values are added to `config.yaml` / `.env.example`, not hardcoded.
- The trade/decision log schema is updated if the phase adds new fields, and old log
  entries remain readable (don't silently break the log format).

## What NOT to build (scope control)

- **No Heston model or Black-Scholes for trade decisions.** BS is used only in the
  backtest to approximate historical options premiums — it is not used in live signal
  generation or order sizing. Using BS to decide whether an option is "cheap" or to
  price live orders is out of scope and would read as a category error to a quant reviewer.
- **No ML-learned alpha.** No neural networks, no RL, no hyperparameter search. The
  edge is the Kalman-filtered cointegration relationship — explainable and reproducible.
  A black box with a suspiciously good equity curve is worth less to a prop desk than
  a clean, honest factor model.
- **No multi-strategy ensemble** — one strategy instance per pair, six pairs total
  (three equity, three crypto). That's it.
- **No equity trading without options.** The equity module uses options (calls + puts)
  as the execution vehicle. Do not add a plain equity spot module — it doesn't satisfy
  the hackathon requirement and muddies the architecture.
- **No expiration management complexity.** Close all options positions before expiry
  (time-stop fires ≥ 1 day before). No assignment handling, no rolling, no pin risk
  management in v1.
- **No real-money code paths, no live trading toggle.** If that day comes, it gets its
  own reviewed change with its own safety checks — never a flag flipped in this repo.
