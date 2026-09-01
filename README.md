# Z-Gate: Deterministic Hybrid Multi-Agent Statistical Arbitrage System

**Repository:** [https://github.com/ARKAISW/Z-Gate](https://github.com/ARKAISW/Z-Gate)  
**Execution Venue:** Alpaca Markets (Paper Trading API)  
**Target Asset Classes:** Equity Options (Market Hours) & Cryptocurrency Spot (24/7)  
**Compliance Guard:** Hard assertion on paper trading endpoint (`ALPACA_PAPER=true`)  

---

## 1. System Overview

Z-Gate is a production-grade, multi-agent statistical arbitrage system engineered for systematic pairs trading across equities and digital assets. The architecture separates statistical modeling, risk management, and order routing across discrete agent boundaries to maximize auditability, reproducibility, and risk containment.

The core alpha engine models continuous synthetic spread dynamics through a State-Space Kalman Filter and fits an Ornstein-Uhlenbeck (OU) mean-reverting stochastic process. A defining feature of the options execution module is the analytical linkage between the estimated OU mean-reversion half-life ($\tau$) and option contract expiration ($DTE$), ensuring directional options are purchased with an optimal duration window relative to the expected speed of reversion.

```
+---------------------------------------------------------------------------------------------+
|                                    Z-GATE ARCHITECTURE                                      |
+---------------------------------------------------------------------------------------------+
|                                                                                             |
|   +--------------------------+                         +--------------------------------+   |
|   |       MARKET DATA        |                         |         LLM PROVIDERS          |   |
|   | Equity (1D) / Crypto (1H)|                         | Featherless / Groq / Ollama    |   |
|   +------------+-------------+                         +---------------+----------------+   |
|                |                                                       |                    |
|                v                                                       v                    |
|   +--------------------------------------------------------------------+----------------+   |
|   |                                  SIGNAL AGENT                                       |   |
|   |  - Engle-Granger Cointegration Gate (p <= 0.05)                                     |   |
|   |  - State-Space Kalman Filter (Dynamic Beta Extraction)                              |   |
|   |  - Ornstein-Uhlenbeck Parameter Fitting (mu, theta, sigma, half-life tau)           |   |
|   |  - Realized Volatility Regime Classification (Normal / High / Extreme)              |   |
|   |  - Asymmetric Sentiment Nudge (+/- 0.15z max)                                       |   |
|   +------------------------------------+------------------------------------------------+   |
|                                        |                                                    |
|                                        v (Signal Model)                                     |
|   +-------------------------------------------------------------------------------------+   |
|   |                                   RISK AGENT                                        |   |
|   |                    100% Deterministic Python / Zero LLM Calls                       |   |
|   |  - 9 Hard Gating Rules (Capital, Half-Life, Cointegration, Volatility, Cooldown)    |   |
|   |  - Fractional Kelly Sizing (1.5% Max Portfolio Risk at Stop)                        |   |
|   +------------------------------------+------------------------------------------------+   |
|                                        |                                                    |
|                                        v (Approved Decision)                                |
|   +-------------------------------------------------------------------------------------+   |
|   |                                 EXECUTION AGENT                                     |   |
|   |                    100% Deterministic Python / Zero LLM Calls                       |   |
|   |  - Module A: Equity Options (+/-15% ATM Strike Band, DTE >= 5, Buy Calls/Puts)      |   |
|   |  - Module B: Crypto Spot (Market-Neutral Long/Short Pairs Sizing)                   |   |
|   |  - Automated 5-Minute Tick Exit Engine (Mean-Reversion, Stop-Loss, Time-Stop)       |   |
|   +------------------------------------+------------------------------------------------+   |
|                                        |                                                    |
|                                        v                                                    |
|   +-------------------------------------------------------------------------------------+   |
|   |                             ALPACA PAPER BROKER API                                 |   |
|   +------------------------------------+------------------------------------------------+   |
|                                        |                                                    |
|                                        v (Nightly Closed Trades)                            |
|   +-------------------------------------------------------------------------------------+   |
|   |                              REFLECTION AGENT                                       |   |
|   |  - Nightly Post-Trade Structured Review & Hypothesis Verification (LLM)             |   |
|   +-------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical & Quantitative Framework

### 2.1 Cointegration Testing (Engle-Granger Two-Step)
For an asset pair $(Y_t, X_t)$, we test for the existence of a stationary linear combination in log-price space:

$$\ln(Y_t) = \alpha + \beta \ln(X_t) + \epsilon_t$$

The residuals $\epsilon_t$ are evaluated using the Augmented Dickey-Fuller (ADF) test:

$$\Delta \epsilon_t = \gamma \epsilon_{t-1} + \sum_{i=1}^{p} \psi_i \Delta \epsilon_{t-i} + u_t$$

A strict rejection threshold of $p \le 0.05$ is enforced. If the hypothesis of a unit root cannot be rejected at the 95% confidence level, the pair is barred from trade generation.

### 2.2 Dynamic Hedge Ratio Estimation (State-Space Kalman Filter)
Traditional static Ordinary Least Squares (OLS) suffers from lookahead bias and fails to account for temporal drift in asset co-movements. We implement a recursive Kalman Filter:

$$\text{State Equation: } \theta_t = \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$\text{Measurement Equation: } y_t = H_t \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

Where:
* $\theta_t = [\alpha_t, \beta_t]^T$ is the unobserved state vector (intercept and time-varying hedge ratio).
* $H_t = [1, x_t]$ is the measurement matrix constructed from the independent asset's log-price.
* $Q$ is the process noise covariance matrix governing the rate of parameter adaptation.
* $R$ is the observation noise variance.

The instantaneous spread $S_t$ is extracted recursively on every closed bar:

$$S_t = \ln(Y_t) - (\alpha_t + \beta_t \ln(X_t))$$

### 2.3 Ornstein-Uhlenbeck (OU) Process Calibration
The continuous spread dynamics are modeled via an Ornstein-Uhlenbeck mean-reverting stochastic differential equation:

$$dS_t = \kappa (\mu - S_t) dt + \sigma dW_t$$

Where:
* $\kappa > 0$ is the mean-reversion speed.
* $\mu$ is the long-term equilibrium spread.
* $\sigma > 0$ is the spread diffusion volatility.
* $W_t$ is a standard Brownian motion.

Discretizing the SDE over interval $\Delta t$ yields an Auto-Regressive AR(1) model:

$$S_t = a + b S_{t-1} + \eta_t$$

The structural parameters are extracted analytically via Ordinary Least Squares:

$$b = e^{-\kappa \Delta t} \implies \kappa = -\frac{\ln(b)}{\Delta t}$$
$$\mu = \frac{a}{1 - b}$$
$$\sigma = \sigma_\eta \sqrt{\frac{-2 \ln(b)}{\Delta t (1 - b^2)}}$$

The half-life of mean-reversion $\tau$ is given by:

$$\tau = \frac{\ln(2)}{\kappa}$$

Pairs with an estimated half-life exceeding risk limits ($\tau > 30\text{ days}$ for Equities, $\tau > 96\text{ hours}$ for Crypto) or demonstrating non-reverting drift ($\kappa \le 0$) are rejected.

### 2.4 Z-Score Standardization
The normalized signal $Z_t$ measures statistical displacement from the equilibrium mean:

$$Z_t = \frac{S_t - \mu}{\sigma_{\text{equilibrium}}}, \quad \text{where } \sigma_{\text{equilibrium}} = \frac{\sigma}{\sqrt{2\kappa}}$$

* **Long Spread Entry ($Z_t \le -1.50\sigma$):** Asset $Y$ is underpriced relative to Asset $X$.
* **Short Spread Entry ($Z_t \ge +1.50\sigma$):** Asset $Y$ is overpriced relative to Asset $X$.
* **Mean-Reversion Exit ($|Z_t| \le 0.30\sigma$):** Spread has normalized to fair value.
* **Stop-Loss ($|Z_t| \ge 3.00\sigma\text{ / }3.50\sigma$):** Structural regime failure cutoff.

---

## 3. Options Expiration Linkage & Execution Mechanics

### 3.1 Expiration Calibration
Options trading in statistical arbitrage often suffers from premature theta decay (buying short-dated contracts) or excessive premium drag (buying long-dated contracts). We link the target Days to Expiration ($DTE$) directly to the calibrated OU half-life:

$$DTE_{\text{target}} = \text{clamp}\left(\tau_{\text{days}} \times 2.5, \, 7, \, 30\right)$$

This guarantees sufficient duration for the spread to complete its reversion cycle while avoiding the terminal gamma and theta acceleration of contracts with $DTE < 5$.

### 3.2 Directional Options Execution (Defined Risk)
To satisfy market neutrality and eliminate naked short margin requirements, options positions are constructed as synthetic directional synthetic wings:
* **Long Spread ($Z \le -1.50\sigma$):** Buy ATM Call on Asset $Y$ + Buy ATM Put on Asset $X$.
* **Short Spread ($Z \ge +1.50\sigma$):** Buy ATM Put on Asset $Y$ + Buy ATM Call on Asset $X$.

Liquid strike selection is enforced by querying a dynamic $\pm 15\%$ strike band centered on the underlying spot price, snapping to the highest open-interest contracts.

---

## 4. Multi-Agent Design & Deterministic Safety Boundary

To ensure complete compliance with institutional risk standards, LLM inference is strictly quarantined:

| Agent Component | Technology | Role & Authority | Gating Nature |
|---|---|---|---|
| **Signal Agent** | Pure Python (`numpy`, `statsmodels`) + LLM Sentiment | Computes Kalman/OU statistics, nudges entry $Z$ by $\pm 0.15$ max | Advisory only |
| **Risk Agent** | 100% Deterministic Python (`pydantic`) | Enforces 9 hard mathematical risk constraints | Absolute Gate (Zero LLM) |
| **Execution Agent** | 100% Deterministic Python (`alpaca-py`) | Dispatches market orders, manages stops, logs trades | Execution Authority (Zero LLM) |
| **Reflection Agent** | LLM (`Featherless 7B` / `Groq` / `Ollama`) | Nightly post-mortem analysis of closed trades | Non-blocking telemetry |

### 4.1 The 9 Hard Risk Gating Rules
1. **Concurrency Cap:** Maximum 5 simultaneous equity pairs, 3 simultaneous crypto pairs.
2. **Duplicate Check:** Rejects new entries on already active pair IDs.
3. **Half-Life Window:** $\tau \in [2, 30]\text{ days}$ (Equity), $\tau \in [4, 96]\text{ hours}$ (Crypto).
4. **Cointegration Gate:** $p\text{-value} \le 0.05$ (Engle-Granger ADF test).
5. **Volatility Regime Filter:** Realized volatility $\le 1.00$ (Equity) / $\le 2.00$ (Crypto).
6. **Sizing Allocation Cap:** 15% notional per crypto pair, 5% premium budget per equity pair.
7. **Buying Power Validation:** Margin requirements must not exceed available capital.
8. **Circuit Breaker:** Rolling 24-hour loss $\ge 5\%$ triggers a 2-hour trading freeze.
9. **Data Freshness Guard:** Rejects bar feeds older than 120 minutes (Crypto) or 4 days (Equity).

---

## 5. Multi-Sector Universe

The strategy monitors 12 liquid, economically cointegrated pairs across 8 sectors:

```
Sector / Class        Pair Identifier       Economic Rationale
---------------------------------------------------------------------------------------------
Commodities           GLD / SLV             Precious metals monetary & industrial co-movement
Energy                XOM / CVX             Integrated global oil majors crack spread basis
Consumer Staples      KO / PEP              Global non-alcoholic beverage duopoly
Tech Megacaps         GOOGL / META          Digital advertising revenue & capex cycle
Semiconductors        NVDA / AMD            Advanced GPU computing & data center hardware
Payment Rails         V / MA                Global electronic transaction network duopoly
Banking               JPM / BAC             Money-center commercial banking net interest margins
Retail                HD / LOW              Home improvement consumer discretionary spending
Digital Assets 24/7   BTC/USD - ETH/USD     Layer 1 digital store-of-value vs smart contract basis
Digital Assets 24/7   ETH/USD - SOL/USD     High-throughput smart contract platform competition
Digital Assets 24/7   BTC/USD - SOL/USD     Crypto macro asset vs high-beta Layer 1
Digital Assets 24/7   LINK/USD - ETH/USD    DeFi oracle infrastructure vs Ethereum network
```

---

## 6. Backtest Performance & Statistical Verification

Historical simulations were executed using a causal rolling backtest engine with exact Black-Scholes options pricing and discrete transaction modeling:

### 6.1 Multi-Year Performance Summary

```
========================================================================================
  MODULE A: Equity Options Backtest (8 Pairs / 1,000 Daily Bars / ~4 Years)
========================================================================================
  Initial Capital:          $100,000.00
  Final Capital:            $178,456.91
  Total Net Return:         +78.46%
  Annualized Return (CAGR): +15.71%
  Sharpe Ratio:             1.57
  Sortino Ratio:            1.10
  Profit Factor:            3.52 (Generated $3.52 in return for every $1.00 lost)
  Win / Loss Payoff Ratio:  2.06 : 1 (Average Win is double the Average Loss)
  Win Rate:                 63.0% (46 Wins / 27 Losses across 73 trades)
  Maximum Drawdown:         5.95%
  Primary Exit Trigger:     Z-Score Mean Reversion (58% of all exits)
========================================================================================

========================================================================================
  MODULE B: Crypto Spot Backtest (4 Pairs / 5,000 Hourly Bars / ~7 Months)
========================================================================================
  Initial Capital:          $100,000.00
  Final Capital:            $102,266.56
  Total Net Return:         +2.27%
  Annualized Return (CAGR): +4.00%
  Sharpe Ratio:             1.75
  Profit Factor:            2.80
  Win Rate:                 63.6% (7 Wins / 4 Losses across 11 trades)
  Maximum Drawdown:         0.68%
  Primary Exit Trigger:     Z-Score Mean Reversion (91% of all exits)
========================================================================================
```

$$\mathbf{\text{Aggregate Multi-Asset Net Return: } \mathbf{+80.72\%} \quad \Big| \quad \text{Composite Sharpe Ratio: } \mathbf{1.62} \quad \Big| \quad \text{Max Portfolio Drawdown: } \mathbf{< 6.0\%}}$$

---

## 7. Installation & Quick Start

### 7.1 Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/ARKAISW/Z-Gate.git
cd Z-Gate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Alpaca Paper Trading API keys
```

### 7.2 Running Unit Tests

```bash
pytest tests/ -v
# 107 passed in ~4.0s
```

### 7.3 Executing Historical Backtests

```bash
# Run multi-year full universe simulation
python -m scripts.run_backtest --module all --bars-equity 1000 --bars-crypto 5000 --save-results
```

### 7.4 Launching Live Paper Trading & Monitoring

```bash
# Terminal 1: Launch 24/7 Live Paper Trading Daemon
python scripts/run_live_paper.py

# Terminal 2: Launch Real-Time CLI Terminal Monitor
python scripts/terminal_monitor.py

# Terminal 3 (Optional): Launch Interactive Web Dashboard
streamlit run scripts/dashboard.py
```

---

## 8. Zero-Cost 24/7 Cloud Deployment Guide

To maintain continuous 24/7 operation without keeping a personal computer powered on, use any of the following 100% free hosting options:

### Option A: Render.com (Free Background Worker)
1. Fork / push this repository to your GitHub account.
2. Sign up at [Render.com](https://render.com) (Free Tier).
3. Create a **New Background Worker** and link your `Z-Gate` repository.
4. Set Environment Variables:
   * `ALPACA_API_KEY`: `your_paper_key`
   * `ALPACA_SECRET_KEY`: `your_paper_secret`
   * `ALPACA_PAPER`: `true`
5. Set Build Command: `pip install -r requirements.txt`
6. Set Start Command: `python scripts/run_live_paper.py`

### Option B: Koyeb Free Serverless Container
1. Sign up at [Koyeb.com](https://www.koyeb.com) (Free Eco Tier).
2. Create an App using the repository's native `Dockerfile`.
3. Add environment variables from `.env`.
4. Deploy the container.

### Option C: Docker Containerization (Any VPS / Server)
```bash
docker compose up -d
```

---

## 9. Repository Structure

```
Z-Gate/
├── config.yaml                     # Master strategy & risk parameter configuration
├── Dockerfile                      # Production container build specification
├── docker-compose.yml              # Trader daemon & dashboard orchestration
├── requirements.txt                # Pinned production dependencies
├── .env.example                    # Template for paper keys & LLM endpoints
├── .gitignore                      # Strict exclusion of secrets, logs, and databases
├── README.md                       # Quantitative specification & system documentation
├── data/                           # Local SQLite persistence (data/trades.db)
├── logs/                           # Structured JSON execution logs
├── results/                        # Exported backtest summary metrics
├── scripts/
│   ├── run_live_paper.py           # 24/7 main live execution loop
│   ├── run_backtest.py             # Multi-year historical simulation engine
│   ├── terminal_monitor.py         # Real-time ASCII live desk monitor
│   └── dashboard.py                # Streamlit quantitative web interface
├── src/
│   ├── broker.py                   # Hard-gated Alpaca Paper Trading SDK wrapper
│   ├── kalman.py                   # Pure Python state-space Kalman Filter
│   ├── indicators.py               # OU parameter fitting, ADF coint, volatility
│   ├── options_selector.py         # DTE half-life snapping & liquid ATM resolution
│   ├── llm_provider.py             # Pluggable multi-provider fallback engine
│   ├── agents/
│   │   ├── signal_agent.py         # Cross-asset econometric signal generator
│   │   ├── risk_agent.py           # Deterministic 9-rule safety & sizing gate
│   │   ├── execution_agent.py      # Alpaca order dispatcher & 5-min exit manager
│   │   └── reflection_agent.py     # Nightly LLM post-mortem review agent
│   └── persistence/
│       ├── schema.py               # Pydantic & SQLModel schema definitions
│       └── db.py                   # SQLite transactional repository layer
└── tests/                          # 107 Unit tests covering all deterministic components
```

---

## 10. License & Compliance Disclaimer

This software is released for educational and research purposes only. Execution is strictly restricted to paper trading environments. Nothing herein constitutes financial, investment, or legal advice.
