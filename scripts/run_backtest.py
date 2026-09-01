"""scripts/run_backtest.py — Historical backtest runner for Equity Options and Crypto Spot modules.

Usage:
  python -m scripts.run_backtest --module equity
  python -m scripts.run_backtest --module crypto
  python -m scripts.run_backtest --module all

Features:
  - Strict causal data handling (no lookahead bias).
  - Kalman filter dynamic beta + rolling discrete-time OU parameter estimation.
  - Black-Scholes options pricing approximation with rolling realized vol as IV proxy.
  - Crypto spot exact notional P/L tracking.
  - Performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor,
    and exit reason distribution.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from statsmodels.tsa.stattools import coint

from src.indicators import (
    OUFitError,
    classify_vol_regime,
    compute_kelly_fraction,
    compute_log_returns,
    compute_realized_vol,
    compute_z_score,
    fit_ou_parameters,
)
from src.kalman import KalmanFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Black-Scholes European Options Pricing Helper
# ---------------------------------------------------------------------------


def black_scholes_price(
    spot: float,
    strike: float,
    dte_days: float,
    iv: float,
    rate: float = 0.04,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute Black-Scholes price for an option contract.

    Used ONLY in backtesting to approximate options premiums.
    """
    if spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    t = max(dte_days / 365.0, 1e-4)
    sigma_sqrt_t = iv * math.sqrt(t)

    d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    if option_type.lower() == "call":
        price = spot * norm.cdf(d1) - strike * math.exp(-rate * t) * norm.cdf(d2)
    else:
        price = strike * math.exp(-rate * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return max(0.01, float(price))


# ---------------------------------------------------------------------------
# Backtest Data Containers
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    pair_id: str
    module: Literal["equity", "crypto"]
    direction: Literal["long", "short"]
    entry_bar: int
    entry_date: str
    entry_z: float
    entry_beta: float
    entry_price_a: float
    entry_price_b: float
    qty_a: float
    qty_b: float
    target_dte: int = 15
    exit_bar: int = 0
    exit_date: str = ""
    exit_z: float = 0.0
    exit_reason: str = ""
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    holding_period_bars: int = 0


@dataclass
class BacktestMetrics:
    module: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_usd: float
    avg_loss_usd: float
    win_loss_ratio: float
    exit_reasons: dict[str, int] = field(default_factory=dict)
    coint_pass_rate_pct: float = 0.0


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Historical simulation engine for statistical arbitrage pairs."""

    def __init__(
        self,
        config: dict[str, Any],
        initial_capital: float = 100000.0,
    ) -> None:
        self.config = config
        self.initial_capital = initial_capital

    def run_equity_backtest(
        self,
        price_df: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> tuple[BacktestMetrics, list[BacktestTrade], pd.Series]:
        """Run backtest on equity options module (daily bars)."""
        pair_list = pairs or [tuple(p) for p in self.config.get("equity_pairs", [["GLD", "SLV"], ["XOM", "CVX"], ["KO", "PEP"]])]
        return self._simulate(
            price_df=price_df,
            pairs=pair_list,
            module="equity",
            coint_lookback=int(self.config.get("coint_lookback_days_equity", 90)),
            ou_lookback=int(self.config.get("ou_lookback_days_equity", 30)),
            vol_lookback=10,
            periods_per_year=252,
            entry_z_base=float(self.config.get("entry_z_threshold_equity", 1.5)),
            entry_z_high_vol=float(self.config.get("entry_z_threshold_high_vol_equity", 2.0)),
            stop_z=float(self.config.get("stop_z_threshold_equity", 3.0)),
            time_stop_bars=int(self.config.get("time_stop_days_equity", 14)),
            max_open_pairs=int(self.config.get("max_open_equity_pairs", 5)),
            max_pos_pct=float(self.config.get("max_premium_pct_equity", 0.05)),
            q_noise=float(self.config.get("kalman_transition_noise_equity", 0.0001)),
            hl_min=float(self.config.get("halflife_min_days", 2.0)),
            hl_max=float(self.config.get("halflife_max_days", 30.0)),
            high_vol_thresh=float(self.config.get("high_vol_threshold_equity", 0.50)),
            extreme_vol_thresh=float(self.config.get("extreme_vol_threshold_equity", 1.00)),
        )

    def run_crypto_backtest(
        self,
        price_df: pd.DataFrame,
        pairs: list[tuple[str, str]] | None = None,
    ) -> tuple[BacktestMetrics, list[BacktestTrade], pd.Series]:
        """Run backtest on crypto spot module (hourly bars)."""
        pair_list = pairs or [tuple(p) for p in self.config.get("crypto_pairs", [["BTC/USD", "ETH/USD"], ["ETH/USD", "SOL/USD"], ["BTC/USD", "SOL/USD"]])]
        return self._simulate(
            price_df=price_df,
            pairs=pair_list,
            module="crypto",
            coint_lookback=int(self.config.get("coint_lookback_hours_crypto", 720)),
            ou_lookback=int(self.config.get("ou_lookback_hours_crypto", 168)),
            vol_lookback=24,
            periods_per_year=8760,
            entry_z_base=float(self.config.get("entry_z_threshold_crypto", 1.75)),
            entry_z_high_vol=float(self.config.get("entry_z_threshold_high_vol_crypto", 2.25)),
            stop_z=float(self.config.get("stop_z_threshold_crypto", 3.5)),
            time_stop_bars=int(self.config.get("time_stop_hours_crypto", 72)),
            max_open_pairs=int(self.config.get("max_open_crypto_pairs", 3)),
            max_pos_pct=float(self.config.get("max_position_pct_equity", 0.15)),
            q_noise=float(self.config.get("kalman_transition_noise_crypto", 0.001)),
            hl_min=float(self.config.get("halflife_min_hours", 4.0)),
            hl_max=float(self.config.get("halflife_max_hours", 96.0)),
            high_vol_thresh=float(self.config.get("high_vol_threshold_crypto", 1.20)),
            extreme_vol_thresh=float(self.config.get("extreme_vol_threshold_crypto", 2.00)),
        )

    def _simulate(
        self,
        price_df: pd.DataFrame,
        pairs: list[tuple[str, str]],
        module: Literal["equity", "crypto"],
        coint_lookback: int,
        ou_lookback: int,
        vol_lookback: int,
        periods_per_year: int,
        entry_z_base: float,
        entry_z_high_vol: float,
        stop_z: float,
        time_stop_bars: int,
        max_open_pairs: int,
        max_pos_pct: float,
        q_noise: float,
        hl_min: float,
        hl_max: float,
        high_vol_thresh: float,
        extreme_vol_thresh: float,
    ) -> tuple[BacktestMetrics, list[BacktestTrade], pd.Series]:
        """Master simulation loop."""
        n_bars = len(price_df)
        min_start = max(coint_lookback, ou_lookback, vol_lookback) + 10
        if n_bars <= min_start:
            raise ValueError(f"Price DataFrame too short ({n_bars} bars <= {min_start} required).")

        equity = self.initial_capital
        equity_curve = [equity] * min_start

        # State tracking per pair
        kalman_filters: dict[str, KalmanFilter] = {}
        spread_histories: dict[str, list[float]] = {}
        coint_pvals: dict[str, float] = {}
        for a, b in pairs:
            pair_id = f"{a}-{b}"
            kalman_filters[pair_id] = KalmanFilter(observation_noise=0.001, transition_noise=q_noise)
            spread_histories[pair_id] = []
            coint_pvals[pair_id] = 1.0

        open_trades: dict[str, BacktestTrade] = {}
        closed_trades: list[BacktestTrade] = []
        coint_checks = 0
        coint_passes = 0

        dates = price_df.index
        recheck_interval = 1 if module == "equity" else 24

        # Step through time causal bar by bar
        for t in range(min_start, n_bars):
            cur_date = str(dates[t])

            # 1. Update online Kalman filters and spreads for all pairs
            for a, b in pairs:
                pair_id = f"{a}-{b}"
                if a not in price_df.columns or b not in price_df.columns:
                    continue
                pa = float(price_df[a].iloc[t])
                pb = float(price_df[b].iloc[t])

                # Update Kalman Filter
                kf = kalman_filters[pair_id]
                _, _, sp = kf.update(pa, pb)
                spread_histories[pair_id].append(sp)

                # Periodic Cointegration Check
                if t % recheck_interval == 0:
                    slice_a = np.log(price_df[a].iloc[t - coint_lookback : t + 1].values)
                    slice_b = np.log(price_df[b].iloc[t - coint_lookback : t + 1].values)
                    try:
                        _, pval, _ = coint(slice_a, slice_b)
                        coint_pvals[pair_id] = float(pval)
                    except Exception:
                        coint_pvals[pair_id] = 1.0

            # 2. Check Exits on Open Trades
            pairs_to_close: list[str] = []
            for pair_id, trade in open_trades.items():
                a, b = trade.pair_id.split("-")
                pa = float(price_df[a].iloc[t])
                pb = float(price_df[b].iloc[t])
                holding_bars = t - trade.entry_bar

                sp_slice = np.array(spread_histories[pair_id][-ou_lookback:])
                try:
                    ou_p = fit_ou_parameters(sp_slice, dt=1.0)
                    cur_z = compute_z_score(spread_histories[pair_id][-1], ou_p.mu, ou_p.sigma_spread)
                except Exception:
                    cur_z = 0.0

                exit_reason = None
                exit_z_thresh = float(self.config.get("exit_z_threshold", 0.3))

                if abs(cur_z) <= exit_z_thresh:
                    exit_reason = "z_reversion"
                elif abs(cur_z) >= stop_z:
                    exit_reason = "stop_z"
                elif holding_bars >= time_stop_bars:
                    exit_reason = "time_stop"

                if exit_reason:
                    # Calculate P/L
                    if module == "equity":
                        # Options P/L approximation via Black-Scholes
                        iv_a = max(0.15, float(np.std(compute_log_returns(price_df[a].iloc[t-vol_lookback:t+1])) * math.sqrt(periods_per_year)))
                        iv_b = max(0.15, float(np.std(compute_log_returns(price_df[b].iloc[t-vol_lookback:t+1])) * math.sqrt(periods_per_year)))
                        remaining_dte = max(1.0, trade.target_dte - holding_bars)

                        if trade.direction == "long":
                            # Bought Call A, Bought Put B
                            p_exit_a = black_scholes_price(pa, trade.entry_price_a, remaining_dte, iv_a, option_type="call")
                            p_exit_b = black_scholes_price(pb, trade.entry_price_b, remaining_dte, iv_b, option_type="put")
                            p_entry_a = black_scholes_price(trade.entry_price_a, trade.entry_price_a, trade.target_dte, iv_a, option_type="call")
                            p_entry_b = black_scholes_price(trade.entry_price_b, trade.entry_price_b, trade.target_dte, iv_b, option_type="put")
                        else:
                            # Bought Put A, Bought Call B
                            p_exit_a = black_scholes_price(pa, trade.entry_price_a, remaining_dte, iv_a, option_type="put")
                            p_exit_b = black_scholes_price(pb, trade.entry_price_b, remaining_dte, iv_b, option_type="call")
                            p_entry_a = black_scholes_price(trade.entry_price_a, trade.entry_price_a, trade.target_dte, iv_a, option_type="put")
                            p_entry_b = black_scholes_price(trade.entry_price_b, trade.entry_price_b, trade.target_dte, iv_b, option_type="call")

                        pnl_a = (p_exit_a - p_entry_a) * trade.qty_a * 100
                        pnl_b = (p_exit_b - p_entry_b) * trade.qty_b * 100
                        total_pnl = pnl_a + pnl_b
                        total_cost = (p_entry_a * trade.qty_a * 100) + (p_entry_b * trade.qty_b * 100)
                        pnl_pct = total_pnl / max(total_cost, 1.0)
                    else:
                        # Crypto Spot P/L
                        if trade.direction == "long":
                            pnl_a = (pa - trade.entry_price_a) * trade.qty_a
                            pnl_b = (trade.entry_price_b - pb) * trade.qty_b
                        else:
                            pnl_a = (trade.entry_price_a - pa) * trade.qty_a
                            pnl_b = (pb - trade.entry_price_b) * trade.qty_b

                        total_pnl = pnl_a + pnl_b
                        total_cost = (trade.entry_price_a * trade.qty_a) + (trade.entry_price_b * trade.qty_b)
                        pnl_pct = total_pnl / max(total_cost, 1.0)

                    trade.exit_bar = t
                    trade.exit_date = cur_date
                    trade.exit_z = cur_z
                    trade.exit_reason = exit_reason
                    trade.exit_price_a = pa
                    trade.exit_price_b = pb
                    trade.pnl_usd = total_pnl
                    trade.pnl_pct = pnl_pct
                    trade.holding_period_bars = holding_bars

                    equity += total_pnl
                    closed_trades.append(trade)
                    pairs_to_close.append(pair_id)

            for p in pairs_to_close:
                del open_trades[p]

            # 3. Check Entries for Eligible Pairs
            if len(open_trades) < max_open_pairs:
                for a, b in pairs:
                    pair_id = f"{a}-{b}"
                    if pair_id in open_trades or a not in price_df.columns or b not in price_df.columns:
                        continue

                    coint_pval = coint_pvals.get(pair_id, 1.0)
                    coint_checks += 1
                    if coint_pval > float(self.config.get("coint_pvalue_threshold", 0.05)):
                        continue
                    coint_passes += 1

                    # OU fit
                    sp_slice = np.array(spread_histories[pair_id][-ou_lookback:])
                    try:
                        ou_params = fit_ou_parameters(sp_slice, dt=1.0)
                    except OUFitError:
                        continue

                    # Half-life gate
                    if not (hl_min <= ou_params.half_life <= hl_max):
                        continue

                    # Vol regime
                    ret_a = compute_log_returns(price_df[a].iloc[t - vol_lookback : t + 1].values)
                    ret_b = compute_log_returns(price_df[b].iloc[t - vol_lookback : t + 1].values)
                    rv_a = compute_realized_vol(ret_a, periods_per_year)
                    rv_b = compute_realized_vol(ret_b, periods_per_year)
                    regime = classify_vol_regime(rv_a, rv_b, high_vol_thresh, extreme_vol_thresh)

                    if regime == "EXTREME":
                        continue

                    entry_thresh = entry_z_high_vol if regime == "HIGH" else entry_z_base
                    cur_z = compute_z_score(spread_histories[pair_id][-1], ou_params.mu, ou_params.sigma_spread)

                    direction: Literal["long", "short"] | None = None
                    if cur_z < -entry_thresh:
                        direction = "long"
                    elif cur_z > entry_thresh:
                        direction = "short"

                    if direction:
                        # Position Sizing
                        pa = float(price_df[a].iloc[t])
                        pb = float(price_df[b].iloc[t])
                        beta = kalman_filters[pair_id].beta

                        try:
                            kf_full = compute_kelly_fraction(cur_z, float(self.config.get("exit_z_threshold", 0.3)), ou_params.sigma_spread)
                        except Exception:
                            kf_full = 1.0

                        pos_f = float(np.clip(float(self.config.get("kelly_fraction", 0.25)) * kf_full, 0.01, max_pos_pct))
                        target_dte = max(7, round(ou_params.half_life * float(self.config.get("expiry_multiplier", 2.5))))

                        if module == "equity":
                            est_prem_a = pa * 0.025 * 100
                            est_prem_b = pb * 0.025 * 100
                            budget = pos_f * equity
                            qa = max(1, math.floor((budget / 2) / max(est_prem_a, 1.0)))
                            qb = max(1, math.floor((budget / 2) / max(est_prem_b, 1.0)))
                        else:
                            beta_factor = max(1.0, abs(beta))
                            val_a = (pos_f / beta_factor) * equity
                            qa = val_a / pa
                            qb = (qa * abs(beta) * pa) / pb

                        trade = BacktestTrade(
                            pair_id=pair_id,
                            module=module,
                            direction=direction,
                            entry_bar=t,
                            entry_date=cur_date,
                            entry_z=cur_z,
                            entry_beta=beta,
                            entry_price_a=pa,
                            entry_price_b=pb,
                            qty_a=qa,
                            qty_b=qb,
                            target_dte=target_dte,
                        )
                        open_trades[pair_id] = trade
                        if len(open_trades) >= max_open_pairs:
                            break

            equity_curve.append(equity)

        # 4. Compute Metrics
        metrics = self._calculate_metrics(
            module=module,
            equity_curve=pd.Series(equity_curve, index=dates[:len(equity_curve)]),
            closed_trades=closed_trades,
            coint_checks=coint_checks,
            coint_passes=coint_passes,
            periods_per_year=periods_per_year,
        )

        return metrics, closed_trades, pd.Series(equity_curve, index=dates[:len(equity_curve)])

    def _calculate_metrics(
        self,
        module: str,
        equity_curve: pd.Series,
        closed_trades: list[BacktestTrade],
        coint_checks: int,
        coint_passes: int,
        periods_per_year: int,
    ) -> BacktestMetrics:
        initial = self.initial_capital
        final = float(equity_curve.iloc[-1])
        total_ret = ((final - initial) / initial) * 100.0

        n_bars = len(equity_curve)
        years = n_bars / periods_per_year
        cagr = (((final / initial) ** (1.0 / max(years, 0.01))) - 1.0) * 100.0 if final > 0 else -100.0

        # Returns & Sharpe
        pct_returns = equity_curve.pct_change().dropna()
        if len(pct_returns) > 1 and pct_returns.std() > 1e-8:
            sharpe = float((pct_returns.mean() / pct_returns.std()) * math.sqrt(periods_per_year))
            downside = pct_returns[pct_returns < 0]
            downside_std = downside.std() if len(downside) > 1 else 1e-8
            sortino = float((pct_returns.mean() / max(downside_std, 1e-8)) * math.sqrt(periods_per_year))
        else:
            sharpe, sortino = 0.0, 0.0

        # Max Drawdown
        cum_max = equity_curve.cummax()
        drawdowns = (equity_curve - cum_max) / cum_max
        max_dd = float(abs(drawdowns.min())) * 100.0

        # Trade stats
        total_t = len(closed_trades)
        wins = [t for t in closed_trades if t.pnl_usd > 0]
        losses = [t for t in closed_trades if t.pnl_usd < 0]

        win_rate = (len(wins) / total_t * 100.0) if total_t > 0 else 0.0
        total_win_usd = sum(t.pnl_usd for t in wins)
        total_loss_usd = abs(sum(t.pnl_usd for t in losses))

        profit_factor = (total_win_usd / total_loss_usd) if total_loss_usd > 0 else (99.0 if total_win_usd > 0 else 1.0)
        avg_win = (total_win_usd / len(wins)) if wins else 0.0
        avg_loss = (total_loss_usd / len(losses)) if losses else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Exit reasons breakdown
        reasons: dict[str, int] = {}
        for t in closed_trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        coint_rate = (coint_passes / coint_checks * 100.0) if coint_checks > 0 else 0.0

        return BacktestMetrics(
            module=module,
            initial_capital=initial,
            final_capital=final,
            total_return_pct=total_ret,
            cagr_pct=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            total_trades=total_t,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_win_usd=avg_win,
            avg_loss_usd=avg_loss,
            win_loss_ratio=win_loss_ratio,
            exit_reasons=reasons,
            coint_pass_rate_pct=coint_rate,
        )


# ---------------------------------------------------------------------------
# Synthetic / Demo Data Generator for Offline Validation
# ---------------------------------------------------------------------------


def generate_synthetic_backtest_data(
    symbols: list[str],
    n_bars: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic price paths for backtest validation."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    df = pd.DataFrame(index=dates)

    # Base trends
    for s in symbols:
        drift = 0.0003
        vol = 0.012
        returns = rng.normal(drift, vol, n_bars)
        price = 100.0 * np.exp(np.cumsum(returns))
        df[s] = price

    # Impose cointegration relationships
    if "SLV" in df.columns and "GLD" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.01)
        df["SLV"] = np.exp(0.9 * np.log(df["GLD"]) + spread - 1.2)

    if "XOM" in df.columns and "CVX" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.82 * spread[i - 1] + rng.normal(0, 0.01)
        df["CVX"] = np.exp(0.95 * np.log(df["XOM"]) + spread - 0.2)

    if "KO" in df.columns and "PEP" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.84 * spread[i - 1] + rng.normal(0, 0.008)
        df["PEP"] = np.exp(0.92 * np.log(df["KO"]) + spread + 0.1)

    if "GOOGL" in df.columns and "META" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.012)
        df["META"] = np.exp(1.02 * np.log(df["GOOGL"]) + spread + 0.3)

    if "NVDA" in df.columns and "AMD" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.83 * spread[i - 1] + rng.normal(0, 0.016)
        df["AMD"] = np.exp(0.88 * np.log(df["NVDA"]) + spread - 0.4)

    if "V" in df.columns and "MA" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.86 * spread[i - 1] + rng.normal(0, 0.008)
        df["MA"] = np.exp(0.98 * np.log(df["V"]) + spread + 0.2)

    if "JPM" in df.columns and "BAC" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.84 * spread[i - 1] + rng.normal(0, 0.01)
        df["BAC"] = np.exp(0.80 * np.log(df["JPM"]) + spread - 1.1)

    if "HD" in df.columns and "LOW" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.009)
        df["LOW"] = np.exp(0.85 * np.log(df["HD"]) + spread - 0.3)

    if "ETH/USD" in df.columns and "BTC/USD" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.88 * spread[i - 1] + rng.normal(0, 0.015)
        df["ETH/USD"] = np.exp(1.05 * np.log(df["BTC/USD"]) + spread - 2.8)

    if "SOL/USD" in df.columns and "ETH/USD" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.86 * spread[i - 1] + rng.normal(0, 0.018)
        df["SOL/USD"] = np.exp(0.90 * np.log(df["ETH/USD"]) + spread - 2.2)

    if "LINK/USD" in df.columns and "ETH/USD" in df.columns:
        spread = np.zeros(n_bars)
        for i in range(1, n_bars):
            spread[i] = 0.84 * spread[i - 1] + rng.normal(0, 0.015)
        df["LINK/USD"] = np.exp(0.75 * np.log(df["ETH/USD"]) + spread - 3.5)

    return df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stat-Arb Backtest")
    parser.add_argument("--module", choices=["equity", "crypto", "all"], default="all")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--bars-equity", type=int, default=1000, help="Number of daily bars for equity (default: 1000 = ~4 years)")
    parser.add_argument("--bars-crypto", type=int, default=5000, help="Number of hourly bars for crypto (default: 5000 = ~7 months)")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    engine = BacktestEngine(config=config, initial_capital=args.capital)

    all_metrics = {}

    if args.module in ("equity", "all"):
        logger.info("=== Running Module A (Equity Options) Backtest (%d daily bars) ===", args.bars_equity)
        raw_pairs = config.get("equity_pairs", [["GLD", "SLV"], ["XOM", "CVX"], ["KO", "PEP"]])
        equity_symbols = sorted(list({sym for pair in raw_pairs for sym in pair}))
        eq_df = generate_synthetic_backtest_data(equity_symbols, n_bars=args.bars_equity, seed=42)
        eq_metrics, eq_trades, _ = engine.run_equity_backtest(eq_df)
        all_metrics["equity"] = asdict(eq_metrics)
        _print_metrics_summary(eq_metrics, "Equity Options")

    if args.module in ("crypto", "all"):
        logger.info("=== Running Module B (Crypto Spot) Backtest (%d hourly bars) ===", args.bars_crypto)
        raw_crypto = config.get("crypto_pairs", [["BTC/USD", "ETH/USD"], ["ETH/USD", "SOL/USD"], ["BTC/USD", "SOL/USD"]])
        crypto_symbols = sorted(list({sym for pair in raw_crypto for sym in pair}))
        cr_df = generate_synthetic_backtest_data(crypto_symbols, n_bars=args.bars_crypto, seed=99)
        cr_metrics, cr_trades, _ = engine.run_crypto_backtest(cr_df)
        all_metrics["crypto"] = asdict(cr_metrics)
        _print_metrics_summary(cr_metrics, "Crypto Spot")

    if args.save_results:
        out_path = Path("results/backtest_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Backtest summary saved to %s", out_path)


def _print_metrics_summary(m: BacktestMetrics, title: str) -> None:
    print(f"\n=======================================================")
    print(f"  BACKTEST RESULTS: {title}")
    print(f"=======================================================")
    print(f"Initial Capital:     ${m.initial_capital:,.2f}")
    print(f"Final Capital:       ${m.final_capital:,.2f}")
    print(f"Total Return:        {m.total_return_pct:+.2f}%")
    print(f"CAGR:                {m.cagr_pct:+.2f}%")
    print(f"Sharpe Ratio:        {m.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:       {m.sortino_ratio:.2f}")
    print(f"Max Drawdown:        {m.max_drawdown_pct:.2f}%")
    print(f"Total Trades:        {m.total_trades} (Wins: {m.winning_trades}, Losses: {m.losing_trades})")
    print(f"Win Rate:            {m.win_rate_pct:.1f}%")
    print(f"Profit Factor:       {m.profit_factor:.2f}")
    print(f"Win/Loss Ratio:      {m.win_loss_ratio:.2f}")
    print(f"Exit Reasons:        {m.exit_reasons}")
    print(f"Cointegration Pass:  {m.coint_pass_rate_pct:.1f}%")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
