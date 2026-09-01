"""signal_agent.py — Shared signal pipeline for equity and crypto modules.

Produces a typed SpreadSignal for every pair at each cycle:
  1. Cointegration check (statsmodels coint) on log-prices
  2. Dynamic hedge ratio beta & spread via Kalman filter
  3. Rolling OU parameter estimation (kappa, mu, sigma, half-life, sigma_spread)
  4. Half-life gate check
  5. Realized volatility computation & regime classification (NORMAL | HIGH | EXTREME)
  6. Z-score calculation
  7. Sentiment modifier application (optional LLM input)
  8. Final direction assignment (long | short | none) & structured rationale logging
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import yaml
from statsmodels.tsa.stattools import coint

from src.broker import BarData
from src.indicators import (
    OUFitError,
    classify_vol_regime,
    compute_log_returns,
    compute_realized_vol,
    compute_z_score,
    fit_ou_parameters,
)
from src.kalman import initialize_filter
from src.models import OUParams as ModelOUParams
from src.models import SentimentResult, SpreadSignal
from src.persistence.db import Database

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SignalAgent:
    """Deterministic signal engine for statistical arbitrage pairs."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db: Database | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.db = db

    def evaluate_pair(
        self,
        asset_a: str,
        asset_b: str,
        module: Literal["equity", "crypto"],
        prices_a: np.ndarray | list[float] | BarData,
        prices_b: np.ndarray | list[float] | BarData,
        sentiment: SentimentResult | None = None,
        data_timestamp: datetime | None = None,
    ) -> SpreadSignal:
        """Run the complete signal pipeline for one asset pair.

        Args:
            asset_a: Symbol for asset A (base asset).
            asset_b: Symbol for asset B.
            module: 'equity' or 'crypto'.
            prices_a: Price array or BarData for asset A (oldest -> newest).
            prices_b: Price array or BarData for asset B (same length).
            sentiment: Optional SentimentResult for asset A from LLM provider.
            data_timestamp: Timestamp of the most recent bar used.

        Returns:
            SpreadSignal with direction ('long' | 'short' | 'none') and full audit rationale.
        """
        pair_id = f"{asset_a}-{asset_b}"
        now_utc = datetime.now(timezone.utc)

        # Extract numpy arrays and latest timestamp
        arr_a, ts_a = self._extract_prices_and_ts(prices_a)
        arr_b, ts_b = self._extract_prices_and_ts(prices_b)

        latest_data_ts = data_timestamp or ts_a or ts_b or now_utc

        if len(arr_a) != len(arr_b):
            raise ValueError(
                f"Mismatched price series lengths for {pair_id}: "
                f"len(A)={len(arr_a)}, len(B)={len(arr_b)}"
            )

        min_required_bars = 20
        if len(arr_a) < min_required_bars:
            logger.warning(
                "Insufficient bars for %s: %d < %d required",
                pair_id, len(arr_a), min_required_bars
            )
            return self._build_empty_signal(
                pair_id=pair_id,
                module=module,
                asset_a=asset_a,
                asset_b=asset_b,
                reason=f"Insufficient bars: {len(arr_a)} < {min_required_bars}",
                latest_data_ts=latest_data_ts,
            )

        # ── 1. Cointegration Gate ─────────────────────────────────────────────
        coint_pvalue = self._check_cointegration(arr_a, arr_b, module)
        coint_threshold = float(self.config.get("coint_pvalue_threshold", 0.05))
        coint_passed = coint_pvalue < coint_threshold

        # ── 2. Kalman Filter Hedge Ratio ──────────────────────────────────────
        obs_noise = float(self.config.get("kalman_observation_noise", 0.001))
        if module == "crypto":
            trans_noise = float(self.config.get("kalman_transition_noise_crypto", 0.001))
        else:
            trans_noise = float(self.config.get("kalman_transition_noise_equity", 0.0001))

        kf, spreads = initialize_filter(
            prices_a=arr_a,
            prices_b=arr_b,
            observation_noise=obs_noise,
            transition_noise=trans_noise,
        )
        current_beta = float(kf.beta)
        current_spread = float(spreads[-1])

        # ── 3. OU Parameter Estimation ────────────────────────────────────────
        if module == "crypto":
            ou_lookback = int(self.config.get("ou_lookback_hours_crypto", 168))
        else:
            ou_lookback = int(self.config.get("ou_lookback_days_equity", 30))

        spread_slice = spreads[-min(len(spreads), ou_lookback):]

        try:
            ou_params = fit_ou_parameters(spread_slice, dt=1.0)
            ou_fit_ok = True
        except OUFitError as err:
            logger.warning("OU fit failed for %s: %s", pair_id, err)
            ou_params = None
            ou_fit_ok = False

        # ── 4. Half-Life Gate ─────────────────────────────────────────────────
        halflife_passed = False
        ou_degraded = False
        if ou_fit_ok and ou_params is not None:
            if module == "crypto":
                hl_min = float(self.config.get("halflife_min_hours", 6))
                hl_max = float(self.config.get("halflife_max_hours", 96))
            else:
                hl_min = float(self.config.get("halflife_min_days", 2))
                hl_max = float(self.config.get("halflife_max_days", 20))

            halflife_passed = hl_min <= ou_params.half_life <= hl_max
            # Mark as degraded if half-life exceeds max (clamped b=0.999 case)
            if ou_params.half_life > hl_max:
                ou_degraded = True

        # ── 5. Volatility Regime Classification ───────────────────────────────
        vol_a, vol_b, vol_regime = self._compute_vol_regime(arr_a, arr_b, module)

        # ── 6. Z-Score Calculation ────────────────────────────────────────────
        # If OU fit is degraded (clamped), sigma_spread is unreliably small and
        # produces inflated z-scores (100+). Use empirical z-score instead:
        # z = (spread - rolling_mean) / rolling_std
        if ou_degraded or not ou_fit_ok or ou_params is None:
            # Empirical z-score from rolling spread statistics
            spread_mu = float(np.mean(spread_slice))
            spread_std = float(np.std(spread_slice))
            if spread_std > 1e-10:
                z_score = (current_spread - spread_mu) / spread_std
            else:
                z_score = 0.0
            logger.debug(
                "Using empirical z-score for %s: z=%.2f (mu=%.4f, std=%.4f)",
                pair_id, z_score, spread_mu, spread_std,
            )
        else:
            z_score = compute_z_score(
                spread_current=current_spread,
                mu_ou=ou_params.mu,
                sigma_spread=ou_params.sigma_spread,
            )

        # ── 7. Thresholds & Sentiment Modifier ────────────────────────────────
        if module == "crypto":
            entry_base = float(self.config.get("entry_z_threshold_crypto", 1.75))
            entry_high_vol = float(self.config.get("entry_z_threshold_high_vol_crypto", 2.25))
            stop_z = float(self.config.get("stop_z_threshold_crypto", 3.5))
        else:
            entry_base = float(self.config.get("entry_z_threshold_equity", 1.5))
            entry_high_vol = float(self.config.get("entry_z_threshold_high_vol_equity", 2.0))
            stop_z = float(self.config.get("stop_z_threshold_equity", 3.0))

        exit_z = float(self.config.get("exit_z_threshold", 0.3))
        entry_threshold_regime = entry_high_vol if vol_regime == "HIGH" else entry_base

        # Determine preliminary direction
        raw_direction: Literal["long", "short", "none"] = "none"
        if z_score < -entry_threshold_regime:
            raw_direction = "long"
        elif z_score > entry_threshold_regime:
            raw_direction = "short"

        # Apply sentiment modifier (if sentiment provided)
        sentiment_mod = 0.0
        if sentiment is not None:
            if raw_direction == "long" and sentiment.sentiment == "negative":
                sentiment_mod = 0.15 * sentiment.confidence
            elif raw_direction == "short" and sentiment.sentiment == "positive":
                sentiment_mod = 0.15 * sentiment.confidence

        effective_entry_z = entry_threshold_regime + sentiment_mod

        # ── 8. Final Direction Assignment ─────────────────────────────────────
        final_direction: Literal["long", "short", "none"] = "none"
        rejection_reasons = []

        # Only EXTREME vol hard-blocks at signal level (very rare with raised thresholds).
        # OU fit failure → z_score stays 0.0 → direction stays "none" naturally.
        if vol_regime == "EXTREME":
            rejection_reasons.append("extreme_vol_regime_block")

        if not rejection_reasons:
            if z_score < -effective_entry_z:
                final_direction = "long"
            elif z_score > effective_entry_z:
                final_direction = "short"
            else:
                rejection_reasons.append(f"z_score_{z_score:.2f}_within_entry_threshold_{effective_entry_z:.2f}")

        # Construct structured rationale
        rationale: dict[str, Any] = {
            "pair_id": pair_id,
            "module": module,
            "coint_passed": coint_passed,
            "coint_pvalue": coint_pvalue,
            "coint_weak": not coint_passed,
            "kalman_beta": current_beta,
            "current_spread": current_spread,
            "ou_fit_ok": ou_fit_ok,
            "ou_degraded": ou_degraded,
            "z_score_type": "empirical" if (ou_degraded or not ou_fit_ok) else "ou_model",
            "ou_half_life": ou_params.half_life if ou_params else None,
            "halflife_passed": halflife_passed,
            "vol_regime": vol_regime,
            "vol_a": vol_a,
            "vol_b": vol_b,
            "z_score": z_score,
            "raw_direction": raw_direction,
            "sentiment_applied": sentiment is not None,
            "sentiment_modifier": sentiment_mod,
            "entry_threshold_used": effective_entry_z,
            "final_direction": final_direction,
            "rejection_reasons": rejection_reasons,
        }

        # Format ModelOUParams
        if ou_params is not None:
            model_ou = ModelOUParams(
                kappa=ou_params.kappa,
                mu=ou_params.mu,
                sigma_ou=ou_params.sigma_ou,
                half_life=ou_params.half_life,
                sigma_spread=ou_params.sigma_spread,
                ar1_r_squared=ou_params.ar1_r_squared,
            )
        else:
            model_ou = ModelOUParams(
                kappa=0.0,
                mu=0.0,
                sigma_ou=0.0,
                half_life=0.0,
                sigma_spread=0.0001,
                ar1_r_squared=0.0,
            )

        signal = SpreadSignal(
            pair_id=pair_id,
            module=module,
            asset_a=asset_a,
            asset_b=asset_b,
            direction=final_direction,
            z_score=z_score,
            beta=current_beta,
            ou_params=model_ou,
            vol_regime=vol_regime,
            vol_a=vol_a,
            vol_b=vol_b,
            coint_pvalue=coint_pvalue,
            sentiment=sentiment,
            sentiment_modifier=sentiment_mod,
            entry_z_threshold_used=effective_entry_z,
            exit_z_threshold=exit_z,
            stop_z_threshold=stop_z,
            signal_rationale=rationale,
            generated_at=now_utc,
            data_timestamp=latest_data_ts,
        )

        if self.db is not None:
            try:
                self.db.insert_signal(str(uuid4()), signal)
            except Exception as exc:
                logger.error("Failed to insert signal into DB: %s", exc)

        logger.info(
            "Signal evaluated: %s (%s) | dir=%s | z=%.2f | beta=%.4f | regime=%s",
            pair_id, module, final_direction, z_score, current_beta, vol_regime,
        )
        return signal

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_prices_and_ts(
        self,
        prices: np.ndarray | list[float] | BarData,
    ) -> tuple[np.ndarray, datetime | None]:
        if isinstance(prices, BarData):
            arr = np.array(prices.closes, dtype=float)
            ts = prices.latest_timestamp
            return arr, ts
        arr = np.asarray(prices, dtype=float)
        return arr, None

    def _check_cointegration(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        module: Literal["equity", "crypto"],
    ) -> float:
        """Run cointegration test on log prices over the module's lookback window."""
        if module == "crypto":
            lookback = int(self.config.get("coint_lookback_hours_crypto", 720))
        else:
            lookback = int(self.config.get("coint_lookback_days_equity", 90))

        slice_a = prices_a[-min(len(prices_a), lookback):]
        slice_b = prices_b[-min(len(prices_b), lookback):]

        log_a = np.log(slice_a)
        log_b = np.log(slice_b)

        try:
            # statsmodels coint test
            _, pvalue, _ = coint(log_a, log_b)
            return float(pvalue)
        except Exception as exc:
            logger.warning("Cointegration test error: %s. Defaulting pvalue=1.0", exc)
            return 1.0

    def _compute_vol_regime(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        module: Literal["equity", "crypto"],
    ) -> tuple[float, float, Literal["NORMAL", "HIGH", "EXTREME"]]:
        """Compute realized vol and regime for the pair."""
        if module == "crypto":
            # 24h realized vol on hourly returns
            lookback = 24
            periods_per_year = 8760
            high_thresh = float(self.config.get("high_vol_threshold_crypto", 0.80))
            extreme_thresh = float(self.config.get("extreme_vol_threshold_crypto", 1.20))
        else:
            # 10-day realized vol on daily returns
            lookback = 10
            periods_per_year = 252
            high_thresh = float(self.config.get("high_vol_threshold_equity", 0.30))
            extreme_thresh = float(self.config.get("extreme_vol_threshold_equity", 0.60))

        slice_a = prices_a[-min(len(prices_a), lookback + 1):]
        slice_b = prices_b[-min(len(prices_b), lookback + 1):]

        try:
            ret_a = compute_log_returns(slice_a)
            vol_a = compute_realized_vol(ret_a, periods_per_year=periods_per_year)
        except Exception:
            vol_a = 0.0

        try:
            ret_b = compute_log_returns(slice_b)
            vol_b = compute_realized_vol(ret_b, periods_per_year=periods_per_year)
        except Exception:
            vol_b = 0.0

        regime = classify_vol_regime(
            rv_a=vol_a,
            rv_b=vol_b,
            high_threshold=high_thresh,
            extreme_threshold=extreme_thresh,
        )
        return vol_a, vol_b, regime

    def _build_empty_signal(
        self,
        pair_id: str,
        module: Literal["equity", "crypto"],
        asset_a: str,
        asset_b: str,
        reason: str,
        latest_data_ts: datetime,
    ) -> SpreadSignal:
        dummy_ou = ModelOUParams(
            kappa=0.0,
            mu=0.0,
            sigma_ou=0.0,
            half_life=0.0,
            sigma_spread=0.0001,
            ar1_r_squared=0.0,
        )
        return SpreadSignal(
            pair_id=pair_id,
            module=module,
            asset_a=asset_a,
            asset_b=asset_b,
            direction="none",
            z_score=0.0,
            beta=1.0,
            ou_params=dummy_ou,
            vol_regime="NORMAL",
            vol_a=0.0,
            vol_b=0.0,
            coint_pvalue=1.0,
            sentiment=None,
            sentiment_modifier=0.0,
            entry_z_threshold_used=1.5 if module == "equity" else 1.75,
            exit_z_threshold=0.3,
            stop_z_threshold=3.0 if module == "equity" else 3.5,
            signal_rationale={"reason": reason, "rejection_reasons": [reason]},
            generated_at=datetime.now(timezone.utc),
            data_timestamp=latest_data_ts,
        )
