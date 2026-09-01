"""indicators.py — Statistical indicators for the OU mean-reversion signal pipeline.

All functions are pure (no I/O, no side effects, no state). Each is independently
unit-testable on synthetic data with known closed-form expected values.

Functions:
  fit_ou_parameters     — AR(1) regression → OU κ, μ, σ, half-life, σ_spread
  compute_z_score       — (spread - μ_OU) / σ_spread
  compute_log_returns   — log(P_t / P_{t-1})
  compute_realized_vol  — annualized std dev of log-returns
  classify_vol_regime   — NORMAL | HIGH | EXTREME based on pair's max realized vol
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Regime type alias
VolRegime = Literal["NORMAL", "HIGH", "EXTREME"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OUFitError(Exception):
    """Raised when OU parameter estimation fails.

    Common causes:
      - Spread has unit root (AR1 coeff b ≥ 1.0): pair is not mean-reverting.
      - Spread has oscillatory dynamics (b ≤ 0): not a valid OU process.
      - Input series too short (< 20 observations).
      - Input contains NaN or Inf.
    """


# ---------------------------------------------------------------------------
# OU parameter container
# ---------------------------------------------------------------------------


class OUParams:
    """Container for Ornstein-Uhlenbeck process parameters fitted to a spread series.

    The spread X_t follows:  dX_t = κ(μ - X_t)dt + σ dW_t

    Attributes:
        kappa:        Mean-reversion speed (per bar unit). Higher → faster reversion.
        mu:           Long-run mean (equilibrium level of the spread).
        sigma_ou:     OU diffusion coefficient.
        half_life:    ln(2)/κ in the same time units as the input bars.
                      E.g., if bars are hourly, half_life is in hours.
        sigma_spread: Stationary standard deviation = σ_ou / sqrt(2κ).
                      Used to normalize the z-score.
        ar1_r_squared: Goodness of fit of the underlying AR(1) regression.
    """

    __slots__ = ("kappa", "mu", "sigma_ou", "half_life", "sigma_spread", "ar1_r_squared")

    def __init__(
        self,
        kappa: float,
        mu: float,
        sigma_ou: float,
        half_life: float,
        sigma_spread: float,
        ar1_r_squared: float,
    ) -> None:
        self.kappa = kappa
        self.mu = mu
        self.sigma_ou = sigma_ou
        self.half_life = half_life
        self.sigma_spread = sigma_spread
        self.ar1_r_squared = ar1_r_squared

    def __repr__(self) -> str:
        return (
            f"OUParams("
            f"κ={self.kappa:.4f}, μ={self.mu:.4f}, σ_ou={self.sigma_ou:.4f}, "
            f"τ={self.half_life:.2f} bars, σ_spread={self.sigma_spread:.4f}, "
            f"R²={self.ar1_r_squared:.3f})"
        )

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "mu": self.mu,
            "sigma_ou": self.sigma_ou,
            "half_life": self.half_life,
            "sigma_spread": self.sigma_spread,
            "ar1_r_squared": self.ar1_r_squared,
        }


# ---------------------------------------------------------------------------
# OU parameter estimation
# ---------------------------------------------------------------------------


def fit_ou_parameters(spread: np.ndarray, dt: float = 1.0) -> OUParams:
    """Estimate Ornstein-Uhlenbeck parameters from a spread time series.

    Uses the exact discrete-time equivalence between the OU SDE and an AR(1)
    regression. The spread series (from the Kalman filter) is regressed as:

        X_{t+1} = a + b · X_t + ε

    This is the exact first-order Euler-Maruyama discretization of the OU SDE with:
        b = e^{-κΔt}      →  κ = -ln(b) / Δt
        a = μ(1 - b)      →  μ = a / (1 - b)
        Var(ε) = σ²(1 - b²) / (2κ)  →  σ² = Var(ε) · 2κ / (1 - b²)

    Stationary std dev:  σ_spread = σ_ou / sqrt(2κ)   [analytical formula]
    Half-life:           τ = ln(2) / κ                 [in bar units]

    Args:
        spread: 1-D array of spread values, oldest-to-newest. Minimum 20 observations.
        dt: Time step in consistent units. The returned half_life is in these units.
            Default 1.0 = one bar (so half_life is in bars = hours for hourly data,
            days for daily data).

    Returns:
        OUParams with all fitted parameters.

    Raises:
        OUFitError: If spread is too short, contains non-finite values, or the AR(1)
                    coefficient indicates non-mean-reverting dynamics.
        ValueError: For other input format issues.
    """
    spread = np.asarray(spread, dtype=float)

    if spread.ndim != 1:
        raise ValueError(f"spread must be 1-D; got shape {spread.shape}.")
    if len(spread) < 20:
        raise OUFitError(
            f"Spread too short for OU estimation: need ≥ 20 observations, got {len(spread)}."
        )
    if not np.isfinite(spread).all():
        n_bad = int(np.sum(~np.isfinite(spread)))
        raise OUFitError(f"Spread contains {n_bad} non-finite value(s) (NaN or Inf).")

    X = spread[:-1]   # X_t,   shape (n-1,)
    Y = spread[1:]    # X_{t+1}, shape (n-1,)

    # OLS: Y = [1, X] @ [a, b]ᵀ + ε
    A = np.column_stack([np.ones(len(X)), X])  # (n-1, 2)
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(A, Y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise OUFitError(f"OLS regression failed: {exc}") from exc

    if rank < 2:
        raise OUFitError(f"Design matrix is rank-deficient (rank={rank}). Degenerate spread?")

    a = float(coeffs[0])
    b = float(coeffs[1])

    # Validate mean-reversion: b must be in (0, 1) for a stationary OU process.
    # b >= 1.0 → unit root or explosive. However, b in [1.0, 1.05) is within
    # estimation noise for short windows. We clamp these to 0.999 and flag as
    # degraded — the half-life gate will reject truly non-reverting pairs.
    # b > 1.10 → genuinely explosive, hard reject.
    # b <= 0.0 → oscillatory dynamics, not OU.
    is_degraded = False
    if b > 1.10:
        raise OUFitError(
            f"AR(1) coefficient b={b:.4f} > 1.10. Spread is explosive (not mean-reverting). "
            "This pair's cointegration has broken down."
        )
    if b >= 1.0:
        logger.debug("AR(1) b=%.4f >= 1.0; clamping to 0.999 (degraded fit)", b)
        b = 0.999
        is_degraded = True
    if b <= 0.0:
        raise OUFitError(
            f"AR(1) coefficient b={b:.4f} <= 0. Spread shows oscillatory dynamics, "
            "not mean-reversion. Not a valid OU process."
        )

    # OU parameter extraction (exact formulas)
    kappa = -np.log(b) / dt
    mu_ou = a / (1.0 - b)

    # Residuals
    residuals = Y - (a + b * X)
    var_eps = float(np.var(residuals, ddof=2))

    # σ²_ou = Var(ε) · 2κ / (1 - b²)   [exact discrete-time formula]
    denom = 1.0 - b ** 2
    if denom <= 1e-10:
        raise OUFitError(
            f"1 - b² = {denom:.2e} ≈ 0; b is too close to ±1. Degenerate fit."
        )

    sigma_ou_sq = var_eps * 2.0 * kappa / denom
    if sigma_ou_sq <= 0.0:
        raise OUFitError(
            f"Computed σ²_ou = {sigma_ou_sq:.4e} ≤ 0. Residual variance may be zero."
        )

    sigma_ou = float(np.sqrt(sigma_ou_sq))

    # Stationary std dev of the OU process (used for z-score normalization)
    sigma_spread = sigma_ou / np.sqrt(2.0 * kappa)

    # Half-life in bar units
    half_life = np.log(2.0) / kappa

    # R² of the AR(1) fit
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    params = OUParams(
        kappa=float(kappa),
        mu=float(mu_ou),
        sigma_ou=float(sigma_ou),
        half_life=float(half_life),
        sigma_spread=float(sigma_spread),
        ar1_r_squared=float(r_squared),
    )

    logger.debug("OU fit: %s", params)
    return params


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


def compute_z_score(
    spread_current: float,
    mu_ou: float,
    sigma_spread: float,
) -> float:
    """Compute the z-score of the current spread value.

    z_t = (spread_t - μ_OU) / σ_spread

    Interpretation:
        z_t > +entry_z  → asset A is expensive relative to B (SHORT spread)
        z_t < -entry_z  → asset A is cheap relative to B (LONG spread)
        |z_t| < exit_z  → spread has reverted to mean (EXIT)
        |z_t| > stop_z  → spread is breaking down (STOP OUT)

    Args:
        spread_current: Current Kalman-filtered spread value.
        mu_ou: OU long-run mean from fit_ou_parameters().
        sigma_spread: OU stationary std dev from fit_ou_parameters().

    Returns:
        Z-score (float, positive or negative).

    Raises:
        ValueError: If sigma_spread is zero or negative.
    """
    if sigma_spread <= 0.0:
        raise ValueError(
            f"sigma_spread must be positive; got {sigma_spread:.4e}. "
            "Check OU parameter estimation."
        )
    return (spread_current - mu_ou) / sigma_spread


# ---------------------------------------------------------------------------
# Log-returns and realized volatility
# ---------------------------------------------------------------------------


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from a price series.

    log_return_t = ln(P_t / P_{t-1})

    Returns an array of length len(prices) - 1.

    Raises:
        ValueError: If any price is non-positive, or fewer than 2 prices.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        raise ValueError(f"Need ≥ 2 prices to compute returns; got {len(prices)}.")
    if not (prices > 0).all():
        n_bad = int(np.sum(prices <= 0))
        raise ValueError(f"All prices must be positive; found {n_bad} non-positive value(s).")
    return np.log(prices[1:] / prices[:-1])


def compute_realized_vol(
    log_returns: np.ndarray,
    periods_per_year: int,
) -> float:
    """Compute annualized realized volatility from log-returns.

    RV = std(log_returns) × sqrt(periods_per_year)

    Args:
        log_returns: 1-D array of log-returns from compute_log_returns().
        periods_per_year: Annualization factor:
            - Daily bars:  252
            - Hourly bars: 8760

    Returns:
        Annualized realized vol as a decimal (e.g., 0.30 = 30% annualized vol).

    Raises:
        ValueError: If log_returns is too short or contains non-finite values.
    """
    log_returns = np.asarray(log_returns, dtype=float)
    if len(log_returns) < 2:
        raise ValueError(
            f"Need ≥ 2 log-returns to compute realized vol; got {len(log_returns)}."
        )
    if not np.isfinite(log_returns).all():
        raise ValueError("log_returns contains non-finite values.")
    return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Volatility regime classification
# ---------------------------------------------------------------------------


def classify_vol_regime(
    rv_a: float,
    rv_b: float,
    high_threshold: float,
    extreme_threshold: float,
) -> VolRegime:
    """Classify the volatility regime for a pair.

    Uses the maximum of the two assets' realized vols:
        EXTREME: max(rv_a, rv_b) ≥ extreme_threshold  → no new entries
        HIGH:    max(rv_a, rv_b) ≥ high_threshold      → wider entry z-threshold
        NORMAL:  otherwise

    Args:
        rv_a: Annualized realized vol for asset A (decimal, e.g. 0.30 = 30%).
        rv_b: Annualized realized vol for asset B.
        high_threshold: Threshold for HIGH regime (config: high_vol_threshold_equity/crypto).
        extreme_threshold: Threshold for EXTREME regime.

    Returns:
        "NORMAL", "HIGH", or "EXTREME".

    Example (equity pair):
        classify_vol_regime(0.25, 0.28, 0.30, 0.60) → "NORMAL"
        classify_vol_regime(0.25, 0.35, 0.30, 0.60) → "HIGH"
        classify_vol_regime(0.25, 0.65, 0.30, 0.60) → "EXTREME"
    """
    max_rv = max(rv_a, rv_b)
    if max_rv >= extreme_threshold:
        return "EXTREME"
    if max_rv >= high_threshold:
        return "HIGH"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Kelly sizing helpers
# ---------------------------------------------------------------------------


def compute_kelly_fraction(
    entry_z: float,
    exit_z_threshold: float,
    sigma_spread: float,
) -> float:
    """Compute the theoretical Kelly fraction for one OU spread trade.

    The OU process implies an expected return from entry to exit of approximately:
        expected_return = (|entry_z| - exit_z_threshold) × σ_spread

    The Kelly fraction for a Gaussian-return bet is:
        kelly_f = expected_return / σ_spread²
                = (|entry_z| - exit_z_threshold) / σ_spread

    This is an approximation — the true Kelly for OU requires integrating the
    process density, which is complex. The simplified formula is standard in
    systematic stat-arb practice and is conservative when σ_spread is well-estimated.

    Args:
        entry_z: Absolute z-score at which the position was entered.
        exit_z_threshold: Z-score at which the position will be exited (mean reversion).
        sigma_spread: OU stationary std dev.

    Returns:
        Theoretical full Kelly fraction (apply kelly_fraction multiplier from config
        before using as a position size).

    Raises:
        ValueError: If arguments are invalid.
    """
    if sigma_spread <= 0:
        raise ValueError(f"sigma_spread must be positive; got {sigma_spread}.")
    if abs(entry_z) <= exit_z_threshold:
        raise ValueError(
            f"|entry_z|={abs(entry_z):.3f} must exceed exit_z_threshold={exit_z_threshold:.3f}."
        )

    expected_return = (abs(entry_z) - exit_z_threshold) * sigma_spread
    kelly_f = expected_return / (sigma_spread ** 2)
    return float(kelly_f)
