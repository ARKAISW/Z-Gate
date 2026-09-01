"""kalman.py — Kalman filter for dynamic hedge ratio estimation.

The filter estimates the time-varying hedge ratio β between the log-prices of two
assets, enabling an adaptive spread rather than a static OLS-fitted relationship.

State vector:  x = [β, μ]ᵀ
  β — hedge ratio:  log(A_t) ≈ β_t · log(B_t) + μ_t
  μ — intercept (absorbs drift / level differences between the two assets)

State-space model:
  Observation:  log(price_A_t) = β_t · log(price_B_t) + μ_t + ε_t,   ε_t ~ N(0, R)
  State:        x_{t+1} = x_t + η_t,                                   η_t ~ N(0, Q)

The random-walk transition (F = I) assumes β and μ drift without a known structure —
appropriate for financial cointegration relationships where the ratio changes slowly
over time due to market microstructure, liquidity, and fundamental shifts.

Why Kalman instead of static OLS:
  Static OLS computes β once over a historical window and assumes it is constant.
  Crypto β can shift materially within 48 hours. Equity β shifts more slowly but
  still meaningfully over months. The Kalman filter adapts online, using a
  parameter Q to control how much β is allowed to change per bar.

  Test: generate a synthetic series where β shifts from 1.3 → 0.9 mid-series.
  Kalman will track the shift; static OLS will be wrong by a fixed amount.
  (This test is in tests/test_kalman.py.)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KalmanFilterError(Exception):
    """Raised when the filter encounters a numerical issue."""


# ---------------------------------------------------------------------------
# KalmanFilter
# ---------------------------------------------------------------------------


@dataclass
class KalmanFilter:
    """Online Kalman filter for adaptive hedge ratio (β) and intercept (μ) estimation.

    Usage:
        kf = KalmanFilter(observation_noise=0.001, transition_noise=0.0001)
        for price_a, price_b in zip(prices_a, prices_b):
            beta, mu, spread = kf.update(price_a, price_b)

    Args:
        observation_noise: R — variance of the measurement noise ε_t.
            Reflects how much we trust each new price observation relative to the
            model's prediction. Larger R → slower filter response.
        transition_noise: Q — variance applied to each state dimension per step.
            Controls how fast β and μ are allowed to drift. Larger Q → faster
            tracking of regime changes, at the cost of more noise sensitivity.
            Use a 10× larger Q for crypto than for equity (see config.yaml).
    """

    observation_noise: float = 0.001    # R
    transition_noise: float = 0.0001    # Q (per state dimension)

    # State estimate: [β, μ]
    _x: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0]),
        repr=False,
    )
    # State covariance (2×2): large initial uncertainty
    _P: np.ndarray = field(
        default_factory=lambda: np.eye(2) * 10.0,
        repr=False,
    )
    _n_updates: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        # F: state transition matrix (identity — random walk)
        self._F = np.eye(2)
        # Q: process noise covariance
        self._Q = np.eye(2) * self.transition_noise

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def beta(self) -> float:
        """Current hedge ratio estimate."""
        return float(self._x[0])

    @property
    def intercept(self) -> float:
        """Current intercept estimate μ."""
        return float(self._x[1])

    @property
    def n_updates(self) -> int:
        """Number of observations processed."""
        return self._n_updates

    def update(self, price_a: float, price_b: float) -> tuple[float, float, float]:
        """Process one price pair and return the updated (beta, mu, spread).

        The spread is defined as:
            spread_t = log(price_A_t) - β_t · log(price_B_t)

        This is the value whose OU dynamics are modelled. When spread_t is much
        higher than the OU long-run mean μ_OU, asset A is expensive relative to B
        (short-spread signal). When much lower, A is cheap (long-spread signal).

        Args:
            price_a: Current price of asset A. Must be strictly positive.
            price_b: Current price of asset B. Must be strictly positive.

        Returns:
            (beta, mu, spread) — all floats.

        Raises:
            KalmanFilterError: If prices are non-positive or a numerical issue occurs.
        """
        if price_a <= 0.0 or price_b <= 0.0:
            raise KalmanFilterError(
                f"Prices must be positive. Got price_a={price_a}, price_b={price_b}."
            )

        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # Observation matrix H — shape (1, 2)
        # Observation model: log_a = β·log_b + μ  ⟹  y = H @ x
        H = np.array([[log_b, 1.0]])  # (1, 2)

        # ── Predict ──────────────────────────────────────────────────────────
        x_pred: np.ndarray = self._F @ self._x          # (2,)
        P_pred: np.ndarray = self._P + self._Q          # (2, 2) — F=I so F@P@F.T = P

        # ── Innovation ───────────────────────────────────────────────────────
        # S: innovation covariance (scalar for 1D observation)
        S_mat: np.ndarray = H @ P_pred @ H.T + self.observation_noise  # (1, 1)
        S: float = float(S_mat[0, 0])

        if S <= 0.0:
            raise KalmanFilterError(
                f"Innovation covariance S={S:.6e} ≤ 0 at step {self._n_updates}. "
                "Check for degenerate input prices."
            )

        # K: Kalman gain — shape (2, 1)
        K: np.ndarray = (P_pred @ H.T) / S  # (2, 1)

        # Scalar innovation: actual obs minus predicted obs
        innovation: float = log_a - float((H @ x_pred)[0])

        # ── Update ───────────────────────────────────────────────────────────
        self._x = x_pred + K.ravel() * innovation

        # Joseph form for numerical stability: (I - KH) P (I - KH)ᵀ + K R Kᵀ
        I_minus_KH: np.ndarray = np.eye(2) - K @ H      # (2, 2)
        self._P = (
            I_minus_KH @ P_pred @ I_minus_KH.T
            + self.observation_noise * (K @ K.T)
        )

        self._n_updates += 1

        beta_t = float(self._x[0])
        mu_t = float(self._x[1])
        spread_t = log_a - beta_t * log_b

        return beta_t, mu_t, spread_t

    def copy(self) -> KalmanFilter:
        """Return a deep copy — useful for backtesting without mutating the live filter."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"KalmanFilter(β={self.beta:.4f}, μ={self.intercept:.4f}, "
            f"R={self.observation_noise}, Q={self.transition_noise}, "
            f"n={self._n_updates})"
        )


# ---------------------------------------------------------------------------
# Warm-start utility
# ---------------------------------------------------------------------------


def initialize_filter(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    observation_noise: float = 0.001,
    transition_noise: float = 0.0001,
) -> tuple[KalmanFilter, np.ndarray]:
    """Warm-start a Kalman filter by replaying historical price data.

    On startup, the live filter needs historical data to converge to a reasonable
    β estimate before generating signals. This function replays the full history
    and returns the filter in a converged state.

    Args:
        prices_a: 1-D array of historical prices for asset A (oldest → newest).
        prices_b: 1-D array of historical prices for asset B (same length).
        observation_noise: R parameter passed to KalmanFilter.
        transition_noise: Q parameter passed to KalmanFilter.

    Returns:
        (filter, spread_history) — the warm-started filter and the spread series
        produced during initialization. The spread_history is used directly by
        fit_ou_parameters() in indicators.py.

    Raises:
        ValueError: If arrays have different lengths or are empty.
        KalmanFilterError: If a numerical issue occurs during initialization.
    """
    prices_a = np.asarray(prices_a, dtype=float)
    prices_b = np.asarray(prices_b, dtype=float)

    if prices_a.shape != prices_b.shape:
        raise ValueError(
            f"prices_a and prices_b must have the same shape. "
            f"Got {prices_a.shape} and {prices_b.shape}."
        )
    if prices_a.ndim != 1 or len(prices_a) == 0:
        raise ValueError("Price arrays must be 1-D and non-empty.")

    kf = KalmanFilter(
        observation_noise=observation_noise,
        transition_noise=transition_noise,
    )

    n = len(prices_a)
    spreads = np.empty(n, dtype=float)

    for i in range(n):
        _, _, spread = kf.update(float(prices_a[i]), float(prices_b[i]))
        spreads[i] = spread

    logger.debug(
        "KalmanFilter warm-started over %d observations. β=%.4f, μ=%.4f.",
        n, kf.beta, kf.intercept,
    )
    return kf, spreads
