"""options_selector.py — Options contract selection from an Alpaca options chain.

Selects the nearest ATM strike and optimal expiry based on OU half-life:
  target_dte = round(tau_days * expiry_multiplier)

The half-life represents expected time to mean reversion. Giving the spread
`expiry_multiplier` (default 2.5×) times its half-life in DTE allows sufficient
time for mean-reversion to complete without suffering excessive early theta decay.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from src.broker import OptionContract

logger = logging.getLogger(__name__)


def select_contract(
    chain: list[OptionContract],
    tau_days: float,
    current_price: float,
    option_type: Literal["call", "put"],
    expiry_multiplier: float = 2.5,
    reference_date: datetime | None = None,
) -> OptionContract | None:
    """Select the best matching options contract from an options chain.

    Args:
        chain: List of OptionContract objects for the underlying.
        tau_days: OU half-life in days (tau = ln(2)/kappa).
        current_price: Current underlying equity spot price.
        option_type: "call" or "put".
        expiry_multiplier: Multiplier for target DTE (default 2.5x tau).
        reference_date: Current date (defaults to UTC now).

    Returns:
        The selected OptionContract, or None if no matching contract is found.
    """
    if not chain:
        logger.warning("Empty options chain provided to select_contract.")
        return None

    ref_dt = reference_date or datetime.now(timezone.utc)
    target_dte = max(1, round(tau_days * expiry_multiplier))

    # 1. Filter by option type (call vs put)
    type_filtered = [c for c in chain if c.option_type.lower() == option_type.lower()]
    if not type_filtered:
        logger.warning("No contracts found with option_type=%s", option_type)
        return None

    # 2. Group available expiry dates and calculate their DTEs
    # Parse expiration dates
    expiries_with_dte: list[tuple[str, int]] = []
    unique_expiries = {c.expiration_date for c in type_filtered}

    min_dte = 5  # Require at least 5 days to expiry to avoid overnight theta collapse
    for exp_str in sorted(unique_expiries):
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dte = (exp_date.date() - ref_dt.date()).days
            if dte >= min_dte:
                expiries_with_dte.append((exp_str, dte))
        except ValueError:
            continue

    # Fallback to any future expiry if no expiries >= min_dte
    if not expiries_with_dte:
        for exp_str in sorted(unique_expiries):
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte = (exp_date.date() - ref_dt.date()).days
                if dte >= 1:
                    expiries_with_dte.append((exp_str, dte))
            except ValueError:
                continue

    if not expiries_with_dte:
        logger.warning("No valid future expiration dates found in chain.")
        return None

    # 3. Select best expiry date:
    # Prefer the earliest expiry date with DTE >= target_dte (clamped to at least min_dte).
    effective_target_dte = max(min_dte, target_dte)
    valid_gte = [item for item in expiries_with_dte if item[1] >= effective_target_dte]
    if valid_gte:
        best_expiry_str, chosen_dte = valid_gte[0]
    else:
        best_expiry_str, chosen_dte = expiries_with_dte[-1]

    # 4. Filter contracts on the selected expiry date
    expiry_contracts = [c for c in type_filtered if c.expiration_date == best_expiry_str]
    if not expiry_contracts:
        return None

    # 5. Select the nearest ATM strike
    best_contract = min(
        expiry_contracts,
        key=lambda c: abs(c.strike_price - current_price),
    )

    logger.info(
        "Selected contract %s | type=%s | strike=%.2f (spot=%.2f) | exp=%s (DTE=%d, target=%d)",
        best_contract.symbol,
        best_contract.option_type,
        best_contract.strike_price,
        current_price,
        best_contract.expiration_date,
        chosen_dte,
        target_dte,
    )
    return best_contract
