"""tests/test_options_selector.py — Unit tests for options contract selection.

Tests:
  1. DTE selection based on tau_days * expiry_multiplier
  2. ATM strike selection nearest to current spot price
  3. Option type filtering (call vs put)
  4. Fallback when target DTE exceeds maximum chain expiry
  5. Empty or malformed chain handling
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.broker import OptionContract
from src.options_selector import select_contract


# ---------------------------------------------------------------------------
# Synthetic Options Chain Generator
# ---------------------------------------------------------------------------


def make_mock_chain(
    underlying: str = "GLD",
    expiries: list[str] | None = None,
    strikes: list[float] | None = None,
) -> list[OptionContract]:
    """Generate a mock options chain with specified expiries and strikes."""
    exp_list = expiries or ["2026-01-10", "2026-01-17", "2026-01-24", "2026-02-21"]
    strike_list = strikes or [170.0, 175.0, 180.0, 185.0, 190.0]

    contracts: list[OptionContract] = []
    for exp in exp_list:
        for strike in strike_list:
            for opt_type in ["call", "put"]:
                sym_type = "C" if opt_type == "call" else "P"
                strike_str = f"{int(strike * 1000):08d}"
                exp_clean = exp.replace("-", "")[2:]
                occ_symbol = f"{underlying}{exp_clean}{sym_type}{strike_str}"
                contracts.append(
                    OptionContract(
                        symbol=occ_symbol,
                        underlying_symbol=underlying,
                        expiration_date=exp,
                        strike_price=strike,
                        option_type=opt_type,
                        close_price=2.50,
                        open_interest=500,
                    )
                )
    return contracts


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestOptionsSelector:
    def test_dte_selection_snaps_to_nearest_ge(self):
        """tau=4.0 days * 2.5 multiplier = 10 DTE target. Reference date is 2026-01-01.
        Available expiries: 2026-01-08 (7 DTE), 2026-01-15 (14 DTE), 2026-01-22 (21 DTE).
        Should select 2026-01-15 (earliest with DTE >= 10).
        """
        ref_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        chain = make_mock_chain(
            underlying="GLD",
            expiries=["2026-01-08", "2026-01-15", "2026-01-22"],
            strikes=[180.0],
        )

        selected = select_contract(
            chain=chain,
            tau_days=4.0,
            current_price=180.0,
            option_type="call",
            expiry_multiplier=2.5,
            reference_date=ref_dt,
        )

        assert selected is not None
        assert selected.expiration_date == "2026-01-15"
        assert selected.option_type == "call"

    def test_atm_strike_selection(self):
        """Current price = 178.40. Available strikes: 170, 175, 180, 185.
        Should select strike 180.0 (abs diff 1.60 vs 175 diff 3.40).
        """
        ref_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        chain = make_mock_chain(
            underlying="GLD",
            expiries=["2026-01-15"],
            strikes=[170.0, 175.0, 180.0, 185.0],
        )

        selected = select_contract(
            chain=chain,
            tau_days=4.0,
            current_price=178.40,
            option_type="call",
            reference_date=ref_dt,
        )

        assert selected is not None
        assert selected.strike_price == 180.0

    def test_put_option_type_selection(self):
        """When option_type='put', returns contract with option_type='put'."""
        ref_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        chain = make_mock_chain(
            underlying="SLV",
            expiries=["2026-01-15"],
            strikes=[22.0],
        )

        selected = select_contract(
            chain=chain,
            tau_days=3.0,
            current_price=22.10,
            option_type="put",
            reference_date=ref_dt,
        )

        assert selected is not None
        assert selected.option_type == "put"
        assert selected.underlying_symbol == "SLV"

    def test_fallback_to_longest_available_expiry(self):
        """When target DTE (e.g. 50 days) exceeds all available expiries, pick the furthest available."""
        ref_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        chain = make_mock_chain(
            underlying="GLD",
            expiries=["2026-01-08", "2026-01-15", "2026-01-22"],  # max is 21 DTE
            strikes=[180.0],
        )

        selected = select_contract(
            chain=chain,
            tau_days=20.0,  # 20 * 2.5 = 50 DTE
            current_price=180.0,
            option_type="call",
            reference_date=ref_dt,
        )

        assert selected is not None
        assert selected.expiration_date == "2026-01-22"

    def test_empty_chain_returns_none(self):
        """Empty chain returns None gracefully."""
        selected = select_contract(
            chain=[],
            tau_days=4.0,
            current_price=180.0,
            option_type="call",
        )
        assert selected is None
