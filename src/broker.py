"""broker.py — Single interface to Alpaca's paper trading API.

PAPER TRADING ONLY. A startup assertion refuses to run if ALPACA_PAPER != 'true'.
This is the only file that imports alpaca-py. All other modules go through this wrapper.

Permission boundary (enforced by test_pipeline_integration.py):
  - get_bars(), get_options_chain(), get_account(), get_positions() — importable anywhere.
  - place_order(), cancel_order() — imported ONLY in execution_agent.py.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.models import BarSet
from alpaca.data.requests import (
    CryptoBarsRequest,
    OptionChainRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order, Position
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    GetOrdersRequest,
    MarketOrderRequest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup assertion
# ---------------------------------------------------------------------------

_PAPER_ENV_VAR = "ALPACA_PAPER"


def _assert_paper_trading() -> None:
    """Hard-fail if ALPACA_PAPER is not explicitly 'true'.

    This is a non-negotiable safety check. If it fails, the system refuses to start.
    There is no override or workaround — that is intentional.
    """
    val = os.environ.get(_PAPER_ENV_VAR, "").strip().lower()
    if val != "true":
        raise RuntimeError(
            f"SAFETY ASSERTION FAILED: {_PAPER_ENV_VAR} is '{val}', expected 'true'. "
            "This system is paper-trading only. Set ALPACA_PAPER=true in your .env "
            "file and restart. Do NOT point this at a live trading account."
        )
    logger.info("Paper trading assertion passed (%s=true).", _PAPER_ENV_VAR)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BarData:
    """Standardized bar data container returned by get_*_bars()."""

    symbol: str
    timestamps: list[datetime]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": self.opens,
                "high": self.highs,
                "low": self.lows,
                "close": self.closes,
                "volume": self.volumes,
            },
            index=pd.DatetimeIndex(self.timestamps, name="timestamp"),
        )

    @property
    def latest_close(self) -> float:
        return self.closes[-1]

    @property
    def latest_timestamp(self) -> datetime:
        return self.timestamps[-1]


@dataclass
class AccountInfo:
    equity: float
    buying_power: float
    cash: float
    currency: str
    snapshot_at: datetime


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    side: str
    market_value: float
    unrealized_pnl: float
    avg_entry_price: float
    asset_class: str


@dataclass
class OptionContract:
    symbol: str           # OCC format e.g. "GLD241220C00175000"
    underlying_symbol: str
    expiration_date: str  # YYYY-MM-DD
    strike_price: float
    option_type: str      # "call" | "put"
    close_price: float | None
    open_interest: int | None


@dataclass
class OrderResult:
    order_id: str
    alpaca_order_id: str
    status: str
    symbol: str
    qty: float
    side: str
    submitted_at: datetime


# ---------------------------------------------------------------------------
# Broker class
# ---------------------------------------------------------------------------


class Broker:
    """Unified interface to Alpaca paper trading (equity + options + crypto)."""

    def __init__(self, api_key: str, secret_key: str) -> None:
        _assert_paper_trading()

        self._trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )
        self._stock_data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self._crypto_data_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        logger.info("Broker initialized (paper=True).")

    # ── Market data ──────────────────────────────────────────────────────────

    def get_equity_bars(
        self,
        symbols: list[str],
        limit: int = 100,
        timeframe: str = "1Day",
    ) -> dict[str, BarData]:
        """Fetch daily bars for equity symbols.

        Args:
            symbols: List of equity tickers (e.g. ['GLD', 'SLV']).
            limit: Maximum number of bars per symbol.
            timeframe: Alpaca timeframe string — '1Day' for daily bars.

        Returns:
            Dict mapping symbol → BarData (oldest-to-newest order).
        """
        tf = self._parse_timeframe(timeframe)
        start_dt = datetime.now(timezone.utc) - timedelta(days=limit * 2)
        result: dict[str, BarData] = {}

        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_dt,
                    limit=limit,
                    feed=DataFeed.IEX,
                    sort="desc",
                )
                response = self._stock_data_client.get_stock_bars(request)
                raw_dict = response.data if hasattr(response, "data") else (response if isinstance(response, dict) else {})
                bars_list = raw_dict.get(symbol, [])
                if not bars_list:
                    logger.warning("No bars returned for %s", symbol)
                    continue
                # sort='desc' returns newest first; reverse so BarData is chronological (oldest to newest)
                bars_list = list(reversed(bars_list))
                result[symbol] = BarData(
                    symbol=symbol,
                    timestamps=[b.timestamp.replace(tzinfo=timezone.utc) for b in bars_list],
                    opens=[float(b.open) for b in bars_list],
                    highs=[float(b.high) for b in bars_list],
                    lows=[float(b.low) for b in bars_list],
                    closes=[float(b.close) for b in bars_list],
                    volumes=[float(b.volume) for b in bars_list],
                )
            except Exception as exc:
                logger.warning("Failed to fetch equity bars for %s: %s", symbol, exc)

        return result

    def get_crypto_bars(
        self,
        symbols: list[str],
        limit: int = 200,
        timeframe: str = "1Hour",
    ) -> dict[str, BarData]:
        """Fetch hourly bars for crypto symbols.

        Args:
            symbols: List of crypto pairs (e.g. ['BTC/USD', 'ETH/USD']).
            limit: Maximum number of bars per symbol.
            timeframe: '1Hour' for hourly bars.

        Returns:
            Dict mapping symbol → BarData (oldest-to-newest order).
        """
        tf = self._parse_timeframe(timeframe)
        start_dt = datetime.now(timezone.utc) - timedelta(hours=limit * 2)
        result: dict[str, BarData] = {}

        for symbol in symbols:
            try:
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_dt,
                    limit=limit,
                    sort="desc",
                )
                response = self._crypto_data_client.get_crypto_bars(request)
                raw_dict = response.data if hasattr(response, "data") else (response if isinstance(response, dict) else {})
                bars_list = raw_dict.get(symbol, [])
                if not bars_list:
                    logger.warning("No bars returned for %s", symbol)
                    continue
                # sort='desc' returns newest first; reverse so BarData is chronological (oldest to newest)
                bars_list = list(reversed(bars_list))
                result[symbol] = BarData(
                    symbol=symbol,
                    timestamps=[b.timestamp.replace(tzinfo=timezone.utc) for b in bars_list],
                    opens=[float(b.open) for b in bars_list],
                    highs=[float(b.high) for b in bars_list],
                    lows=[float(b.low) for b in bars_list],
                    closes=[float(b.close) for b in bars_list],
                    volumes=[float(b.volume) for b in bars_list],
                )
            except Exception as exc:
                logger.warning("Failed to fetch crypto bars for %s: %s", symbol, exc)

        return result

    def get_options_chain(
        self,
        underlying_symbol: str,
        expiry_date_gte: str | None = None,
        expiry_date_lte: str | None = None,
        option_type: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
    ) -> list[OptionContract]:
        """Fetch an options chain from Alpaca.

        Args:
            underlying_symbol: Equity ticker (e.g. 'GLD').
            expiry_date_gte: Minimum expiry date in YYYY-MM-DD format.
            expiry_date_lte: Maximum expiry date in YYYY-MM-DD format.
            option_type: 'call' or 'put' (None = both).
            strike_price_gte: Minimum strike.
            strike_price_lte: Maximum strike.

        Returns:
            List of OptionContract objects.
        """
        kwargs: dict[str, Any] = {
            "underlying_symbols": [underlying_symbol],
        }
        if expiry_date_gte:
            kwargs["expiration_date_gte"] = expiry_date_gte
        else:
            # Default to tomorrow so we only fetch future active options
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
            kwargs["expiration_date_gte"] = tomorrow
        if expiry_date_lte:
            kwargs["expiration_date_lte"] = expiry_date_lte
        if option_type:
            kwargs["type"] = option_type
        if strike_price_gte is not None:
            kwargs["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            kwargs["strike_price_lte"] = str(strike_price_lte)

        try:
            request = GetOptionContractsRequest(**kwargs)
            response = self._trading_client.get_option_contracts(request)
        except Exception as exc:
            logger.error(
                "get_options_chain failed for %s: %s", underlying_symbol, exc
            )
            raise BrokerError(f"Failed to fetch options chain: {exc}") from exc

        contracts = []
        for c in response.option_contracts:
            contracts.append(
                OptionContract(
                    symbol=c.symbol,
                    underlying_symbol=c.underlying_symbol,
                    expiration_date=str(c.expiration_date),
                    strike_price=float(c.strike_price),
                    option_type=c.type.value.lower(),
                    close_price=float(c.close_price) if c.close_price else None,
                    open_interest=int(c.open_interest) if c.open_interest else None,
                )
            )
        return contracts

    # ── Account / positions ───────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        """Fetch current account state.

        Raises BrokerError on any API failure — callers (Risk Agent) must treat
        this as a fail-closed condition.
        """
        try:
            account = self._trading_client.get_account()
        except Exception as exc:
            logger.error("get_account failed: %s", exc)
            raise BrokerError(f"Failed to fetch account info: {exc}") from exc

        return AccountInfo(
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            cash=float(account.cash),
            currency=account.currency,
            snapshot_at=datetime.utcnow(),
        )

    def get_positions(self) -> list[PositionInfo]:
        """Fetch all open positions."""
        try:
            positions: list[Position] = self._trading_client.get_all_positions()
        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc

        return [
            PositionInfo(
                symbol=p.symbol,
                qty=float(p.qty),
                side=p.side.value,
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                avg_entry_price=float(p.avg_entry_price),
                asset_class=p.asset_class.value,
            )
            for p in positions
        ]

    # ── Order placement — RESTRICTED to execution_agent.py ───────────────────
    # grep-checkable: the string 'place_order' appears ONLY here and in execution_agent.py

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> OrderResult:
        """Place a paper market order.

        Args:
            symbol: Ticker or options contract OCC symbol or crypto pair.
            qty: Quantity (shares, contracts, or crypto units).
            side: 'buy' or 'sell'.
            time_in_force: 'day' (default) or 'gtc' (for crypto).

        Returns:
            OrderResult with Alpaca order ID and status.

        Raises:
            BrokerError: On any API error. Caller must handle and log.
        """
        order_side = OrderSide.BUY if side.lower() in ("buy", "buy_to_open") else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
        )
        try:
            order: Order = self._trading_client.submit_order(order_data=request)
        except Exception as exc:
            logger.error(
                "place_order FAILED: symbol=%s qty=%s side=%s | %s",
                symbol, qty, side, exc,
            )
            raise BrokerError(f"Order placement failed for {symbol}: {exc}") from exc

        logger.info(
            "Order placed: alpaca_id=%s symbol=%s qty=%s side=%s status=%s",
            order.id, symbol, qty, side, order.status,
        )
        return OrderResult(
            order_id=str(order.client_order_id) if order.client_order_id else str(order.id),
            alpaca_order_id=str(order.id),
            status=order.status.value,
            symbol=symbol,
            qty=float(qty),
            side=side,
            submitted_at=order.submitted_at.replace(tzinfo=timezone.utc),
        )

    def cancel_order(self, alpaca_order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled, False if already filled."""
        try:
            self._trading_client.cancel_order_by_id(alpaca_order_id)
            logger.info("Cancelled order %s", alpaca_order_id)
            return True
        except Exception as exc:
            logger.warning("cancel_order %s: %s", alpaca_order_id, exc)
            return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_timeframe(timeframe: str) -> TimeFrame:
        mapping = {
            "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
            "1Day":  TimeFrame(1,  TimeFrameUnit.Day),
        }
        if timeframe not in mapping:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Valid: {list(mapping.keys())}"
            )
        return mapping[timeframe]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrokerError(Exception):
    """Raised on any Alpaca API error. Callers treat this as fail-closed."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_broker() -> Broker:
    """Create a Broker from environment variables.

    Reads ALPACA_API_KEY and ALPACA_SECRET_KEY from the environment.
    Fails hard if ALPACA_PAPER != 'true' (via the Broker constructor).
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
            "Copy .env.example to .env and fill in your paper trading credentials."
        )

    return Broker(api_key=api_key, secret_key=secret_key)
