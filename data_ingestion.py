# data_ingestion.py — QuantEdge MT5: Data Connection & Ingestion
# RULE: This is the ONLY module that may import MetaTrader5.
# All other modules must call functions from here.

import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz

from config import MT5_BASE_DELAY_SECONDS, MT5_MAX_RETRIES
from logger_config import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _exponential_backoff(attempt: int, base: float = MT5_BASE_DELAY_SECONDS) -> None:
    """Sleep for base * 2^attempt seconds."""
    delay = base * (2 ** attempt)
    logger.warning(f"Backing off for {delay:.1f}s (attempt {attempt + 1}/{MT5_MAX_RETRIES})")
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

def initialize_mt5(
    login: int,
    password: str,
    server: str,
    path: str,
) -> bool:
    """
    Initialize the MetaTrader 5 terminal connection.

    Retries with exponential backoff on failure.

    Args:
        login:    MT5 account number.
        password: MT5 account password.
        server:   Broker server name (e.g. "BrokerName-Live").
        path:     Full path to terminal64.exe.

    Returns:
        True if connected successfully, False after all retries exhausted.
    """
    for attempt in range(MT5_MAX_RETRIES):
        initialized = mt5.initialize(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=30_000,
        )
        if initialized:
            info = mt5.terminal_info()
            logger.info(
                f"MT5 connected | Server: {server} | Login: {login} | "
                f"Build: {info.build if info else 'unknown'}"
            )
            return True

        error = mt5.last_error()
        logger.error(f"MT5 init failed: {error}")
        if attempt < MT5_MAX_RETRIES - 1:
            _exponential_backoff(attempt)

    logger.critical("MT5 initialization failed after all retries. System halting.")
    return False


def shutdown_mt5() -> None:
    """
    Gracefully disconnect from MetaTrader 5.
    Always call this at program exit.
    """
    mt5.shutdown()
    logger.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# Historical Data
# ---------------------------------------------------------------------------

def get_historical_data(
    symbol: str,
    timeframe: int,
    start: datetime,
    end: datetime,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars from MT5 for a given symbol and time range.

    Args:
        symbol:    Trading symbol (e.g. "EURUSD").
        timeframe: MT5 timeframe constant (e.g. mt5.TIMEFRAME_H1).
        start:     Start datetime (timezone-aware recommended).
        end:       End datetime (timezone-aware recommended).

    Returns:
        DataFrame with columns [open, high, low, close, tick_volume, spread, real_volume]
        and a UTC DatetimeIndex. Returns None on failure.
    """
    utc = pytz.utc
    start_utc = start.astimezone(utc) if start.tzinfo else utc.localize(start)
    end_utc   = end.astimezone(utc)   if end.tzinfo   else utc.localize(end)

    for attempt in range(MT5_MAX_RETRIES):
        rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            logger.info(f"Fetched {len(df)} bars for {symbol} [{start_utc} → {end_utc}]")
            return df

        error = mt5.last_error()
        logger.warning(f"get_historical_data failed for {symbol}: {error}")
        if attempt < MT5_MAX_RETRIES - 1:
            _exponential_backoff(attempt)

    logger.error(f"Could not fetch historical data for {symbol} after {MT5_MAX_RETRIES} retries.")
    return None


def get_historical_data_n_bars(
    symbol: str,
    timeframe: int,
    n_bars: int,
) -> Optional[pd.DataFrame]:
    """
    Fetch the most recent N bars from MT5.

    Args:
        symbol:    Trading symbol.
        timeframe: MT5 timeframe constant.
        n_bars:    Number of bars to retrieve.

    Returns:
        DataFrame with UTC DatetimeIndex, or None on failure.
    """
    for attempt in range(MT5_MAX_RETRIES):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            logger.debug(f"Fetched {len(df)} bars (N-bars) for {symbol}")
            return df

        error = mt5.last_error()
        logger.warning(f"get_historical_data_n_bars failed for {symbol}: {error}")
        if attempt < MT5_MAX_RETRIES - 1:
            _exponential_backoff(attempt)

    logger.error(f"Could not fetch {n_bars} bars for {symbol}.")
    return None


# ---------------------------------------------------------------------------
# Live Market Data
# ---------------------------------------------------------------------------

def get_live_tick(symbol: str) -> Optional[dict]:
    """
    Fetch the latest tick for a symbol.

    Returns:
        Dict with keys: symbol, bid, ask, last, volume, time (UTC datetime).
        Returns None on failure.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.warning(f"Could not fetch tick for {symbol}: {mt5.last_error()}")
        return None

    return {
        "symbol": symbol,
        "bid":    tick.bid,
        "ask":    tick.ask,
        "last":   tick.last,
        "volume": tick.volume,
        "time":   datetime.fromtimestamp(tick.time, tz=pytz.utc),
    }


def get_symbol_info(symbol: str) -> Optional[dict]:
    """
    Fetch trading specification for a symbol (pip size, contract size, etc.).

    Returns:
        Dict of relevant symbol properties, or None on failure.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(f"Symbol info unavailable for {symbol}: {mt5.last_error()}")
        return None

    return {
        "symbol":        info.name,
        "digits":        info.digits,
        "point":         info.point,
        "trade_tick_size":   info.trade_tick_size,
        "trade_tick_value":  info.trade_tick_value,
        "trade_contract_size": info.trade_contract_size,
        "volume_min":    info.volume_min,
        "volume_max":    info.volume_max,
        "volume_step":   info.volume_step,
        "spread":        info.spread,
        "currency_base": info.currency_base,
        "currency_profit": info.currency_profit,
    }


# ---------------------------------------------------------------------------
# Account & Positions
# ---------------------------------------------------------------------------

def get_account_info() -> Optional[dict]:
    """
    Fetch current account state from MT5.

    Returns:
        Dict with balance, equity, margin, free_margin, profit, leverage.
        Returns None on failure.
    """
    info = mt5.account_info()
    if info is None:
        logger.error(f"Could not fetch account info: {mt5.last_error()}")
        return None

    return {
        "login":        info.login,
        "balance":      info.balance,
        "equity":       info.equity,
        "margin":       info.margin,
        "free_margin":  info.margin_free,
        "profit":       info.profit,
        "leverage":     info.leverage,
        "currency":     info.currency,
        "server":       info.server,
    }


def get_open_positions(symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all open positions, optionally filtered by symbol.

    Args:
        symbol: If provided, filter to only this symbol's positions.

    Returns:
        DataFrame of open positions. Empty DataFrame if none.
    """
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

    if positions is None or len(positions) == 0:
        logger.debug(f"No open positions{f' for {symbol}' if symbol else ''}.")
        return pd.DataFrame()

    df = pd.DataFrame([p._asdict() for p in positions])
    df["time"]     = pd.to_datetime(df["time"],     unit="s", utc=True)
    df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    logger.debug(f"Found {len(df)} open position(s).")
    return df


def get_pending_orders(symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all pending (limit/stop) orders, optionally filtered by symbol.

    Returns:
        DataFrame of pending orders. Empty DataFrame if none.
    """
    orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()

    if orders is None or len(orders) == 0:
        return pd.DataFrame()

    df = pd.DataFrame([o._asdict() for o in orders])
    df["time_setup"] = pd.to_datetime(df["time_setup"], unit="s", utc=True)
    return df
