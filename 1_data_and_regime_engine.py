# 1_data_and_regime_engine.py — QuantEdge MT5: Data Ingestion & Regime Detection
# Combines safe MT5 connection with rolling Hurst exponent (DFA) and GARCH(1,1) volatility
# RULE: This module may import MetaTrader5 ONLY for data ingestion purposes.
# All regime calculations use pure functions from feature_engineering where possible.

import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from arch import arch_model

from config import (
    DEFAULT_TIMEFRAME_STR,
    FEATURE_WARMUP_BARS,
    GARCH_P,
    GARCH_Q,
    HURST_MAX_LAG,
    HURST_MIN_LAG,
    HURST_TRENDING_THRESHOLD,
    LOOKBACK_BARS,
    MT5_BASE_DELAY_SECONDS,
    MT5_MAX_RETRIES,
)
from feature_engineering import compute_hurst_exponent, compute_log_returns, fit_garch
from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# MT5 Connection Helpers (same safety patterns as data_ingestion.py)
# ---------------------------------------------------------------------------

def _mt5_exponential_backoff(attempt: int, base: float = MT5_BASE_DELAY_SECONDS) -> None:
    """Sleep for base * 2^attempt seconds with logging."""
    delay = base * (2 ** attempt)
    logger.warning(f"MT5 operation backing off for {delay:.1f}s (attempt {attempt + 1}/{MT5_MAX_RETRIES})")
    time.sleep(delay)


def safe_mt5_initialize(
    login: int,
    password: str,
    server: str,
    path: str,
) -> bool:
    """
    Safely initialize MT5 connection with exponential backoff retry.

    Args:
        login: MT5 account number
        password: Account password
        server: Broker server name
        path: Full path to terminal64.exe

    Returns:
        True if connection successful, False if all retries exhausted
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
        logger.error(f"MT5 initialize failed (attempt {attempt + 1}): {error}")

        if attempt < MT5_MAX_RETRIES - 1:
            _mt5_exponential_backoff(attempt)

    logger.critical("MT5 initialization failed after all retries.")
    return False


def safe_mt5_shutdown() -> None:
    """Gracefully shutdown MT5 connection."""
    try:
        mt5.shutdown()
        logger.info("MT5 connection closed.")
    except Exception as e:
        logger.error(f"Error during MT5 shutdown: {e}")


# ---------------------------------------------------------------------------
# Data Acquisition Functions
# ---------------------------------------------------------------------------

def get_m1_historical_data(
    symbol: str = "EURUSD",
    lookback_bars: int = LOOKBACK_BARS,
) -> Optional[pd.DataFrame]:
    """
    Fetch M1 (1-minute) historical data for regime analysis.

    Args:
        symbol: Trading symbol (default: EURUSD)
        lookback_bars: Number of M1 bars to fetch

    Returns:
        DataFrame with OHLCV data and UTC DatetimeIndex, or None on failure
    """
    timeframe = mt5.TIMEFRAME_M1
    utc = pytz.utc
    end_dt = datetime.now(utc)
    start_dt = end_dt - timedelta(minutes=lookback_bars)

    for attempt in range(MT5_MAX_RETRIES):
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)

            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
                df.rename(columns={"tick_volume": "volume"}, inplace=True)

                logger.info(f"Fetched {len(df)} M1 bars for {symbol}")
                return df

        except Exception as e:
            error = mt5.last_error()
            logger.warning(f"M1 data fetch failed (attempt {attempt + 1}): {error} - {e}")

            if attempt < MT5_MAX_RETRIES - 1:
                _mt5_exponential_backoff(attempt)

    logger.error(f"Could not fetch M1 data for {symbol} after {MT5_MAX_RETRIES} retries.")
    return None


def get_live_tick_safe(symbol: str = "EURUSD") -> Optional[dict]:
    """
    Safely fetch latest tick with error handling.

    Args:
        symbol: Trading symbol

    Returns:
        Tick dictionary or None on failure
    """
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"No tick data for {symbol}: {mt5.last_error()}")
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time, tz=pytz.utc),
        }
    except Exception as e:
        logger.error(f"Exception fetching tick for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Regime Detection Calculations
# ---------------------------------------------------------------------------

def rolling_hurst_exponent(
    price_series: pd.Series,
    window: int = 100,
    min_lag: int = HURST_MIN_LAG,
    max_lag: int = HURST_MAX_LAG,
) -> pd.Series:
    """
    Calculate rolling Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Args:
        price_series: Price series (typically close prices)
        window: Rolling window size in periods (default: 100)
        min_lag: Minimum lag for DFA calculation
        max_lag: Maximum lag for DFA calculation

    Returns:
        Series of Hurst exponent values aligned with input index
    """
    def hurst_func(x):
        if len(x) < max_lag * 2:
            return np.nan
        return compute_hurst_exponent(pd.Series(x), min_lag, max_lag)

    hurst_series = price_series.rolling(window).apply(hurst_func, raw=False)
    logger.debug(f"Calculated rolling Hurst with window={window}")
    return hurst_series


def is_trending_signal(hurst_value: float) -> bool:
    """
    Determine if market is trending based on Hurst exponent.

    Args:
        hurst_value: Calculated Hurst exponent (0.0 to 1.0)

    Returns:
        True if H > 0.55 (trending), False otherwise
    """
    return hurst_value > HURST_TRENDING_THRESHOLD


def rolling_garch_volatility(
    returns_series: pd.Series,
    window: int = 100,
    p: int = GARCH_P,
    q: int = GARCH_Q,
) -> pd.Series:
    """
    Calculate rolling GARCH(1,1) volatility forecasts.

    Mathematical formulation: σ²_t = ω + αε²_{t-1} + βσ²_{t-1}

    Args:
        returns_series: Log-return series
        window: Rolling window for GARCH fitting (default: 100)
        p: GARCH lag order
        q: ARCH lag order

    Returns:
        Series of forecasted volatility values (standard deviation, not variance)
    """
    def garch_func(x):
        if len(x) < 30:  # Minimum for GARCH stability
            return np.nan
        vol, _ = fit_garch(pd.Series(x), p, q)
        return vol  # Returns conditional volatility (std dev)

    garch_series = returns_series.rolling(window).apply(garch_func, raw=False)
    logger.debug(f"Calculated rolling GARCH({p},{q}) volatility with window={window}")
    return garch_series


# ---------------------------------------------------------------------------
# Main Data & Regime Pipeline
# ---------------------------------------------------------------------------

def fetch_and_process_regime_data(
    symbol: str = "EURUSD",
    lookback_bars: int = LOOKBACK_BARS,
) -> Optional[pd.DataFrame]:
    """
    Main pipeline: Fetch M1 data, calculate Hurst exponent and GARCH volatility.

    Returns synchronized DataFrame with:
    - Raw price data (open, high, low, close, volume)
    - Hurst exponent values (rolling 100-period)
    - Boolean is_trending flag (H > 0.55)
    - GARCH volatility forecasts (standard deviation)
    - Log returns (used for GARCH calculation)

    Args:
        symbol: Trading symbol to analyze
        lookback_bars: Number of M1 bars to fetch for analysis

    Returns:
        Processed DataFrame with regime indicators, or None on failure
    """
    # Step 1: Fetch raw M1 data
    logger.info(f"Fetching {lookback_bars} M1 bars for {symbol} regime analysis")
    df_raw = get_m1_historical_data(symbol, lookback_bars)

    if df_raw is None or df_raw.empty:
        logger.error("Failed to fetch M1 data - cannot proceed with regime analysis")
        return None

    # Step 2: Calculate log returns (needed for GARCH)
    df_raw["log_return"] = compute_log_returns(df_raw["close"])

    # Step 3: Rolling Hurst exponent (100-period window)
    logger.info("Calculating rolling Hurst exponent (DFA method)")
    df_raw["hurst"] = rolling_hurst_exponent(df_raw["close"], window=100)

    # Step 4: Boolean trending signal
    df_raw["is_trending"] = df_raw["hurst"].apply(is_trending_signal)

    # Step 5: Rolling GARCH(1,1) volatility on log returns
    logger.info("Calculating rolling GARCH(1,1) volatility")
    # Use only non-null returns for GARCH calculation
    clean_returns = df_raw["log_return"].dropna()
    if len(clean_returns) < 30:
        logger.warning("Insufficient returns data for GARCH calculation")
        df_raw["garch_vol"] = np.nan
        df_raw["garch_var"] = np.nan
    else:
        garch_vol = rolling_garch_volatility(clean_returns, window=100)
        # Align back to original dataframe
        df_raw["garch_vol"] = garch_vol.reindex(df_raw.index)
        df_raw["garch_var"] = df_raw["garch_vol"] ** 2  # Variance = volatility²

    # Step 6: Clean up and prepare final output
    # Keep only essential columns for regime detection pipeline
    required_columns = [
        "open", "high", "low", "close", "volume",
        "log_return", "hurst", "is_trending", "garch_vol", "garch_var"
    ]

    # Ensure all required columns exist
    missing_cols = [col for col in required_columns if col not in df_raw.columns]
    if missing_cols:
        logger.warning(f"Missing columns in regime data: {missing_cols}")
        # Add missing columns with NaN values
        for col in missing_cols:
            df_raw[col] = np.nan

    # Select and return only the regime-relevant columns
    regime_df = df_raw[required_columns].copy()

    # Log summary statistics
    valid_hurst = regime_df["hurst"].dropna()
    valid_garch = regime_df["garch_vol"].dropna()
    trending_pct = (regime_df["is_trending"].sum() / len(regime_df)) * 100 if len(regime_df) > 0 else 0

    logger.info(
        f"Regime data pipeline complete: {len(regime_df)} rows | "
        f"Hurst: {valid_hurst.mean():.3f}±{valid_hurst.std():.3f} ({len(valid_hurst)} valid) | "
        f"GARCH Vol: {valid_garch.mean():.6f}±{valid_garch.std():.6f} ({len(valid_garch)} valid) | "
        f"Trending: {trending_pct:.1f}% of periods"
    )

    return regime_df


def get_current_regime_state(symbol: str = "EURUSD") -> Optional[dict]:
    """
    Get the most recent regime state for real-time trading decisions.

    Args:
        symbol: Trading symbol to analyze

    Returns:
        Dictionary with current regime indicators or None on failure
    """
    # Fetch recent data (enough for rolling calculations)
    regime_df = fetch_and_process_regime_data(symbol, lookback_bars=150)

    if regime_df is None or regime_df.empty:
        logger.error("Cannot get current regime state - data pipeline failed")
        return None

    # Get the most recent complete row (drop NaN values from warmup period)
    clean_df = regime_df.dropna()

    if clean_df.empty:
        logger.warning("Insufficient data for current regime state (warmup period)")
        return None

    latest = clean_df.iloc[-1]

    regime_state = {
        "symbol": symbol,
        "timestamp": latest.name,  # DatetimeIndex value
        "price": float(latest["close"]),
        "hurst": float(latest["hurst"]),
        "is_trending": bool(latest["is_trending"]),
        "garch_volatility": float(latest["garch_vol"]),
        "garch_variance": float(latest["garch_var"]),
        "log_return": float(latest["log_return"]),
    }

    logger.debug(f"Current regime state: {regime_state}")
    return regime_state


# ---------------------------------------------------------------------------
# Module Health Check
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """
    Perform health check on the data and regime engine.

    Returns:
        Dictionary with component status
    """
    health = {
        "module": "1_data_and_regime_engine",
        "timestamp": datetime.now(pytz.utc).isoformat(),
        "mt5_initialized": False,
        "can_fetch_data": False,
        "hurst_available": False,
        "garch_available": False,
    }

    # Check MT5 connection
    try:
        health["mt5_initialized"] = mt5.terminal_info() is not None
    except:
        health["mt5_initialized"] = False

    # Check if we can fetch data
    if health["mt5_initialized"]:
        try:
            test_data = get_m1_historical_data("EURUSD", 10)
            health["can_fetch_data"] = test_data is not None and not test_data.empty
        except:
            health["can_fetch_data"] = False

    # Check dependencies
    try:
        from hurst import compute_Hc
        health["hurst_available"] = True
    except ImportError:
        health["hurst_available"] = False

    try:
        from arch import arch_model
        health["garch_available"] = True
    except ImportError:
        health["garch_available"] = False

    # Overall status
    health["healthy"] = (
        health["mt5_initialized"] and
        health["can_fetch_data"] and
        health["hurst_available"] and
        health["garch_available"]
    )

    logger.info(f"Health check: {health}")
    return health


# ---------------------------------------------------------------------------
# Example Usage (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Example usage and testing of the data and regime engine."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv(r"D:\Trading\.env")

    # Example MT5 credentials (should be in .env, not hardcoded)
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")
    MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

    if MT5_LOGIN == 0 or not MT5_PASSWORD or not MT5_SERVER:
        logger.error("MT5 credentials not configured in .env file")
        logger.info("Please copy .env.template to .env and fill in your credentials")
    else:
        # Initialize MT5
        if safe_mt5_initialize(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH):
            try:
                # Test the regime pipeline
                logger.info("Testing regime data pipeline...")
                regime_data = fetch_and_process_regime_data("EURUSD", 200)

                if regime_data is not None:
                    print(f"\nRegime Data Shape: {regime_data.shape}")
                    print(f"Columns: {list(regime_data.columns)}")
                    print(f"\nLatest regime state:")
                    latest = regime_data.dropna().iloc[-1] if not regime_data.dropna().empty else None
                    if latest is not None:
                        print(f"  Time: {latest.name}")
                        print(f"  Price: {latest['close']:.5f}")
                        print(f"  Hurst: {latest['hurst']:.4f} ({'TRENDING' if latest['is_trending'] else 'NOT TRENDING'})")
                        print(f"  GARCH Vol: {latest['garch_vol']:.6f}")
                        print(f"  GARCH Var: {latest['garch_var']:.6f}")

                # Test current regime state
                print(f"\nCurrent regime state:")
                current_state = get_current_regime_state("EURUSD")
                if current_state:
                    for key, value in current_state.items():
                        if key != "timestamp":
                            print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")

            finally:
                safe_mt5_shutdown()
        else:
            logger.error("Failed to initialize MT5 connection")