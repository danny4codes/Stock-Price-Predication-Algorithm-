# feature_engineering.py — QuantEdge MT5: Technical Indicators & Custom Metrics
# RULES:
#   - All functions are PURE (no side effects, no global state)
#   - No module-level imports of MT5 — only pandas/numpy/ta/arch/hurst
#   - build_feature_matrix() is the single entry point for the ML pipeline

from typing import Tuple

import numpy as np
import pandas as pd
from arch import arch_model

try:
    from hurst import compute_Hc
    _HURST_AVAILABLE = True
except ImportError:
    _HURST_AVAILABLE = False

import ta  # Technical Analysis library

from config import (
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    GARCH_P, GARCH_Q, HURST_MAX_LAG, HURST_MIN_LAG,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD,
    FEATURE_WARMUP_BARS,
)
from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Core Indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Compute Relative Strength Index.

    Args:
        series: Close price series.
        period: Lookback period. Default: 14.

    Returns:
        RSI series (0–100). Contains NaN for first `period` bars.
    """
    return ta.momentum.RSIIndicator(close=series, window=period).rsi()


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Compute MACD line, signal line, and histogram.

    Args:
        series: Close price series.
        fast:   Fast EMA period. Default: 12.
        slow:   Slow EMA period. Default: 26.
        signal: Signal EMA period. Default: 9.

    Returns:
        DataFrame with columns: [macd, macd_signal, macd_diff].
    """
    indicator = ta.trend.MACD(
        close=series, window_fast=fast, window_slow=slow, window_sign=signal
    )
    return pd.DataFrame({
        "macd":        indicator.macd(),
        "macd_signal": indicator.macd_signal(),
        "macd_diff":   indicator.macd_diff(),
    })


def compute_bollinger_bands(
    series: pd.Series,
    period: int = BOLLINGER_PERIOD,
    std_dev: float = BOLLINGER_STD,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands and derived metrics.

    Args:
        series:  Close price series.
        period:  Rolling window. Default: 20.
        std_dev: Number of standard deviations. Default: 2.0.

    Returns:
        DataFrame with columns: [bb_upper, bb_mid, bb_lower, bb_width, bb_pct].
    """
    indicator = ta.volatility.BollingerBands(
        close=series, window=period, window_dev=std_dev
    )
    return pd.DataFrame({
        "bb_upper": indicator.bollinger_hband(),
        "bb_mid":   indicator.bollinger_mavg(),
        "bb_lower": indicator.bollinger_lband(),
        "bb_width": indicator.bollinger_wband(),  # Width as % of mid
        "bb_pct":   indicator.bollinger_pband(),  # % position within bands (0–1)
    })


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Compute Average True Range.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: Lookback period. Default: 14.

    Returns:
        ATR series. Contains NaN for first `period` bars.
    """
    return ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=period
    ).average_true_range()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute Volume-Weighted Average Price (session VWAP).

    Note: Resets each day. Requires 'high', 'low', 'close', 'volume' columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        VWAP series aligned to df.index.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Cumulative within each calendar day
    cumulative_tp_vol = tp_vol.groupby(df.index.date).cumsum()
    cumulative_vol    = df["volume"].groupby(df.index.date).cumsum()

    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    return vwap.rename("vwap")


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator (%K and %D).

    Returns:
        DataFrame with columns: [stoch_k, stoch_d].
    """
    indicator = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=k_period, smooth_window=d_period
    )
    return pd.DataFrame({
        "stoch_k": indicator.stoch(),
        "stoch_d": indicator.stoch_signal(),
    })


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Args:
        series: Price series.
        period: EMA period.

    Returns:
        EMA series.
    """
    return ta.trend.EMAIndicator(close=series, window=period).ema_indicator()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Compute Average Directional Index (trend strength).

    Returns:
        DataFrame with columns: [adx, adx_pos (DI+), adx_neg (DI-)].
    """
    indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period)
    return pd.DataFrame({
        "adx":     indicator.adx(),
        "adx_pos": indicator.adx_pos(),
        "adx_neg": indicator.adx_neg(),
    })


# ---------------------------------------------------------------------------
# Advanced / Quantitative Features
# ---------------------------------------------------------------------------

def compute_hurst_exponent(
    series: pd.Series,
    min_lag: int = HURST_MIN_LAG,
    max_lag: int = HURST_MAX_LAG,
) -> float:
    """
    Compute the Hurst Exponent to characterize market regime.

    Interpretation:
        H > 0.55 → Trending (persistent)
        H < 0.45 → Mean-reverting (anti-persistent)
        0.45 ≤ H ≤ 0.55 → Random walk (no edge)

    Args:
        series:  Price series (close prices recommended).
        min_lag: Minimum lag for R/S analysis. Default: 2.
        max_lag: Maximum lag for R/S analysis. Default: 100.

    Returns:
        Hurst exponent (float, typically 0.0–1.0). Returns 0.5 on failure.
    """
    if not _HURST_AVAILABLE:
        logger.warning("hurst library not available. Returning H=0.5 (random walk).")
        return 0.5

    clean = series.dropna()
    if len(clean) < max_lag * 2:
        logger.warning(
            f"Insufficient data for Hurst ({len(clean)} bars, need {max_lag * 2}). "
            "Returning H=0.5."
        )
        return 0.5

    try:
        H, _, _ = compute_Hc(clean.values, kind="price", simplified=True)
        logger.debug(f"Hurst exponent: {H:.4f}")
        return float(H)
    except Exception as e:
        logger.warning(f"Hurst computation failed: {e}. Returning H=0.5.")
        return 0.5


def fit_garch(
    returns: pd.Series,
    p: int = GARCH_P,
    q: int = GARCH_Q,
) -> Tuple[float, object]:
    """
    Fit a GARCH(p, q) model to returns and extract conditional volatility.

    Args:
        returns: Log-return series (not price levels).
        p:       GARCH lag order for past variances. Default: 1.
        q:       ARCH lag order for past residuals. Default: 1.

    Returns:
        Tuple of (latest_conditional_volatility, fitted_model_result).
        Returns (np.nan, None) on failure.
    """
    clean = returns.dropna()
    if len(clean) < 50:
        logger.warning("Insufficient returns for GARCH (need ≥ 50). Returning NaN.")
        return np.nan, None

    try:
        model = arch_model(
            clean * 100,  # Scale to % returns for numerical stability
            vol="Garch",
            p=p,
            q=q,
            dist="Normal",
        )
        result = model.fit(disp="off", show_warning=False)
        cond_vol = result.conditional_volatility.iloc[-1] / 100  # Scale back
        logger.debug(f"GARCH fitted. Latest conditional vol: {cond_vol:.6f}")
        return float(cond_vol), result
    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}. Returning NaN.")
        return np.nan, None


def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Compute log returns from a close price series.

    Returns:
        Log-return series: ln(P_t / P_{t-1}). First value is NaN.
    """
    return np.log(close / close.shift(1)).rename("log_return")


def compute_price_momentum(close: pd.Series, periods: list) -> pd.DataFrame:
    """
    Compute price momentum (rate of change) over multiple lookback periods.

    Args:
        close:   Close price series.
        periods: List of lookback periods (e.g. [5, 10, 20]).

    Returns:
        DataFrame with one column per period: mom_5, mom_10, etc.
    """
    result = {}
    for p in periods:
        result[f"mom_{p}"] = close.pct_change(p)
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Feature Matrix Builder — Main Entry Point
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compose all technical indicators into a single ML-ready feature matrix.

    This is the SINGLE entry point for the ML pipeline.
    Input must have columns: [open, high, low, close, volume].
    Output has all NaN rows (warmup period) dropped.

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        Feature DataFrame with no NaN values. May have fewer rows than input.
    """
    if df is None or df.empty:
        logger.error("build_feature_matrix received empty DataFrame.")
        return pd.DataFrame()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"build_feature_matrix missing columns: {missing}")
        return pd.DataFrame()

    features = pd.DataFrame(index=df.index)

    # --- Price-derived ---
    features["log_return"]   = compute_log_returns(df["close"])
    features["close"]        = df["close"]
    features["hl_ratio"]     = (df["high"] - df["low"]) / df["close"]  # Bar range
    features["oc_ratio"]     = (df["close"] - df["open"]) / df["open"]  # Candle body

    # --- Trend Indicators ---
    features["rsi"]          = compute_rsi(df["close"])
    features["ema_20"]       = compute_ema(df["close"], 20)
    features["ema_50"]       = compute_ema(df["close"], 50)
    features["ema_200"]      = compute_ema(df["close"], 200)
    features["ema_cross"]    = features["ema_20"] - features["ema_50"]  # Golden/Death cross

    macd_df = compute_macd(df["close"])
    features = pd.concat([features, macd_df], axis=1)

    adx_df = compute_adx(df["high"], df["low"], df["close"])
    features = pd.concat([features, adx_df], axis=1)

    # --- Volatility Indicators ---
    bb_df = compute_bollinger_bands(df["close"])
    features = pd.concat([features, bb_df], axis=1)

    features["atr"] = compute_atr(df["high"], df["low"], df["close"])
    features["atr_pct"] = features["atr"] / df["close"]  # Normalised ATR

    # --- Momentum ---
    stoch_df = compute_stochastic(df["high"], df["low"], df["close"])
    features = pd.concat([features, stoch_df], axis=1)

    mom_df = compute_price_momentum(df["close"], periods=[5, 10, 20, 60])
    features = pd.concat([features, mom_df], axis=1)

    # --- Volume ---
    if df["volume"].sum() > 0:
        features["vwap"]       = compute_vwap(df)
        features["vwap_dist"]  = (df["close"] - features["vwap"]) / features["vwap"]
        features["vol_ratio"]  = df["volume"] / df["volume"].rolling(20).mean()

    # --- Regime (Hurst — scalar, applied as rolling window) ---
    # Rolling Hurst is expensive; compute on last N bars only for live use
    hurst_window = 100
    features["hurst"] = (
        df["close"]
        .rolling(hurst_window)
        .apply(lambda x: compute_hurst_exponent(pd.Series(x)), raw=False)
    )

    # --- GARCH Volatility (rolling, last 100 bars) ---
    log_ret = features["log_return"]
    features["garch_vol"] = (
        log_ret
        .rolling(100)
        .apply(lambda x: fit_garch(pd.Series(x))[0], raw=False)
    )

    # Drop warmup NaN rows
    n_before = len(features)
    features.dropna(inplace=True)
    n_after = len(features)
    logger.info(
        f"Feature matrix built: {n_after} rows "
        f"({n_before - n_after} warmup rows dropped, {len(features.columns)} features)"
    )

    return features
