# feature_engineering.py — QuantEdge MT5: Econometric & Microstructure Features
# Rules:
#   - All functions are PURE (no side effects, no global state)
#   - NO module-level imports of MT5 — only pandas/numpy/arch/hurst
#   - NO retail technical indicators (no RSI, MACD, ADX, Bollinger Bands, EMA, etc.)
#   - Microstructure logic (DC, OFI, Primary Signals) lives in 2_microstructure_engine.py
#   - build_feature_matrix() is the single entry point for the ML pipeline
#
# PERMITTED FEATURE FAMILIES:
#   - Hurst Exponent (regime detection via DFA)
#   - GARCH(1,1) conditional volatility
#   - Directional Changes (DC) — delegated to 2_microstructure_engine.py
#   - Order Flow Imbalance (OFI) — delegated to 2_microstructure_engine.py
#   - VWAP deviation
#   - Log returns and volatility z-scores

from typing import Tuple

from datetime import datetime as _dt  # noqa: F401 (used in type hints downstream)
import numpy as np
import pandas as pd
from arch import arch_model

try:
    from hurst import compute_Hc
    _HURST_AVAILABLE = True
except ImportError:
    _HURST_AVAILABLE = False

from config import (
    ATR_PERIOD,
    GARCH_P,
    GARCH_Q,
    GARCH_ROLLING_WINDOW,
    HURST_MAX_LAG,
    HURST_MEAN_REVERT_THRESHOLD,
    HURST_MIN_LAG,
    HURST_ROLLING_WINDOW,
    HURST_TRENDING_THRESHOLD,
    VOL_ZSCORE_THRESHOLD,
    VOL_ZSCORE_WINDOW,
    VWAP_DIST_THRESHOLD,
    FEATURE_WARMUP_BARS,
)
from logger_config import setup_logger

# Import microstructure components from the dedicated engine (numeric prefix requires importlib)
import importlib as _il
_mse = _il.import_module("2_microstructure_engine")
DirectionalChangeDetector = _mse.DirectionalChangeDetector
calculate_order_flow_imbalance = _mse.calculate_order_flow_imbalance
evaluate_primary_signals_batch = _mse.evaluate_primary_signals_batch
evaluate_pullback_signals = _mse.evaluate_pullback_signals

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Core Econometric Indicators
# ---------------------------------------------------------------------------

def compute_log_returns(close: pd.Series) -> pd.Series:
    """
    Compute log returns from a close price series.

    Returns:
        Log-return series: ln(P_t / P_{t-1}). First value is NaN.
    """
    return np.log(close / close.shift(1)).rename("log_return")


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
        logger.warning("Insufficient returns for GARCH (need >= 50). Returning NaN.")
        return np.nan, None

    try:
        import warnings
        model = arch_model(
            clean * 100,  # Scale to % returns for numerical stability
            vol="Garch",
            p=p,
            q=q,
            dist="Normal",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)
        cond_vol = result.conditional_volatility.iloc[-1] / 100  # Scale back
        return float(cond_vol), result
    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}. Returning NaN.")
        return np.nan, None


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Compute Average True Range for SL/TP sizing.

    Args:
        high:   High price series.
        low:    Low price series.
        close:  Close price series.
        period: Lookback period. Default: 14.

    Returns:
        ATR series. Contains NaN for first `period` bars.
    """
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr.rename("atr")


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute Volume-Weighted Average Price (session VWAP).

    Resets each day. Requires 'high', 'low', 'close', 'volume' columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        VWAP series aligned to df.index.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    cumulative_tp_vol = tp_vol.groupby(df.index.date).cumsum()
    cumulative_vol = df["volume"].groupby(df.index.date).cumsum()

    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    return vwap.rename("vwap")


# ---------------------------------------------------------------------------
# Feature Matrix Builder — Main Entry Point
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compose all econometric & microstructure features into a single ML-ready feature matrix.

    PERMITTED feature families:
        - Log returns (price dynamics)
        - Hurst Exponent (regime persistence via DFA)
        - GARCH conditional volatility (volatility clustering)
        - Directional Changes (microstructure events)
        - Order Flow Imbalance (order flow pressure)
        - ATR (volatility for position sizing)
        - VWAP deviation (fair-value deviation)

    PROHIBITED: RSI, MACD, EMA crossovers, ADX, Bollinger Bands, Stochastic.

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

    # --- Microstructure Step 1: Directional Changes (major) ---
    # Must happen early — only needs close + volume, no other features required.
    dc_detector = DirectionalChangeDetector()
    dc_df = dc_detector.process_series(
        prices=df["close"], volumes=df["volume"]
    )
    # Forward-fill DC events and signal strength so non-DC rows have previous values
    features["dc_event"] = dc_df["event_type"].ffill()
    features["dc_signal_strength"] = dc_df["signal_strength"].ffill().fillna(0.0)
    features["dc_extremum_type"] = dc_df["extremum_type"].ffill()
    features["extremum_price"] = dc_df["extremum_price"].ffill()

    # --- Microstructure Step 2: Order Flow Imbalance ---
    features["ofi"] = calculate_order_flow_imbalance(df["close"], df["volume"])

    # --- Price-derived features ---
    features["log_return"] = compute_log_returns(df["close"])
    features["close"] = df["close"]
    features["hl_ratio"] = (df["high"] - df["low"]) / df["close"]  # Bar range
    features["oc_ratio"] = (df["close"] - df["open"]) / df["open"]  # Candle body

    # --- ATR (for SL/TP sizing — not a predictive feature per se) ---
    features["atr"] = compute_atr(df["high"], df["low"], df["close"])
    features["atr_pct"] = features["atr"] / df["close"]  # Normalised ATR

    # --- Volume & VWAP ---
    if df["volume"].sum() > 0:
        features["vwap"] = compute_vwap(df)
        features["vwap_dist"] = (df["close"] - features["vwap"]) / features["vwap"].replace(0, np.nan)
        features["vol_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()

    # --- Regime: Rolling Hurst Exponent ---
    features["hurst"] = (
        df["close"]
        .rolling(
            HURST_ROLLING_WINDOW,
            min_periods=min(HURST_MAX_LAG * 2, HURST_ROLLING_WINDOW),
        )
        .apply(lambda x: compute_hurst_exponent(pd.Series(x)), raw=False)
    )

    # --- is_trending flag (required by evaluate_primary_signals_batch) ---
    features["is_trending"] = features["hurst"] > HURST_TRENDING_THRESHOLD

    # --- Volatility: Rolling GARCH ---
    log_ret = features["log_return"]
    features["garch_vol"] = (
        log_ret
        .rolling(GARCH_ROLLING_WINDOW, min_periods=50)
        .apply(lambda x: fit_garch(pd.Series(x))[0], raw=False)
    )

    # --- Microstructure Step 3: Primary signal generation ---
    # Now all required columns (hurst, dc_event, ofi, is_trending) are present.
    features = evaluate_primary_signals_batch(features)

    # --- Step 4: Pullback signal detection (Phase 9.5) ---
    # Adds pullback_signal, pullback_direction, pullback_strength, primed columns
    features = evaluate_pullback_signals(features)

    # --- Volatility Z-Score (GARCH) ---
    vol_rolling_mean = features["garch_vol"].rolling(VOL_ZSCORE_WINDOW, min_periods=20).mean()
    vol_rolling_std = features["garch_vol"].rolling(VOL_ZSCORE_WINDOW, min_periods=20).std()
    features["vol_zscore"] = (features["garch_vol"] - vol_rolling_mean) / vol_rolling_std.replace(0, np.nan)

    # Drop warmup NaN rows (but preserve pullback signal rows by filling them first)
    # Pullback signals must be preserved even if other columns have NaN
    n_before = len(features)
    features.dropna(inplace=True)
    n_after = len(features)
    logger.info(
        f"Feature matrix built: {n_after} rows "
        f"({n_before - n_after} warmup rows dropped, {len(features.columns)} features)"
    )

    return features