# feature_engineering.py — QuantEdge MT5: Econometric & Microstructure Features ONLY
# RULES:
#   - All functions are PURE (no side effects, no global state)
#   - NO module-level imports of MT5 — only pandas/numpy/arch/hurst
#   - NO retail technical indicators (no RSI, MACD, ADX, Bollinger Bands, Stochastic, EMA)
#   - build_feature_matrix() is the single entry point for the ML pipeline
#
# PERMITTED FEATURE FAMILIES:
#   - Hurst Exponent (regime detection via DFA)
#   - GARCH(1,1) conditional volatility
#   - Directional Changes (DC) — microstructure event detection
#   - Order Flow Imbalance (OFI) — microstructure order flow proxy
#   - VWAP deviation
#   - Log returns and volatility z-scores

from typing import Tuple

from datetime import datetime as _dt
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
    DC_MIN_PRICE,
    DC_THETA,
    GARCH_P,
    GARCH_Q,
    GARCH_ROLLING_WINDOW,
    HURST_MAX_LAG,
    HURST_MEAN_REVERT_THRESHOLD,
    HURST_MIN_LAG,
    HURST_ROLLING_WINDOW,
    HURST_TRENDING_THRESHOLD,
    OFI_THRESHOLD,
    OFI_WINDOW,
    VOL_ZSCORE_THRESHOLD,
    VOL_ZSCORE_WINDOW,
    VWAP_DIST_THRESHOLD,
    FEATURE_WARMUP_BARS,
)
from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Directional Change Detector (from 2_microstructure_engine.py)
# ---------------------------------------------------------------------------

class DirectionalChangeDetector:
    """
    Detects Directional Changes (DC) in price series by identifying significant
    price movements from local extrema.

    Discards physical time — processes events based on price thresholds only.
    """

    def __init__(self, theta: float = DC_THETA):
        """
        Initialize DC detector.

        Args:
            theta: Threshold for significant price movement (default: DC_THETA).
        """
        self.theta = theta
        self.last_extremum_price = None
        self.last_extremum_type = None  # 'min' or 'max'

    def process_tick(self, price: float, volume: float, timestamp) -> dict:
        """
        Process a single tick and detect directional change events.

        Args:
            price: Current price
            volume: Tick volume
            timestamp: Timestamp of the tick

        Returns:
            Dict with keys: event_type, price, volume, timestamp, signal_strength.
            event_type is None (no event), 'upturn', or 'downturn'.
        """
        result = {
            'event_type': None,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'signal_strength': 0.0
        }

        # First tick — initialize extremum
        if self.last_extremum_price is None:
            self.last_extremum_price = price
            self.last_extremum_type = 'min'
            return result

        # Calculate price change from last extremum
        if self.last_extremum_price == 0:
            return result
        price_change = (price - self.last_extremum_price) / self.last_extremum_price

        # Check for upturn event (price rising from local minimum)
        if self.last_extremum_type == 'min' and price_change >= self.theta:
            result['event_type'] = 'upturn'
            result['signal_strength'] = min(price_change / self.theta, 2.0)
            self.last_extremum_price = price
            self.last_extremum_type = 'max'
            logger.debug(f"UPTURN EVENT: Price {price:.5f}, Change {price_change:.4f}")

        # Check for downturn event (price falling from local maximum)
        elif self.last_extremum_type == 'max' and price_change <= -self.theta:
            result['event_type'] = 'downturn'
            result['signal_strength'] = min(abs(price_change) / self.theta, 2.0)
            self.last_extremum_price = price
            self.last_extremum_type = 'min'
            logger.debug(f"DOWNTURN EVENT: Price {price:.5f}, Change {price_change:.4f}")

        # Update extremum if we found a new local extreme
        elif self.last_extremum_type == 'min' and price < self.last_extremum_price:
            self.last_extremum_price = price

        elif self.last_extremum_type == 'max' and price > self.last_extremum_price:
            self.last_extremum_price = price

        return result


# ---------------------------------------------------------------------------
# Order Flow Imbalance (from 2_microstructure_engine.py)
# ---------------------------------------------------------------------------

def calculate_order_flow_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = OFI_WINDOW,
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI) proxy using tick volume.

    OFI = (Volume of up-ticks - Volume of down-ticks) / rolling window.
    Positive OFI indicates aggressive buying pressure.
    Negative OFI indicates aggressive selling pressure.

    Args:
        prices: Series of prices.
        volumes: Series of corresponding volumes.
        window: Rolling window for OFI calculation.

    Returns:
        Series of OFI values (normalized between -1 and 1).
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")

    if len(prices) < 2:
        return pd.Series(index=prices.index, dtype=float).fillna(0.0)

    # Calculate price changes to determine tick direction
    price_changes = prices.diff()

    # Classify ticks: up-tick (>0), down-tick (<0), zero-tick (=0)
    up_tick = price_changes > 0
    down_tick = price_changes < 0
    zero_tick = price_changes == 0

    # Assign volume to tick direction
    up_volume = volumes.where(up_tick, 0)
    down_volume = volumes.where(down_tick, 0)

    # For zero ticks, split volume equally
    zero_volume_split = volumes.where(zero_tick, 0) / 2
    up_volume += zero_volume_split
    down_volume += zero_volume_split

    # Calculate rolling OFI: (up_volume - down_volume) / (up_volume + down_volume)
    net_volume = up_volume - down_volume
    total_volume = up_volume + down_volume
    ofi_raw = np.where(total_volume > 0, net_volume / total_volume, 0)

    # Apply rolling window smoothing
    ofi_series = pd.Series(ofi_raw, index=prices.index)
    ofi_smoothed = ofi_series.rolling(window=window, min_periods=1).mean()

    logger.debug(f"OFI calculated: mean={ofi_smoothed.mean():.4f}, std={ofi_smoothed.std():.4f}")
    return ofi_smoothed


def generate_primary_signals(
    data_df: pd.DataFrame,
    theta: float = DC_THETA,
    ofi_window: int = OFI_WINDOW,
    ofi_threshold: float = OFI_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate Primary Buy/Sell signals based on microstructure analysis.

    Signal Logic:
        - Primary_Buy: is_trending=True AND Upturn DC Event AND OFI significantly positive
        - Primary_Sell: is_trending=True AND Downturn DC Event AND OFI significantly negative

    Args:
        data_df: DataFrame with columns: ['close', 'volume', 'is_trending', ...]
        theta: Threshold for directional change events.
        ofi_window: Rolling window for OFI calculation.
        ofi_threshold: Minimum OFI magnitude for signal generation.

    Returns:
        DataFrame with signal columns: dc_event, dc_signal_strength, ofi,
        primary_buy, primary_sell, signal_strength.
    """
    if data_df is None or data_df.empty:
        logger.warning("Empty data provided to signal generation")
        return pd.DataFrame()

    df = data_df.copy()

    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    if 'is_trending' not in df.columns:
        logger.warning("is_trending column not found, assuming False for all rows")
        df['is_trending'] = False

    # Initialize signal columns
    df['dc_event'] = None
    df['dc_signal_strength'] = 0.0
    df['ofi'] = 0.0
    df['primary_buy'] = False
    df['primary_sell'] = False
    df['signal_strength'] = 0.0

    # Process data sequentially to detect directional changes
    dc_detector = DirectionalChangeDetector(theta=theta)
    event_list = []
    strength_list = []

    logger.info(f"Processing {len(df)} ticks for directional change detection...")

    for idx, row in df.iterrows():
        price = row['close']
        volume = row['volume'] if 'volume' in row and not pd.isna(row['volume']) else 1.0
        timestamp = idx if isinstance(idx, (pd.Timestamp, _dt)) else None

        dc_result = dc_detector.process_tick(price, volume, timestamp)
        event_list.append(dc_result['event_type'])
        strength_list.append(dc_result['signal_strength'])

    # Assign DC events and strengths to dataframe
    df['dc_event'] = event_list
    df['dc_signal_strength'] = strength_list

    # Calculate Order Flow Imbalance
    logger.info("Calculating Order Flow Imbalance...")
    df['ofi'] = calculate_order_flow_imbalance(df['close'], df['volume'], window=ofi_window)

    # Generate Primary Signals
    logger.info("Generating primary signals...")

    trending_condition = df['is_trending'] == True
    upturn_condition = df['dc_event'] == 'upturn'
    downturn_condition = df['dc_event'] == 'downturn'
    ofi_positive = df['ofi'] > ofi_threshold
    ofi_negative = df['ofi'] < -ofi_threshold

    # Primary Buy: Trending + Upturn DC + Positive OFI
    df['primary_buy'] = trending_condition & upturn_condition & ofi_positive

    # Primary Sell: Trending + Downturn DC + Negative OFI
    df['primary_sell'] = trending_condition & downturn_condition & ofi_negative

    # Calculate combined signal strength (0 to 1)
    dc_strength_norm = np.clip(df['dc_signal_strength'] / 2.0, 0, 1)
    ofi_magnitude = np.clip(np.abs(df['ofi']) / 1.0, 0, 1)
    regime_factor = df['is_trending'].astype(float)

    ofi_relevant = np.where(
        df['primary_buy'],
        np.clip(df['ofi'] / 1.0, 0, 1),
        np.where(
            df['primary_sell'],
            np.clip(np.abs(df['ofi']) / 1.0, 0, 1),
            0
        )
    )

    df['signal_strength'] = dc_strength_norm * ofi_magnitude * regime_factor

    buy_signals = df['primary_buy'].sum()
    sell_signals = df['primary_sell'].sum()
    total_events = (df['dc_event'] != None).sum()

    logger.info(
        f"Signal generation complete: {total_events} DC events | "
        f"{buy_signals} Primary Buy signals | {sell_signals} Primary Sell signals | "
        f"OFI mean: {df['ofi'].mean():.4f}, std: {df['ofi'].std():.4f}"
    )

    return df


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
        logger.warning("Insufficient returns for GARCH (need >= 50). Returning NaN.")
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

    # --- Price-derived ---
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

    # --- is_trending flag (required by generate_primary_signals) ---
    features["is_trending"] = features["hurst"] > HURST_TRENDING_THRESHOLD

    # --- Volatility: Rolling GARCH ---
    log_ret = features["log_return"]
    features["garch_vol"] = (
        log_ret
        .rolling(GARCH_ROLLING_WINDOW, min_periods=50)
        .apply(lambda x: fit_garch(pd.Series(x))[0], raw=False)
    )

    # --- Microstructure: Directional Changes & OFI ---
    dc_results = generate_primary_signals(df)
    if not dc_results.empty:
        features["dc_event"] = dc_results["dc_event"]
        features["dc_signal_strength"] = dc_results["dc_signal_strength"]
        features["ofi"] = dc_results["ofi"]
    else:
        features["dc_event"] = None
        features["dc_signal_strength"] = 0.0
        features["ofi"] = 0.0

    # --- Volatility Z-Score (GARCH) ---
    vol_rolling_mean = features["garch_vol"].rolling(VOL_ZSCORE_WINDOW, min_periods=20).mean()
    vol_rolling_std = features["garch_vol"].rolling(VOL_ZSCORE_WINDOW, min_periods=20).std()
    features["vol_zscore"] = (features["garch_vol"] - vol_rolling_mean) / vol_rolling_std.replace(0, np.nan)

    # Drop warmup NaN rows
    n_before = len(features)
    features.dropna(inplace=True)
    n_after = len(features)
    logger.info(
        f"Feature matrix built: {n_after} rows "
        f"({n_before - n_after} warmup rows dropped, {len(features.columns)} features)"
    )

    return features