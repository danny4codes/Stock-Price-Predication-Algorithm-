# 2_microstructure_engine.py — QuantEdge MT5: Market Microstructure Processing Engine
#
# This module processes raw OHLCV tick data into microstructure features:
#   1. Directional Changes (DC) — Intrinsic Time event detection
#   2. Order Flow Imbalance (OFI) — Tick-volume order flow proxy
#   3. Primary Signal Logic — Boolean buy/sell signal generation
#
# Architecture:
#   - All functions are PURE (no side effects, no global state)
#   - DirectionalChangeDetector tracks state internally (per-instance)
#   - NO MetaTrader5 imports — this module only processes data frames
#   - NO retail indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
#
# Input:  OHLCV DataFrames from data_ingestion.py
# Output: DC events, OFI series, boolean primary signals
# ---------------------------------------------------------------------------

from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    DC_THETA,
    DC_MIN_PRICE,
    OFI_WINDOW,
    OFI_THRESHOLD,
    HURST_TRENDING_THRESHOLD,
    HURST_MEAN_REVERT_THRESHOLD,
)
from logger_config import setup_logger

logger = setup_logger(__name__)


# ===========================================================================
# 1. DIRECTIONAL CHANGES (DC) — Intrinsic Time Framework
# ===========================================================================

class DirectionalChangeDetector:
    """
    Detects Directional Changes (DC) in a price series using the Intrinsic
    Time framework. Physical time constraints are discarded — events are
    defined purely by cumulative price movement relative to a threshold.

    Mechanics:
        - An 'Upturn Event' fires when price rises by ≥ theta from the
          last recorded local minimum.
        - A 'Downturn Event' fires when price falls by ≥ theta from the
          last recorded local maximum.
        - Between events, the detector tracks evolving local extrema.

    This is the standard DC detector per the quantitative microstructure
    literature (Golub et al., 2018; Muni Toke, 2011).
    """

    def __init__(self, theta: float = DC_THETA, min_price: float = DC_MIN_PRICE):
        """
        Initialize the Directional Change detector.

        Args:
            theta:     Minimum fractional price change to trigger an event.
                       e.g. theta=0.001 means a 0.1% move is required.
            min_price: Minimum price threshold to skip detection (avoids
                       division-by-zero on penny stocks or zero-price ticks).
        """
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if min_price <= 0:
            raise ValueError(f"min_price must be positive, got {min_price}")

        self.theta = theta
        self.min_price = min_price
        self.last_extremum_price: Optional[float] = None
        self.last_extremum_type: Optional[str] = None  # 'min' or 'max'
        self._event_counter: int = 0

    def reset(self) -> None:
        """Reset detector state to process a new series from scratch."""
        self.last_extremum_price = None
        self.last_extremum_type = None
        self._event_counter = 0

    def process_tick(self, price: float, volume: float = 0.0,
                     timestamp: Optional[datetime] = None) -> dict:
        """
        Process a single tick through the DC detector.

        Args:
            price:     Current trade price.
            volume:    Tick volume (not used in DC logic but preserved for
                       downstream OFI integration).
            timestamp: Optional timestamp for the tick.

        Returns:
            dict with keys:
                event_type:      None | 'upturn' | 'downturn'
                price:           Current price
                volume:          Current volume
                timestamp:       Current timestamp
                signal_strength: Normalised strength ∈ [0, 2]
                                  (ratio of actual move to theta, capped at 2)
                extremum_price:  The last recorded extremum price
                extremum_type:   'min' | 'max' | None
        """
        result = {
            'event_type': None,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'signal_strength': 0.0,
            'extremum_price': self.last_extremum_price,
            'extremum_type': self.last_extremum_type,
        }

        # Guard: skip invalid prices
        if price <= self.min_price or price <= 0:
            logger.debug(f"DC: skipping invalid price {price}")
            return result

        # First tick — initialise extremum as a local minimum
        if self.last_extremum_price is None:
            self.last_extremum_price = price
            self.last_extremum_type = 'min'
            result['extremum_price'] = price
            result['extremum_type'] = 'min'
            return result

        # Guard: avoid division by zero on degenerate extremum
        if self.last_extremum_price == 0:
            return result

        # Fractional price change from last extremum
        price_change = (price - self.last_extremum_price) / self.last_extremum_price

        # --- Upturn Event: price rose by ≥ theta from a local minimum ---
        if self.last_extremum_type == 'min' and price_change >= self.theta:
            result['event_type'] = 'upturn'
            result['signal_strength'] = min(price_change / self.theta, 2.0)
            self.last_extremum_price = price
            self.last_extremum_type = 'max'
            self._event_counter += 1
            logger.debug(
                f"DC UPTURN #{self._event_counter}: "
                f"price={price:.5f}, Δ={price_change:+.4f} ({price_change/self.theta:.1f}×θ)"
            )

        # --- Downturn Event: price fell by ≥ theta from a local maximum ---
        elif self.last_extremum_type == 'max' and price_change <= -self.theta:
            result['event_type'] = 'downturn'
            result['signal_strength'] = min(abs(price_change) / self.theta, 2.0)
            self.last_extremum_price = price
            self.last_extremum_type = 'min'
            self._event_counter += 1
            logger.debug(
                f"DC DOWNTURN #{self._event_counter}: "
                f"price={price:.5f}, Δ={price_change:+.4f} ({abs(price_change)/self.theta:.1f}×θ)"
            )

        # --- Track evolving local extrema (no event fired) ---
        elif self.last_extremum_type == 'min' and price < self.last_extremum_price:
            self.last_extremum_price = price

        elif self.last_extremum_type == 'max' and price > self.last_extremum_price:
            self.last_extremum_price = price

        result['extremum_price'] = self.last_extremum_price
        result['extremum_type'] = self.last_extremum_type
        return result

    def process_series(self, prices: pd.Series, volumes: Optional[pd.Series] = None,
                       timestamps: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Process an entire price series, returning a DataFrame of DC events.

        Args:
            prices:    Series of close prices with a DatetimeIndex.
            volumes:   Optional series of tick volumes (same index).
            timestamps: Optional explicit timestamps (overrides index).

        Returns:
            DataFrame with columns:
                ['event_type', 'price', 'volume', 'timestamp',
                 'signal_strength', 'extremum_price', 'extremum_type']
        """
        self.reset()
        n = len(prices)
        if n == 0:
            return pd.DataFrame(columns=[
                'event_type', 'price', 'volume', 'timestamp',
                'signal_strength', 'extremum_price', 'extremum_type'
            ])

        events = []
        vols = volumes.values if volumes is not None else np.zeros(n)
        ts = timestamps if timestamps is not None else prices.index

        for i in range(n):
            result = self.process_tick(
                price=float(prices.iloc[i]),
                volume=float(vols[i]) if vols[i] == vols[i] else 0.0,  # NaN check
                timestamp=ts[i] if ts is not None else None,
            )
            events.append(result)

        return pd.DataFrame(events, index=prices.index)

    @property
    def event_count(self) -> int:
        """Total number of DC events detected since last reset."""
        return self._event_counter


# ===========================================================================
# 2. ORDER FLOW IMBALANCE (OFI) — Tick-Volume Proxy
# ===========================================================================

def calculate_order_flow_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = OFI_WINDOW,
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI) — a proxy for aggressive
    institutional buying versus selling pressure.

    OFI is computed per tick as:
        OFI_t = (V_up - V_down) / (V_up + V_down)

    where V_up is volume on up-ticks (close > previous close) and V_down
    is volume on down-ticks. The result is smoothed with a rolling mean.

    Interpretation:
        OFI > 0  → Net buying pressure (institutions accumulating)
        OFI < 0  → Net selling pressure (institutions distributing)
        OFI ≈ 0  → Balanced flow / noise

    Args:
        prices: OHLCV close prices.
        volumes: Corresponding tick volumes.
        window: Rolling window size for smoothing (default: OFI_WINDOW=20).

    Returns:
        Smoothed OFI series, values ∈ [-1, 1]. Same index as input.
    """
    if prices is None or volumes is None:
        raise ValueError("prices and volumes must not be None")

    if len(prices) != len(volumes):
        raise ValueError(
            f"prices ({len(prices)}) and volumes ({len(volumes)}) must have same length"
        )

    if len(prices) < 2:
        logger.warning("OFI: fewer than 2 data points, returning zeros.")
        return pd.Series(0.0, index=prices.index, dtype=float)

    # Price direction classification
    price_diff = prices.diff()

    up_tick_mask   = price_diff > 0
    down_tick_mask = price_diff < 0
    zero_tick_mask = price_diff == 0

    # Allocate volume to direction
    up_volume   = volumes.where(up_tick_mask, 0.0)
    down_volume = volumes.where(down_tick_mask, 0.0)

    # Zero ticks: split volume equally between up and down
    zero_volume = volumes.where(zero_tick_mask, 0.0) / 2.0
    up_volume   = up_volume + zero_volume
    down_volume = down_volume + zero_volume

    # OFI = (up - down) / (up + down), guard against ÷0
    net_volume     = up_volume - down_volume
    total_volume   = up_volume + down_volume
    ofi_raw = np.where(total_volume > 0, net_volume / total_volume, 0.0)

    # Rolling smoothing
    ofi_series = pd.Series(ofi_raw, index=prices.index, dtype=float)
    ofi_smoothed = ofi_series.rolling(window=window, min_periods=1).mean()

    logger.debug(
        f"OFI computed: mean={ofi_smoothed.mean():.4f}, "
        f"std={ofi_smoothed.std():.4f}, "
        f"range=[{ofi_smoothed.min():.4f}, {ofi_smoothed.max():.4f}]"
    )
    return ofi_smoothed


# ===========================================================================
# 3. PRIMARY SIGNAL LOGIC — Boolean Event Generator
# ===========================================================================

class PrimarySignalResult:
    """
    Container for a single primary signal evaluation result.

    Attributes:
        buy:           True if all Primary_Buy conditions are met.
        sell:          True if all Primary_Sell conditions are met.
        buy_strength:  Signal strength for buy (0.0 to 1.0).
        sell_strength: Signal strength for sell (0.0 to 1.0).
        regime:        Current regime: 'trending', 'mean_reverting', or 'random'.
        reason:        Human-readable explanation of why signal was/wasn't generated.
    """
    __slots__ = ('buy', 'sell', 'buy_strength', 'sell_strength', 'regime', 'reason')

    def __init__(self, buy: bool = False, sell: bool = False,
                 buy_strength: float = 0.0, sell_strength: float = 0.0,
                 regime: str = 'random', reason: str = ''):
        self.buy = buy
        self.sell = sell
        self.buy_strength = buy_strength
        self.sell_strength = sell_strength
        self.regime = regime
        self.reason = reason

    def __repr__(self) -> str:
        parts = [f"buy={self.buy}", f"sell={self.sell}"]
        if self.buy:
            parts.append(f"buy_str={self.buy_strength:.3f}")
        if self.sell:
            parts.append(f"sell_str={self.sell_strength:.3f}")
        parts.append(f"regime={self.regime}")
        if self.reason:
            parts.append(f"reason='{self.reason}'")
        return f"PrimarySignalResult({', '.join(parts)})"

    def to_dict(self) -> dict:
        return {
            'buy': self.buy,
            'sell': self.sell,
            'buy_strength': self.buy_strength,
            'sell_strength': self.sell_strength,
            'regime': self.regime,
            'reason': self.reason,
        }


def classify_regime(hurst: float) -> str:
    """
    Classify market regime from the Hurst exponent.

    Args:
        hurst: Hurst exponent value (0.0 to 1.0).

    Returns:
        'trending'       if H > 0.55
        'mean_reverting' if H < 0.45
        'random'         otherwise
    """
    if hurst > HURST_TRENDING_THRESHOLD:
        return 'trending'
    elif hurst < HURST_MEAN_REVERT_THRESHOLD:
        return 'mean_reverting'
    return 'random'


def evaluate_primary_signal(
    hurst: float,
    dc_event: Optional[str],
    ofi: float,
    ofi_threshold: float = OFI_THRESHOLD,
) -> PrimarySignalResult:
    """
    Evaluate whether a primary buy or sell signal should be generated.

    Signal Logic (STRICT — all conditions must be True):

        Primary_Buy:
            1. Hurst Exponent > 0.55  (trending regime confirmed)
            2. DC Event == 'upturn'   (price surged theta from local min)
            3. OFI > +ofi_threshold   (significant net buying pressure)

        Primary_Sell (exact inverse):
            1. Hurst Exponent > 0.55  (trending regime confirmed)
            2. DC Event == 'downturn' (price dropped theta from local max)
            3. OFI < -ofi_threshold  (significant net selling pressure)

    If regime is 'mean_reverting' or 'random', no primary signals are
    generated — the system defers to mean-reversion strategies instead.

    Args:
        hurst:          Current Hurst exponent value.
        dc_event:       Most recent DC event type: None, 'upturn', or 'downturn'.
        ofi:            Current Order Flow Imbalance value.
        ofi_threshold:  Minimum |OFI| required to confirm signal (default: 0.1).

    Returns:
        PrimarySignalResult with buy/sell booleans, strengths, regime, and reason.
    """
    result = PrimarySignalResult(regime=classify_regime(hurst))

    # ------------------------------------------------------------------
    # Guard: Must be in trending regime for primary signals
    # ------------------------------------------------------------------
    if result.regime != 'trending':
        result.reason = (
            f"No primary signal: regime={result.regime} "
            f"(H={hurst:.3f}, need H>{HURST_TRENDING_THRESHOLD} for trending)"
        )
        logger.debug(result.reason)
        return result

    # ------------------------------------------------------------------
    # Guard: Must have a valid DC event
    # ------------------------------------------------------------------
    if dc_event is None:
        result.reason = "No primary signal: no DC event detected"
        logger.debug(result.reason)
        return result

    # ------------------------------------------------------------------
    # Primary Buy: Upturn DC + Positive OFI in trending regime
    # ------------------------------------------------------------------
    if dc_event == 'upturn' and ofi > ofi_threshold:
        result.buy = True
        # Strength: product of Hurst conviction and OFI magnitude
        hurst_conviction = min((hurst - 0.5) * 2, 1.0)  # H: 0.55-1.0 → 0-1
        ofi_conviction   = min(abs(ofi), 1.0)
        result.buy_strength = round(
            (hurst_conviction * 0.5 + ofi_conviction * 0.5), 3
        )
        result.reason = (
            f"Primary_BUY: H={hurst:.3f} (trending) + "
            f"DC=upturn + OFI={ofi:.4f} > {ofi_threshold}"
        )
        logger.info(f"  ✅ {result.reason} | strength={result.buy_strength:.3f}")
        return result

    # ------------------------------------------------------------------
    # Primary Sell (exact inverse): Downturn DC + Negative OFI in trending
    # ------------------------------------------------------------------
    if dc_event == 'downturn' and ofi < -ofi_threshold:
        result.sell = True
        hurst_conviction = min((hurst - 0.5) * 2, 1.0)
        ofi_conviction   = min(abs(ofi), 1.0)
        result.sell_strength = round(
            (hurst_conviction * 0.5 + ofi_conviction * 0.5), 3
        )
        result.reason = (
            f"Primary_SELL: H={hurst:.3f} (trending) + "
            f"DC=downturn + OFI={ofi:.4f} < {-ofi_threshold}"
        )
        logger.info(f"  ✅ {result.reason} | strength={result.sell_strength:.3f}")
        return result

    # ------------------------------------------------------------------
    # Conditions not met — no primary signal
    # ------------------------------------------------------------------
    if dc_event == 'upturn':
        result.reason = (
            f"No primary signal: upturn detected but OFI={ofi:.4f} "
            f"below threshold {ofi_threshold}"
        )
    elif dc_event == 'downturn':
        result.reason = (
            f"No primary signal: downturn detected but OFI={ofi:.4f} "
            f"above threshold {-ofi_threshold}"
        )
    logger.debug(result.reason)
    return result


def evaluate_primary_signals_batch(
    features: pd.DataFrame,
    ofi_threshold: float = OFI_THRESHOLD,
) -> pd.DataFrame:
    """
    Vectorised batch evaluation of primary signals across a feature DataFrame.

    Adds columns: primary_buy, primary_sell, signal_strength, regime.

    Args:
        features:   DataFrame with columns: 'hurst', 'dc_event', 'ofi', 'is_trending'.
        ofi_threshold: Minimum |OFI| to confirm buy/sell.

    Returns:
        The input DataFrame with signal columns appended.
    """
    if features.empty:
        logger.warning("evaluate_primary_signals_batch: empty DataFrame")
        return features

    df = features.copy()

    # Ensure required columns exist
    required = {'hurst', 'dc_event', 'ofi'}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing required columns for signal eval: {missing}")
        return df

    # Default is_trending if absent
    if 'is_trending' not in df.columns:
        df['is_trending'] = df['hurst'] > HURST_TRENDING_THRESHOLD

    # --- Regime classification ---
    df['regime'] = df['hurst'].apply(classify_regime)

    # --- Boolean signal generation ---
    trending  = df['is_trending'] == True
    upturn    = df['dc_event'] == 'upturn'
    downturn  = df['dc_event'] == 'downturn'
    ofi_pos   = df['ofi'] > ofi_threshold
    ofi_neg   = df['ofi'] < -ofi_threshold

    df['primary_buy']  = trending & upturn & ofi_pos
    df['primary_sell'] = trending & downturn & ofi_neg

    # --- Signal strength ---
    hurst_conv = np.clip((df['hurst'] - 0.5) * 2, 0, 1)
    ofi_magn   = np.clip(np.abs(df['ofi']), 0, 1)

    df['signal_strength'] = np.where(
        df['primary_buy'] | df['primary_sell'],
        np.round(hurst_conv * 0.5 + ofi_magn * 0.5, 3),
        0.0
    )

    n_buy  = df['primary_buy'].sum()
    n_sell = df['primary_sell'].sum()
    logger.info(
        f"Batch signal eval: {n_buy} primary buys, {n_sell} primary sells "
        f"out of {len(df)} rows"
    )

    return df


# ===========================================================================
# Convenience: Full Pipeline — DC + OFI + Primary Signals from OHLCV
# ===========================================================================

def run_microstructure_pipeline(
    df: pd.DataFrame,
    theta: float = DC_THETA,
    ofi_window: int = OFI_WINDOW,
    ofi_threshold: float = OFI_THRESHOLD,
    hurst_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run the complete microstructure pipeline on raw OHLCV data:
        1. Detect Directional Change events (per-tick)
        2. Calculate Order Flow Imbalance (rolling)
        3. Generate primary boolean signals

    Args:
        df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex.
        theta: DC threshold (fractional price move).
        ofi_window: Rolling window for OFI smoothing.
        ofi_threshold: Minimum |OFI| for signal confirmation.
        hurst_value: Optional fixed Hurst exponent to apply to all rows.
                     If None, 'is_trending' must already be in the DataFrame.

    Returns:
        DataFrame with original data plus:
            ['dc_event', 'dc_signal_strength', 'ofi',
             'primary_buy', 'primary_sell', 'signal_strength', 'regime']
    """
    if df is None or df.empty:
        logger.warning("run_microstructure_pipeline: empty input")
        return pd.DataFrame()

    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return pd.DataFrame()

    result = df.copy()

    # ---- Hurst / is_trending ----
    if hurst_value is not None:
        result['hurst'] = hurst_value
        result['is_trending'] = hurst_value > HURST_TRENDING_THRESHOLD
    elif 'hurst' not in result.columns:
        logger.warning("No 'hurst' column or hurst_value provided — assuming non-trending.")
        result['hurst'] = 0.5
        result['is_trending'] = False

    # ---- Step 1: Directional Changes ----
    detector = DirectionalChangeDetector(theta=theta)
    dc_df = detector.process_series(
        prices=result['close'],
        volumes=result['volume'],
    )
    result['dc_event']           = dc_df['event_type']
    result['dc_signal_strength'] = dc_df['signal_strength']

    # ---- Step 2: OFI ----
    result['ofi'] = calculate_order_flow_imbalance(
        result['close'], result['volume'], window=ofi_window
    )

    # ---- Step 3: Primary Signals ----
    result = evaluate_primary_signals_batch(result, ofi_threshold=ofi_threshold)

    return result