# 2_microstructure_engine.py — QuantEdge MT5: Market Microstructure Engine
# Discarding physical time, processing tick-by-tick events for signal generation
# RULE: This module processes data from 1_data_and_regime_engine.py and generates trading signals
# based on directional changes, order flow imbalance, and regime conditions.

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from logger_config import setup_logger

logger = setup_logger(__name__)


class DirectionalChangeDetector:
    """
    Detects Directional Changes (DC) in price series by identifying significant
    price movements from local extrema.

    Discards physical time - processes events based on price thresholds only.
    """

    def __init__(self, theta: float = 0.001):
        """
        Initialize DC detector.

        Args:
            theta: Threshold for significant price movement (0.001 = 0.1%)
        """
        self.theta = theta
        self.last_extremum_price = None
        self.last_extremum_type = None  # 'min' or 'max'
        self.events = []  # Store detected events

    def process_tick(self, price: float, volume: float, timestamp) -> dict:
        """
        Process a single tick and detect directional change events.

        Args:
            price: Current price
            volume: Tick volume
            timestamp: Timestamp of the tick

        Returns:
            Dictionary with event information:
            {
                'event_type': None/'upturn'/'downturn',
                'price': price,
                'volume': volume,
                'timestamp': timestamp,
                'signal_strength': float  # Normalized strength of the event
            }
        """
        result = {
            'event_type': None,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'signal_strength': 0.0
        }

        # First tick - initialize extremum
        if self.last_extremum_price is None:
            self.last_extremum_price = price
            self.last_extremum_type = 'min'  # Start by looking for a minimum
            return result

        # Calculate price change from last extremum
        price_change = (price - self.last_extremum_price) / self.last_extremum_price

        # Check for upturn event (price rising from local minimum)
        if self.last_extremum_type == 'min' and price_change >= self.theta:
            # Confirmed upturn: price rose by theta from local minimum
            result['event_type'] = 'upturn'
            result['signal_strength'] = min(price_change / self.theta, 2.0)  # Cap at 2x theta

            # Update extremum to this maximum for next downturn detection
            self.last_extremum_price = price
            self.last_extremum_type = 'max'

            logger.debug(f"UPTURN EVENT: Price {price:.5f}, Change {price_change:.4f}")

        # Check for downturn event (price falling from local maximum)
        elif self.last_extremum_type == 'max' and price_change <= -self.theta:
            # Confirmed downturn: price fell by theta from local maximum
            result['event_type'] = 'downturn'
            result['signal_strength'] = min(abs(price_change) / self.theta, 2.0)  # Cap at 2x theta

            # Update extremum to this minimum for next upturn detection
            self.last_extremum_price = price
            self.last_extremum_type = 'min'

            logger.debug(f"DOWNTURN EVENT: Price {price:.5f}, Change {price_change:.4f}")

        # Update extremum if we found a new local extreme
        elif self.last_extremum_type == 'min' and price < self.last_extremum_price:
            # New lower minimum found
            self.last_extremum_price = price

        elif self.last_extremum_type == 'max' and price > self.last_extremum_price:
            # New higher maximum found
            self.last_extremum_price = price

        return result


def calculate_order_flow_imbalance(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI) proxy using tick volume.

    OFI = (Volume of up-ticks - Volume of down-ticks) / rolling window
    Positive OFI indicates aggressive buying pressure
    Negative OFI indicates aggressive selling pressure

    Args:
        prices: Series of prices
        volumes: Series of corresponding volumes
        window: Rolling window for OFI calculation

    Returns:
        Series of OFI values (normalized between -1 and 1)
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

    # For zero ticks, split volume equally (could be improved with more sophisticated methods)
    zero_volume_split = volumes.where(zero_tick, 0) / 2
    up_volume += zero_volume_split
    down_volume += zero_volume_split

    # Calculate rolling OFI: (up_volume - down_volume) / (up_volume + down_volume)
    # This gives a value between -1 and 1
    net_volume = up_volume - down_volume
    total_volume = up_volume + down_volume

    # Avoid division by zero
    ofi_raw = np.where(total_volume > 0, net_volume / total_volume, 0)

    # Apply rolling window smoothing
    ofi_series = pd.Series(ofi_raw, index=prices.index)
    ofi_smoothed = ofi_series.rolling(window=window, min_periods=1).mean()

    logger.debug(f"OFI calculated: mean={ofi_smoothed.mean():.4f}, std={ofi_smoothed.std():.4f}")

    return ofi_smoothed


def generate_primary_signals(
    data_df: pd.DataFrame,
    theta: float = 0.001,
    ofi_window: int = 20,
    ofi_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Generate Primary Buy/Sell signals based on microstructure analysis.

    Signal Logic:
    - Primary_Buy: is_trending=True AND Upturn DC Event AND OFI significantly positive
    - Primary_Sell: is_trending=True AND Downturn DC Event AND OFI significantly negative

    Args:
        data_df: DataFrame from 1_data_and_regime_engine.py with columns:
                ['close', 'volume', 'is_trending', ...]
        theta: Threshold for directional change events (default: 0.001)
        ofi_window: Rolling window for OFI calculation (default: 20)
        ofi_threshold: Minimum OFI magnitude for signal generation (default: 0.1)

    Returns:
        DataFrame with additional columns:
        - dc_event: Directional change event type ('upturn', 'downturn', None)
        - dc_signal_strength: Strength of the DC event
        - ofi: Order flow imbalance value
        - primary_buy: Boolean signal for primary buy
        - primary_sell: Boolean signal for primary sell
        - signal_strength: Combined signal strength (0 to 1)
    """
    if data_df is None or data_df.empty:
        logger.warning("Empty data provided to signal generation")
        return pd.DataFrame()

    # Make a copy to avoid modifying original data
    df = data_df.copy()

    # Ensure required columns exist
    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Ensure is_trending column exists (from regime engine)
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
    # Note: We process row by row to maintain state in DC detector
    dc_detector = DirectionalChangeDetector(theta=theta)

    event_list = []
    strength_list = []

    logger.info(f"Processing {len(df)} ticks for directional change detection...")

    for idx, row in df.iterrows():
        price = row['close']
        volume = row['volume'] if 'volume' in row and not pd.isna(row['volume']) else 1.0
        timestamp = idx if isinstance(idx, (pd.Timestamp, datetime)) else None

        # Process tick for directional change detection
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

    # Condition 1: Must be in trending regime
    trending_condition = df['is_trending'] == True

    # Condition 2: Directional change event occurred
    upturn_condition = df['dc_event'] == 'upturn'
    downturn_condition = df['dc_event'] == 'downturn'

    # Condition 3: OFI significantly positive/negative
    ofi_positive = df['ofi'] > ofi_threshold
    ofi_negative = df['ofi'] < -ofi_threshold

    # Primary Buy: Trending + Upturn DC + Positive OFI
    df['primary_buy'] = trending_condition & upturn_condition & ofi_positive

    # Primary Sell: Trending + Downturn DC + Negative OFI
    df['primary_sell'] = trending_condition & downturn_condition & ofi_negative

    # Calculate combined signal strength (0 to 1)
    # Strength based on: DC strength * OFI magnitude * regime confidence
    dc_strength_norm = np.clip(df['dc_signal_strength'] / 2.0, 0, 1)  # Normalize 0-2 to 0-1
    ofi_magnitude = np.clip(np.abs(df['ofi']) / 1.0, 0, 1)  # Normalize OFI to 0-1 (capped at 1.0)
    regime_factor = df['is_trending'].astype(float)  # 1.0 if trending, 0.0 if not

    # For buy signals: use OFI positive, for sell signals: use OFI negative magnitude
    ofi_relevant = np.where(
        df['primary_buy'],
        np.clip(df['ofi'] / 1.0, 0, 1),  # Positive OFI for buys
        np.where(
            df['primary_sell'],
            np.clip(np.abs(df['ofi']) / 1.0, 0, 1),  # Absolute OFI for sells
            0  # No signal
        )
    )

    df['signal_strength'] = dc_strength_norm * ofi_magnitude * regime_factor

    # Log signal statistics
    buy_signals = df['primary_buy'].sum()
    sell_signals = df['primary_sell'].sum()
    total_events = (df['dc_event'] != None).sum()

    logger.info(
        f"Signal generation complete: {total_events} DC events | "
        f"{buy_signals} Primary Buy signals | {sell_signals} Primary Sell signals | "
        f"OFI mean: {df['ofi'].mean():.4f}, std: {df['ofi'].std():.4f}"
    )

    return df


def process_microstructure_signals(
    regime_data: pd.DataFrame,
    theta: float = 0.001,
    ofi_window: int = 20,
    ofi_threshold: float = 0.1
) -> Tuple[pd.DataFrame, dict]:
    """
    Main processing function that takes regime data and returns microstructure signals.

    Args:
        regime_data: DataFrame from 1_data_and_regime_engine.py
        theta: Threshold for directional change events
        ofi_window: Rolling window for OFI calculation
        ofi_threshold: Minimum OFI magnitude for signal generation

    Returns:
        Tuple of (signals_dataframe, statistics_dict)
    """
    if regime_data is None or regime_data.empty:
        logger.warning("No regime data provided for microstructure processing")
        return pd.DataFrame(), {}

    logger.info("Starting microstructure signal processing...")

    # Generate signals
    signals_df = generate_primary_signals(
        regime_data,
        theta=theta,
        ofi_window=ofi_window,
        ofi_threshold=ofi_threshold
    )

    # Calculate statistics
    stats = {
        'total_ticks': len(signals_df),
        'trending_periods': signals_df['is_trending'].sum(),
        'dc_events_total': (signals_df['dc_event'] != None).sum(),
        'dc_upturn_events': (signals_df['dc_event'] == 'upturn').sum(),
        'dc_downturn_events': (signals_df['dc_event'] == 'downturn').sum(),
        'primary_buy_signals': signals_df['primary_buy'].sum(),
        'primary_sell_signals': signals_df['primary_sell'].sum(),
        'avg_signal_strength': signals_df['signal_strength'].mean(),
        'max_signal_strength': signals_df['signal_strength'].max(),
        'ofi_mean': signals_df['ofi'].mean(),
        'ofi_std': signals_df['ofi'].std(),
        'ofi_sharpe': signals_df['ofi'].mean() / signals_df['ofi'].std() if signals_df['ofi'].std() > 0 else 0
    }

    logger.info(f"Microstructure processing complete: {stats}")

    return signals_df, stats


# ---------------------------------------------------------------------------
# Example Usage and Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Example usage and testing of the microstructure engine."""
    import os
    from dotenv import load_dotenv
    from 1_data_and_regime_engine import (
        safe_mt5_initialize,
        safe_mt5_shutdown,
        fetch_and_process_regime_data
    )

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
                # Get regime data from module 1
                logger.info("Fetching regime data from 1_data_and_regime_engine...")
                regime_data = fetch_and_process_regime_data("EURUSD", lookback_bars=500)

                if regime_data is not None and not regime_data.empty:
                    print(f"\nRegime Data Shape: {regime_data.shape}")
                    print(f"Date Range: {regime_data.index[0]} to {regime_data.index[-1]}")

                    # Process microstructure signals
                    signals_df, stats = process_microstructure_signals(regime_data)

                    if not signals_df.empty:
                        print(f"\nMicrostructure Signals Shape: {signals_df.shape}")
                        print(f"\nStatistics:")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")

                        # Show recent signals
                        recent_signals = signals_df[
                            signals_df['primary_buy'] | signals_df['primary_sell']
                        ].tail(10)

                        if not recent_signals.empty:
                            print(f"\nRecent Primary Signals:")
                            for idx, row in recent_signals.iterrows():
                                signal_type = "BUY" if row['primary_buy'] else "SELL"
                                print(f"  {idx}: {signal_type} | "
                                      f"Price: {row['close']:.5f} | "
                                      f"Hurst: {row.get('hurst', 0):.3f} | "
                                      f"OFI: {row['ofi']:.3f} | "
                                      f"DC: {row['dc_event']} | "
                                      f"Strength: {row['signal_strength']:.3f}")
                        else:
                            print("\nNo primary signals generated in the recent data.")

                        # Show DC events
                        dc_events = signals_df[signals_df['dc_event'] != None]
                        if not dc_events.empty:
                            print(f"\nDirectional Change Events Summary:")
                            print(f"  Total DC Events: {len(dc_events)}")
                            print(f"  Upturn Events: {len(dc_events[dc_events['dc_event'] == 'upturn'])}")
                            print(f"  Downturn Events: {len(dc_events[dc_events['dc_event'] == 'downturn'])}")

                    else:
                        print("Failed to generate microstructure signals")
                else:
                    print("Failed to fetch regime data")

            finally:
                safe_mt5_shutdown()
        else:
            logger.error("Failed to initialize MT5 connection")