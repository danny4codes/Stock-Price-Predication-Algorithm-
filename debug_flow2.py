from pathlib import Path
from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check intermediate state - before dropna
# Let me trace through manually

from microstructure_engine import (
    DirectionalChangeDetector, calculate_order_flow_imbalance,
    evaluate_primary_signals_batch, evaluate_pullback_signals
)
from config import DC_THETA, DC_MIN_PRICE, OFI_WINDOW, OFI_THRESHOLD

# Step by step
from config import GARCH_ROLLING_WINDOW, VOL_ZSCORE_WINDOW
import numpy as np

# Load data again for step-by-step
df2 = load_historical_data(symbol)
df2.sort_index(inplace=True)

# Build features step by step
features2 = df2.copy()

# Add log returns
features2['log_return'] = np.log(features2['close'] / features2['close'].shift(1))
features2.dropna(inplace=True)

# Hurst
from microstructure_engine import compute_hurst_exponent
features2['hurst'] = compute_hurst_exponent(features2['close'])

# DC events
detector = DirectionalChangeDetector(theta=DC_THETA, min_price=DC_MIN_PRICE)
features2['dc_event'] = detector.process_series(features2['close'])

# OFI
features2['ofi'] = calculate_order_flow_imbalance(
    features2['close'], features2['volume'], window=OFI_WINDOW
)

# Regime
from microstructure_engine import classify_regime
features2['regime'] = features2['hurst'].apply(classify_regime)

print(f"Before dropna: {len(features2)} rows")
print(f"Pullback signals before dropna: {features2.get('pullback_signal', pd.Series()).sum()}")

# Check pullback after running the function
features3 = evaluate_pullback_signals(features2.copy())
print(f"Pullback signals after evaluate_pullback_signals: {features3['pullback_signal'].sum()}")

# Check primed
print(f"Primed: {features3['primed'].sum()}")