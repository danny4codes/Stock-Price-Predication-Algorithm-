from backtester import load_historical_data
from feature_engineering import build_feature_matrix, evaluate_pullback_signals
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check what we have
print(f"Features has dc_event: {'dc_event' in features.columns}")
print(f"Features has ofi: {'ofi' in features.columns}")
print(f"Features has hurst: {'hurst' in features.columns}")
print(f"Features has regime: {'regime' in features.columns}")

if 'dc_event' in features.columns:
    print(f"\nDC events in features: {features['dc_event'].notna().sum()}")
    print(f"DC event values: {features['dc_event'].dropna().unique()}")

# Run pullback on features
result = evaluate_pullback_signals(features.copy())
print(f"\nPullback signals in result: {result['pullback_signal'].sum()}")
print(f"Primed in result: {result['primed'].sum()}")