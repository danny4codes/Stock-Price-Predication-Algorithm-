from pathlib import Path
from feature_engineering import build_feature_matrix
from backtester import _generate_signal_from_row, load_historical_data, apply_signal_filters
import pandas as pd

# Load data using same file as earlier debug
symbol = "EURUSD"
df = pd.read_parquet('data/historical/EURUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()  # Same downsampling
features = build_feature_matrix(df)

print(f"Loaded {len(df)} bars, features: {len(features)} rows")
print(f"Feature index range: {features.index[0]} to {features.index[-1]}")
print(f"Warmup needed: {min(50, len(features) // 4)} = {min(50, len(features) // 4)}")

# Check for pullback signals and their positions
pullback_rows = features[features['pullback_signal'] == True]
print(f"\nPullback signals: {len(pullback_rows)}")

for idx, row in pullback_rows.iterrows():
    # Find position in features
    pos = features.index.get_loc(idx)
    print(f"\nPullback at position {pos}, index {idx}")
    print(f"  hurst: {row['hurst']:.4f}")
    print(f"  regime: {row['regime']}")
    print(f"  pullback_direction: {row['pullback_direction']}")

    # Check if position >= warmup
    warmup = min(50, len(features) // 4)
    print(f"  warmup = {warmup}, position >= warmup? {pos >= warmup}")

    if pos >= warmup:
        # Generate signal and check filters
        signal = _generate_signal_from_row(row, symbol, idx)
        print(f"  Signal direction: {signal['direction']}")

        if signal['direction'] != 0:
            account_info = {"equity": 10000.0, "balance": 10000.0}
            filtered = apply_signal_filters(signal, account_info, 0)
            print(f"  Filtered: {filtered is not None}")