from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check OFI values at primed positions
primed_rows = features[features['primed'] == True]
print(f"Primed rows: {len(primed_rows)}")

# Check OFI transitions around primed rows
for idx, row in primed_rows.head(3).iterrows():
    pos = features.index.get_loc(idx)
    print(f"\nPrimed at position {pos}:")
    print(f"  dc_event: {row['dc_event']}")
    print(f"  ofi: {row['ofi']:.4f}")
    print(f"  hurst: {row['hurst']:.4f}")
    print(f"  regime: {row['regime']}")

    # Check OFI before and after
    if pos > 0:
        prev_ofi = features.iloc[pos-1]['ofi']
        print(f"  prev_ofi (pos-1): {prev_ofi:.4f}")
    if pos < len(features) - 1:
        next_ofi = features.iloc[pos+1]['ofi']
        print(f"  next_ofi (pos+1): {next_ofi:.4f}")