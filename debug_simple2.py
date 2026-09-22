from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Debug the exact backoff detection logic
# The issue: primed states exist but pullback signals don't

# Check each primed row's OFI sequence
primed_idx = features[features['primed'] == True].index.tolist()

for idx in primed_idx[:3]:
    pos = features.index.get_loc(idx)
    print(f"\nPrimed at position {pos}:")

    # Show OFI around this position
    for i in range(max(0, pos-2), min(len(features), pos+5)):
        row = features.iloc[i]
        marker = " <-- PRIMED" if i == pos else ""
        print(f"  {i}: OFI={row['ofi']:.4f}, dc={row['dc_event']}, regime={row['regime']}{marker}")