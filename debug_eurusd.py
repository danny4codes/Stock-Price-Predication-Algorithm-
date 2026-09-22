from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check pullback signals
pullbacks = features[features['pullback_signal'] == True]
print(f"Pullback signals in final features: {len(pullbacks)}")

for idx, row in pullbacks.iterrows():
    pos = features.index.get_loc(idx)
    print(f"  Position {pos}: dir={row['pullback_direction']}, ofi={row['ofi']:.3f}, hurst={row['hurst']:.3f}")

# Check primed rows
primed = features[features['primed'] == True]
print(f"\nPrimed rows: {len(primed)}")

# Check OFI around first primed row to understand why no pullback
if len(primed) > 0:
    first_primed_idx = primed.index[0]
    first_primed_pos = features.index.get_loc(first_primed_idx)
    print(f"\nFirst primed at position {first_primed_pos}:")
    row = primed.iloc[0]
    primed_ofi = row['ofi']
    print(f"  primed_ofi={primed_ofi:.3f}")

    # Show next 15 OFI values
    print(f"  Next 15 OFI values:")
    for i in range(15):
        if first_primed_pos + i + 1 < len(features):
            next_ofi = features.iloc[first_primed_pos + i + 1]['ofi']
            threshold = 0.10
            weakened = primed_ofi * 0.5
            print(f"    {first_primed_pos + i + 1}: ofi={next_ofi:.3f}, weakened_threshold={weakened:.3f}, recovered={next_ofi > threshold}")