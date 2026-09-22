from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "GBPUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check pullback signals that survived
pullbacks = features[features['pullback_signal'] == True]
print(f"Pullback signals in final features: {len(pullbacks)}")

for idx, row in pullbacks.iterrows():
    pos = features.index.get_loc(idx)
    print(f"  Position {pos}: pullback_direction={row['pullback_direction']}, ofi={row['ofi']:.3f}")

# Check primed states
primed = features[features['primed'] == True]
print(f"\nPrimed rows: {len(primed)}")

# Check dc_event values
dc_events = features[features['dc_event'].notna()]
print(f"\nRows with non-null dc_event: {len(dc_events)}")

# Check NaN per column
print("\nNaN counts:")
for col in features.columns:
    nan_count = features[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count}")