from pathlib import Path
from feature_engineering import build_feature_matrix
import pandas as pd

# Run the actual backtest logic step by step
symbol = "GBPUSD"
df = pd.read_parquet('data/historical/GBPUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()
print(f"Loaded {len(df)} bars (downsampled)")

features = build_feature_matrix(df)
print(f"Features: {len(features)} rows")

# Check dc_event distribution
print(f"\nDC event distribution:")
dc_counts = features['dc_event'].value_counts()
print(dc_counts)

print(f"\nTrending bars: {(features['regime'] == 'trending').sum()}")
print(f"Primed bars: {features['primed'].sum()}")
print(f"Pullback signals: {features['pullback_signal'].sum()}")

# Check first few primed rows
primed_rows = features[features['primed'] == True]
print(f"\nPrimed row details:")
for idx, row in primed_rows.head(5).iterrows():
    pos = features.index.get_loc(idx)
    print(f"  {pos}: dc_event={row['dc_event']}, ofi={row['ofi']:.4f}, regime={row['regime']}")