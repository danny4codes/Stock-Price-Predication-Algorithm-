from pathlib import Path
from feature_engineering import build_feature_matrix
import pandas as pd

df = pd.read_parquet('data/historical/GBPUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()
features = build_feature_matrix(df)

# Check trending bars
trending = features[features['regime'] == 'trending']
print(f"Trending bars: {len(trending)}")

for idx, row in trending.iterrows():
    print(f"\nTrending bar:")
    print(f"  dc_event: {row['dc_event']}")
    print(f"  ofi: {row['ofi']:.4f}")
    print(f"  hurst: {row['hurst']:.4f}")
    print(f"  Would set primed (dc_event==upturn and ofi>0.15): {row['dc_event'] == 'upturn' and row['ofi'] > 0.15}")
    print(f"  Would set primed (dc_event==downturn and ofi<-0.15): {row['dc_event'] == 'downturn' and row['ofi'] < -0.15}")