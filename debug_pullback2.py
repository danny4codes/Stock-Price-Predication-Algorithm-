from pathlib import Path
from feature_engineering import build_feature_matrix
import pandas as pd

# Load EURUSD data
df = pd.read_parquet('data/historical/EURUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()
features = build_feature_matrix(df)

# Check rows where pullback_signal is True
pullback_rows = features[features['pullback_signal'] == True]
print(f"Pullback signals detected: {len(pullback_rows)}")

if len(pullback_rows) > 0:
    for idx, row in pullback_rows.iterrows():
        print(f"\nRow {idx}:")
        print(f"  regime: {row['regime']}")
        print(f"  hurst: {row['hurst']:.3f}")
        print(f"  dc_event: {row['dc_event']}")
        print(f"  ofi: {row['ofi']:.4f}")
        print(f"  pullback_direction: {row['pullback_direction']}")
        print(f"  pullback_signal: {row['pullback_signal']}")

# Check OFI transitions around primed states
primed_rows = features[features['primed'] == True]
print(f"\n\nPrimed states: {len(primed_rows)}")
if len(primed_rows) > 0:
    for idx, row in primed_rows.head(10).iterrows():
        print(f"\nRow {idx}:")
        print(f"  regime: {row['regime']}")
        print(f"  hurst: {row['hurst']:.3f}")
        print(f"  ofi: {row['ofi']:.4f}")
        print(f"  dc_event: {row['dc_event']}")
        print(f"  primed_direction: {row['primed_direction']}")