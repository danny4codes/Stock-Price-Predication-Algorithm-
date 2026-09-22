from pathlib import Path
from feature_engineering import build_feature_matrix
import pandas as pd

# Load EURUSD data
df = pd.read_parquet('data/historical/EURUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()
features = build_feature_matrix(df)

ofi = features['ofi']
print('OFI stats:')
print(f'  Mean: {ofi.mean():.4f}')
print(f'  Std: {ofi.std():.4f}')
print(f'  Min: {ofi.min():.4f}')
print(f'  Max: {ofi.max():.4f}')
print(f'  Near zero (|ofi|<0.1): {(ofi.abs() < 0.1).sum()} bars')
print(f'  Near zero (|ofi|<0.05): {(ofi.abs() < 0.05).sum()} bars')
print(f'DC events: {features["dc_event"].notna().sum()}')
print(f'Primed states: {features["primed"].sum()}')
print(f'Pullback signals: {features["pullback_signal"].sum()}')

# Check regime distribution
print(f'\nRegime distribution:')
print(features['regime'].value_counts())

# Check trending periods with pullback potential
trending = features[features['regime'] == 'trending']
print(f'\nTrending bars: {len(trending)}')
print(f'Trending OFI crossing zero: {(trending["ofi"].abs() < 0.1).sum()}')