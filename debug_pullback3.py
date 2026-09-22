from pathlib import Path
from feature_engineering import build_feature_matrix
from signal_generation import detect_market_regime
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
        hurst = row.get('hurst', 0.5)
        regime = detect_market_regime(hurst)
        pullback_signal = row.get('pullback_signal', False)
        pullback_direction = row.get('pullback_direction', 0)

        print(f"\nRow {idx}:")
        print(f"  hurst: {hurst:.4f}")
        print(f"  regime from detect_market_regime: {regime}")
        print(f"  pullback_signal: {pullback_signal}")
        print(f"  pullback_direction: {pullback_direction}")

        # Check the backtester condition
        condition = pullback_signal and pullback_direction != 0 and regime == "trending"
        print(f"  Backtester condition (pullback_signal AND pullback_direction AND regime==trending): {condition}")

        # Also check in the full features
        print(f"  features['regime'][{idx}]: {features.loc[idx, 'regime']}")