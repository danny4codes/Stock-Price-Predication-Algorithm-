"""Debug script to analyze OFI patterns during pullback scenarios."""
import pandas as pd
from pathlib import Path
from feature_engineering import build_feature_matrix
from config import OFI_THRESHOLD, OFI_PULLBACK_THRESHOLD

# Load EURUSD data
data_path = Path(r"D:\Trading\data\historical/EURUSD_M1_20260518_131338.parquet")
df = pd.read_parquet(data_path)
df = df.iloc[::180].copy()  # Downsample like backtester

# Build features
features = build_feature_matrix(df)

# Find some primed states and examine OFI around them
print("Analyzing pullback conditions...")
print(f"OFI threshold: {OFI_THRESHOLD}")
print(f"OFI pullup threshold: {OFI_PULLBACK_THRESHOLD}")
print()

# Look for rows where pullback_signal fired
pullback_rows = features[features['pullback_signal'] == True]
print(f"Pullback signals found: {len(pullback_rows)}")
print()

if len(pullback_rows) > 0:
    for idx in pullback_rows.index[:3]:
        pos = features.index.get_loc(idx)
        print(f"\n--- Pullback at {idx} ---")
        print(f"Pullback direction: {features.at[idx, 'pullback_direction']}")
        print(f"OFI at signal: {features.at[idx, 'ofi']:.4f}")
        print(f"Hurst: {features.at[idx, 'hurst']:.3f}")

        # Look at surrounding 15 bars
        start = max(0, pos - 15)
        end = min(len(features), pos + 5)
        window = features.iloc[start:end]
        print("\nSurrounding bars (dc_event, ofi, regime):")
        for i, r in window.iterrows():
            marker = " <-- PULLBACK" if i == idx else ""
            print(f"  {r.name}: dc={r['dc_event']}, ofi={r['ofi']:.4f}, regime={r['regime']}{marker}")

# Also check primed states
primed_rows = features[features['primed'] == True]
print(f"\n\nPrimed states found: {len(primed_rows)}")
for idx in primed_rows.index[:3]:
    pos = features.index.get_loc(idx)
    print(f"\n--- Primed at {idx} ---")
    print(f"Primed direction: {features.at[idx, 'primed_direction']}")
    print(f"OFI: {features.at[idx, 'ofi']:.4f}")
    print(f"DC event: {features.at[idx, 'dc_event']}")
    print(f"Hurst: {features.at[idx, 'hurst']:.3f}")

    # Look at next 15 bars for pullback pattern
    start = pos
    end = min(len(features), pos + 15)
    window = features.iloc[start:end]
    print("\nNext 15 bars (looking for OFI pullback):")
    for i, r in window.iterrows():
        primed_marker = " <-- PRIMED" if i == idx else ""
        print(f"  {r.name}: ofi={r['ofi']:.4f}, dc={r['dc_event']}{primed_marker}")