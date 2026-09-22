"""Debug script to analyze individual trades."""
import pandas as pd
from pathlib import Path
from feature_engineering import build_feature_matrix
from config import OFI_THRESHOLD, HURST_TRENDING_THRESHOLD

# Load EURUSD data
data_path = Path(r"D:\Trading\data\historical/EURUSD_M1_20260518_131338.parquet")
df_raw = pd.read_parquet(data_path)
df = df_raw.iloc[::180].copy()  # Downsample like backtester

# Build features
features = build_feature_matrix(df)

# Find the first pullback signal with detailed analysis
pullback_rows = features[features['pullback_signal'] == True]
print(f"Total pullback signals: {len(pullback_rows)}")
print()

if len(pullback_rows) > 0:
    # Analyze the first trade
    idx = pullback_rows.index[0]
    pos = features.index.get_loc(idx)

    print(f"=== First pullback signal at {idx} ===")
    row = features.loc[idx]
    print(f"Direction: {row['pullback_direction']} ({'LONG' if row['pullback_direction'] == 1 else 'SHORT'})")
    print(f"Entry (close): {row['close']:.5f}")
    print(f"ATR: {row['atr']:.5f}")
    print(f"Hurst: {row['hurst']:.3f}")
    print(f"OFI: {row['ofi']:.4f}")
    print(f"DC event: {row['dc_event']}")

    # Get the original OHLC data for the trade window
    df_aligned = df_raw.iloc[::180].copy()
    start_pos = df_aligned.index.get_loc(features.index[0])
    df_aligned = df_aligned.iloc[start_pos:].copy()

    # Check what happens in the next 20 bars
    print("\n=== Next 20 bars (OHLC) ===")
    for i in range(pos, min(pos + 20, len(features))):
        r = features.iloc[i]
        ohlc = df_aligned.iloc[i]
        print(f"{r.name}: close={r['close']:.5f}, high={ohlc['high']:.5f}, low={ohlc['low']:.5f}, ofi={r['ofi']:.4f}, dc={r.get('dc_event')}")

    # Check what happens in the previous 20 bars
    print("\n=== Previous 20 bars (primed state context) ===")
    for i in range(max(0, pos - 20), pos):
        r = features.iloc[i]
        OHLC = df_aligned.iloc[i]
        primed = " **PRIMED**" if r['primed'] else ""
        print(f"{r.name}: close={r['close']:.5f}, low={OHLC['low']:.5f}, ofi={r['ofi']:.4f}, dc={r['dc_event']}, regime={r['regime']}{primed}")