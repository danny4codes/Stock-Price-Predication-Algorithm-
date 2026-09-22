from pathlib import Path
from backtester import load_historical_data, generate_signals_vectorized
import pandas as pd

# Run the actual backtest logic step by step
symbol = "GBPUSD"
df = load_historical_data(symbol)
print(f"Loaded {len(df)} bars")

from feature_engineering import build_feature_matrix
features = build_feature_matrix(df)
print(f"Features: {len(features)} rows")

warmup = min(5, len(features) // 10)
print(f"Warmup: {warmup}")

# Check features directly
print(f"\nFeatures with pullback_signal=True:")
for idx, row in features.iterrows():
    if row.get('pullback_signal', False):
        pos = features.index.get_loc(idx)
        print(f"  Position {pos}: pullback_direction={row['pullback_direction']}, regime={row['regime']}")
        print(f"    hurst={row['hurst']:.3f}, ofi={row['ofi']:.4f}")

# Generate signals
signals = generate_signals_vectorized(features, symbol)

# Check all signals (not just after warmup)
print(f"\nAll signals with direction != 0:")
for idx, sig in enumerate(signals):
    if sig['direction'] != 0:
        print(f"  Position {idx}: direction={sig['direction']}, confidence={sig['confidence']}")
        print(f"    After warmup ({warmup})? {idx >= warmup}")