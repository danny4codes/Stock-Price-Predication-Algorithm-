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

# Generate signals
signals = generate_signals_vectorized(features, symbol)

# Check for non-zero signals after warmup
count = 0
for idx in range(warmup, len(signals)):
    if signals[idx]['direction'] != 0:
        count += 1
        print(f"\nSignal at position {idx}:")
        print(f"  direction: {signals[idx]['direction']}")
        print(f"  confidence: {signals[idx]['confidence']}")

print(f"\nTotal signals after warmup: {count}")