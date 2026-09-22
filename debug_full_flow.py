from pathlib import Path
from backtester import load_historical_data, generate_signals_vectorized
from feature_engineering import build_feature_matrix
import pandas as pd

# Run the actual backtest logic step by step
symbol = "EURUSD"
df = load_historical_data(symbol)
print(f"Loaded {len(df)} bars")

features = build_feature_matrix(df)
print(f"Features: {len(features)} rows")

# Check pullback signals in features
pullback_rows = features[features['pullback_signal'] == True]
print(f"\nPullback signals in features: {len(pullback_rows)}")

for idx, row in pullback_rows.iterrows():
    pos = features.index.get_loc(idx)
    print(f"  Position {pos}: pullback_direction={row['pullback_direction']}")

# Generate signals
signals = generate_signals_vectorized(features, symbol)

# Count non-zero signals
non_zero = sum(1 for s in signals if s['direction'] != 0)
print(f"\nNon-zero signals: {non_zero}")

# Check warmup
warmup = min(5, len(features) // 10)
print(f"Warmup: {warmup}")

# Check signals at pullback positions
for idx, row in pullback_rows.iterrows():
    pos = features.index.get_loc(idx)
    sig = signals[pos]
    print(f"\nSignal at pullback position {pos}:")
    print(f"  direction: {sig['direction']}")
    print(f"  After warmup: {pos >= warmup}")