from backtester import load_historical_data, generate_signals_vectorized
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)
features = build_feature_matrix(df)

# Check pullback signals already in features
print(f"Pullback signals in features: {features['pullback_signal'].sum()}")
print(f"Primed in features: {features['primed'].sum()}")

# Generate signals
signals = generate_signals_vectorized(features, symbol)

# Count non-zero
non_zero = [(i, s) for i, s in enumerate(signals) if s['direction'] != 0]
print(f"\nNon-zero signals: {len(non_zero)}")

for pos, sig in non_zero[:5]:
    print(f"  Position {pos}: direction={sig['direction']}")