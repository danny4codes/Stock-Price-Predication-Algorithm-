from backtester import load_historical_data
from feature_engineering import (
    build_feature_matrix, evaluate_primary_signals_batch, evaluate_pullback_signals
)
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)

# Build features manually to trace
features = build_feature_matrix(df)

# Check what evaluate_pullback_signals does to the features DataFrame directly
print(f"Features shape: {features.shape}")
print(f"Primed in features: {features['primed'].sum()}")
print(f"Pullback in features: {features['pullback_signal'].sum()}")

# Check if primed rows have different regime
primed = features[features['primed'] == True]
print(f"\nPrimed rows regime distribution:")
print(primed['regime'].value_counts())