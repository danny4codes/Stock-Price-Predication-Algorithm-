from backtester import load_historical_data
from feature_engineering import build_feature_matrix
import pandas as pd

symbol = "EURUSD"
df = load_historical_data(symbol)

# Check where pullback signals appear in the data
df_sorted = df.sort_index()
df_subset = df_sorted.copy()

# Simulate what build_feature_matrix does
import numpy as np
df_subset['log_return'] = np.log(df_subset['close'] / df_subset['close'].shift(1))
df_subset.dropna(inplace=True)

# Check pullbacks at different stages
from feature_engineering import evaluate_pullback_signals

# Check if pullback signals are in the raw output
features = build_feature_matrix(df)
print(f"Pullback signals after build_feature_matrix: {features['pullback_signal'].sum()}")

# Check primed rows in the final features
primed_in_final = features[features['primed'] == True]
print(f"Primed in final features: {len(primed_in_final)}")

# Check what rows pullback signals are on
pullback_positions = []
for idx, row in features.iterrows():
    if row['pullback_signal']:
        pullback_positions.append(features.index.get_loc(idx))

print(f"Pullback positions: {pullback_positions}")