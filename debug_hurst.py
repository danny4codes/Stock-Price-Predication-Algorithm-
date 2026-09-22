from backtester import load_historical_data
from feature_engineering import build_feature_matrix
from config import HURST_TRENDING_THRESHOLD, OFI_THRESHOLD

symbol = 'EURUSD'
df = load_historical_data(symbol)

features = build_feature_matrix(df)

# Check where pullback signals are
pullbacks = features[features['pullback_signal'] == True]
print(f'Pullback signals found: {len(pullbacks)}')

# Check primed rows
primed = features[features['primed'] == True]
print(f'Primed rows: {len(primed)}')

# Check at what row indices pullbacks appear
for idx, row in pullbacks.iterrows():
    pos = features.index.get_loc(idx)
    print(f'Pullback at position {pos}: direction={row["pullback_direction"]}')