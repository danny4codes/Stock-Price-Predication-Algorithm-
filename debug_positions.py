from backtester import load_historical_data
import pandas as pd
import numpy as np
import importlib as _il
_mse = _il.import_module('2_microstructure_engine')
DirectionalChangeDetector = _mse.DirectionalChangeDetector
calculate_order_flow_imbalance = _mse.calculate_order_flow_imbalance
evaluate_primary_signals_batch = _mse.evaluate_primary_signals_batch
evaluate_pullback_signals = _mse.evaluate_pullback_signals
classify_regime = _mse.classify_regime

from config import DC_THETA, OFI_THRESHOLD, HURST_ROLLING_WINDOW, HURST_MAX_LAG, HURST_TRENDING_THRESHOLD

symbol = 'EURUSD'
df = load_historical_data(symbol)

features = df.copy()
features['log_return'] = np.log(features['close'] / features['close'].shift(1))

detector = DirectionalChangeDetector(theta=DC_THETA)
dc_df = detector.process_series(features['close'], features['volume'])
features['dc_event'] = dc_df['event_type']
features['ofi'] = calculate_order_flow_imbalance(features['close'], features['volume'])

# Add hurst
from feature_engineering import compute_hurst_exponent
features['hurst'] = features['close'].rolling(HURST_ROLLING_WINDOW, min_periods=HURST_MAX_LAG*2).apply(
    lambda x: compute_hurst_exponent(pd.Series(x)), raw=False
)
features['regime'] = features['hurst'].apply(classify_regime)

# Primary signals
features = evaluate_primary_signals_batch(features, ofi_threshold=OFI_THRESHOLD)

# Pullback
features = evaluate_pullback_signals(features)

# Check where primed and pullback occur
print("Positions of primed states before dropna:")
for i in range(len(features)):
    if features.iloc[i]['primed']:
        print(f"  Row {i}: ofi={features.iloc[i]['ofi']:.3f}")

print("\nPositions of pullback signals before dropna:")
for i in range(len(features)):
    if features.iloc[i]['pullback_signal']:
        print(f"  Row {i}: pullback_direction={features.iloc[i]['pullback_direction']}")

# Check GARCH NaN threshold
features['garch_vol'] = 1.0  # placeholder
features['vol_zscore'] = 0.0

# Dropna
n_before = len(features)
features.dropna(inplace=True)
n_after = len(features)
print(f"\nRows dropped by dropna: {n_before - n_after}")