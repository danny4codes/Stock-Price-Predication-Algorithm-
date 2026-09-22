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

from config import DC_THETA, OFI_THRESHOLD, HURST_ROLLING_WINDOW, HURST_MAX_LAG, HURST_TRENDING_THRESHOLD, GARCH_ROLLING_WINDOW, VOL_ZSCORE_WINDOW

symbol = 'EURUSD'
df = load_historical_data(symbol)

# Build FULL features same as build_feature_matrix
features = df.copy()
features['log_return'] = np.log(features['close'] / features['close'].shift(1))
features['hl_ratio'] = (features['high'] - features['low']) / features['close']
features['oc_ratio'] = (features['close'] - features['open']) / features['open']

detector = DirectionalChangeDetector(theta=DC_THETA)
dc_df = detector.process_series(features['close'], features['volume'])
features['dc_event'] = dc_df['event_type']
features['dc_signal_strength'] = dc_df['signal_strength']
features['ofi'] = calculate_order_flow_imbalance(features['close'], features['volume'])

from feature_engineering import compute_hurst_exponent
features['hurst'] = features['close'].rolling(HURST_ROLLING_WINDOW, min_periods=HURST_MAX_LAG*2).apply(
    lambda x: compute_hurst_exponent(pd.Series(x)), raw=False
)
features['regime'] = features['hurst'].apply(classify_regime)
features['is_trending'] = features['hurst'] > HURST_TRENDING_THRESHOLD

# GARCH
features['garch_vol'] = features['log_return'].rolling(GARCH_ROLLING_WINDOW, min_periods=50).apply(
    lambda x: 1.0, raw=False
)

# Vol zscore
features['vol_zscore'] = 0.0

# Primary signals
features = evaluate_primary_signals_batch(features, ofi_threshold=OFI_THRESHOLD)

# Pullback
features = evaluate_pullback_signals(features)

# Check the exact rows
for check_row in [1025, 1030, 1035, 1140, 1145, 1150]:
    row = features.iloc[check_row]
    print(f"Row {check_row}: pullback={row.get('pullback_signal', False)}, primed={row.get('primed', False)}, hurst={row.get('hurst', 0):.3f}, ofi={row.get('ofi', 0):.3f}")

# Now check NaN counts per column
print("\nNaN counts per column:")
for col in ['pullback_signal', 'primed', 'hurst', 'ofi', 'dc_event', 'garch_vol', 'vol_zscore', 'log_return', 'hl_ratio', 'oc_ratio']:
    nan_count = features[col].isna().sum()
    print(f"  {col}: {nan_count}")

# Dropna and check
n_before = len(features)
features.dropna(inplace=True)
n_after = len(features)
print(f"\nAfter dropna: {n_after} rows")

# Check if pullback signals survived
pullbacks = features[features['pullback_signal'] == True]
print(f"Pullback signals after dropna: {len(pullbacks)}")