from backtester import load_historical_data
import pandas as pd
import numpy as np
import importlib as _il
_mse = _il.import_module('2_microstructure_engine')
DirectionalChangeDetector = _mse.DirectionalChangeDetector
calculate_order_flow_imbalance = _mse.calculate_order_flow_imbalance

from config import DC_THETA, OFI_THRESHOLD

symbol = 'EURUSD'
df = load_historical_data(symbol)

features = df.copy()
features['log_return'] = np.log(features['close'] / features['close'].shift(1))

detector = DirectionalChangeDetector(theta=DC_THETA)
dc_df = detector.process_series(features['close'], features['volume'])
features['dc_event'] = dc_df['event_type']
features['ofi'] = calculate_order_flow_imbalance(features['close'], features['volume'])

# Check rows around primed positions
print('Looking for primed rows...')
for i in range(len(features)-10):
    row = features.iloc[i]
    if not pd.isna(row['dc_event']) and row['dc_event'] in ['upturn', 'downturn'] and abs(row['ofi']) > OFI_THRESHOLD:
        print(f'Row {i}: dc={row["dc_event"]}, ofi={row["ofi"]:.3f}')
        for j in range(i+1, min(i+16, len(features))):
            next_row = features.iloc[j]
            print(f'  {j}: ofi={next_row["ofi"]:.3f}')