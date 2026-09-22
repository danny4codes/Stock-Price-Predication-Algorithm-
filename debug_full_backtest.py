from pathlib import Path
from feature_engineering import build_feature_matrix
from backtester import _generate_signal_from_row, detect_market_regime, apply_signal_filters
import pandas as pd

# Load EURUSD data
df = pd.read_parquet('data/historical/EURUSD_M1_20260518_131338.parquet')
df = df.iloc[::180].copy()
features = build_feature_matrix(df)

# Check rows where pullback_signal is True
pullback_rows = features[features['pullback_signal'] == True]

for idx, row in pullback_rows.iterrows():
    print(f"\n=== Checking row {idx} ===")

    # Generate signal
    signal = _generate_signal_from_row(row, "EURUSD", idx)
    print(f"Signal direction: {signal['direction']}")
    print(f"Signal confidence: {signal['confidence']}")

    if signal['direction'] != 0:
        # Check filters
        account_info = {"equity": 10000.0, "balance": 10000.0}
        filtered = apply_signal_filters(signal, account_info, 0)
        if filtered is None:
            print("Filtered: REJECTED (None)")
        else:
            print(f"Filtered direction: {filtered['direction']}")