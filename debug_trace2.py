from backtester import load_historical_data
import pandas as pd
import numpy as np

symbol = "EURUSD"
df = load_historical_data(symbol)

# Manually trace the full build_feature_matrix
from feature_engineering import (
    DirectionalChangeDetector, calculate_order_flow_imbalance,
    compute_log_returns, compute_atr, compute_vwap, compute_hurst_exponent,
    evaluate_primary_signals_batch, evaluate_pullback_signals, classify_regime
)
from config import HURST_ROLLING_WINDOW, HURST_MAX_LAG, HURST_TRENDING_THRESHOLD, GARCH_ROLLING_WINDOW, VOL_ZSCORE_WINDOW, OFI_THRESHOLD, HURST_MEAN_REVERT_THRESHOLD

features = pd.DataFrame(index=df.index)

# Step 1: DC
detector = DirectionalChangeDetector()
dc_df = detector.process_series(df["close"], df["volume"])
features["dc_event"] = dc_df["event_type"]
features["dc_signal_strength"] = dc_df["signal_strength"]

# Step 2: OFI
features["ofi"] = calculate_order_flow_imbalance(df["close"], df["volume"])

# Step 3: Other features
features["log_return"] = compute_log_returns(df["close"])
features["close"] = df["close"]
features["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
features["oc_ratio"] = (df["close"] - df["open"]) / df["open"]
features["atr"] = compute_atr(df["high"], df["low"], df["close"])
features["atr_pct"] = features["atr"] / df["close"]
if df["volume"].sum() > 0:
    features["vwap"] = compute_vwap(df)
    features["vwap_dist"] = (df["close"] - features["vwap"]) / features["vwap"].replace(0, np.nan)
    features["vol_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()

# Step 4: Hurst
features["hurst"] = df["close"].rolling(
    HURST_ROLLING_WINDOW,
    min_periods=min(HURST_MAX_LAG * 2, HURST_ROLLING_WINDOW),
).apply(lambda x: compute_hurst_exponent(pd.Series(x)), raw=False)

features["is_trending"] = features["hurst"] > HURST_TRENDING_THRESHOLD
features["regime"] = features["hurst"].apply(classify_regime)

# Step 5: GARCH
from microstructure_engine import fit_garch
log_ret = features["log_return"]
features["garch_vol"] = log_ret.rolling(GARCH_ROLLING_WINDOW, min_periods=50).apply(
    lambda x: fit_garch(pd.Series(x))[0], raw=False
)

print(f"Before primary signals: {len(features)} rows")

# Step 6: Primary signals
features = evaluate_primary_signals_batch(features, ofi_threshold=OFI_THRESHOLD)
print(f"Primary signals done: {len(features)} rows")

# Step 7: Pullback signals
features = evaluate_pullback_signals(features, ofi_threshold=OFI_THRESHOLD)
print(f"Pullback signals done: {len(features)} rows")
print(f"Pullback signals: {features['pullback_signal'].sum()}")
print(f"Primed: {features['primed'].sum()}")

# Check before dropna
print(f"\nNaN per column:")
for col in features.columns:
    nan_count = features[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count} NaN")