# tests/test_feature_engineering.py — Unit Tests for feature_engineering.py
# Updated: Retail indicators removed, only econometric/microstructure features tested

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    build_feature_matrix,
    compute_atr,
    compute_hurst_exponent,
    compute_log_returns,
    compute_vwap,
    fit_garch,
    DirectionalChangeDetector,
    calculate_order_flow_imbalance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    close  = 1.1000 + np.cumsum(rng.normal(0, 0.0010, n))
    high   = close + rng.uniform(0.0005, 0.0020, n)
    low    = close - rng.uniform(0.0005, 0.0020, n)
    open_  = close + rng.normal(0, 0.0005, n)
    volume = rng.integers(100, 10000, n).astype(float)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_trending_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate trending OHLCV data (persistent upward movement)."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 0.20, n)
    noise = rng.normal(0, 0.0002, n)
    close = 1.1000 + np.cumsum(trend + noise)
    high = close + rng.uniform(0.0003, 0.0015, n)
    low = close - rng.uniform(0.0003, 0.0015, n)
    open_ = close + rng.normal(0, 0.0003, n)
    volume = rng.integers(100, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def close(ohlcv):
    return ohlcv["close"]


@pytest.fixture
def trending_ohlcv():
    return make_trending_ohlcv()


# ---------------------------------------------------------------------------
# Log Returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_returns_series(self, close):
        result = compute_log_returns(close)
        assert isinstance(result, pd.Series)

    def test_first_value_nan(self, close):
        result = compute_log_returns(close)
        assert pd.isna(result.iloc[0])

    def test_correct_formula(self, close):
        result = compute_log_returns(close)
        expected = np.log(close.iloc[1] / close.iloc[0])
        assert abs(result.iloc[1] - expected) < 1e-10


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestComputeATR:
    def test_returns_series(self, ohlcv):
        result = compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_non_negative(self, ohlcv):
        result = compute_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class TestComputeVWAP:
    def test_returns_series(self, ohlcv):
        result = compute_vwap(ohlcv)
        assert isinstance(result, pd.Series)

    def test_same_length(self, ohlcv):
        result = compute_vwap(ohlcv)
        assert len(result) == len(ohlcv)

    def test_positive_values(self, ohlcv):
        result = compute_vwap(ohlcv).dropna()
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------

class TestComputeHurstExponent:
    def test_returns_float(self, close):
        result = compute_hurst_exponent(close)
        assert isinstance(result, float)

    def test_value_between_0_and_1(self, close):
        result = compute_hurst_exponent(close)
        assert 0.0 <= result <= 1.0

    def test_insufficient_data_returns_half(self):
        tiny = pd.Series([1.0, 1.1, 1.2, 1.1])
        result = compute_hurst_exponent(tiny, max_lag=100)
        assert result == 0.5

    def test_trending_data_positive_hurst(self, trending_ohlcv):
        """Trending data should yield Hurst > 0.5."""
        result = compute_hurst_exponent(trending_ohlcv["close"])
        assert result > 0.5


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

class TestFitGARCH:
    def test_returns_tuple(self, close):
        returns = compute_log_returns(close).dropna()
        vol, result = fit_garch(returns)
        assert isinstance(vol, float) or np.isnan(vol)

    def test_insufficient_data_returns_nan(self):
        tiny = pd.Series([0.001, -0.002, 0.003] * 5)
        vol, _ = fit_garch(tiny)
        assert np.isnan(vol)

    def test_volatility_positive(self, close):
        returns = compute_log_returns(close).dropna()
        vol, _ = fit_garch(returns)
        if not np.isnan(vol):
            assert vol > 0


# ---------------------------------------------------------------------------
# Directional Change Detector
# ---------------------------------------------------------------------------

class TestDirectionalChangeDetector:
    def test_initialization(self):
        detector = DirectionalChangeDetector(theta=0.001)
        assert detector.theta == 0.001
        assert detector.last_extremum_price is None

    def test_first_tick_no_event(self):
        detector = DirectionalChangeDetector(theta=0.001)
        result = detector.process_tick(1.1000, 1000, None)
        assert result["event_type"] is None
        assert result["price"] == 1.1000

    def test_upturn_event(self):
        detector = DirectionalChangeDetector(theta=0.01)
        # Start at 1.1000, need price > 1.1000 * 1.01 = 1.111 to trigger upturn
        detector.process_tick(1.1000, 1000, None)  # init
        result = detector.process_tick(1.1200, 1500, None)
        assert result["event_type"] == "upturn"
        assert result["signal_strength"] > 0

    def test_downturn_event(self):
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000, 1000, None)  # init as min
        detector.process_tick(1.1200, 1000, None)  # upturn (1.12 > 1.111), now max at 1.12
        result = detector.process_tick(1.1000, 1500, None)  # drop > 1% from max → downturn
        assert result["event_type"] == "downturn"

    def test_new_minimum_update(self):
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000, 1000, None)
        result = detector.process_tick(1.0900, 1000, None)  # New lower min
        assert result["event_type"] is None
        assert detector.last_extremum_price == 1.0900


# ---------------------------------------------------------------------------
# Order Flow Imbalance
# ---------------------------------------------------------------------------

class TestOrderFlowImbalance:
    def test_up_ticks_positive_ofi(self):
        prices = pd.Series([1.0, 1.01, 1.02, 1.03])
        volumes = pd.Series([100, 100, 100, 100])
        result = calculate_order_flow_imbalance(prices, volumes, window=3)
        assert result.iloc[-1] > 0  # Upward ticks → positive OFI

    def test_down_ticks_negative_ofi(self):
        prices = pd.Series([1.03, 1.02, 1.01, 1.0])
        volumes = pd.Series([100, 100, 100, 100])
        result = calculate_order_flow_imbalance(prices, volumes, window=3)
        assert result.iloc[-1] < 0  # Downward ticks → negative OFI

    def test_mixed_ticks_near_zero(self):
        prices = pd.Series([1.0, 1.01, 1.0, 1.01, 1.0])
        volumes = pd.Series([100, 100, 100, 100, 100])
        result = calculate_order_flow_imbalance(prices, volumes, window=5)
        assert abs(result.iloc[-1]) < 0.5  # Mixed → near zero

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_order_flow_imbalance(pd.Series([1, 2, 3]), pd.Series([1, 2]))


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_dataframe(self, ohlcv):
        result = build_feature_matrix(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_values(self, ohlcv):
        result = build_feature_matrix(ohlcv)
        assert not result.isna().any().any(), "Feature matrix contains NaN values"

    def test_fewer_rows_than_input(self, ohlcv):
        result = build_feature_matrix(ohlcv)
        assert len(result) < len(ohlcv)  # Warmup rows dropped

    def test_empty_input_returns_empty(self):
        result = build_feature_matrix(pd.DataFrame())
        assert result.empty

    def test_missing_column_returns_empty(self, ohlcv):
        bad_df = ohlcv.drop(columns=["volume"])
        result = build_feature_matrix(bad_df)
        assert result.empty

    def test_expected_key_columns_present(self, ohlcv):
        result = build_feature_matrix(ohlcv)
        expected = {"log_return", "atr", "hurst", "garch_vol", "ofi", "dc_signal_strength"}
        assert expected.issubset(set(result.columns))

    def test_no_retail_indicators(self, ohlcv):
        """Verify prohibited indicators are absent."""
        result = build_feature_matrix(ohlcv)
        prohibited = {"rsi", "macd", "macd_signal", "macd_diff", "ema_20",
                       "ema_50", "ema_200", "ema_cross", "adx", "adx_pos",
                       "adx_neg", "bb_upper", "bb_mid", "bb_lower", "bb_width",
                       "bb_pct", "stoch_k", "stoch_d"}
        present = prohibited & set(result.columns)
        assert len(present) == 0, f"Prohibited indicators found: {present}"