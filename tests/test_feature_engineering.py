# tests/test_feature_engineering.py — Unit Tests for feature_engineering.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    build_feature_matrix,
    compute_atr,
    compute_bollinger_bands,
    compute_hurst_exponent,
    compute_log_returns,
    compute_macd,
    compute_price_momentum,
    compute_rsi,
    compute_vwap,
    fit_garch,
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


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def close(ohlcv):
    return ohlcv["close"]


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestComputeRSI:
    def test_returns_series(self, close):
        result = compute_rsi(close)
        assert isinstance(result, pd.Series)

    def test_same_length(self, close):
        result = compute_rsi(close)
        assert len(result) == len(close)

    def test_values_in_range(self, close):
        result = compute_rsi(close).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_leading_nans(self, close):
        result = compute_rsi(close, period=14)
        assert result.iloc[:13].isna().all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestComputeMACD:
    def test_returns_dataframe(self, close):
        result = compute_macd(close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, close):
        result = compute_macd(close)
        assert set(result.columns) == {"macd", "macd_signal", "macd_diff"}

    def test_same_length(self, close):
        result = compute_macd(close)
        assert len(result) == len(close)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestComputeBollingerBands:
    def test_returns_dataframe(self, close):
        result = compute_bollinger_bands(close)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, close):
        result = compute_bollinger_bands(close)
        assert {"bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct"} == set(result.columns)

    def test_upper_above_lower(self, close):
        result = compute_bollinger_bands(close).dropna()
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_mid_between_bands(self, close):
        result = compute_bollinger_bands(close).dropna()
        assert (result["bb_mid"] >= result["bb_lower"]).all()
        assert (result["bb_mid"] <= result["bb_upper"]).all()


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
# Price Momentum
# ---------------------------------------------------------------------------

class TestComputePriceMomentum:
    def test_returns_dataframe(self, close):
        result = compute_price_momentum(close, [5, 10, 20])
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, close):
        result = compute_price_momentum(close, [5, 10, 20])
        assert set(result.columns) == {"mom_5", "mom_10", "mom_20"}


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
        expected = {"rsi", "macd", "bb_pct", "atr", "adx", "log_return"}
        assert expected.issubset(set(result.columns))
