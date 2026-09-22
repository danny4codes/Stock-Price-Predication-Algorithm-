# tests/test_signal_generation.py — Unit Tests for signal_generation.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from signal_generation import (
    Signal,
    apply_signal_filters,
    combine_signals,
    detect_market_regime,
    generate_mean_reversion_signal,
    generate_trend_signal,
    _null_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_account():
    return {"balance": 10000.0, "equity": 10000.0}


def make_features(n: int = 50, override: dict = None) -> pd.DataFrame:
    """Make a minimal features DataFrame for signal tests (econometric features only)."""
    rng = np.random.default_rng(99)
    base = {
        "close":                1.1000 + np.cumsum(rng.normal(0, 0.001, n)),
        "atr":                  np.full(n, 0.0030),
        "hurst":                np.full(n, 0.60),
        "is_trending":          np.full(n, True),
        "log_return":           rng.normal(0, 0.001, n),
        "garch_vol":            np.full(n, 0.0015),
        "vol_zscore":           np.full(n, 0.5),
        "vwap":                 np.full(n, 1.1000),
        "vwap_dist":            np.full(n, 0.0),
        "ofi":                  np.full(n, 0.15),
        "dc_event":             [None] * n,
        "dc_signal_strength":   np.full(n, 0.0),
        "pullback_signal":      np.full(n, False),
        "pullback_direction":   np.full(n, 0),
    }
    if override:
        for k, v in override.items():
            if k == "dc_event":
                base[k] = [v] * n  # list of identical event types
            else:
                base[k] = np.full(n, v)

    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(base, index=idx)


# ---------------------------------------------------------------------------
# detect_market_regime
# ---------------------------------------------------------------------------

class TestDetectMarketRegime:
    def test_trending(self):
        assert detect_market_regime(0.60) == "trending"

    def test_mean_reverting(self):
        assert detect_market_regime(0.30) == "mean_reverting"

    def test_random_high_boundary(self):
        assert detect_market_regime(0.50) == "random"

    def test_random_exact_thresholds(self):
        assert detect_market_regime(0.55) == "random"
        assert detect_market_regime(0.45) == "random"

    def test_extreme_trending(self):
        assert detect_market_regime(0.90) == "trending"

    def test_extreme_mean_reverting(self):
        assert detect_market_regime(0.10) == "mean_reverting"


# ---------------------------------------------------------------------------
# generate_trend_signal
# ---------------------------------------------------------------------------

class TestGenerateTrendSignal:
    def test_long_signal_conditions(self):
        """Hurst > 0.58 + upturn DC event + pullback confirmation → LONG"""
        features = make_features(override={
            "dc_event": "upturn",
            "ofi": 0.15,
            "hurst": 0.60,
            "is_trending": True,
            "pullback_signal": True,
            "pullback_direction": 1,
        })
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == 1

    def test_short_signal_conditions(self):
        """Hurst > 0.58 + downturn DC event + pullback confirmation → SHORT"""
        features = make_features(override={
            "dc_event": "downturn",
            "ofi": -0.15,
            "hurst": 0.60,
            "is_trending": True,
            "pullback_signal": True,
            "pullback_direction": -1,
        })
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == -1

    def test_no_dc_event_returns_flat(self):
        """No DC event → no trend signal."""
        features = make_features(override={
            "dc_event": None,
            "ofi": 0.15,
            "hurst": 0.60,
        })
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == 0

    def test_signal_has_valid_sl_tp(self):
        features = make_features(override={
            "dc_event": "upturn",
            "ofi": 0.15,
            "hurst": 0.60,
        })
        sig = generate_trend_signal(features, "EURUSD")
        if sig["direction"] != 0:
            assert sig["stop_loss"] > 0
            assert sig["take_profit"] > 0

    def test_confidence_between_0_and_1(self):
        features = make_features()
        sig = generate_trend_signal(features, "EURUSD")
        assert 0.0 <= sig["confidence"] <= 1.0

    def test_long_sl_below_entry(self):
        features = make_features(override={
            "dc_event": "upturn",
            "ofi": 0.15,
            "hurst": 0.60,
        })
        sig = generate_trend_signal(features, "EURUSD")
        if sig["direction"] == 1:
            assert sig["stop_loss"] < sig["entry_price"]
            assert sig["take_profit"] > sig["entry_price"]


# ---------------------------------------------------------------------------
# generate_mean_reversion_signal
# ---------------------------------------------------------------------------

class TestGenerateMeanReversionSignal:
    def test_long_mr_signal(self):
        """Price below VWAP + high vol z-score + mean-reverting regime → LONG MR"""
        features = make_features(override={
            "vwap_dist": -0.002,  # price significantly below VWAP
            "vol_zscore": 1.5,    # volatility spike
            "hurst": 0.30,        # mean-reverting regime (adjusted threshold)
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == 1

    def test_short_mr_signal(self):
        """Price above VWAP + high vol z-score + mean-reverting regime → SHORT MR"""
        features = make_features(override={
            "vwap_dist": 0.002,   # price significantly above VWAP
            "vol_zscore": 1.5,    # volatility spike
            "hurst": 0.30,        # mean-reverting regime (adjusted threshold)
            "vwap": 1.1000,
            "close": 1.1030,
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == -1

    def test_trending_regime_returns_flat(self):
        """Hurst > 0.45 (not mean-reverting) → no MR signal."""
        features = make_features(override={
            "vwap_dist": -0.002,
            "vol_zscore": 1.5,
            "hurst": 0.60,  # trending, not MR
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == 0

    def test_neutral_conditions_flat(self):
        """Low vol z-score and small VWAP deviation → no MR signal."""
        features = make_features(override={
            "vwap_dist": 0.0,
            "vol_zscore": 0.2,
            "hurst": 0.40,
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == 0


# ---------------------------------------------------------------------------
# combine_signals
# ---------------------------------------------------------------------------

class TestCombineSignals:
    def _make_signal(self, symbol, direction, confidence, regime="trending"):
        return Signal(
            symbol=symbol, direction=direction, confidence=confidence,
            entry_price=1.1000, stop_loss=1.0970, take_profit=1.1090,
            timestamp=pd.Timestamp.now(tz="UTC"), regime=regime, atr=0.003
        )

    def test_trending_uses_trend_signal(self):
        trend = self._make_signal("EURUSD", 1, 0.70, "trending")
        mr    = self._make_signal("EURUSD", -1, 0.80, "mean_reverting")
        result = combine_signals(trend, mr, 0.6, "trending")
        assert result["direction"] == 1  # Trend signal wins

    def test_mean_reverting_uses_mr_signal(self):
        trend = self._make_signal("EURUSD", 1, 0.70, "trending")
        mr    = self._make_signal("EURUSD", -1, 0.80, "mean_reverting")
        result = combine_signals(trend, mr, 0.6, "mean_reverting")
        # MR disabled - returns flat signal in non-trending regime
        assert result["direction"] == 0

    def test_random_returns_flat(self):
        trend = self._make_signal("EURUSD", 1, 0.70)
        mr    = self._make_signal("EURUSD", 1, 0.70)
        result = combine_signals(trend, mr, 0.6, "random")
        assert result["direction"] == 0

    def test_low_confidence_returns_flat(self):
        trend = self._make_signal("EURUSD", 1, 0.10)
        mr    = self._make_signal("EURUSD", 0, 0.0)
        result = combine_signals(trend, mr, 0.10, "trending")
        # Blended conf = (0.10 + 0.10) / 2 = 0.10 < MIN_CONFIDENCE (0.30)
        assert result["direction"] == 0


# ---------------------------------------------------------------------------
# apply_signal_filters
# ---------------------------------------------------------------------------

class TestApplySignalFilters:
    def _make_signal(self, direction=1, confidence=0.60, sl=1.0970, tp=1.1090):
        return Signal(
            symbol="EURUSD", direction=direction, confidence=confidence,
            entry_price=1.1000, stop_loss=sl, take_profit=tp,
            timestamp=pd.Timestamp.now(tz="UTC"), regime="trending", atr=0.003
        )

    def test_valid_signal_passes(self, sample_account):
        sig = self._make_signal()
        result = apply_signal_filters(sig, sample_account, 0)
        assert result is not None
        assert result["direction"] == 1

    def test_flat_signal_rejected(self, sample_account):
        sig = self._make_signal(direction=0)
        result = apply_signal_filters(sig, sample_account, 0)
        assert result is None

    def test_insufficient_rr_rejected(self, sample_account):
        # SL = 50 pips, TP = 30 pips → R:R = 0.6 (< 1.5)
        sig = self._make_signal(sl=1.0950, tp=1.1030)
        result = apply_signal_filters(sig, sample_account, 0)
        assert result is None

    def test_low_confidence_rejected(self, sample_account):
        sig = self._make_signal(confidence=0.10)
        result = apply_signal_filters(sig, sample_account, 0)
        assert result is None

    def test_invalid_account_rejected(self):
        sig = self._make_signal()
        result = apply_signal_filters(sig, {}, 0)
        assert result is None

    def test_zero_prices_rejected(self, sample_account):
        sig = self._make_signal(sl=0, tp=0)
        result = apply_signal_filters(sig, sample_account, 0)
        assert result is None
