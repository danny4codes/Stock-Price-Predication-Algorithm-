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
    """Make a minimal features DataFrame for signal tests."""
    rng = np.random.default_rng(99)
    base = {
        "close":      1.1000 + np.cumsum(rng.normal(0, 0.001, n)),
        "rsi":        np.full(n, 45.0),
        "macd":       np.full(n, 0.0002),
        "macd_signal":np.full(n, 0.0001),
        "macd_diff":  np.full(n, 0.0001),
        "ema_20":     np.full(n, 1.1010),
        "ema_50":     np.full(n, 1.1000),
        "ema_200":    np.full(n, 1.0980),
        "ema_cross":  np.full(n, 0.0010),   # ema_20 > ema_50 → bullish
        "adx":        np.full(n, 30.0),      # Strong trend
        "adx_pos":    np.full(n, 20.0),
        "adx_neg":    np.full(n, 10.0),
        "bb_upper":   np.full(n, 1.1050),
        "bb_mid":     np.full(n, 1.1000),
        "bb_lower":   np.full(n, 1.0950),
        "bb_pct":     np.full(n, 0.5),
        "atr":        np.full(n, 0.0030),
        "stoch_k":    np.full(n, 50.0),
        "stoch_d":    np.full(n, 48.0),
        "hurst":      np.full(n, 0.60),
        "log_return": rng.normal(0, 0.001, n),
    }
    if override:
        for k, v in override.items():
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
        assert detect_market_regime(0.40) == "mean_reverting"

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
        """ADX > 25, EMA cross positive, MACD diff positive, RSI < 70 → LONG"""
        features = make_features(override={
            "adx": 35.0, "ema_cross": 0.0010, "macd_diff": 0.0002, "rsi": 50.0
        })
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == 1

    def test_short_signal_conditions(self):
        """ADX > 25, EMA cross negative, MACD diff negative, RSI > 30 → SHORT"""
        features = make_features(override={
            "adx": 35.0, "ema_cross": -0.0010, "macd_diff": -0.0002, "rsi": 50.0
        })
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == -1

    def test_weak_adx_returns_flat(self):
        """ADX < 25 = no trend signal."""
        features = make_features(override={"adx": 15.0})
        sig = generate_trend_signal(features, "EURUSD")
        assert sig["direction"] == 0

    def test_signal_has_valid_sl_tp(self):
        features = make_features(override={
            "adx": 35.0, "ema_cross": 0.0010, "macd_diff": 0.0002, "rsi": 50.0
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
            "adx": 35.0, "ema_cross": 0.0010, "macd_diff": 0.0002, "rsi": 50.0
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
        """Price below lower band, RSI oversold, stoch_k crossing UP above stoch_d → LONG MR"""
        features = make_features(override={
            "bb_pct": 0.02, "rsi": 28.0, "stoch_k": 22.0, "stoch_d": 18.0  # stoch_k > stoch_d = bullish cross
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == 1

    def test_short_mr_signal(self):
        """Price above upper band, RSI overbought, stoch_k crossing DOWN below stoch_d → SHORT MR"""
        features = make_features(override={
            "bb_pct": 0.97, "rsi": 72.0, "stoch_k": 76.0, "stoch_d": 82.0  # stoch_k < stoch_d = bearish cross, stoch_k > 75
        })
        sig = generate_mean_reversion_signal(features, "EURUSD")
        assert sig["direction"] == -1

    def test_neutral_conditions_flat(self):
        features = make_features(override={"bb_pct": 0.5, "rsi": 50.0, "stoch_k": 50.0})
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
        assert result["direction"] == -1  # MR signal wins

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
