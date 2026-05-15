# tests/test_microstructure_engine.py — Unit Tests for 2_microstructure_engine.py
#
# Covers:
#   - DirectionalChangeDetector: upturn/downturn events, extremum tracking, edge cases
#   - calculate_order_flow_imbalance: up/down/mixed tick flows, edge cases
#   - evaluate_primary_signal: all boolean logic paths
#   - evaluate_primary_signals_batch: vectorised signal generation
#   - classify_regime: Hurst-based regime classification
#   - run_microstructure_pipeline: full end-to-end pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import importlib
_mc = importlib.import_module("2_microstructure_engine")
DirectionalChangeDetector    = _mc.DirectionalChangeDetector
calculate_order_flow_imbalance = _mc.calculate_order_flow_imbalance
PrimarySignalResult           = _mc.PrimarySignalResult
classify_regime               = _mc.classify_regime
evaluate_primary_signal       = _mc.evaluate_primary_signal
evaluate_primary_signals_batch = _mc.evaluate_primary_signals_batch
run_microstructure_pipeline   = _mc.run_microstructure_pipeline
from config import DC_THETA, HURST_TRENDING_THRESHOLD, HURST_MEAN_REVERT_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_trending_prices(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate a trending price series with guaranteed directional changes."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 0.10, n)
    noise = rng.normal(0, 0.001, n)
    prices = 1.1000 + np.cumsum(trend + noise)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.Series(prices, index=idx, name="close")


def make_mean_reverting_prices(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate an oscillating price series (no strong trend)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.001, n)
    prices = 1.1000 + np.cumsum(noise)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture
def trending_prices():
    return make_trending_prices()


@pytest.fixture
def ohlcv_from_prices(trending_prices):
    """Build a minimal OHLCV DataFrame from a close price series."""
    rng = np.random.default_rng(42)
    close = trending_prices.values
    high = close + rng.uniform(0.0002, 0.0010, len(close))
    low = close - rng.uniform(0.0002, 0.0010, len(close))
    open_ = close + rng.normal(0, 0.0003, len(close))
    volume = rng.integers(100, 5000, len(close)).astype(float)

    idx = trending_prices.index
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ===========================================================================
# 1. DIRECTIONAL CHANGE DETECTOR
# ===========================================================================

class TestDirectionalChangeDetector:
    """Tests for the DirectionalChangeDetector class."""

    # --- Initialization ---
    def test_initialization(self):
        detector = DirectionalChangeDetector(theta=0.005)
        assert detector.theta == 0.005
        assert detector.last_extremum_price is None
        assert detector.last_extremum_type is None
        assert detector.event_count == 0

    def test_default_theta(self):
        detector = DirectionalChangeDetector()
        assert detector.theta == DC_THETA

    def test_invalid_theta_raises(self):
        with pytest.raises(ValueError, match="theta must be positive"):
            DirectionalChangeDetector(theta=0)
        with pytest.raises(ValueError, match="theta must be positive"):
            DirectionalChangeDetector(theta=-0.01)

    def test_invalid_min_price_raises(self):
        with pytest.raises(ValueError, match="min_price must be positive"):
            DirectionalChangeDetector(min_price=0)

    # --- First tick ---
    def test_first_tick_no_event(self):
        detector = DirectionalChangeDetector(theta=0.01)
        result = detector.process_tick(1.1000, 1000, None)
        assert result["event_type"] is None
        assert result["price"] == 1.1000
        assert result["volume"] == 1000
        assert result["signal_strength"] == 0.0
        assert result["extremum_type"] == "min"
        assert detector.event_count == 0

    def test_first_tick_sets_min(self):
        detector = DirectionalChangeDetector(theta=0.01)
        result = detector.process_tick(1.1000)
        assert detector.last_extremum_price == 1.1000
        assert detector.last_extremum_type == "min"

    # --- Upturn events ---
    def test_upturn_event_basic(self):
        """Price rises by > theta from local min → upturn fires."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)      # init min
        result = detector.process_tick(1.1200)  # +1.818% > 1%
        assert result["event_type"] == "upturn"
        # signal_strength = min(Δ/θ, 2.0)  where Δ = 0.02/1.10 = 0.01818
        # strength = min(0.01818/0.01, 2.0) = min(1.818, 2.0) = 1.818
        assert result["signal_strength"] == pytest.approx(1.818, abs=0.01)
        assert detector.last_extremum_type == "max"
        assert detector.last_extremum_price == pytest.approx(1.1200)
        assert detector.event_count == 1

    def test_downturn_after_upturn(self):
        """Full cycle: upturn → price falls → downturn fires."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)  # init min
        r1 = detector.process_tick(1.1200)  # upturn
        assert r1["event_type"] == "upturn"
        # Now extremum is max at 1.12. Drop by 1.09% → barely below theta
        r2 = detector.process_tick(1.1078)  # Δ = -0.0109 → downtrun
        assert r2["event_type"] == "downturn"
        assert detector.event_count == 2

    def test_upturn_below_threshold_no_event(self):
        """Price rises but not enough → no event."""
        detector = DirectionalChangeDetector(theta=0.05)
        detector.process_tick(1.1000)
        # Need price > 1.1000 * 1.05 = 1.155 to fire
        result = detector.process_tick(1.1030)  # +0.27% < 5%
        assert result["event_type"] is None
        assert detector.event_count == 0

    def test_upturn_then_track_new_max(self):
        """After upturn, continuing rise updates the max without firing event."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        r1 = detector.process_tick(1.1200)  # upturn, max=1.12
        assert r1["event_type"] == "upturn"
        # Price continues up — just extends the max
        r2 = detector.process_tick(1.1300)  # still from 1.12 base
        assert r2["event_type"] is None
        assert detector.last_extremum_price == pytest.approx(1.1300)
        assert detector.last_extremum_type == "max"
        assert detector.event_count == 1  # only the first upturn

    # --- Downturn events ---
    def test_downturn_event_basic(self):
        """Price falls by > theta from local max → downturn fires."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)       # init min
        detector.process_tick(1.1200)       # upturn → now max at 1.12
        # Δ = (1.105 - 1.12) / 1.12 = -0.01339  (> 1% in abs)
        result = detector.process_tick(1.1050)
        assert result["event_type"] == "downturn"
        assert result["signal_strength"] == pytest.approx(1.339, abs=0.01)
        assert detector.last_extremum_type == "min"
        assert detector.event_count == 2  # upturn + downturn

    def test_downturn_below_threshold_no_event(self):
        """Price falls but not enough → no event."""
        detector = DirectionalChangeDetector(theta=0.05)
        detector.process_tick(1.1000)
        detector.process_tick(1.1200)  # upturn
        result = detector.process_tick(1.1150)  # -0.45% from max < 5%
        assert result["event_type"] is None

    # --- Extremum tracking ---
    def test_new_minimum_updated(self):
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        result = detector.process_tick(1.0900)
        assert result["event_type"] is None
        assert detector.last_extremum_price == pytest.approx(1.0900)
        assert detector.last_extremum_type == "min"

    def test_new_maximum_updated(self):
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        detector.process_tick(1.1200)  # upturn, now max
        r = detector.process_tick(1.1300)
        assert r["event_type"] is None  # no threshold breach yet
        assert detector.last_extremum_price == pytest.approx(1.1300)
        assert detector.last_extremum_type == "max"

    # --- Signal strength clamping ---
    def test_signal_strength_capped_at_2(self):
        """signal_strength is min(Δ/θ, 2.0)."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        # 3% move with 1% theta → 3.0 capped to 2.0
        result = detector.process_tick(1.1300)  # Δ=0.03/1.10=0.02727, /0.01=2.727 → 2.0
        assert result["signal_strength"] == 2.0

    def test_signal_strength_proportional(self):
        """signal_strength = Δ/θ when Δ/θ ≤ 2 and Δ ≥ θ."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        # Need a move ≥ 1% (= theta); use 1.112 to avoid float-rounding edge
        # Δ = 0.012/1.100 ≈ 0.01091  →  strength ≈ 1.091
        result = detector.process_tick(1.1120)
        assert result["signal_strength"] == pytest.approx(1.091, abs=0.01)

    def test_sub_theta_move_no_signal(self):
        """Moves below theta produce zero signal strength."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        result = detector.process_tick(1.1050)  # 0.45% < 1%
        assert result["signal_strength"] == 0.0

    def test_signal_strength_symmetric_downturn(self):
        """Downturn strength uses abs(Δ)/θ."""
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        detector.process_tick(1.1200)  # upturn, max=1.12
        result = detector.process_tick(1.1050)  # -1.34% → strength≈1.34
        assert result["signal_strength"] == pytest.approx(1.339, abs=0.01)

    # --- Reset ---
    def test_reset_clears_state(self):
        detector = DirectionalChangeDetector(theta=0.01)
        detector.process_tick(1.1000)
        detector.process_tick(1.1200)
        assert detector.event_count == 1
        detector.reset()
        assert detector.last_extremum_price is None
        assert detector.last_extremum_type is None
        assert detector.event_count == 0

    # --- Edge cases ---
    def test_zero_price_skipped(self):
        detector = DirectionalChangeDetector(theta=0.01)
        result = detector.process_tick(0.0, 0, None)
        assert result["event_type"] is None

    def test_negative_price_skipped(self):
        detector = DirectionalChangeDetector(theta=0.01)
        result = detector.process_tick(-1.0, 0, None)
        assert result["event_type"] is None

    def test_existing_min_price_boundary(self):
        detector = DirectionalChangeDetector(theta=0.01, min_price=0.0001)
        result = detector.process_tick(0.0001, 0, None)
        assert result["event_type"] is None

    def test_below_min_price_skipped(self):
        detector = DirectionalChangeDetector(theta=0.01, min_price=0.01)
        result = detector.process_tick(0.005, 0, None)
        assert result["event_type"] is None

    # --- process_series end-to-end ---
    def test_process_series_returns_dataframe(self):
        detector = DirectionalChangeDetector(theta=0.01)
        prices = pd.Series([1.0, 1.05, 1.10, 1.08, 1.13],
                           index=pd.date_range("2024-01-01", periods=5, tz="UTC"))
        result = detector.process_series(prices)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert set(result.columns) >= {
            "event_type", "price", "signal_strength", "extremum_type"
        }

    def test_process_series_with_volumes(self):
        detector = DirectionalChangeDetector(theta=0.01)
        prices = pd.Series([1.0, 1.05, 1.10, 1.08, 1.13],
                           index=pd.date_range("2024-01-01", periods=5, tz="UTC"))
        volumes = pd.Series([100, 200, 150, 300, 250], index=prices.index)
        result = detector.process_series(prices, volumes=volumes)
        assert "volume" in result.columns
        assert list(result["volume"]) == [100, 200, 150, 300, 250]

    def test_process_series_empty(self):
        detector = DirectionalChangeDetector(theta=0.01)
        prices = pd.Series([], dtype=float)
        result = detector.process_series(prices)
        assert result.empty

    def test_process_series_realistic_trending(self, trending_prices):
        """On a strongly trending series, expect multiple upturn events."""
        detector = DirectionalChangeDetector(theta=0.002)
        result = detector.process_series(trending_prices)
        events = result["event_type"].dropna()
        assert len(events) > 0
        first_event = events.iloc[0]
        assert first_event == "upturn"


# ===========================================================================
# 2. ORDER FLOW IMBALANCE (OFI)
# ===========================================================================

class TestOrderFlowImbalance:
    """Tests for the calculate_order_flow_imbalance function."""

    def test_up_ticks_positive_ofi(self):
        prices = pd.Series([1.0, 1.01, 1.02, 1.03])
        volumes = pd.Series([100.0, 100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=3)
        assert result.iloc[-1] > 0

    def test_up_ticks_all_volume_up(self):
        """Monotonically rising prices → OFI should be 1.0 per tick."""
        prices = pd.Series([1.0, 1.01, 1.02])
        volumes = pd.Series([100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=1)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_down_ticks_negative_ofi(self):
        prices = pd.Series([1.03, 1.02, 1.01, 1.0])
        volumes = pd.Series([100.0, 100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=3)
        assert result.iloc[-1] < 0

    def test_down_ticks_all_volume_down(self):
        prices = pd.Series([1.02, 1.01, 1.0])
        volumes = pd.Series([100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=1)
        assert result.iloc[-1] == pytest.approx(-1.0)

    def test_mixed_ticks_near_zero(self):
        prices = pd.Series([1.0, 1.01, 1.0, 1.01, 1.0])
        volumes = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=5)
        assert abs(result.iloc[-1]) < 0.5

    def test_zero_change_ticks_split_volume(self):
        """Zero-price ticks should split volume equally."""
        prices = pd.Series([1.0, 1.0, 1.0, 1.02])
        volumes = pd.Series([100.0, 100.0, 100.0, 100.0])
        result = calculate_order_flow_imbalance(prices, volumes, window=4)
        assert result.iloc[-1] > 0
        assert result.iloc[-1] < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have same length"):
            calculate_order_flow_imbalance(
                pd.Series([1, 2, 3]), pd.Series([1, 2])
            )

    def test_single_price_returns_zero(self):
        result = calculate_order_flow_imbalance(
            pd.Series([1.0]), pd.Series([100.0])
        )
        assert len(result) == 1
        assert result.iloc[0] == 0.0

    def test_empty_input(self):
        result = calculate_order_flow_imbalance(
            pd.Series([], dtype=float), pd.Series([], dtype=float)
        )
        assert result.empty

    def test_none_input_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            calculate_order_flow_imbalance(None, pd.Series([1.0]))
        with pytest.raises(ValueError, match="must not be None"):
            calculate_order_flow_imbalance(pd.Series([1.0]), None)

    def test_large_window_smooths(self):
        prices_up = pd.Series([1.0 + i * 0.01 for i in range(50)])
        vols = pd.Series([100.0] * 50)
        prices_spike = prices_up.copy()
        prices_spike.iloc[-1] += 0.05

        ofi_small = calculate_order_flow_imbalance(prices_spike, vols, window=3)
        ofi_large = calculate_order_flow_imbalance(prices_spike, vols, window=20)

        assert abs(ofi_large.iloc[-1]) <= abs(ofi_small.iloc[-1]) + 1e-10


# ===========================================================================
# 3. CLASSIFY REGIME
# ===========================================================================

class TestClassifyRegime:
    """Tests for the Hurst-based regime classification."""

    def test_trending_above_threshold(self):
        assert classify_regime(0.60) == "trending"
        assert classify_regime(0.90) == "trending"
        assert classify_regime(0.55 + 1e-9) == "trending"

    def test_mean_reverting_below_threshold(self):
        assert classify_regime(0.40) == "mean_reverting"
        assert classify_regime(0.10) == "mean_reverting"
        assert classify_regime(0.45 - 1e-9) == "mean_reverting"

    def test_random_between_thresholds(self):
        assert classify_regime(0.50) == "random"
        assert classify_regime(0.55) == "random"
        assert classify_regime(0.45) == "random"

    def test_boundary_consistency(self):
        assert classify_regime(HURST_TRENDING_THRESHOLD + 0.001) == "trending"
        assert classify_regime(HURST_MEAN_REVERT_THRESHOLD - 0.001) == "mean_reverting"


# ===========================================================================
# 4. EVALUATE PRIMARY SIGNAL (single-tick)
# ===========================================================================

class TestEvaluatePrimarySignal:
    """Tests for the evaluate_primary_signal function."""

    # --- Primary Buy conditions ---
    def test_primary_buy_all_conditions_met(self):
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="upturn", ofi=0.15
        )
        assert result.buy is True
        assert result.sell is False
        assert result.regime == "trending"
        assert "Primary_BUY" in result.reason

    def test_primary_buy_strength_calculation(self):
        result = evaluate_primary_signal(
            hurst=0.70, dc_event="upturn", ofi=0.50
        )
        assert result.buy is True
        # hurst_conviction = (0.70 - 0.5) * 2 = 0.40
        # ofi_conviction = min(0.50, 1.0) = 0.50
        # strength = round(0.40 * 0.5 + 0.50 * 0.5, 3) = 0.45
        assert result.buy_strength == pytest.approx(0.45, abs=0.01)
        assert 0.0 < result.buy_strength <= 1.0

    def test_primary_buy_trending_but_no_dc(self):
        """No DC event → no buy even if trending."""
        result = evaluate_primary_signal(
            hurst=0.60, dc_event=None, ofi=0.15
        )
        assert result.buy is False

    def test_primary_buy_trending_but_ofi_too_low(self):
        """DC event present but OFI below threshold."""
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="upturn", ofi=0.01
        )
        assert result.buy is False
        assert "OFI" in result.reason.lower() or "below" in result.reason.lower()

    def test_primary_buy_trending_but_negative_ofi(self):
        """DC upturn but negative OFI → no buy."""
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="upturn", ofi=-0.15
        )
        assert result.buy is False

    # --- Primary Sell conditions ---
    def test_primary_sell_all_conditions_met(self):
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="downturn", ofi=-0.15
        )
        assert result.sell is True
        assert result.buy is False
        assert result.regime == "trending"
        assert "Primary_SELL" in result.reason

    def test_primary_sell_strength_calculation(self):
        result = evaluate_primary_signal(
            hurst=0.75, dc_event="downturn", ofi=-0.60
        )
        assert result.sell is True
        # hurst_conviction = (0.75 - 0.5) * 2 = 0.50
        # ofi_conviction = min(0.60, 1.0) = 0.60
        # strength = round(0.50 * 0.5 + 0.60 * 0.5, 3) = 0.55
        assert result.sell_strength == pytest.approx(0.55, abs=0.01)

    def test_primary_sell_no_dc_event(self):
        result = evaluate_primary_signal(
            hurst=0.60, dc_event=None, ofi=-0.15
        )
        assert result.sell is False

    def test_primary_sell_ofi_too_high(self):
        """Downturn but positive OFI → no sell."""
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="downturn", ofi=0.10
        )
        assert result.sell is False

    # --- Regime guards ---
    def test_mean_reverting_no_primary_signal(self):
        """Mean-reverting regime → no primary signals."""
        result = evaluate_primary_signal(
            hurst=0.35, dc_event="upturn", ofi=0.50
        )
        assert result.buy is False
        assert result.sell is False
        assert result.regime == "mean_reverting"

    def test_random_no_primary_signal(self):
        """Random regime → no primary signals."""
        result = evaluate_primary_signal(
            hurst=0.50, dc_event="upturn", ofi=0.50
        )
        assert result.buy is False
        assert result.sell is False
        assert result.regime == "random"

    def test_downturn_in_mean_revert(self):
        """Even with downturn + negative OFI, mean-revert kills the signal."""
        result = evaluate_primary_signal(
            hurst=0.40, dc_event="downturn", ofi=-0.50
        )
        assert result.sell is False

    # --- Custom OFI threshold ---
    def test_custom_ofi_threshold(self):
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="upturn", ofi=0.05,
            ofi_threshold=0.03
        )
        assert result.buy is True

    # --- Result object ---
    def test_result_to_dict(self):
        result = evaluate_primary_signal(
            hurst=0.60, dc_event="upturn", ofi=0.15
        )
        d = result.to_dict()
        assert d["buy"] is True
        assert d["regime"] == "trending"
        assert isinstance(d["buy_strength"], float)

    def test_result_default_state(self):
        result = PrimarySignalResult()
        assert result.buy is False
        assert result.sell is False
        assert result.buy_strength == 0.0
        assert result.sell_strength == 0.0
        assert result.regime == "random"
        assert result.reason == ""


# ===========================================================================
# 5. BATCH EVALUATION — evaluate_primary_signals_batch
# ===========================================================================

class TestEvaluatePrimarySignalsBatch:
    """Tests for the vectorised batch signal evaluator."""

    def _make_features(self, n=50, overrides=None):
        rng = np.random.default_rng(99)
        base = {
            "close":              1.1000 + np.cumsum(rng.normal(0, 0.001, n)),
            "volume":             rng.integers(100, 5000, n).astype(float),
            "hurst":              np.full(n, 0.60),
            "dc_event":           [None] * n,
            "ofi":                np.full(n, 0.0),
            "is_trending":        np.full(n, True),
        }
        if overrides:
            for k, v in overrides.items():
                base[k] = np.full(n, v) if not isinstance(v, list) else v
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        return pd.DataFrame(base, index=idx)

    def test_buy_signals_generated(self):
        df = self._make_features(overrides={"dc_event": "upturn", "ofi": 0.2})
        result = evaluate_primary_signals_batch(df)
        assert result["primary_buy"].sum() > 0

    def test_sell_signals_generated(self):
        df = self._make_features(overrides={"dc_event": "downturn", "ofi": -0.2})
        result = evaluate_primary_signals_batch(df)
        assert result["primary_sell"].sum() > 0

    def test_no_signal_without_dc(self):
        df = self._make_features(overrides={"dc_event": None, "ofi": 0.5})
        result = evaluate_primary_signals_batch(df)
        assert result["primary_buy"].sum() == 0
        assert result["primary_sell"].sum() == 0

    def test_no_signal_without_trending(self):
        df = self._make_features(overrides={
            "dc_event": "upturn", "ofi": 0.5,
            "hurst": 0.40, "is_trending": False,
        })
        result = evaluate_primary_signals_batch(df)
        assert result["primary_buy"].sum() == 0

    def test_no_signal_with_wrong_ofi_direction(self):
        """Upturn + negative OFI should NOT produce buy."""
        df = self._make_features(overrides={
            "dc_event": "upturn", "ofi": -0.3,
        })
        result = evaluate_primary_signals_batch(df)
        assert result["primary_buy"].sum() == 0

    def test_signal_strength_nonzero_on_signals(self):
        df = self._make_features(overrides={"dc_event": "upturn", "ofi": 0.3})
        result = evaluate_primary_signals_batch(df)
        signal_rows = result[result["primary_buy"]]
        assert (signal_rows["signal_strength"] > 0).all()

    def test_signal_strength_zero_on_no_signals(self):
        df = self._make_features(overrides={"dc_event": None})
        result = evaluate_primary_signals_batch(df)
        assert (result["signal_strength"] == 0).all()

    def test_regime_column_added(self):
        df = self._make_features()
        result = evaluate_primary_signals_batch(df)
        assert "regime" in result.columns
        assert (result["regime"] == "trending").all()

    def test_inherits_input_index(self):
        df = self._make_features()
        result = evaluate_primary_signals_batch(df)
        assert list(result.index) == list(df.index)


# ===========================================================================
# 6. FULL PIPELINE — run_microstructure_pipeline
# ===========================================================================

class TestRunMicrostructurePipeline:
    """End-to-end tests for the full microstructure pipeline."""

    def test_pipeline_returns_dataframe(self, ohlcv_from_prices):
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.60
        )
        assert isinstance(result, pd.DataFrame)

    def test_pipeline_adds_expected_columns(self, ohlcv_from_prices):
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.60
        )
        expected_new = {
            "dc_event", "dc_signal_strength", "ofi",
            "primary_buy", "primary_sell", "signal_strength", "regime",
            "hurst", "is_trending",
        }
        assert expected_new.issubset(set(result.columns))

    def test_pipeline_preserves_original_columns(self, ohlcv_from_prices):
        original_cols = set(ohlcv_from_prices.columns)
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.60
        )
        assert original_cols.issubset(set(result.columns))

    def test_pipeline_output_fewer_rows(self, ohlcv_from_prices):
        """Some rows may be lost to rolling window warmup."""
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.60
        )
        assert len(result) <= len(ohlcv_from_prices)

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = run_microstructure_pipeline(df)
        assert result.empty

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"close": [1.0, 1.1], "volume": [100, 200]})
        result = run_microstructure_pipeline(df)
        assert result.empty

    def test_none_input_returns_empty(self):
        result = run_microstructure_pipeline(None)
        assert result.empty

    def test_nontrending_produces_no_primary_signals(self, ohlcv_from_prices):
        """With H=0.40 (mean-reverting), no primary signals should fire."""
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.40
        )
        assert result["primary_buy"].sum() == 0
        assert result["primary_sell"].sum() == 0

    def test_signal_consistency(self, ohlcv_from_prices):
        """Primary signals only fire when all three conditions align."""
        result = run_microstructure_pipeline(
            ohlcv_from_prices, hurst_value=0.60
        )
        if "primary_buy" not in result.columns:
            pytest.fail("primary_buy column missing from pipeline output")

        buy_rows = result[result["primary_buy"]]
        sell_rows = result[result["primary_sell"]]

        if len(buy_rows) > 0:
            assert (buy_rows["dc_event"] == "upturn").all()
            assert (buy_rows["regime"] == "trending").all()
        if len(sell_rows) > 0:
            assert (sell_rows["dc_event"] == "downturn").all()
            assert (sell_rows["regime"] == "trending").all()