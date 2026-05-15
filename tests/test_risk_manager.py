# tests/test_risk_manager.py — Unit Tests for risk_manager.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

import risk_manager
from risk_manager import (
    calculate_position_size,
    check_daily_drawdown,
    check_position_limit,
    check_slippage,
    reset_daily_session,
    validate_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_account():
    return {
        "login":       123456,
        "balance":     10000.0,
        "equity":      10000.0,
        "margin":      0.0,
        "free_margin": 10000.0,
        "profit":      0.0,
        "leverage":    100,
        "currency":    "USD",
        "server":      "Test-Server",
    }


@pytest.fixture
def sample_symbol_info():
    return {
        "symbol":              "EURUSD",
        "digits":              5,
        "point":               0.00001,
        "trade_tick_size":     0.00001,
        "trade_tick_value":    1.0,
        "trade_contract_size": 100000,
        "volume_min":          0.01,
        "volume_max":          100.0,
        "volume_step":         0.01,
        "spread":              10,
        "currency_base":       "EUR",
        "currency_profit":     "USD",
    }


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

class TestCalculatePositionSize:
    def test_returns_float(self, sample_symbol_info):
        result = calculate_position_size(10000.0, 20.0, sample_symbol_info)
        assert isinstance(result, float)

    def test_1pct_risk(self, sample_symbol_info):
        """1% of 10,000 = $100 risk. With 20 pip SL, result should be > 0."""
        result = calculate_position_size(10000.0, 20.0, sample_symbol_info)
        assert result > 0

    def test_zero_sl_returns_zero(self, sample_symbol_info):
        result = calculate_position_size(10000.0, 0.0, sample_symbol_info)
        assert result == 0.0

    def test_negative_sl_returns_zero(self, sample_symbol_info):
        result = calculate_position_size(10000.0, -5.0, sample_symbol_info)
        assert result == 0.0

    def test_result_within_volume_bounds(self, sample_symbol_info):
        result = calculate_position_size(10000.0, 20.0, sample_symbol_info)
        assert sample_symbol_info["volume_min"] <= result <= sample_symbol_info["volume_max"]

    def test_larger_balance_larger_position(self, sample_symbol_info):
        small = calculate_position_size(10000.0, 20.0, sample_symbol_info)
        large = calculate_position_size(100000.0, 20.0, sample_symbol_info)
        assert large > small

    def test_larger_sl_smaller_position(self, sample_symbol_info):
        tight = calculate_position_size(10000.0, 10.0, sample_symbol_info)
        wide  = calculate_position_size(10000.0, 50.0, sample_symbol_info)
        assert tight >= wide  # Wider SL = smaller lot


# ---------------------------------------------------------------------------
# Daily Drawdown
# ---------------------------------------------------------------------------

class TestCheckDailyDrawdown:
    def setup_method(self):
        reset_daily_session(10000.0)

    def test_no_drawdown_allows_trade(self):
        assert check_daily_drawdown(10000.0) is True

    def test_small_drawdown_allows_trade(self):
        assert check_daily_drawdown(9700.0) is True  # 3% drawdown, under 4% limit

    def test_exact_limit_blocks_trade(self):
        assert check_daily_drawdown(9600.0) is False  # Exactly 4% drawdown

    def test_beyond_limit_blocks_trade(self):
        assert check_daily_drawdown(9500.0) is False  # 5% drawdown

    def test_profit_always_allows_trade(self):
        assert check_daily_drawdown(10500.0) is True  # In profit


# ---------------------------------------------------------------------------
# Position Limit
# ---------------------------------------------------------------------------

class TestCheckPositionLimit:
    def test_below_limit_allowed(self):
        assert check_position_limit(0) is True
        assert check_position_limit(4) is True

    def test_at_limit_blocked(self):
        assert check_position_limit(5) is False

    def test_above_limit_blocked(self):
        assert check_position_limit(10) is False


# ---------------------------------------------------------------------------
# Slippage Guard
# ---------------------------------------------------------------------------

class TestCheckSlippage:
    def test_no_slippage_passes(self):
        assert check_slippage(1.10000, 1.10000, 0.00001) is True

    def test_tiny_slippage_passes(self):
        assert check_slippage(1.10000, 1.10002, 0.00001) is True  # 0.2 pips

    def test_excessive_slippage_fails(self):
        # 5 pips = 0.00050. MAX_SLIPPAGE_PIPS = 3 → should fail
        assert check_slippage(1.10000, 1.10050, 0.00001) is False

    def test_exact_limit_passes(self):
        # Exactly 3 pips
        assert check_slippage(1.10000, 1.10030, 0.00001) is True


# ---------------------------------------------------------------------------
# validate_order
# ---------------------------------------------------------------------------

class TestValidateOrder:
    def setup_method(self):
        reset_daily_session(10000.0)

    @patch("risk_manager._get_current_price")
    def test_valid_long_order_passes(self, mock_get_price, sample_account, sample_symbol_info):
        mock_get_price.return_value = 1.10050

        result = validate_order(
            symbol="EURUSD",
            direction=1,
            stop_loss=1.09800,  # ~25 pips SL
            account_info=sample_account,
            open_positions_count=0,
            symbol_info=sample_symbol_info,
        )
        assert result is True

    @patch("risk_manager._get_current_price")
    def test_invalid_direction_fails(self, mock_get_price, sample_account, sample_symbol_info):
        result = validate_order(
            symbol="EURUSD",
            direction=0,  # FLAT — invalid
            stop_loss=1.09800,
            account_info=sample_account,
            open_positions_count=0,
            symbol_info=sample_symbol_info,
        )
        assert result is False

    @patch("risk_manager._get_current_price")
    def test_no_stop_loss_fails(self, mock_get_price, sample_account, sample_symbol_info):
        result = validate_order(
            symbol="EURUSD",
            direction=1,
            stop_loss=0,  # No SL — FORBIDDEN
            account_info=sample_account,
            open_positions_count=0,
            symbol_info=sample_symbol_info,
        )
        assert result is False

    @patch("risk_manager._get_current_price")
    def test_drawdown_limit_blocks_order(self, mock_get_price, sample_account, sample_symbol_info):
        reset_daily_session(10000.0)
        sample_account_dd = {**sample_account, "equity": 9500.0}  # 5% DD

        result = validate_order(
            symbol="EURUSD",
            direction=1,
            stop_loss=1.09800,
            account_info=sample_account_dd,
            open_positions_count=0,
            symbol_info=sample_symbol_info,
        )
        assert result is False

    @patch("risk_manager._get_current_price")
    def test_position_limit_blocks_order(self, mock_get_price, sample_account, sample_symbol_info):
        mock_get_price.return_value = 1.10050

        result = validate_order(
            symbol="EURUSD",
            direction=1,
            stop_loss=1.09800,
            account_info=sample_account,
            open_positions_count=5,  # At MAX_CONCURRENT_POSITIONS
            symbol_info=sample_symbol_info,
        )
        assert result is False
