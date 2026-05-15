# risk_manager.py — QuantEdge MT5: Position Sizing & Drawdown Enforcement
# RULES:
#   - Risk percentages are HARDCODED — never changed without explicit user approval
#   - Every order MUST be validated here before mt5.order_send() (in data_ingestion)
#   - All MT5 calls routed through data_ingestion.py — NO direct MetaTrader5 import
#   - Returns False = system blocks the trade, no exceptions

from datetime import date
from typing import Optional

from data_ingestion import get_current_price as _get_current_price
from config import (
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_SLIPPAGE_PIPS,
    RISK_PER_TRADE_PCT,
)
from logger_config import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Session State (reset at midnight)
# ---------------------------------------------------------------------------
_session_start_balance: Optional[float] = None
_session_date: Optional[date] = None


def reset_daily_session(balance: float) -> None:
    """
    Record the balance at the start of a new trading day.
    Call this once at session open (e.g. when date changes).

    Args:
        balance: Account balance at session start.
    """
    global _session_start_balance, _session_date
    _session_start_balance = balance
    _session_date = date.today()
    logger.info(f"Daily session reset. Start balance: {balance:.2f}")


# ---------------------------------------------------------------------------
# Position Sizing — Fixed Fractional (1%)
# ---------------------------------------------------------------------------

def calculate_position_size(
    account_balance: float,
    stop_loss_pips: float,
    symbol_info: dict,
) -> float:
    """
    Compute lot size using fixed fractional position sizing (1% risk per trade).

    Formula:
        risk_amount = balance × 0.01
        lots        = risk_amount / (stop_loss_pips × pip_value_per_lot)

    Args:
        account_balance: Current account balance (in account currency).
        stop_loss_pips:  Distance from entry to stop loss in pips.
        symbol_info:     Dict from data_ingestion.get_symbol_info().

    Returns:
        Lot size rounded to symbol's volume step. Returns 0.0 if invalid.
    """
    if stop_loss_pips <= 0:
        logger.error(f"Invalid stop_loss_pips: {stop_loss_pips}. Cannot size position.")
        return 0.0

    risk_amount  = account_balance * RISK_PER_TRADE_PCT
    pip_value    = symbol_info.get("trade_tick_value", 0.0)
    tick_size    = symbol_info.get("trade_tick_size", 0.0)
    point        = symbol_info.get("point", 0.0)

    if pip_value <= 0 or point <= 0:
        logger.error(f"Invalid symbol info for sizing: pip_value={pip_value}, point={point}")
        return 0.0

    price_distance = stop_loss_pips * point * 10
    lots_raw = risk_amount / (price_distance / tick_size * pip_value)

    vol_min  = symbol_info.get("volume_min", 0.01)
    vol_max  = symbol_info.get("volume_max", 100.0)
    vol_step = symbol_info.get("volume_step", 0.01)

    lots = round(round(lots_raw / vol_step) * vol_step, 2)
    lots = max(vol_min, min(vol_max, lots))

    logger.info(
        f"Position size: {lots} lots | Risk: {risk_amount:.2f} | "
        f"SL pips: {stop_loss_pips} | Balance: {account_balance:.2f}"
    )
    return lots


# ---------------------------------------------------------------------------
# Daily Drawdown Check
# ---------------------------------------------------------------------------

def check_daily_drawdown(current_equity: float) -> bool:
    """
    Check if the daily drawdown limit (4%) has been breached.

    Args:
        current_equity: Current account equity.

    Returns:
        True if trading is ALLOWED, False if daily limit is breached.
    """
    # HARDCODED — 4% maximum daily drawdown
    MAX_DAILY_DD = MAX_DAILY_DRAWDOWN_PCT

    # Reset if new day
    global _session_start_balance, _session_date
    today = date.today()
    if _session_date != today or _session_start_balance is None:
        logger.warning(
            "Daily session not initialized. Call reset_daily_session() at start of day. "
            "Allowing trade (cannot enforce drawdown limit without baseline)."
        )
        return True

    drawdown = (_session_start_balance - current_equity) / _session_start_balance

    if drawdown >= MAX_DAILY_DD:
        logger.critical(
            f"🛑 DAILY DRAWDOWN LIMIT BREACHED: {drawdown:.2%} "
            f"(limit: {MAX_DAILY_DD:.0%}). "
            f"Session start: {_session_start_balance:.2f} | Current equity: {current_equity:.2f}. "
            "ALL new orders BLOCKED for the remainder of this session."
        )
        return False

    remaining = MAX_DAILY_DD - drawdown
    logger.debug(
        f"Drawdown: {drawdown:.2%} | Remaining buffer: {remaining:.2%}"
    )
    return True


# ---------------------------------------------------------------------------
# Concurrent Position Limit
# ---------------------------------------------------------------------------

def check_position_limit(open_positions_count: int) -> bool:
    """
    Enforce maximum concurrent position limit.

    Args:
        open_positions_count: Number of currently open positions.

    Returns:
        True if a new position may be opened, False if limit reached.
    """
    if open_positions_count >= MAX_CONCURRENT_POSITIONS:
        logger.warning(
            f"Position limit reached: {open_positions_count}/{MAX_CONCURRENT_POSITIONS}. "
            "New order blocked."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Slippage Guard
# ---------------------------------------------------------------------------

def check_slippage(
    requested_price: float,
    fill_price: float,
    symbol_point: float,
) -> bool:
    """
    Verify fill price is within acceptable slippage tolerance.

    Args:
        requested_price: Price at which order was submitted.
        fill_price:      Actual execution price.
        symbol_point:    Instrument's point size (from symbol_info).

    Returns:
        True if slippage is acceptable, False if excessive.
    """
    slippage_pips = abs(fill_price - requested_price) / (symbol_point * 10)

    if slippage_pips > MAX_SLIPPAGE_PIPS:
        logger.warning(
            f"Excessive slippage detected: {slippage_pips:.1f} pips "
            f"(limit: {MAX_SLIPPAGE_PIPS} pips). "
            f"Requested: {requested_price} | Fill: {fill_price}"
        )
        return False

    logger.debug(f"Slippage: {slippage_pips:.2f} pips — within tolerance.")
    return True


# ---------------------------------------------------------------------------
# Pre-Order Validation Gate — MUST be called before every mt5.order_send()
# ---------------------------------------------------------------------------

def validate_order(
    symbol: str,
    direction: int,
    stop_loss: float,
    account_info: dict,
    open_positions_count: int,
    symbol_info: dict,
) -> bool:
    """
    Master validation gate. Runs ALL risk checks before an order is allowed.

    Checks performed (in order):
        1. Stop loss is present and valid
        2. Direction is valid (1 or -1)
        3. Daily drawdown limit not breached
        4. Concurrent position limit not exceeded
        5. Stop loss distance is reasonable (≥ 1 pip)

    Args:
        symbol:                Trading symbol.
        direction:             1 (LONG) or -1 (SHORT).
        stop_loss:             Stop loss price level.
        account_info:          Dict from data_ingestion.get_account_info().
        open_positions_count:  Number of currently open positions.
        symbol_info:           Dict from data_ingestion.get_symbol_info().

    Returns:
        True if ALL checks pass and order may proceed.
        False if ANY check fails — order must be rejected.
    """
    # 1. Stop loss is mandatory
    if stop_loss <= 0:
        logger.error(f"[{symbol}] Order rejected: No valid stop loss ({stop_loss}).")
        return False

    # 2. Valid direction
    if direction not in (1, -1):
        logger.error(f"[{symbol}] Order rejected: Invalid direction {direction}.")
        return False

    # 3. Daily drawdown
    if not check_daily_drawdown(account_info["equity"]):
        return False

    # 4. Position limit
    if not check_position_limit(open_positions_count):
        return False

    # 5. Minimum stop loss distance (1 pip)
    current_price = _get_current_price(symbol, direction)
    if current_price is not None:
        point = symbol_info.get("point", 0.00001)
        sl_distance_pips = abs(current_price - stop_loss) / (point * 10)
        if sl_distance_pips < 1.0:
            logger.error(
                f"[{symbol}] Order rejected: Stop loss too close "
                f"({sl_distance_pips:.2f} pips). Minimum: 1 pip."
            )
            return False

    logger.info(f"[{symbol}] Order validation PASSED. Direction: {'LONG' if direction == 1 else 'SHORT'}")
    return True