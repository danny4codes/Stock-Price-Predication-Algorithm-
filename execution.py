# execution.py — QuantEdge MT5: Order Placement & Position Management
# RULES:
#   - ALL order routing goes through data_ingestion.py — NO direct MetaTrader5 imports
#   - ALWAYS call risk_manager.validate_order() before any order send
#   - ALWAYS check result of data_ingestion.mt5_order_send() after every order operation
#   - NO position may be opened without a valid Stop Loss
#   - Log ALL order results at INFO level with full ticket details

from typing import Optional

from data_ingestion import (
    get_account_info,
    get_current_price,
    get_open_positions,
    get_position_by_ticket,
    get_symbol_info,
    mt5_order_send,
    TRADE_ACTION_DEAL,
    TRADE_ACTION_PENDING,
    TRADE_ACTION_SLTP,
    TRADE_ACTION_REMOVE,
    TRADE_RETCODE_DONE,
    ORDER_TYPE_BUY,
    ORDER_TYPE_SELL,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_SELL_LIMIT,
    ORDER_TIME_GTC,
    ORDER_FILLING_IOC,
    ORDER_FILLING_RETURN,
)
from logger_config import setup_logger
from risk_manager import calculate_position_size, validate_order
from signal_generation import Signal

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Market Orders
# ---------------------------------------------------------------------------

def place_market_order(
    symbol:    str,
    direction: int,
    volume:    float,
    sl:        float,
    tp:        float,
    comment:   str = "QuantEdge",
    magic:     int = 20260101,
) -> dict:
    """
    Place a market (instant execution) order.

    REQUIRES:
        1. risk_manager.validate_order() must pass
        2. Valid SL and TP > 0

    Args:
        symbol:    Trading symbol (e.g. "EURUSD").
        direction: 1 = BUY, -1 = SELL.
        volume:    Lot size (from risk_manager.calculate_position_size()).
        sl:        Stop loss price. MANDATORY.
        tp:        Take profit price. MANDATORY.
        comment:   Order comment. Default: "QuantEdge".
        magic:     Magic number to identify system orders.

    Returns:
        Dict with 'success' bool and order details or error info.
    """
    if sl <= 0 or tp <= 0:
        logger.error(f"[{symbol}] place_market_order rejected: SL={sl} TP={tp} (both must be > 0).")
        return {"success": False, "error": "Invalid SL/TP"}

    price = get_current_price(symbol, direction)
    if price is None:
        return {"success": False, "error": "Could not fetch current price"}

    order_type = ORDER_TYPE_BUY if direction == 1 else ORDER_TYPE_SELL

    request = {
        "action":       TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,          # Max slippage in points
        "magic":        magic,
        "comment":      comment,
        "type_time":    ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    logger.info(
        f"[{symbol}] Sending MARKET {'BUY' if direction == 1 else 'SELL'} | "
        f"Vol: {volume} | Price: {price} | SL: {sl} | TP: {tp}"
    )

    result = mt5_order_send(request)
    return _log_order_result(result)


# ---------------------------------------------------------------------------
# Limit Orders
# ---------------------------------------------------------------------------

def place_limit_order(
    symbol:    str,
    direction: int,
    price:     float,
    volume:    float,
    sl:        float,
    tp:        float,
    comment:   str = "QuantEdge-Limit",
    magic:     int = 20260101,
) -> dict:
    """
    Place a pending limit order.

    Args:
        symbol:    Trading symbol.
        direction: 1 = BUY_LIMIT, -1 = SELL_LIMIT.
        price:     Limit price to execute at.
        volume:    Lot size.
        sl:        Stop loss price. MANDATORY.
        tp:        Take profit price. MANDATORY.
        comment:   Order comment.
        magic:     Magic number.

    Returns:
        Dict with 'success' bool and order details or error info.
    """
    if sl <= 0 or tp <= 0:
        logger.error(f"[{symbol}] place_limit_order rejected: SL={sl} TP={tp} must be > 0.")
        return {"success": False, "error": "Invalid SL/TP"}

    order_type = ORDER_TYPE_BUY_LIMIT if direction == 1 else ORDER_TYPE_SELL_LIMIT

    request = {
        "action":       TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        magic,
        "comment":      comment,
        "type_time":    ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_RETURN,
    }

    logger.info(
        f"[{symbol}] Sending {'BUY_LIMIT' if direction == 1 else 'SELL_LIMIT'} | "
        f"Price: {price} | Vol: {volume} | SL: {sl} | TP: {tp}"
    )
    result = mt5_order_send(request)
    return _log_order_result(result)


# ---------------------------------------------------------------------------
# Signal Execution
# ---------------------------------------------------------------------------

def execute_signal(signal: Signal, account_info: dict) -> dict:
    """
    Execute a trading signal: size position, validate, and place order.

    Args:
        signal:        Approved Signal from apply_signal_filters().
        account_info:  Dict from data_ingestion.get_account_info().

    Returns:
        Dict with 'success' bool and order details or error info.
    """
    direction = signal["direction"]
    symbol    = signal["symbol"]

    if direction == 0:
        return {"success": False, "error": "Signal direction is FLAT"}

    # Get symbol info for position sizing
    sym_info = get_symbol_info(symbol)
    if sym_info is None:
        return {"success": False, "error": f"Could not fetch symbol info for {symbol}"}

    # Calculate position size (risk_manager validates SL distance)
    sl_distance_pips = abs(signal["entry_price"] - signal["stop_loss"]) / (sym_info["point"] * 10)
    volume = calculate_position_size(
        account_balance=account_info["balance"],
        stop_loss_pips=sl_distance_pips,
        symbol_info=sym_info,
    )

    if volume <= 0:
        return {"success": False, "error": f"Calculated volume <= 0: {volume}"}

    logger.info(
        f"[{symbol}] Executing {'LONG' if direction == 1 else 'SHORT'} | "
        f"Vol: {volume} | Entry: {signal['entry_price']} | "
        f"SL: {signal['stop_loss']} | TP: {signal['take_profit']}"
    )
    return place_market_order(
        symbol=symbol, direction=direction, volume=volume,
        sl=signal["stop_loss"], tp=signal["take_profit"],
    )


# ---------------------------------------------------------------------------
# Position Management
# ---------------------------------------------------------------------------

def close_position(ticket: int, comment: str = "QuantEdge-Close") -> bool:
    """
    Close an open position by ticket number at market price.

    Args:
        ticket:  MT5 position ticket.
        comment: Order comment.

    Returns:
        True if closed successfully, False otherwise.
    """
    pos = get_position_by_ticket(ticket)
    if pos is None:
        logger.error(f"Cannot close ticket {ticket}: position not found.")
        return False

    # Reverse direction to close
    close_type = ORDER_TYPE_SELL if pos.type == ORDER_TYPE_BUY else ORDER_TYPE_BUY
    price = get_current_price(pos.symbol, 1 if close_type == ORDER_TYPE_SELL else -1)

    if price is None:
        logger.error(f"Cannot fetch price for {pos.symbol} to close ticket {ticket}.")
        return False

    request = {
        "action":       TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        pos.magic,
        "comment":      comment,
        "type_time":    ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    logger.info(f"Closing position | Ticket: {ticket} | Symbol: {pos.symbol} | Vol: {pos.volume}")
    result = mt5_order_send(request)
    outcome = _log_order_result(result)
    return outcome.get("success", False)


def close_all_positions(symbol: Optional[str] = None) -> int:
    """
    Close all open positions, optionally filtered by symbol.

    Args:
        symbol: If provided, close only positions for this symbol.

    Returns:
        Number of positions successfully closed.
    """
    positions = get_open_positions(symbol)
    if positions is None or len(positions) == 0:
        logger.info("No open positions to close.")
        return 0

    closed = 0
    for _, pos_row in positions.iterrows():
        if close_position(pos_row["ticket"]):
            closed += 1

    logger.info(f"Closed {closed}/{len(positions)} positions.")
    return closed


def modify_sl_tp(
    ticket:  int,
    new_sl:  float,
    new_tp:  float,
) -> bool:
    """
    Modify Stop Loss and Take Profit of an existing open position.

    Args:
        ticket: MT5 position ticket.
        new_sl: New stop loss price.
        new_tp: New take profit price.

    Returns:
        True if modified successfully, False otherwise.
    """
    pos = get_position_by_ticket(ticket)
    if pos is None:
        logger.error(f"Cannot modify ticket {ticket}: position not found.")
        return False

    request = {
        "action":   TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": ticket,
        "sl":       new_sl,
        "tp":       new_tp,
    }

    result = mt5_order_send(request)
    outcome = _log_order_result(result)
    if outcome.get("success"):
        logger.info(f"SL/TP modified | Ticket: {ticket} | New SL: {new_sl} | New TP: {new_tp}")
    return outcome.get("success", False)


def cancel_pending_order(ticket: int) -> bool:
    """
    Cancel a pending (limit/stop) order.

    Args:
        ticket: MT5 pending order ticket.

    Returns:
        True if cancelled successfully, False otherwise.
    """
    request = {
        "action": TRADE_ACTION_REMOVE,
        "order":  ticket,
    }
    result  = mt5_order_send(request)
    outcome = _log_order_result(result)
    if outcome.get("success"):
        logger.info(f"Pending order cancelled | Ticket: {ticket}")
    return outcome.get("success", False)


def get_position_by_symbol(symbol: str) -> Optional[dict]:
    """
    Get the first open position for a given symbol.

    Returns:
        Dict of position details, or None if no open position exists.
    """
    positions = get_open_positions(symbol)
    if positions is None or len(positions) == 0:
        return None

    pos = positions.iloc[0]
    return {
        "ticket":      pos.ticket,
        "symbol":      pos.symbol,
        "volume":      pos.volume,
        "type":        pos.type,
        "open_price":  pos.price_open,
        "current_sl":  pos.sl,
        "current_tp":  pos.tp,
        "profit":      pos.profit,
        "magic":       pos.magic,
        "comment":     pos.comment,
    }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _log_order_result(result) -> dict:
    """Parse OrderSendResult and log it. Returns dict."""
    from data_ingestion import mt5_last_error

    if result is None:
        error = mt5_last_error()
        logger.error(f"Order send returned None. MT5 error: {error}")
        return {"success": False, "error": str(error)}

    if result.retcode == TRADE_RETCODE_DONE:
        info = {
            "success":    True,
            "ticket":     result.order,
            "volume":     result.volume,
            "price":      result.price,
            "bid":        result.bid,
            "ask":        result.ask,
            "comment":    result.comment,
            "retcode":    result.retcode,
        }
        logger.info(
            f"✅ Order filled | Ticket: {result.order} | "
            f"Price: {result.price} | Volume: {result.volume} | "
            f"Comment: {result.comment}"
        )
        return info
    else:
        error = mt5_last_error()
        logger.error(
            f"❌ Order failed | retcode: {result.retcode} | "
            f"comment: {result.comment} | MT5 error: {error}"
        )
        return {
            "success":  False,
            "retcode":  result.retcode,
            "comment":  result.comment,
            "error":    str(error),
        }