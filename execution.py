# execution.py — QuantEdge MT5: Order Placement & Position Management
# RULES:
#   - ALWAYS call risk_manager.validate_order() before any mt5.order_send()
#   - ALWAYS check mt5.last_error() after every order operation
#   - NO position may be opened without a valid Stop Loss
#   - Log ALL order results at INFO level with full ticket details

from typing import Optional

import MetaTrader5 as mt5

import risk_manager
from data_ingestion import get_account_info, get_open_positions, get_symbol_info
from logger_config import setup_logger
from signal_generation import Signal

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _log_order_result(result) -> dict:
    """Parse mt5.OrderSendResult and log it. Returns dict."""
    if result is None:
        error = mt5.last_error()
        logger.error(f"Order send returned None. MT5 error: {error}")
        return {"success": False, "error": str(error)}

    if result.retcode == mt5.TRADE_RETCODE_DONE:
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
        error = mt5.last_error()
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


def _direction_to_mt5(direction: int) -> int:
    """Convert direction int to mt5.ORDER_TYPE_* constant."""
    return mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL


def _get_current_price(symbol: str, direction: int) -> Optional[float]:
    """Get ask (for BUY) or bid (for SELL) price."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Cannot get tick for {symbol}: {mt5.last_error()}")
        return None
    return tick.ask if direction == 1 else tick.bid


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

    REQUIRES: risk_manager.validate_order() must pass before calling this.

    Args:
        symbol:    Trading symbol (e.g. "EURUSD").
        direction: 1 = BUY, -1 = SELL.
        volume:    Lot size (from risk_manager.calculate_position_size()).
        sl:        Stop loss price. MANDATORY — must be > 0.
        tp:        Take profit price. MANDATORY — must be > 0.
        comment:   Order comment. Default: "QuantEdge".
        magic:     Magic number to identify system orders.

    Returns:
        Dict with 'success' bool and order details or error info.
    """
    if sl <= 0 or tp <= 0:
        logger.error(f"[{symbol}] place_market_order rejected: SL={sl} TP={tp} (both must be > 0).")
        return {"success": False, "error": "Invalid SL/TP"}

    price = _get_current_price(symbol, direction)
    if price is None:
        return {"success": False, "error": "Could not fetch current price"}

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     volume,
        "type":       _direction_to_mt5(direction),
        "price":      price,
        "sl":         sl,
        "tp":         tp,
        "deviation":  20,           # Max slippage in points
        "magic":      magic,
        "comment":    comment,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(
        f"[{symbol}] Sending MARKET {'BUY' if direction == 1 else 'SELL'} | "
        f"Vol: {volume} | Price: {price} | SL: {sl} | TP: {tp}"
    )

    result = mt5.order_send(request)
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

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == 1 else mt5.ORDER_TYPE_SELL_LIMIT

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       volume,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    logger.info(
        f"[{symbol}] Sending {'BUY_LIMIT' if direction == 1 else 'SELL_LIMIT'} | "
        f"Price: {price} | Vol: {volume} | SL: {sl} | TP: {tp}"
    )
    result = mt5.order_send(request)
    return _log_order_result(result)


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
    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error(f"Cannot close ticket {ticket}: position not found.")
        return False

    pos = position[0]
    # Reverse direction to close
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price      = mt5.symbol_info_tick(pos.symbol)

    if price is None:
        logger.error(f"Cannot fetch price for {pos.symbol} to close ticket {ticket}.")
        return False

    fill_price = price.bid if close_type == mt5.ORDER_TYPE_SELL else price.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        fill_price,
        "deviation":    20,
        "magic":        pos.magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    logger.info(f"Closing position | Ticket: {ticket} | Symbol: {pos.symbol} | Vol: {pos.volume}")
    result = mt5.order_send(request)
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
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if not positions:
        logger.info("No open positions to close.")
        return 0

    closed = 0
    for pos in positions:
        if close_position(pos.ticket):
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
    position = mt5.positions_get(ticket=ticket)
    if not position:
        logger.error(f"Cannot modify ticket {ticket}: position not found.")
        return False

    pos     = position[0]
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": ticket,
        "sl":       new_sl,
        "tp":       new_tp,
    }

    result = mt5.order_send(request)
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
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  ticket,
    }
    result  = mt5.order_send(request)
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
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    pos = positions[0]
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
# Signal → Execution Pipeline
# ---------------------------------------------------------------------------

def execute_signal(signal: Signal) -> dict:
    """
    Full pipeline: validate → size → send order for an approved Signal.

    This is the primary entry point from signal_generation.apply_signal_filters().

    Args:
        signal: An approved Signal TypedDict (direction != 0).

    Returns:
        Dict with execution result including 'success' bool.
    """
    symbol    = signal["symbol"]
    direction = signal["direction"]
    sl        = signal["stop_loss"]
    tp        = signal["take_profit"]

    # 1. Fetch live account & symbol data
    account_info = get_account_info()
    symbol_info  = get_symbol_info(symbol)
    positions_df = get_open_positions()

    if account_info is None or symbol_info is None:
        logger.error(f"[{symbol}] Cannot execute: failed to fetch account/symbol info.")
        return {"success": False, "error": "Data fetch failure"}

    open_count = len(positions_df)

    # 2. Risk validation gate (mandatory)
    if not risk_manager.validate_order(
        symbol=symbol,
        direction=direction,
        stop_loss=sl,
        account_info=account_info,
        open_positions_count=open_count,
        symbol_info=symbol_info,
    ):
        logger.warning(f"[{symbol}] Order blocked by risk_manager.")
        return {"success": False, "error": "Risk validation failed"}

    # 3. Check for existing position in same symbol (avoid doubling)
    existing = get_position_by_symbol(symbol)
    if existing:
        logger.warning(
            f"[{symbol}] Already have an open position (ticket {existing['ticket']}). "
            "Skipping new order."
        )
        return {"success": False, "error": "Existing position"}

    # 4. Calculate position size
    entry_price    = signal["entry_price"]
    atr            = signal["atr"]
    sl_pips        = abs(entry_price - sl) / (symbol_info["point"] * 10)
    volume         = risk_manager.calculate_position_size(
        account_balance=account_info["balance"],
        stop_loss_pips=sl_pips,
        symbol_info=symbol_info,
    )

    if volume <= 0:
        logger.error(f"[{symbol}] Calculated volume is 0 — order rejected.")
        return {"success": False, "error": "Zero volume"}

    # 5. Place the order
    result = place_market_order(
        symbol=symbol,
        direction=direction,
        volume=volume,
        sl=sl,
        tp=tp,
        comment=f"QE|{signal['regime'][:2].upper()}|{signal['confidence']:.2f}",
    )

    if result.get("success"):
        logger.info(
            f"[{symbol}] EXECUTION COMPLETE ✅ | "
            f"{'LONG' if direction == 1 else 'SHORT'} {volume} lots | "
            f"Regime: {signal['regime']} | Conf: {signal['confidence']:.2f}"
        )

    return result
