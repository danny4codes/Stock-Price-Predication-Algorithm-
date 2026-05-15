# signal_generation.py — QuantEdge MT5: Entry/Exit Signal Logic
# RULES:
#   - No ML models here — pure rule-based logic only
#   - All signals must pass apply_signal_filters() before being acted on
#   - Returns Signal TypedDict or None (never raises)

from datetime import datetime
from typing import Optional, TypedDict

import pandas as pd
import pytz

from config import (
    HURST_MEAN_REVERT_THRESHOLD,
    HURST_TRENDING_THRESHOLD,
)
from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Signal Contract
# ---------------------------------------------------------------------------

class Signal(TypedDict):
    symbol:      str
    direction:   int        # 1 = LONG, -1 = SHORT, 0 = FLAT
    confidence:  float      # 0.0 – 1.0
    entry_price: float
    stop_loss:   float
    take_profit: float
    timestamp:   datetime
    regime:      str        # "trending" | "mean_reverting" | "random"
    atr:         float      # ATR at signal time (used for SL/TP sizing)


def _null_signal(symbol: str, regime: str = "random") -> Signal:
    """Return a neutral FLAT signal — no action."""
    return Signal(
        symbol=symbol,
        direction=0,
        confidence=0.0,
        entry_price=0.0,
        stop_loss=0.0,
        take_profit=0.0,
        timestamp=datetime.now(tz=pytz.utc),
        regime=regime,
        atr=0.0,
    )


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def detect_market_regime(hurst: float) -> str:
    """
    Classify market regime from Hurst exponent.

    Args:
        hurst: Hurst exponent (0.0 – 1.0).

    Returns:
        "trending"       if H > HURST_TRENDING_THRESHOLD (0.55)
        "mean_reverting" if H < HURST_MEAN_REVERT_THRESHOLD (0.45)
        "random"         if H is between thresholds
    """
    if hurst > HURST_TRENDING_THRESHOLD:
        return "trending"
    elif hurst < HURST_MEAN_REVERT_THRESHOLD:
        return "mean_reverting"
    else:
        return "random"


# ---------------------------------------------------------------------------
# Trend Signal (uses ADX + EMA crossover + MACD)
# ---------------------------------------------------------------------------

def generate_trend_signal(
    features: pd.DataFrame,
    symbol: str,
    atr_sl_multiplier: float = 1.5,
    atr_tp_multiplier: float = 3.0,
) -> Signal:
    """
    Generate a trend-following signal based on:
        - EMA crossover (20 > 50 = bullish bias)
        - MACD histogram direction
        - ADX > 25 (trend is strong enough)
        - RSI not overbought/oversold at entry

    Args:
        features:           Feature DataFrame from build_feature_matrix().
        symbol:             Trading symbol (for logging).
        atr_sl_multiplier:  SL = entry ± (ATR × multiplier). Default: 1.5.
        atr_tp_multiplier:  TP = entry ± (ATR × multiplier). Default: 3.0.

    Returns:
        Signal dict. direction=0 if no valid trend signal.
    """
    try:
        row = features.iloc[-1]  # Latest completed bar
        entry = row["close"]
        atr   = row.get("atr", 0.0)

        # --- Trend Conditions ---
        ema_cross    = row.get("ema_cross", 0)          # ema_20 - ema_50
        macd_diff    = row.get("macd_diff", 0)          # MACD histogram
        adx          = row.get("adx", 0)
        rsi          = row.get("rsi", 50)

        trend_strong = adx > 25
        rsi_ok_long  = rsi < 70  # Not overbought for long
        rsi_ok_short = rsi > 30  # Not oversold for short

        if trend_strong and ema_cross > 0 and macd_diff > 0 and rsi_ok_long:
            direction = 1   # LONG
            sl = entry - (atr * atr_sl_multiplier)
            tp = entry + (atr * atr_tp_multiplier)
            confidence = _trend_confidence(adx, abs(ema_cross) / entry, rsi)

        elif trend_strong and ema_cross < 0 and macd_diff < 0 and rsi_ok_short:
            direction = -1  # SHORT
            sl = entry + (atr * atr_sl_multiplier)
            tp = entry - (atr * atr_tp_multiplier)
            confidence = _trend_confidence(adx, abs(ema_cross) / entry, rsi)

        else:
            logger.debug(f"[{symbol}] No trend signal. ADX:{adx:.1f} EMA_X:{ema_cross:.5f} MACD:{macd_diff:.5f}")
            return _null_signal(symbol, "trending")

        signal = Signal(
            symbol=symbol, direction=direction, confidence=confidence,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            timestamp=datetime.now(tz=pytz.utc), regime="trending", atr=atr,
        )
        logger.info(
            f"[{symbol}] TREND signal: {'LONG' if direction == 1 else 'SHORT'} | "
            f"Confidence: {confidence:.2f} | Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"
        )
        return signal

    except Exception as e:
        logger.error(f"[{symbol}] generate_trend_signal error: {e}", exc_info=True)
        return _null_signal(symbol)


def _trend_confidence(adx: float, ema_cross_norm: float, rsi: float) -> float:
    """Score trend signal confidence 0.0–1.0 from ADX, EMA cross strength, RSI neutrality."""
    adx_score     = min(adx / 50.0, 1.0)             # 50+ ADX = max score
    cross_score   = min(ema_cross_norm * 1000, 1.0)  # Normalize EMA cross
    rsi_distance  = abs(rsi - 50) / 50.0              # 0 at neutral, 1 at extremes
    rsi_score     = 1.0 - rsi_distance                # Better signal when RSI is neutral
    return round((adx_score * 0.5 + cross_score * 0.3 + rsi_score * 0.2), 3)


# ---------------------------------------------------------------------------
# Mean Reversion Signal (uses Bollinger Bands + RSI + Stochastic)
# ---------------------------------------------------------------------------

def generate_mean_reversion_signal(
    features: pd.DataFrame,
    symbol: str,
    atr_sl_multiplier: float = 1.0,
    atr_tp_multiplier: float = 1.5,
) -> Signal:
    """
    Generate a mean-reversion signal based on:
        - Price outside Bollinger Bands (bb_pct < 0.05 or > 0.95)
        - RSI oversold (< 30) or overbought (> 70)
        - Stochastic confirmation

    Args:
        features:           Feature DataFrame from build_feature_matrix().
        symbol:             Trading symbol.
        atr_sl_multiplier:  SL distance = ATR × multiplier. Default: 1.0.
        atr_tp_multiplier:  TP distance = ATR × multiplier. Default: 1.5.

    Returns:
        Signal dict. direction=0 if no valid MR signal.
    """
    try:
        row   = features.iloc[-1]
        entry = row["close"]
        atr   = row.get("atr", 0.0)

        bb_pct   = row.get("bb_pct", 0.5)
        rsi      = row.get("rsi", 50)
        stoch_k  = row.get("stoch_k", 50)
        stoch_d  = row.get("stoch_d", 50)

        # Long: price below lower band, RSI oversold, stochastic crossing up
        if bb_pct < 0.05 and rsi < 32 and stoch_k < 25 and stoch_k > stoch_d:
            direction  = 1
            sl         = entry - (atr * atr_sl_multiplier)
            tp         = row.get("bb_mid", entry + atr * atr_tp_multiplier)
            confidence = _mr_confidence(bb_pct, rsi, stoch_k, side="long")

        # Short: price above upper band, RSI overbought, stochastic crossing down
        elif bb_pct > 0.95 and rsi > 68 and stoch_k > 75 and stoch_k < stoch_d:
            direction  = -1
            sl         = entry + (atr * atr_sl_multiplier)
            tp         = row.get("bb_mid", entry - atr * atr_tp_multiplier)
            confidence = _mr_confidence(bb_pct, rsi, stoch_k, side="short")

        else:
            logger.debug(
                f"[{symbol}] No MR signal. bb_pct:{bb_pct:.2f} RSI:{rsi:.1f} StochK:{stoch_k:.1f}"
            )
            return _null_signal(symbol, "mean_reverting")

        signal = Signal(
            symbol=symbol, direction=direction, confidence=confidence,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            timestamp=datetime.now(tz=pytz.utc), regime="mean_reverting", atr=atr,
        )
        logger.info(
            f"[{symbol}] MR signal: {'LONG' if direction == 1 else 'SHORT'} | "
            f"Confidence: {confidence:.2f} | BB%: {bb_pct:.2f} | RSI: {rsi:.1f}"
        )
        return signal

    except Exception as e:
        logger.error(f"[{symbol}] generate_mean_reversion_signal error: {e}", exc_info=True)
        return _null_signal(symbol)


def _mr_confidence(bb_pct: float, rsi: float, stoch_k: float, side: str) -> float:
    """Score mean-reversion confidence 0.0–1.0."""
    if side == "long":
        bb_score    = max(0, (0.05 - bb_pct) / 0.05)   # Further below band = better
        rsi_score   = max(0, (35 - rsi) / 35)           # Lower RSI = better
        stoch_score = max(0, (30 - stoch_k) / 30)
    else:
        bb_score    = max(0, (bb_pct - 0.95) / 0.05)
        rsi_score   = max(0, (rsi - 65) / 35)
        stoch_score = max(0, (stoch_k - 70) / 30)
    return round((bb_score * 0.4 + rsi_score * 0.35 + stoch_score * 0.25), 3)


# ---------------------------------------------------------------------------
# Signal Combiner & Filter
# ---------------------------------------------------------------------------

def combine_signals(
    trend_sig: Signal,
    mr_sig: Signal,
    ml_confidence: float,
    regime: str,
) -> Signal:
    """
    Combine trend and mean-reversion signals based on current market regime.

    Strategy:
        - "trending"       → use trend_sig if confidence > 0.3
        - "mean_reverting" → use mr_sig if confidence > 0.3
        - "random"         → return null signal (no trade)

    ML confidence acts as a multiplier: final_conf = signal_conf × (1 + ml_conf) / 2

    Args:
        trend_sig:      Signal from generate_trend_signal().
        mr_sig:         Signal from generate_mean_reversion_signal().
        ml_confidence:  Probability from ML model (0.0–1.0).
        regime:         Current market regime string.

    Returns:
        The selected Signal, or null signal if no clear edge.
    """
    MIN_CONFIDENCE = 0.30

    if regime == "trending":
        base = trend_sig
    elif regime == "mean_reverting":
        base = mr_sig
    else:
        logger.debug("Random walk regime — no signal generated.")
        return _null_signal(trend_sig["symbol"], "random")

    if base["direction"] == 0:
        return base

    # Blend with ML confidence
    blended_conf = (base["confidence"] + ml_confidence) / 2
    if blended_conf < MIN_CONFIDENCE:
        logger.info(
            f"[{base['symbol']}] Signal filtered: combined confidence {blended_conf:.2f} "
            f"< threshold {MIN_CONFIDENCE}."
        )
        return _null_signal(base["symbol"], regime)

    # Return updated signal with blended confidence
    return Signal(**{**base, "confidence": round(blended_conf, 3)})


def apply_signal_filters(
    signal: Signal,
    account_info: dict,
    open_positions_count: int,
) -> Optional[Signal]:
    """
    Final gate: apply all pre-trade filters before execution.

    Checks:
        1. Signal direction is not FLAT (direction != 0)
        2. Valid entry, SL, TP prices
        3. Risk/reward ratio ≥ 1.5
        4. Account info is valid
        5. Minimum confidence threshold

    Args:
        signal:               Signal from combine_signals().
        account_info:         Dict from data_ingestion.get_account_info().
        open_positions_count: Current number of open positions.

    Returns:
        The Signal if all filters pass, None if any filter rejects it.
    """
    sym = signal.get("symbol", "UNKNOWN")

    # 1. Not a flat signal
    if signal["direction"] == 0:
        return None

    # 2. Valid prices
    if signal["entry_price"] <= 0 or signal["stop_loss"] <= 0 or signal["take_profit"] <= 0:
        logger.warning(f"[{sym}] Signal rejected: invalid prices.")
        return None

    # 3. Risk/reward ratio
    sl_dist = abs(signal["entry_price"] - signal["stop_loss"])
    tp_dist = abs(signal["entry_price"] - signal["take_profit"])
    if sl_dist == 0:
        logger.warning(f"[{sym}] Signal rejected: zero SL distance.")
        return None
    rr_ratio = tp_dist / sl_dist
    if rr_ratio < 1.5:
        logger.warning(
            f"[{sym}] Signal rejected: R:R {rr_ratio:.2f} < 1.5 minimum."
        )
        return None

    # 4. Account info valid
    if not account_info or account_info.get("equity", 0) <= 0:
        logger.error(f"[{sym}] Signal rejected: invalid account info.")
        return None

    # 5. Minimum confidence
    if signal["confidence"] < 0.25:
        logger.info(f"[{sym}] Signal rejected: confidence {signal['confidence']:.2f} < 0.25.")
        return None

    logger.info(
        f"[{sym}] Signal APPROVED ✅ | "
        f"{'LONG' if signal['direction'] == 1 else 'SHORT'} | "
        f"Conf: {signal['confidence']:.2f} | R:R {rr_ratio:.2f}"
    )
    return signal
