# signal_generation.py — QuantEdge MT5: Entry/Exit Signal Logic (Econometric & Microstructure)
# RULES:
#   - No ML models here — pure rule-based logic only
#   - NO retail indicators (RSI, MACD, ADX, Bollinger Bands, Stochastic, EMA crossovers)
#   - All signals must pass apply_signal_filters() before being acted on
#   - Returns Signal TypedDict or None (never raises)
#
# PERMITTED SIGNAL SOURCES:
#   - Hurst Exponent (regime detection)
#   - GARCH conditional volatility (volatility spikes / mean reversion)
#   - Directional Changes (DC) — microstructure event detection
#   - Order Flow Imbalance (OFI) — microstructure order flow proxy
#   - VWAP deviation (fair-value mean reversion)

from datetime import datetime
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import pytz

from config import (
    CONFIDENCE_THRESHOLD,
    DC_THETA,
    HURST_MEAN_REVERT_THRESHOLD,
    HURST_TRENDING_THRESHOLD,
    MIN_RISK_REWARD_RATIO,
    MIN_SIGNAL_CONFIDENCE,
    MR_SL_ATR,
    MR_TP_ATR,
    OFI_THRESHOLD,
    TREND_SL_ATR,
    TREND_TP_ATR,
    VWAP_DIST_THRESHOLD,
    VOL_ZSCORE_THRESHOLD,
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
# Regime Detection (Hurst Exponent)
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
# Trend Signal — Microstructure (DC + OFI in trending regime)
# ---------------------------------------------------------------------------

def generate_trend_signal(
    features: pd.DataFrame,
    symbol: str,
    atr_sl_multiplier: float = TREND_SL_ATR,
    atr_tp_multiplier: float = TREND_TP_ATR,
) -> Signal:
    """
    Generate a trend-following signal based on microstructure:
        - Directional Change (DC) event CONFIRMS trend direction
        - BUT entry waits for MICROSTRUCTURE PULLBACK (Phase 9.5)
        - Pullback trigger: OFI drops below 0, then recovers above 0 (for LONG)
        - Pullback trigger: OFI rises above 0, then drops below 0 (for SHORT)
        - Hurst exponent > 0.58 (trending regime)

    Entry Logic (Phase 9.5 - Microstructure Pullback):
        1. "Primed" state: DC fires with H > 0.58 → Market is primed, DON'T ENTER YET
        2. Pullback: OFI shows profit-taking (opposite of trend), then flips back
        3. Entry: Only when pullback confirmation occurs

    Args:
        features:           Feature DataFrame from build_feature_matrix().
        symbol:             Trading symbol (for logging).
        atr_sl_multiplier:  SL = entry - (ATR × multiplier).
        atr_tp_multiplier:  TP = entry + (ATR × multiplier).

    Returns:
        Signal dict. direction=0 if no valid trend signal.
    """
    if features.empty:
        logger.debug(f"[{symbol}] Empty features in generate_trend_signal.")
        return _null_signal(symbol, "trending")

    try:
        row = features.iloc[-1]
        entry = row["close"]
        atr = row.get("atr", 0.0)
        hurst = row.get("hurst", 0.5)
        pullback_signal = row.get("pullback_signal", False)
        pullback_direction = row.get("pullback_direction", 0)
        ofi = row.get("ofi", 0.0)

        # Must be in a trending regime
        if hurst <= HURST_TRENDING_THRESHOLD:
            logger.debug(f"[{symbol}] Not trending (H={hurst:.3f}). No trend signal.")
            return _null_signal(symbol, "trending")

        # NEW (Phase 9.5): Only generate signal on pullback confirmation
        direction = 0
        if pullback_signal and pullback_direction != 0:
            direction = pullback_direction

        if direction == 0:
            logger.debug(f"[{symbol}] No pullback confirmation. No trend signal.")
            return _null_signal(symbol, "trending")

        # Set SL/TP based on direction
        if direction == 1:
            sl = entry - (atr * atr_sl_multiplier)
            tp = entry + (atr * atr_tp_multiplier)
        else:
            sl = entry + (atr * atr_sl_multiplier)
            tp = entry - (atr * atr_tp_multiplier)

        # Compute confidence from OFI magnitude and Hurst strength
        ofi_magnitude = min(abs(ofi), 1.0)
        hurst_strength = min((hurst - 0.5) * 2, 1.0)  # Scale 0.5-1.0 → 0-1
        confidence = round((ofi_magnitude * 0.5 + hurst_strength * 0.5), 3)
        confidence = max(MIN_SIGNAL_CONFIDENCE, confidence)

        signal = Signal(
            symbol=symbol, direction=direction, confidence=confidence,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            timestamp=datetime.now(tz=pytz.utc), regime="trending", atr=atr,
        )
        logger.info(
            f"[{symbol}] TREND signal (PULLBACK ENTRY): {'LONG' if direction == 1 else 'SHORT'} | "
            f"Conf: {confidence:.2f} | OFI: {ofi:.4f} | Hurst: {hurst:.3f} | "
            f"Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"
        )
        return signal

    except Exception as e:
        logger.error(f"[{symbol}] generate_trend_signal error: {e}", exc_info=True)
        return _null_signal(symbol)


def _trend_confidence(ofi_magnitude: float, hurst: float) -> float:
    """Score trend signal confidence 0.0–1.0 from OFI and Hurst."""
    ofi_score = min(ofi_magnitude, 1.0)
    hurst_score = min((hurst - 0.5) * 2, 1.0)
    return round((ofi_score * 0.5 + hurst_score * 0.5), 3)


# ---------------------------------------------------------------------------
# Mean Reversion Signal — Volatility & VWAP Based
# ---------------------------------------------------------------------------

def generate_mean_reversion_signal(
    features: pd.DataFrame,
    symbol: str,
    atr_sl_multiplier: float = MR_SL_ATR,
    atr_tp_multiplier: float = MR_TP_ATR,
) -> Signal:
    """
    Generate a mean-reversion signal based on:
        - GARCH volatility spike (vol_zscore exceeds threshold)
        - Price deviating from VWAP beyond threshold
        - Hurst exponent < 0.45 (mean-reverting regime)

    Args:
        features:           Feature DataFrame from build_feature_matrix().
        symbol:             Trading symbol.
        atr_sl_multiplier:  SL distance = ATR × multiplier.
        atr_tp_multiplier:  TP distance = ATR × multiplier.

    Returns:
        Signal dict. direction=0 if no valid MR signal.
    """
    if features.empty:
        logger.debug(f"[{symbol}] Empty features in generate_mean_reversion_signal.")
        return _null_signal(symbol, "mean_reverting")

    try:
        row = features.iloc[-1]
        entry = row["close"]
        atr = row.get("atr", 0.0)
        hurst = row.get("hurst", 0.5)
        vol_zscore = row.get("vol_zscore", 0.0)
        vwap = row.get("vwap", 0.0)
        vwap_dist = row.get("vwap_dist", 0.0)

        # Must be in a mean-reverting regime
        if hurst >= HURST_MEAN_REVERT_THRESHOLD:
            logger.debug(f"[{symbol}] Not mean-reverting (H={hurst:.3f}). No MR signal.")
            return _null_signal(symbol, "mean_reverting")

        direction = 0

        # LONG: price significantly below VWAP + high vol (oversold conditions)
        if vwap > 0 and vwap_dist < -VWAP_DIST_THRESHOLD and vol_zscore > VOL_ZSCORE_THRESHOLD:
            direction = 1
            sl = entry - (atr * atr_sl_multiplier)
            tp = vwap  # Target: revert back to VWAP
            if tp <= sl:
                tp = entry + (atr * atr_tp_multiplier)

        # SHORT: price significantly above VWAP + high vol (overbought conditions)
        elif vwap > 0 and vwap_dist > VWAP_DIST_THRESHOLD and vol_zscore > VOL_ZSCORE_THRESHOLD:
            direction = -1
            sl = entry + (atr * atr_sl_multiplier)
            tp = vwap  # Target: revert back to VWAP
            if tp >= sl:
                tp = entry - (atr * atr_tp_multiplier)

        if direction == 0:
            logger.debug(
                f"[{symbol}] H={hurst:.3f} VZ={vol_zscore:.2f} VWAP_D={vwap_dist:.4f} — no MR signal."
            )
            return _null_signal(symbol, "mean_reverting")

        # Compute confidence from z-score magnitude and VWAP distance
        z_score_conf = min(abs(vol_zscore) / (VOL_ZSCORE_THRESHOLD * 2), 1.0)
        vwap_conf = min(abs(vwap_dist) / (VWAP_DIST_THRESHOLD * 3), 1.0)
        hurst_conf = 1.0 - min(hurst * 2, 1.0)  # Lower Hurst = higher MR confidence
        confidence = round((z_score_conf * 0.4 + vwap_conf * 0.3 + hurst_conf * 0.3), 3)
        confidence = max(MIN_SIGNAL_CONFIDENCE, confidence)

        signal = Signal(
            symbol=symbol, direction=direction, confidence=confidence,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            timestamp=datetime.now(tz=pytz.utc), regime="mean_reverting", atr=atr,
        )
        logger.info(
            f"[{symbol}] MR signal: {'LONG' if direction == 1 else 'SHORT'} | "
            f"Conf: {confidence:.2f} | VolZ: {vol_zscore:.2f} | VWAP_D: {vwap_dist:.4f} | "
            f"Hurst: {hurst:.3f} | Entry: {entry:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"
        )
        return signal

    except Exception as e:
        logger.error(f"[{symbol}] generate_mean_reversion_signal error: {e}", exc_info=True)
        return _null_signal(symbol)


def _mr_confidence(vol_zscore: float, vwap_dist: float, hurst: float) -> float:
    """Score mean-reversion confidence 0.0–1.0."""
    z_conf = min(abs(vol_zscore) / (VOL_ZSCORE_THRESHOLD * 2), 1.0)
    vwap_conf = min(abs(vwap_dist) / (VWAP_DIST_THRESHOLD * 3), 1.0)
    h_conf = 1.0 - min(hurst * 2, 1.0)
    return round((z_conf * 0.4 + vwap_conf * 0.3 + h_conf * 0.3), 3)


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

    Strategy (TREND-FOLLOWING MODE - Option A):
        - Only "trending" regime generates signals using trend_sig
        - "mean_reverting" and "random" regimes return null signal (no trade)
        - Mean-reversion strategy is DISABLED per prop firm pivot

    ML confidence acts as a multiplier: final_conf = signal_conf × (1 + ml_conf) / 2

    Args:
        trend_sig:      Signal from generate_trend_signal().
        mr_sig:         Signal from generate_mean_reversion_signal() (IGNORED).
        ml_confidence:  Probability from ML model (0.0–1.0).
        regime:         Current market regime string.

    Returns:
        The selected Signal, or null signal if no clear edge.
    """
    # TREND-FOLLOWING ONLY: Mean-reversion is DISABLED
    if regime != "trending":
        logger.debug(f"Non-trending regime ({regime}) — no signal generated (MR disabled).")
        return _null_signal(trend_sig["symbol"] if trend_sig else "UNKNOWN", regime)

    base = trend_sig

    if base["direction"] == 0:
        return base

    # Blend with ML confidence
    blended_conf = (base["confidence"] + ml_confidence) / 2
    if blended_conf < MIN_SIGNAL_CONFIDENCE:
        logger.info(
            f"[{base['symbol']}] Signal filtered: combined confidence {blended_conf:.2f} "
            f"< threshold {MIN_SIGNAL_CONFIDENCE}."
        )
        return _null_signal(base["symbol"], regime)

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
        3. Risk/reward ratio ≥ MIN_RISK_REWARD_RATIO
        4. Account info is valid
        5. Minimum confidence threshold (CONFIDENCE_THRESHOLD)

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
    if rr_ratio < MIN_RISK_REWARD_RATIO - 0.001:  # Allow small floating-point tolerance
        logger.warning(
            f"[{sym}] Signal rejected: R:R {rr_ratio:.2f} < {MIN_RISK_REWARD_RATIO} minimum."
        )
        return None

    # 4. Account info valid
    if not account_info or account_info.get("equity", 0) <= 0:
        logger.error(f"[{sym}] Signal rejected: invalid account info.")
        return None

    # 5. Minimum confidence
    if signal["confidence"] < CONFIDENCE_THRESHOLD:
        logger.info(f"[{sym}] Signal rejected: confidence {signal['confidence']:.2f} < {CONFIDENCE_THRESHOLD}.")
        return None

    logger.info(
        f"[{sym}] Signal APPROVED | "
        f"{'LONG' if signal['direction'] == 1 else 'SHORT'} | "
        f"Conf: {signal['confidence']:.2f} | R:R {rr_ratio:.2f}"
    )
    return signal