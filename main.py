# main.py — QuantEdge MT5: Main Async Trading Loop
# Entry point for the live trading system.
# Usage: python main.py

import asyncio
import os
import signal as os_signal
from datetime import datetime, date
from pathlib import Path

import MetaTrader5 as mt5
import pytz
from dotenv import load_dotenv

import risk_manager
from config import (
    LOOKBACK_BARS,
    LOOP_INTERVAL_SECONDS,
    SYMBOLS,
)
from data_ingestion import (
    get_account_info,
    get_historical_data_n_bars,
    get_open_positions,
    initialize_mt5,
    shutdown_mt5,
)
from execution import execute_signal
from feature_engineering import build_feature_matrix, compute_hurst_exponent
from logger_config import setup_logger
from machine_learning import get_latest_signal_confidence, load_model
from signal_generation import (
    apply_signal_filters,
    combine_signals,
    detect_market_regime,
    generate_mean_reversion_signal,
    generate_trend_signal,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")
logger = setup_logger("main")

# Global shutdown flag
_shutdown_requested = False


def _handle_shutdown(sig, frame):
    """Graceful shutdown handler for Ctrl+C and SIGTERM."""
    global _shutdown_requested
    logger.info(f"Shutdown signal received ({sig}). Finishing current cycle...")
    _shutdown_requested = True


os_signal.signal(os_signal.SIGINT,  _handle_shutdown)
os_signal.signal(os_signal.SIGTERM, _handle_shutdown)


# ---------------------------------------------------------------------------
# Per-Symbol Processing
# ---------------------------------------------------------------------------

async def process_symbol(
    symbol:    str,
    timeframe: int,
    model_path: str | None = None,
) -> None:
    """
    Run one full signal-generation and execution cycle for a single symbol.

    Pipeline:
        1. Fetch latest bars
        2. Build feature matrix
        3. Detect regime (Hurst)
        4. Generate trend + MR signals
        5. Optionally run ML inference
        6. Combine & filter signals
        7. Execute if approved

    Args:
        symbol:     Trading symbol (e.g. "EURUSD").
        timeframe:  MT5 timeframe constant.
        model_path: Optional path to .joblib model file. None = no ML.
    """
    logger.debug(f"[{symbol}] Processing cycle start.")

    # 1. Fetch OHLCV
    df = get_historical_data_n_bars(symbol, timeframe, LOOKBACK_BARS)
    if df is None or df.empty:
        logger.warning(f"[{symbol}] No data — skipping cycle.")
        return

    # 2. Feature engineering
    features = build_feature_matrix(df)
    if features.empty or len(features) < 10:
        logger.warning(f"[{symbol}] Insufficient features ({len(features)} rows) — skipping.")
        return

    # 3. Regime detection
    hurst  = float(features["hurst"].iloc[-1]) if "hurst" in features.columns else 0.5
    regime = detect_market_regime(hurst)
    logger.info(f"[{symbol}] Regime: {regime} (H={hurst:.3f})")

    # 4. Generate rule-based signals
    trend_sig = generate_trend_signal(features, symbol)
    mr_sig    = generate_mean_reversion_signal(features, symbol)

    # 5. ML inference (optional)
    ml_confidence = 0.5  # Default: neutral
    if model_path:
        try:
            model, scaler = load_model(model_path)
            # Remove label/non-feature columns if present
            feat_cols = [c for c in features.columns if c not in ("close",)]
            _, ml_confidence = get_latest_signal_confidence(model, scaler, features[feat_cols])
            logger.info(f"[{symbol}] ML confidence: {ml_confidence:.3f}")
        except Exception as e:
            logger.warning(f"[{symbol}] ML inference failed: {e}. Using default confidence.")

    # 6. Combine signals
    final_signal = combine_signals(trend_sig, mr_sig, ml_confidence, regime)

    # 7. Filter & execute
    account_info = get_account_info()
    if account_info is None:
        logger.error(f"[{symbol}] Cannot fetch account info — skipping execution.")
        return

    positions_df  = get_open_positions()
    open_count    = len(positions_df)
    approved      = apply_signal_filters(final_signal, account_info, open_count)

    if approved:
        result = execute_signal(approved, account_info)
        if not result.get("success"):
            logger.warning(f"[{symbol}] Execution failed: {result.get('error')}")
    else:
        logger.debug(f"[{symbol}] No tradeable signal this cycle.")


# ---------------------------------------------------------------------------
# Main Trading Loop
# ---------------------------------------------------------------------------

async def trading_loop(timeframe: int, model_path: str | None = None) -> None:
    """
    Main async loop. Processes all SYMBOLS on each tick. Runs indefinitely.

    Args:
        timeframe:  MT5 timeframe constant (e.g. mt5.TIMEFRAME_H1).
        model_path: Optional path to a trained .joblib model file.
    """
    logger.info("=" * 60)
    logger.info("QuantEdge MT5 — Trading System STARTING")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Loop interval: {LOOP_INTERVAL_SECONDS}s")
    logger.info("=" * 60)

    # Initialize daily session tracking
    account_info = get_account_info()
    if account_info:
        risk_manager.reset_daily_session(account_info["balance"])
        logger.info(f"Session opened | Balance: {account_info['balance']} {account_info['currency']}")

    last_session_date = date.today()

    while not _shutdown_requested:
        loop_start = datetime.now(tz=pytz.utc)

        # --- Reset daily session at midnight ---
        today = date.today()
        if today != last_session_date:
            account_info = get_account_info()
            if account_info:
                risk_manager.reset_daily_session(account_info["balance"])
                logger.info(f"New trading day | Session reset | Balance: {account_info['balance']}")
            last_session_date = today

        # --- Process all symbols concurrently ---
        tasks = [
            process_symbol(sym, timeframe, model_path)
            for sym in SYMBOLS
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Trading loop tasks cancelled.")
            break
        except Exception as e:
            logger.error(f"Unhandled error in trading loop: {e}", exc_info=True)

        # --- Sleep until next cycle ---
        elapsed = (datetime.now(tz=pytz.utc) - loop_start).total_seconds()
        sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed)

        logger.info(
            f"Cycle complete in {elapsed:.1f}s. "
            f"Sleeping {sleep_time:.1f}s until next cycle."
        )

        if sleep_time > 0 and not _shutdown_requested:
            await asyncio.sleep(sleep_time)

    logger.info("Trading loop exited cleanly.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Configure and launch the trading system.

    Reads credentials from .env, connects to MT5,
    then runs the async trading loop until Ctrl+C.
    """
    # --- Load credentials ---
    mt5_login    = int(os.getenv("MT5_LOGIN", "0"))
    mt5_password = os.getenv("MT5_PASSWORD", "")
    mt5_server   = os.getenv("MT5_SERVER", "")
    mt5_path     = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
    model_path   = os.getenv("MODEL_PATH", None)  # Optional

    if not mt5_login or not mt5_password or not mt5_server:
        logger.critical(
            "MT5 credentials missing! "
            "Create .env from .env.template and fill in MT5_LOGIN, MT5_PASSWORD, MT5_SERVER."
        )
        return

    # --- Connect to MT5 ---
    if not initialize_mt5(mt5_login, mt5_password, mt5_server, mt5_path):
        logger.critical("Failed to connect to MT5. Exiting.")
        return

    # --- Set timeframe ---
    timeframe = mt5.TIMEFRAME_H1  # Default: 1-hour bars

    # --- Run ---
    try:
        asyncio.run(trading_loop(timeframe=timeframe, model_path=model_path))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        logger.info("Shutting down MT5 connection...")
        shutdown_mt5()
        logger.info("QuantEdge MT5 — System STOPPED cleanly. ✅")


if __name__ == "__main__":
    main()
