#!/usr/bin/env python3
"""
verify_mt5_connection.py — Standalone script to test MetaTrader 5 connectivity.

Usage:
    1. Edit .env with your actual MT5 credentials first
    2. Make sure MetaTrader 5 terminal is OPEN (running)
    3. Run: python verify_mt5_connection.py

This script will:
    - Load credentials from .env
    - Connect to MT5 via data_ingestion.initialize_mt5()
    - Fetch account info and print balance/equity
    - Fetch symbol info for EURUSD
    - Pull a small sample of historical data
    - Disconnect cleanly

If any step fails, the error is logged clearly.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from data_ingestion import (
    initialize_mt5,
    shutdown_mt5,
    get_account_info,
    get_symbol_info,
    get_historical_data,
    get_live_tick,
)
from logger_config import setup_logger

logger = setup_logger("verify_connection")


def print_banner(text: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    # ------------------------------------------------------------------
    # Step 0: Load credentials
    # ------------------------------------------------------------------
    print_banner("MT5 Connection Verification")
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    mt5_login    = os.getenv("MT5_LOGIN", "").strip()
    mt5_password = os.getenv("MT5_PASSWORD", "").strip()
    mt5_server   = os.getenv("MT5_SERVER", "").strip()
    mt5_path     = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe").strip()

    if not all([mt5_login, mt5_password, mt5_server]):
        print("❌ ERROR: Missing credentials in .env file.")
        print("   Please edit .env with your MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER.")
        sys.exit(1)

    print(f"  Login:    {mt5_login}")
    print(f"  Server:   {mt5_server}")
    print(f"  Path:     {mt5_path}")
    print(f"  Password: {'*' * 8}")

    # ------------------------------------------------------------------
    # Step 1: Connect to MT5
    # ------------------------------------------------------------------
    print("\n[1/5] Connecting to MetaTrader 5...")
    if not initialize_mt5(int(mt5_login), mt5_password, mt5_server, mt5_path):
        print("❌ FAILED: Could not connect to MT5.")
        print("   Troubleshooting:")
        print("   - Is MetaTrader 5 terminal open? (must be running)")
        print("   - Are your credentials correct in .env?")
        print("   - Is the terminal64.exe path correct?")
        sys.exit(1)
    print("✅ Connected successfully!")

    # ------------------------------------------------------------------
    # Step 2: Fetch account info
    # ------------------------------------------------------------------
    print("\n[2/5] Fetching account info...")
    info = get_account_info()
    if info is None:
        print("❌ Could not fetch account info.")
    else:
        print(f"  Balance:    {info['balance']:>15,.2f} {info['currency']}")
        print(f"  Equity:     {info['equity']:>15,.2f} {info['currency']}")
        print(f"  Margin:     {info['margin']:>15,.2f}")
        print(f"  Free Margin:{info['free_margin']:>15,.2f}")
        print(f"  Leverage:   1:{info['leverage']}")
    print("✅ Account info fetched.")

    # ------------------------------------------------------------------
    # Step 3: Fetch symbol info
    # ------------------------------------------------------------------
    print("\n[3/5] Fetching symbol info for EURUSD...")
    sym_info = get_symbol_info("EURUSD")
    if sym_info is None:
        print("❌ Could not fetch symbol info.")
    else:
        print(f"  Digits:         {sym_info['digits']}")
        print(f"  Point:          {sym_info['point']}")
        print(f"  Tick Size:      {sym_info['trade_tick_size']}")
        print(f"  Tick Value:     {sym_info['trade_tick_value']}")
        print(f"  Min Volume:     {sym_info['volume_min']}")
        print(f"  Max Volume:     {sym_info['volume_max']}")
        print(f"  Spread:         {sym_info['spread']} pts")
    print("✅ Symbol info fetched.")

    # ------------------------------------------------------------------
    # Step 4: Fetch a tick
    # ------------------------------------------------------------------
    print("\n[4/5] Fetching live tick for EURUSD...")
    tick = get_live_tick("EURUSD")
    if tick is None:
        print("❌ Could not fetch live tick.")
    else:
        print(f"  Bid: {tick['bid']:.5f}  |  Ask: {tick['ask']:.5f}  |  Last: {tick['last']}")
        print(f"  Volume: {tick['volume']}  |  Time: {tick['time']}")
    print("✅ Live tick fetched.")

    # ------------------------------------------------------------------
    # Step 5: Fetch historical data
    # ------------------------------------------------------------------
    print("\n[5/5] Fetching 10 bars of H1 history for EURUSD...")
    from datetime import datetime, timedelta
    from config import HIGHER_TIMEFRAME_STR
    import MetaTrader5 as mt5

    end = datetime.now()
    start = end - timedelta(hours=10)
    df = get_historical_data("EURUSD", mt5.TIMEFRAME_H1, start, end)
    if df is None or df.empty:
        print("❌ Could not fetch historical data.")
    else:
        print(f"  Fetched {len(df)} bars:")
        print(f"  {df[['open', 'high', 'low', 'close', 'volume']].tail(3).to_string()}")
    print("✅ Historical data fetched.")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    shutdown_mt5()

    print_banner("All Checks Passed ✅")
    print("MT5 connection is fully functional.")
    print("You can now run: python main.py")


if __name__ == "__main__":
    main()