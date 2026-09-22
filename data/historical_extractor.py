# historical_extractor.py — QuantEdge MT5: Historical Data Extraction Pipeline
# Downloads 6 months of M1 and H1 data for specified trading pairs.
# RULE: Uses data_ingestion.py for ALL MT5 operations - never imports MetaTrader5 directly.

import sys
from pathlib import Path
# Add parent directory to sys.path to allow running script directly from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

from config import RAW_DIR
from data_ingestion import (
    TIMEFRAME_H1,
    TIMEFRAME_M1,
    get_historical_data,
    get_symbol_info,
    initialize_mt5,
    mt5_last_error,
    shutdown_mt5,
)
from logger_config import setup_logger

logger = setup_logger(__name__)

# Output directory for historical data
HISTORICAL_DIR = RAW_DIR.parent / "historical"
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

# Target symbols for backtesting (extended list from master list)
TARGET_SYMBOLS = [
    "XAUUSD",  # Gold
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
]

# Timeframes to download (using re-exported mt5 constants)
TIMEFRAME_MAP = {
    "M1": TIMEFRAME_M1,
    "H1": TIMEFRAME_H1,
}


def process_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Process raw OHLCV dataframe: rename columns, ensure spread is numeric.

    Args:
        df: Raw DataFrame from MT5 with DatetimeIndex.
        symbol: Symbol name for logging.

    Returns:
        Processed DataFrame with proper column types.
    """
    df = df.copy()

    # Rename tick_volume to volume for clarity
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # Ensure spread is numeric and handle missing values
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    else:
        # Calculate spread from high-low if spread column missing
        df["spread"] = df["high"] - df["low"]

    # Drop rows with critical missing data
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    logger.debug(f"Processed {len(df)} rows for {symbol}")
    return df


def save_dataframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: Path = HISTORICAL_DIR,
    format: str = "parquet",
) -> bool:
    """
    Save DataFrame to disk as parquet or CSV.

    Args:
        df: Processed DataFrame with DatetimeIndex.
        symbol: Trading symbol.
        timeframe: Timeframe string (M1, H1).
        output_dir: Output directory path.
        format: 'parquet' or 'csv'.

    Returns:
        True if saved successfully, False otherwise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timeframe}_{timestamp}"

    try:
        if format == "parquet":
            filepath = output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, compression="snappy")
        else:
            filepath = output_dir / f"{filename}.csv"
            df.to_csv(filepath)

        logger.info(f"Saved {len(df)} rows to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {symbol} {timeframe} data: {e}")
        return False


def _chunked_historical_fetch(
    symbol: str,
    timeframe: int,
    start: datetime,
    end: datetime,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch historical data in chunks to avoid MT5 limits on large date ranges.
    M1 timeframe with 6+ months of data requires chunking (~180k bars exceeds limit).

    Args:
        symbol: Trading symbol.
        timeframe: MT5 timeframe constant.
        start: Start datetime.
        end: End datetime.
        chunk_days: Days per chunk (default 30 for M1 compatibility).

    Returns:
        Concatenated DataFrame with all chunks.
    """
    chunks = []
    current_start = start

    while current_start < end:
        chunk_end = min(current_start + timedelta(days=chunk_days), end)
        logger.debug(f"Fetching chunk: {symbol} {current_start.date()} to {chunk_end.date()}")

        df = get_historical_data(symbol, timeframe, current_start, chunk_end)
        if df is not None and len(df) > 0:
            chunks.append(df)
        else:
            logger.warning(f"Chunk returned no data: {current_start.date()} to {chunk_end.date()}")

        current_start = chunk_end

    if chunks:
        return pd.concat(chunks)
    return pd.DataFrame()


def download_symbol_data(
    symbol: str,
    months: int = 6,
    timeframes: list = None,
    output_dir: Path = HISTORICAL_DIR,
    format: str = "parquet",
) -> dict:
    """
    Download historical data for a symbol across multiple timeframes.

    Args:
        symbol: Trading symbol.
        months: Number of months of history to fetch.
        timeframes: List of timeframe strings (e.g., ["M1", "H1"]).
        output_dir: Output directory for saved files.
        format: 'parquet' or 'csv'.

    Returns:
        Dict with download results per timeframe.
    """
    if timeframes is None:
        timeframes = list(TIMEFRAME_MAP.keys())

    results = {}

    # Calculate date range
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(days=30 * months)

    logger.info(f"Downloading {months} months of data for {symbol}")

    for tf_name in timeframes:
        tf_const = TIMEFRAME_MAP[tf_name]
        logger.info(f"Fetching {symbol} {tf_name} data ({start_date.date()} to {end_date.date()})")

        # M1 requires chunked fetching for 6+ months (~180k bars exceeds MT5 limit)
        if tf_name == "M1" and months > 2:
            df = _chunked_historical_fetch(symbol, tf_const, start_date, end_date, chunk_days=30)
        else:
            df = get_historical_data(symbol, tf_const, start_date, end_date)

        if df is None or len(df) == 0:
            logger.warning(f"No data returned for {symbol} {tf_name}")
            results[tf_name] = {"status": "failed", "rows": 0, "error": str(mt5_last_error())}
            continue

        df = process_dataframe(df, symbol)
        saved = save_dataframe(df, symbol, tf_name, output_dir, format)

        results[tf_name] = {"status": "success" if saved else "failed", "rows": len(df)}

    return results


def run_extraction_pipeline(
    symbols: list = None,
    months: int = 6,
    output_dir: Path = HISTORICAL_DIR,
    format: str = "parquet",
) -> dict:
    """
    Run the full historical data extraction pipeline.

    Args:
        symbols: List of symbols to download. Uses TARGET_SYMBOLS if None.
        months: Number of months of history.
        output_dir: Output directory for saved files.
        format: 'parquet' or 'csv'.

    Returns:
        Summary dict with counts of successful/failed downloads.
    """
    if symbols is None:
        symbols = TARGET_SYMBOLS

    summary = {
        "total_symbols": len(symbols),
        "successful": 0,
        "failed": 0,
        "timeframes": {},
    }

    # Ensure MT5 is connected (try connecting to existing terminal first)
    if not initialize_mt5(login=0, password="", server="", path=""):
        logger.critical("MT5 connection failed. Ensure terminal is running.")
        summary["error"] = "MT5 connection failed"
        return summary

    logger.info(f"Starting historical data extraction for {len(symbols)} symbols")
    logger.info(f"Output directory: {output_dir}")

    for symbol in symbols:
        # Check if symbol is available in MT5
        symbol_info = get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Symbol {symbol} not available in MT5. Skipping.")
            summary["failed"] += 1
            continue

        results = download_symbol_data(symbol, months, list(TIMEFRAME_MAP.keys()), output_dir, format)

        # Check if at least one timeframe succeeded
        if results.get("M1", {}).get("status") == "success" or results.get("H1", {}).get("status") == "success":
            summary["successful"] += 1
        else:
            summary["failed"] += 1

        summary["timeframes"][symbol] = results

    logger.info(
        f"Extraction complete: {summary['successful']}/{summary['total_symbols']} symbols successful"
    )
    shutdown_mt5()
    return summary


def main():
    """Entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract historical MT5 data for backtesting")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TARGET_SYMBOLS,
        help="Symbols to download (default: all target symbols)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months of history (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HISTORICAL_DIR,
        help=f"Output directory (default: {HISTORICAL_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    summary = run_extraction_pipeline(
        symbols=args.symbols,
        months=args.months,
        output_dir=args.output_dir,
        format=args.format,
    )

    # Print summary
    print(f"\nExtraction Summary:")
    print(f"  Successful: {summary['successful']}/{summary['total_symbols']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Output: {args.output_dir}")

    if "error" in summary:
        print(f"  Error: {summary['error']}")


if __name__ == "__main__":
    main()