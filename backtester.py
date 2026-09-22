# backtester.py — QuantEdge MT5: Historical Backtesting Engine
# Runs vectorized backtests on historical parquet data with full risk management.

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    HURST_TRENDING_THRESHOLD,
    HURST_MEAN_REVERT_THRESHOLD,
    MAX_DAILY_DRAWDOWN_PCT,
    MIN_SIGNAL_CONFIDENCE,
    MR_SL_ATR,
    MR_TP_ATR,
    OFI_THRESHOLD,
    RISK_PER_TRADE_PCT,
    TREND_SL_ATR,
    TREND_TP_ATR,
    VOL_ZSCORE_THRESHOLD,
    VWAP_DIST_THRESHOLD,
)
from feature_engineering import build_feature_matrix
from logger_config import setup_logger
from signal_generation import (
    Signal,
    apply_signal_filters,
    detect_market_regime,
)

logger = setup_logger(__name__)

# Default paths
HISTORICAL_DATA_DIR = Path(r"D:\Trading\data\historical")

# Fixed spread penalty per trade (2 pips)
SPREAD_PIPS = 2.0


class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.equity_curve: list[float] = []  # Track equity for proper drawdown calc
        self.trades_log: list[dict] = []

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "total_pnl": self.total_pnl,
        }


class SimulatedAccount:
    """Simulated trading account with risk management."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.daily_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        self.date = datetime.now().date()
        self.equity_curve: list[float] = [initial_balance]  # Track ALL equity points

    def reset_daily(self, new_date: datetime) -> None:
        """Reset daily tracking at start of new day."""
        if new_date.date() != self.date:
            self.date = new_date.date()
            self.daily_start_balance = self.balance

    def can_trade(self) -> bool:
        """Check if daily drawdown limit allows trading."""
        drawdown = (self.daily_start_balance - self.balance) / self.daily_start_balance
        self.current_drawdown = drawdown
        return drawdown < MAX_DAILY_DRAWDOWN_PCT

    def calculate_position_size(self, stop_loss_pips: float, pip_value: float = 1.0) -> float:
        """Calculate position size using fixed fractional (1%) risk."""
        risk_amount = self.balance * RISK_PER_TRADE_PCT
        lots = risk_amount / (stop_loss_pips * pip_value)
        return round(max(lots, 0.01), 2)

    def execute_trade(self, signal: Signal, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Execute a simulated trade with TRAILING STOP for trend following.
        Entry at next bar open. Trail stop behind price for trend capture.
        """
        from config import TREND_SL_ATR

        entry = signal["entry_price"]
        sl = signal["stop_loss"]
        direction = signal["direction"]
        symbol = signal["symbol"]
        atr = signal.get("atr", 0.001)

        # Determine pip value based on symbol
        if symbol == "XAUUSD":
            pip_value = 0.1
            pip_value_dollar = 1.0
        else:
            pip_value = 0.0001 if "JPY" not in symbol else 0.01
            pip_value_dollar = 10.0

        # Calculate position size
        sl_pips = abs(entry - sl) / pip_value
        risk_amount = self.balance * RISK_PER_TRADE_PCT
        lots = round(max(risk_amount / (sl_pips * pip_value_dollar), 0.01), 2)

        lookahead = min(100, len(df) - idx - 1)
        if lookahead <= 0:
            return None

        future_data = df.iloc[idx + 1 : idx + 1 + lookahead].copy()
        pnl = 0.0
        exit_price = entry
        exit_reason = "unknown"

        # Trailing stop parameters: trail at 1.5x ATR behind price for trend capture
        trail_distance = atr * 2.0

        if direction == 1:  # LONG
            first_row = future_data.iloc[0]
            actual_entry = first_row["open"] + (SPREAD_PIPS * pip_value)

            # Initial stop loss
            current_sl = sl
            highest_price = actual_entry

            for _, row in future_data.iterrows():
                low = row["low"]
                high = row["high"]

                # Update highest price and trail stop
                if high > highest_price:
                    highest_price = high
                    new_sl = highest_price - trail_distance
                    if new_sl > current_sl:
                        current_sl = new_sl

                # Check stop loss hit
                if low <= current_sl:
                    exit_price = current_sl - (SPREAD_PIPS * pip_value)
                    pnl = -risk_amount
                    exit_reason = "sl_trail"
                    break
            else:
                # Timeout - close at last close
                last_close = future_data["close"].iloc[-1]
                exit_price = last_close
                exit_pips = (last_close - actual_entry) / pip_value
                pnl = max(exit_pips, 0) * lots * pip_value_dollar
                exit_reason = "timeout"

        else:  # SHORT
            first_row = future_data.iloc[0]
            actual_entry = first_row["open"] - (SPREAD_PIPS * pip_value)

            # Initial stop loss
            current_sl = sl
            lowest_price = actual_entry

            for _, row in future_data.iterrows():
                low = row["low"]
                high = row["high"]

                # Update lowest price and trail stop
                if low < lowest_price:
                    lowest_price = low
                    new_sl = lowest_price + trail_distance
                    if new_sl < current_sl:
                        current_sl = new_sl

                # Check stop loss hit
                if high >= current_sl:
                    exit_price = current_sl + (SPREAD_PIPS * pip_value)
                    pnl = -risk_amount
                    exit_reason = "sl_trail"
                    break
            else:
                # Timeout - close at last close
                last_close = future_data["close"].iloc[-1]
                exit_price = last_close
                exit_pips = (actual_entry - last_close) / pip_value
                pnl = max(exit_pips, 0) * lots * pip_value_dollar
                exit_reason = "timeout"

        self.balance += pnl
        self.equity_curve.append(self.balance)
        self.peak_balance = max(self.peak_balance, self.balance)

        return {
            "pnl": pnl,
            "entry": entry,
            "exit": exit_price,
            "direction": "LONG" if direction == 1 else "SHORT",
            "reason": exit_reason,
            "lots": lots,
        }


def load_historical_data(symbol: str, data_dir: Path = HISTORICAL_DATA_DIR) -> Optional[pd.DataFrame]:
    """Load M1 data for a symbol from parquet files, downsampled to ~10K bars for backtesting."""
    parquet_files = list(data_dir.glob(f"{symbol}_M1_*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found for {symbol}")
        return None

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=False)

    # Ensure proper index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)

    # Downsample to H1-equivalent by taking every ~180th bar
    original_len = len(df)
    df = df.iloc[::180].copy()
    logger.info(f"Loaded {original_len} bars for {symbol}, downsampled to {len(df)} bars")
    return df


def generate_signals_vectorized(features: pd.DataFrame, symbol: str) -> list[Signal]:
    """
    Generate signals for all rows efficiently.
    Returns list of Signal dicts, one per row (most will be flat/None).
    """
    signals = []
    for idx in range(len(features)):
        row = features.iloc[idx]
        signal = _generate_signal_from_row(row, symbol, features.index[idx])
        signals.append(signal)
    return signals


def _generate_signal_from_row(row: pd.Series, symbol: str, timestamp: datetime) -> Signal:
    """Generate a single TREND-FOLLOWING signal from one feature row.
    Phase 9.5: Uses pullback confirmation logic instead of direct DC entry.
    - DC event + trending Hurst → Primed state (don't enter yet)
    - OFI pullback (crossing 0) → Actual entry trigger
    """
    entry = row["close"]
    atr = row.get("atr", 0.0)
    hurst = row.get("hurst", 0.5)
    ofi = row.get("ofi", 0.0)
    pullback_signal = row.get("pullback_signal", False)
    pullback_direction = row.get("pullback_direction", 0)

    regime = detect_market_regime(hurst)

    # ONLY TREND FOLLOWING: Require pullback confirmation
    # Phase 9.5: Enter only after pullback, not immediately after DC event
    # Note: pullback detection already verified trending regime at primed state
    direction = 0
    if pullback_signal and pullback_direction != 0:
        direction = pullback_direction

    if direction == 0:
        return Signal(symbol=symbol, direction=0, confidence=0.0, entry_price=0.0,
                     stop_loss=0.0, take_profit=0.0, timestamp=timestamp.replace(tzinfo=timezone.utc),
                     regime=regime, atr=atr)

    if direction == 1:
        sl = entry - (atr * TREND_SL_ATR)
        conf = max(MIN_SIGNAL_CONFIDENCE, min(abs(ofi), 1.0) * 0.5 + min((hurst - 0.5) * 2, 1.0) * 0.5)
        return Signal(symbol=symbol, direction=1, confidence=conf, entry_price=entry,
                     stop_loss=sl, take_profit=entry + (atr * TREND_TP_ATR),
                     timestamp=timestamp.replace(tzinfo=timezone.utc),
                     regime="trending", atr=atr)
    else:
        sl = entry + (atr * TREND_SL_ATR)
        conf = max(MIN_SIGNAL_CONFIDENCE, min(abs(ofi), 1.0) * 0.5 + min((hurst - 0.5) * 2, 1.0) * 0.5)
        return Signal(symbol=symbol, direction=-1, confidence=conf, entry_price=entry,
                     stop_loss=sl, take_profit=entry - (atr * TREND_TP_ATR),
                     timestamp=timestamp.replace(tzinfo=timezone.utc),
                     regime="trending", atr=atr)


def run_backtest(symbol: str, initial_balance: float = 10000.0) -> BacktestResult:
    """Run a backtest for a single symbol."""
    result = BacktestResult()
    account = SimulatedAccount(initial_balance)

    # Load data
    df = load_historical_data(symbol)
    if df is None or len(df) < 300:
        logger.warning(f"Insufficient data for {symbol}")
        return result

    # Build features once
    features = build_feature_matrix(df)
    if features.empty:
        logger.warning(f"No features generated for {symbol}")
        return result

    # CRITICAL: Align df to features (build_feature_matrix drops warmup rows)
    start_pos = df.index.get_loc(features.index[0])
    df_aligned = df.iloc[start_pos:].copy()

    # Pre-generate all signals (O(n) instead of O(n²))
    signals = generate_signals_vectorized(features, symbol)

    # Iterate through features and simulate trades
    # Note: features already have warmup rows dropped by build_feature_matrix,
    # so we use a small minimum warmup
    warmup = min(5, len(features) // 10)
    for idx in range(warmup, len(features)):
        current_time = features.index[idx]
        signal = signals[idx]

        # Track equity at every time step for proper drawdown calculation
        if len(account.equity_curve) <= idx:
            account.equity_curve.append(account.balance)

        # Skip flat signals
        if signal["direction"] == 0:
            continue

        # Check daily drawdown limit
        account.reset_daily(current_time)
        if not account.can_trade():
            continue

        # Apply filters
        account_info = {"equity": account.balance, "balance": account.balance}
        filtered = apply_signal_filters(signal, account_info, 0)

        if filtered is None or filtered["direction"] == 0:
            continue

        # Execute trade
        trade_result = account.execute_trade(filtered, df_aligned, idx)
        if trade_result:
            result.total_trades += 1
            result.total_pnl += trade_result["pnl"]

            if trade_result["pnl"] > 0:
                result.wins += 1
            else:
                result.losses += 1

            result.trades_log.append(trade_result)

    # Calculate max drawdown from equity curve (peak-to-trough)
    if account.equity_curve:
        equity = np.array(account.equity_curve)
        running_max = np.maximum.accumulate(equity)
        # FIX: Divide by running_max (peak), not initial_balance
        drawdowns = (running_max - equity) / running_max
        result.max_drawdown = float(drawdowns.max())

    return result


def main():
    """Run backtests on all available historical data."""
    import time

    start_time = time.time()

    # Find available symbols from parquet files
    parquet_files = list(HISTORICAL_DATA_DIR.glob("*_M1_*.parquet"))
    symbols = list(set(f.stem.split("_M1_")[0] for f in parquet_files))
    # Skip XAUUSD - removed from trading due to excessive volatility
    symbols = [s for s in symbols if s != "XAUUSD"]

    if not symbols:
        print("No historical data found. Run data/historical_extractor.py first.")
        return

    print(f"\n{'='*60}")
    print("QUANTEDGE MT5 BACKTEST ENGINE")
    print(f"{'='*60}")
    print(f"Symbols found: {symbols}")
    print(f"{'='*60}\n")

    all_results = {}

    for symbol in symbols:
        logger.info(f"Backtesting {symbol}...")
        result = run_backtest(symbol)
        all_results[symbol] = result

    # Print scoreboard
    print(f"\n{'='*60}")
    print("BACKTEST SCOREBOARD")
    print(f"{'='*60}")
    print(f"{'Symbol':<12} {'Trades':>8} {'Win Rate':>10} {'PnL':>12} {'Max DD':>10}")
    print("-" * 60)

    for symbol, result in all_results.items():
        stats = result.to_dict()
        print(
            f"{symbol:<12} {stats['total_trades']:>8} "
            f"{stats['win_rate']*100:>9.1f}% "
            f"${stats['total_pnl']:>10.2f} "
            f"{stats['max_drawdown']*100:>9.1f}%"
        )

    # Aggregate stats
    total_trades = sum(r.total_trades for r in all_results.values())
    total_wins = sum(r.wins for r in all_results.values())
    total_pnl = sum(r.total_pnl for r in all_results.values())
    execution_time = time.time() - start_time

    print("-" * 60)
    print(f"{'TOTAL':<12} {total_trades:>8} {total_wins/max(total_trades,1)*100:>9.1f}% ${total_pnl:>10.2f}")
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()