# config.py — QuantEdge MT5 Configuration Constants
# All non-secret configuration lives here.
# Secrets (login, password, server) go in .env — NEVER here.

from pathlib import Path

# ===========================================================================
# PATHS
# ===========================================================================
BASE_DIR    = Path(r"D:\Trading")
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR   = BASE_DIR / "models"
LOG_DIR     = BASE_DIR / "logs"

# Auto-create required directories on import
for _dir in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# RISK PARAMETERS — DO NOT CHANGE WITHOUT EXPLICIT USER APPROVAL
# ===========================================================================
RISK_PER_TRADE_PCT      = 0.01    # 1% of account balance per trade — FIXED FRACTIONAL
MAX_DAILY_DRAWDOWN_PCT  = 0.04    # 4% max drawdown per day — system halts at breach
MAX_CONCURRENT_POSITIONS = 5      # Maximum open positions at any time
MAX_SLIPPAGE_PIPS       = 3.0     # Reject fill if slippage exceeds this

# ===========================================================================
# TRADING PARAMETERS
# ===========================================================================
# Symbols to trade (add/remove as needed)
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "XAUUSD",  # Gold
]

# MetaTrader5 timeframe constants (import mt5 to use these)
# DO NOT hardcode integer values — use mt5.TIMEFRAME_* at runtime
DEFAULT_TIMEFRAME_STR   = "H1"    # Human-readable label for logging
HIGHER_TIMEFRAME_STR    = "H4"    # For trend context
LOWER_TIMEFRAME_STR     = "M15"   # For entry timing

# Historical data lookback for feature engineering
LOOKBACK_BARS           = 500     # Number of bars to fetch for feature calculation
FEATURE_WARMUP_BARS     = 100     # Bars discarded due to indicator warmup

# ===========================================================================
# MAIN LOOP PARAMETERS
# ===========================================================================
LOOP_INTERVAL_SECONDS   = 60      # Main async loop tick rate (seconds)
MT5_MAX_RETRIES         = 5       # Exponential backoff max attempts
MT5_BASE_DELAY_SECONDS  = 1.0     # Base delay for exponential backoff

# ===========================================================================
# MACHINE LEARNING PARAMETERS
# ===========================================================================
ML_FORWARD_PERIODS      = 5       # Bars ahead to predict
ML_PRICE_THRESHOLD      = 0.001   # Minimum return to label as signal (0.1%)
ML_WALK_FORWARD_SPLITS  = 5       # TimeSeriesSplit n_splits
ML_MODEL_TYPE           = "random_forest"  # Options: "random_forest", "gradient_boost", "svm"
ML_TEST_SIZE_RATIO      = 0.2     # Ratio of data reserved for final test set

# ===========================================================================
# FEATURE ENGINEERING PARAMETERS
# ===========================================================================
RSI_PERIOD              = 14
MACD_FAST               = 12
MACD_SLOW               = 26
MACD_SIGNAL             = 9
BOLLINGER_PERIOD        = 20
BOLLINGER_STD           = 2.0
ATR_PERIOD              = 14
HURST_MIN_LAG           = 2
HURST_MAX_LAG           = 100
GARCH_P                 = 1
GARCH_Q                 = 1

# ===========================================================================
# REGIME DETECTION THRESHOLDS (Hurst Exponent)
# ===========================================================================
HURST_TRENDING_THRESHOLD      = 0.55   # H > 0.55 → trending
HURST_MEAN_REVERT_THRESHOLD   = 0.45   # H < 0.45 → mean reverting
# H between 0.45 and 0.55 → random walk (no trade)

# ===========================================================================
# LOGGING
# ===========================================================================
LOG_MAX_BYTES           = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT        = 5                 # Keep 5 rotated log files
LOG_FORMAT              = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT         = "%Y-%m-%d %H:%M:%S"
