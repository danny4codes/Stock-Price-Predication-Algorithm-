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
    # XAUUSD removed - volatility too high for current signal logic
    # Re-add after optimizing volatility handling
]

# MetaTrader5 timeframe constants (import mt5 to use these)
# DO NOT hardcode integer values — use mt5.TIMEFRAME_* at runtime
DEFAULT_TIMEFRAME_STR   = "H1"    # Human-readable label for logging
HIGHER_TIMEFRAME_STR    = "H4"    # For trend context
LOWER_TIMEFRAME_STR     = "M15"   # For entry timing

# Historical data lookback for feature engineering
LOOKBACK_BARS           = 500     # Number of bars to fetch for feature calculation
FEATURE_WARMUP_BARS     = 250     # Bars discarded due to indicator warmup (must exceed HURST_ROLLING_WINDOW)

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
# FEATURE ENGINEERING PARAMETERS — Econometric & Microstructure Only
# ===========================================================================
# RETAIL INDICATORS PROHIBITED: No SMA/EMA crossover, RSI, MACD, ADX,
# Bollinger Bands, or Stochastic oscillators anywhere in the pipeline.

ATR_PERIOD              = 14      # ATR lookback — used for SL/TP sizing only
HURST_MIN_LAG           = 2       # Min lag for R/S Hurst analysis
HURST_MAX_LAG           = 100     # Max lag for R/S Hurst analysis
HURST_ROLLING_WINDOW    = 200     # Rolling window for Hurst computation (must be >= HURST_MAX_LAG * 2)
GARCH_P                 = 1       # GARCH lag order for past variances
GARCH_Q                 = 1       # ARCH lag order for past residuals
GARCH_ROLLING_WINDOW    = 100     # Rolling window for GARCH volatility

# Directional Changes (DC) parameters
DC_THETA                = 0.002   # Threshold for significant price movement (0.2%) - TIGHTENED
DC_MIN_PRICE            = 0.0001  # Min price for DC detection
DC_MICRO_THETA          = 0.0005  # Micro-DC threshold for pullback detection (0.05%)

# Order Flow Imbalance (OFI) parameters
OFI_WINDOW              = 20      # Rolling window for OFI smoothing
OFI_THRESHOLD           = 0.10    # Min |OFI| magnitude to confirm signal (adjusted for pullback logic)
OFI_PULLBACK_THRESHOLD  = 0.0     # OFI crossing zero for pullback detection

# Volatility Z-Score for mean-reversion entry
VOL_ZSCORE_WINDOW       = 50      # Window for GARCH vol z-score calc
VOL_ZSCORE_THRESHOLD    = 1.0     # Z-score threshold for vol spike

# VWAP deviation for mean-reversion entry
VWAP_DIST_THRESHOLD     = 0.001   # Min |price - VWAP| / VWAP for MR entry

# Microstructure Pullback entry parameters
PULLBACK_WAIT_BARS      = 10      # Max bars to wait for pullback after Primed state
OFI_CONFIRMATION_BARS   = 3       # Bars OFI must sustain above/below zero for confirmation

# ===========================================================================
# REGIME DETECTION THRESHOLDS (Hurst Exponent)
# ===========================================================================
HURST_TRENDING_THRESHOLD      = 0.58   # H > 0.58 → trending (TIGHTENED for quality)
HURST_MEAN_REVERT_THRESHOLD   = 0.35   # H < 0.35 → mean reverting
# H between 0.42 and 0.58 → random walk (no trade)

# ===========================================================================
# SIGNAL GENERATION THRESHOLDS
# ===========================================================================
MIN_SIGNAL_CONFIDENCE        = 0.30   # Minimum combined confidence to trade
CONFIDENCE_THRESHOLD         = 0.25   # Minimum confidence at final filter gate
MIN_RISK_REWARD_RATIO        = 1.5    # Minimum R:R ratio for signal approval

# ATR Multipliers for stop-loss / take-profit sizing
TREND_SL_ATR                = 1.5    # Trend SL = ATR × this
TREND_TP_ATR                = 3.0    # Trend TP = ATR × this
MR_SL_ATR                   = 1.0    # Mean-reversion SL = ATR × this
MR_TP_ATR                   = 1.5    # Mean-reversion TP = ATR × this

# ===========================================================================
# LOGGING
# ===========================================================================
LOG_MAX_BYTES           = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT        = 5                 # Keep 5 rotated log files
LOG_FORMAT              = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT         = "%Y-%m-%d %H:%M:%S"