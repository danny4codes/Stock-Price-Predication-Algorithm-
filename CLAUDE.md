# CLAUDE.md — Institutional-Grade Algorithmic Trading System
## Operating Manual for AI-Assisted Development

> **This file is the authoritative specification for this project.**
> All AI agents (Claude Code, Antigravity) MUST read and adhere to every rule defined here before writing, modifying, or deleting any code.

---

## 0. Project Overview

| Property         | Value                                                      |
|------------------|------------------------------------------------------------|
| **Project Name** | QuantEdge MT5 — Algorithmic Trading System                 |
| **Language**     | Python 3.14+                                               |
| **Platform**     | MetaTrader 5 (MT5) on Windows                              |
| **Root Path**    | `D:\Trading\`                                              |
| **OS**           | Windows 11 / Windows 10                                    |
| **AI Agents**    | Claude Code (CLI), Antigravity                             |
| **Shell**        | PowerShell (Windows)                                       |

---

## 1. System Architecture

The system is **strictly modular**. Each responsibility is isolated into its own file. No business logic may bleed across module boundaries. All inter-module communication happens through well-defined interfaces (function calls or shared data contracts).

```
D:\Trading\
│
├── CLAUDE.md                  ← This file (AI operating manual)
├── CONTEXT.md                 ← Project state & session memory
├── .env                       ← Secrets & environment variables (never commit)
├── requirements.txt           ← Pinned dependency manifest
├── setup.py                   ← Project installation script
│
├── data_ingestion.py          ← MT5 connection & historical/live data
├── feature_engineering.py     ← Technical indicators & custom metrics
├── signal_generation.py       ← Entry/exit signal logic
├── machine_learning.py        ← Model training, validation & inference
├── execution.py               ← Order placement & position tracking
│
├── risk_manager.py            ← Position sizing, drawdown enforcement
├── logger_config.py           ← Centralized structured logging setup
├── config.py                  ← All configuration constants (not secrets)
│
├── models/                    ← Serialized trained models (.pkl, .joblib)
├── data/
│   ├── raw/                   ← Raw OHLCV data from MT5
│   └── processed/             ← Feature-engineered datasets
├── logs/                      ← Rotating log files (auto-created)
└── tests/                     ← Unit & integration tests
```

---

## 2. Module Specifications

### 2.1 `data_ingestion.py` — MT5 Connection & Data

**Responsibility**: All MetaTrader 5 connectivity and data retrieval. No other module may import `MetaTrader5` directly.

**Required Functions**:
```python
def initialize_mt5(login: int, password: str, server: str, path: str) -> bool
def shutdown_mt5() -> None
def get_historical_data(symbol: str, timeframe: int, start: datetime, end: datetime) -> pd.DataFrame
def get_live_tick(symbol: str) -> dict
def get_account_info() -> dict
def get_open_positions() -> pd.DataFrame
```

**Rules**:
- Always call `mt5.initialize()` inside a `try/except` block
- Implement **exponential backoff** (max 5 retries, starting at 1s) on connection failure
- Return `None` or empty `pd.DataFrame` (never raise) on data fetch failure — log the error
- All returned DataFrames MUST have a `DatetimeIndex` in UTC
- Use `mt5.TIMEFRAME_*` constants — never hardcode timeframe integers

---

### 2.2 `feature_engineering.py` — Indicators & Metrics

**Responsibility**: Transform raw OHLCV data into ML-ready features. Pure functions only — no side effects.

**Required Functions**:
```python
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame
def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series
def compute_vwap(df: pd.DataFrame) -> pd.Series
def compute_hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float
def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> arch.arch_model
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame
```

**Rules**:
- All functions must be **pure** (same input → same output, no global state)
- Handle `NaN` propagation gracefully — document which functions drop NaNs
- `build_feature_matrix` is the single entry point that composes all features
- No hardcoded lookback periods outside of function default arguments

---

### 2.3 `signal_generation.py` — Entry/Exit Logic

**Responsibility**: Convert feature matrix into discrete trading signals. No ML models here.

**Signal Contract**:
```python
# Signal values: 1 = LONG, -1 = SHORT, 0 = FLAT/NO ACTION
class Signal(TypedDict):
    symbol: str
    direction: int          # 1, -1, or 0
    confidence: float       # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    regime: str             # "trending" | "mean_reverting" | "random"
```

**Required Functions**:
```python
def detect_market_regime(hurst: float) -> str
def generate_trend_signal(features: pd.DataFrame) -> Signal
def generate_mean_reversion_signal(features: pd.DataFrame) -> Signal
def combine_signals(trend_sig: Signal, mr_sig: Signal, ml_confidence: float) -> Signal
def apply_signal_filters(signal: Signal, account_info: dict) -> Signal | None
```

**Rules**:
- `apply_signal_filters` is the final gate — it must check against `risk_manager.py` before returning
- Never generate a signal if `account_info` cannot be fetched
- All signal logic must be unit-testable with mock DataFrames

---

### 2.4 `machine_learning.py` — Model Training & Inference

**Responsibility**: Train, validate, serialize, load, and run ML models.

**Required Functions**:
```python
def prepare_labels(df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.001) -> pd.Series
def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> BaseEstimator
def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5) -> dict
def save_model(model: BaseEstimator, path: str) -> None
def load_model(path: str) -> BaseEstimator
def predict(model: BaseEstimator, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]
```

**Rules**:
- **NEVER** use future data in training (strict temporal split, no shuffle)
- Use `walk_forward_validate` — standard K-Fold cross-validation is **forbidden** for time series
- Models must be saved to `D:\Trading\models\` with timestamp in filename
- `predict` returns `(predictions, probabilities)` always as a tuple
- Log training metrics (accuracy, F1, Sharpe of signals) via `logger_config`

---

### 2.5 `execution.py` — Order Placement & Position Tracking

**Responsibility**: All MetaTrader 5 trade execution. Must route through `risk_manager.py` first.

**Required Functions**:
```python
def place_market_order(symbol: str, direction: int, volume: float, sl: float, tp: float) -> dict
def place_limit_order(symbol: str, direction: int, price: float, volume: float, sl: float, tp: float) -> dict
def close_position(ticket: int) -> bool
def modify_sl_tp(ticket: int, new_sl: float, new_tp: float) -> bool
def get_position_by_symbol(symbol: str) -> dict | None
def cancel_pending_order(ticket: int) -> bool
```

**Rules**:
- ALWAYS call `risk_manager.validate_order()` before any `mt5.order_send()`
- Check `mt5.last_error()` after every order operation and log it
- Implement **slippage guard**: if fill price deviates > 3 pips from requested, log WARNING
- No position may be opened without a valid Stop Loss — this is **non-negotiable**
- All order results must be logged at INFO level with full ticket details

---

## 3. Core Dependencies

```
# requirements.txt — PINNED versions required for reproducibility
MetaTrader5>=5.0.4885
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
arch>=6.3.0
hurst>=0.0.5
python-dotenv>=1.0.0
pytz>=2024.1
joblib>=1.3.0
ta>=0.11.0
```

**Installation**:
```powershell
# Run from D:\Trading\ in PowerShell as Administrator
pip install -r requirements.txt
```

**Compatibility Notes**:
- `MetaTrader5` library is **Windows-only** — no Linux/macOS support
- `arch` requires `numpy` to be installed first
- `hurst` package may require `numpy<2.0` — verify after install

---

## 4. Execution Rules

### 4.1 Asynchronous Execution

```python
# REQUIRED pattern for the main trading loop
import asyncio
import logging

async def trading_loop():
    """Main async trading loop. Runs indefinitely until KeyboardInterrupt."""
    logger = logging.getLogger("trading_loop")
    while True:
        try:
            # 1. Fetch live data
            # 2. Engineer features
            # 3. Generate signal
            # 4. Execute if valid
            await asyncio.sleep(60)  # 1-minute cadence default
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled gracefully.")
            break
        except Exception as e:
            logger.error(f"Unhandled error in trading loop: {e}", exc_info=True)
            await asyncio.sleep(5)  # brief pause before retry

if __name__ == "__main__":
    asyncio.run(trading_loop())
```

### 4.2 Error Handling — MT5 Connection Drops

```python
# REQUIRED pattern for MT5 reconnection
def safe_mt5_call(func, *args, max_retries=5, base_delay=1.0, **kwargs):
    """Wraps any MT5 function call with exponential backoff retry."""
    for attempt in range(max_retries):
        result = func(*args, **kwargs)
        if result is not None:
            return result
        error = mt5.last_error()
        logger.warning(f"MT5 call failed (attempt {attempt+1}): {error}")
        time.sleep(base_delay * (2 ** attempt))
    logger.critical("MT5 call failed after max retries. Halting execution.")
    return None
```

### 4.3 Structured Logging

```python
# logger_config.py — centralized setup
import logging
import logging.handlers
from pathlib import Path

def setup_logger(name: str, log_dir: str = r"D:\Trading\logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Rotating file handler — 10MB max, 5 backups
    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

---

## 5. Risk Parameters

### 5.1 Position Sizing — Fixed Fractional (1% Per Trade)

```python
# risk_manager.py — core sizing logic
def calculate_position_size(
    account_balance: float,
    risk_pct: float,        # Fixed at 0.01 (1%)
    stop_loss_pips: float,
    pip_value: float,
    symbol: str
) -> float:
    """
    Fixed fractional position sizing.
    Risk per trade = 1% of account balance. NEVER exceed this.
    """
    MAX_RISK_PCT = 0.01     # HARDCODED — do not parameterize further
    risk_amount = account_balance * MAX_RISK_PCT
    lots = risk_amount / (stop_loss_pips * pip_value)
    return round(lots, 2)
```

### 5.2 Daily Drawdown Limit (4%)

```python
def check_daily_drawdown(
    account_info: dict,
    daily_start_balance: float
) -> bool:
    """
    Returns True if trading is ALLOWED, False if daily limit is breached.
    Maximum daily drawdown: 4% of balance at day start.
    """
    MAX_DAILY_DD = 0.04     # HARDCODED — 4% absolute maximum
    current_equity = account_info["equity"]
    drawdown = (daily_start_balance - current_equity) / daily_start_balance
    if drawdown >= MAX_DAILY_DD:
        logger.critical(
            f"DAILY DRAWDOWN LIMIT REACHED: {drawdown:.2%}. "
            "All new orders BLOCKED for the rest of the session."
        )
        return False
    return True
```

### 5.3 Hard Rules — Non-Negotiable

| Rule                        | Enforcement                                    |
|-----------------------------|------------------------------------------------|
| Max risk per trade          | **1%** of account balance — fixed fractional   |
| Max daily drawdown          | **4%** — system halts new orders at breach     |
| Stop Loss requirement       | **Mandatory** on every position — no exceptions|
| Martingale logic            | **FORBIDDEN** — never double down on losses    |
| Grid trading                | **FORBIDDEN** — no overlapping same-direction orders |
| Averaging down              | **FORBIDDEN** — losing positions are not added to |
| Max concurrent positions    | **5** — configurable in `config.py`            |

---

## 6. Environment Specifics (Windows)

### 6.1 Path Conventions

```python
# ALWAYS use pathlib.Path for file operations on Windows
from pathlib import Path

BASE_DIR = Path(r"D:\Trading")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# NEVER hardcode forward slashes for Windows paths
# NEVER use os.path.join — use pathlib exclusively
```

### 6.2 MT5 Path Configuration

```python
# In .env file (never commit this file):
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# Load with:
from dotenv import load_dotenv
import os
load_dotenv(r"D:\Trading\.env")
mt5_path = os.getenv("MT5_PATH")
```

### 6.3 PowerShell Execution Policy

```powershell
# If scripts fail to run, execute once in admin PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 6.4 Virtual Environment

```powershell
# Create and activate (from D:\Trading\):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Always activate before running any Python files
# Always install packages into the venv, not system Python
```

---

## 7. Claude Code Status & Troubleshooting

### Current Status
- **Claude Code CLI**: ✅ Installed — version 2.1.141
- **VS Code Extension**: ❌ `code` command not in PATH — VS Code may not be installed or PATH not configured
- **Python**: ✅ 3.14.4 available
- **Dependencies**: ❌ None installed yet — run `pip install -r requirements.txt`

### Why Claude Code CLI Works But VS Code Doesn't

The `claude` CLI is installed as a standalone Node.js package (globally via npm). VS Code (`code` command) is a separate application. If `code` is not recognized in PowerShell:

1. **Option A — Install VS Code**: Download from https://code.visualstudio.com/ and check "Add to PATH" during installation
2. **Option B — Add manually**: Find VS Code at `C:\Users\Admin\AppData\Local\Programs\Microsoft VS Code\bin\` and add to system PATH
3. **Option C — Use Claude Code CLI directly**: `claude` in PowerShell already works — use it from `D:\Trading\`

### Using Claude Code CLI

```powershell
# Navigate to project root
cd D:\Trading

# Start Claude Code session
claude

# Or run with a specific prompt
claude "Review data_ingestion.py and check for MT5 error handling gaps"
```

---

## 8. Development Workflow

### Starting a New Development Session

```powershell
# 1. Navigate to project
cd D:\Trading

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Verify MT5 is running (must be open before Python connects)
# Open MetaTrader 5 manually

# 4. Run the system
python main.py
```

### Code Quality Rules

- **Type hints are mandatory** on all public functions
- **Docstrings are mandatory** on all public functions (Google style)
- **No magic numbers** — all constants go in `config.py`
- **No print statements** — use `logger` exclusively
- **Every function > 20 lines** must have a corresponding unit test in `tests/`
- **Git commit** after every working milestone (no broken commits)

---

## 5. STRICT PROHIBITIONS

> **These rules are ABSOLUTE. Violations will be rejected at code review.**

| Rule | Enforcement |
|------|-------------|
| **NEVER** implement or import retail technical indicators such as Moving Averages (SMA/EMA), RSI, MACD, ADX, or Bollinger Bands. | Architecture review — blocked on merge. |
| **ALWAYS** strictly rely on the econometric and microstructure models defined: **Hurst Exponent**, **GARCH(1,1)**, **Directional Changes (DC)**, and **Order Flow Imbalance (OFI)**. | Enforced by module boundary rules. |
| **NEVER** import `MetaTrader5` in any module other than `data_ingestion.py`. | Code linting gate — `from MetaTrader5` or `import MetaTrader5` only allowed in `data_ingestion.py`. |
| **NEVER** use future data in training labels — `prepare_labels()` must drop the last N rows. | Test assertion enforced. |
| **NEVER** use standard K-Fold cross-validation for time series — walk-forward validation only. | Code review gate. |

---

## 6. Environment Specifics (Windows)

> **READ THIS BEFORE MAKING ANY CODE CHANGES**

1. **Always read `CONTEXT.md`** at the start of every session to understand current state
2. **Never modify `risk_manager.py` risk percentages** without explicit user confirmation
3. **Never commit `.env`** — it contains secrets
4. **Always update `CONTEXT.md`** after completing a task to reflect new state
5. **Follow the module boundaries** strictly — data flows in one direction only:
   ```
   data_ingestion → feature_engineering → signal_generation → execution
                                    ↗
                    machine_learning
   ```
6. **Test before declaring done** — run the relevant test in `tests/` after every change
7. **Log, don't print** — all output goes through `logger_config.setup_logger()`
8. **Ask before deleting** — never remove existing functions without user confirmation

---

*Last Updated: 2026-05-14 | Maintainer: QuantEdge System*
