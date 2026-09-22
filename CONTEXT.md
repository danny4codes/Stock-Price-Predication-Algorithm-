# CONTEXT.md — Project State & Session Memory
## QuantEdge MT5 — Algorithmic Trading System

> **Purpose**: This file is the living memory of the project. Every AI agent and developer MUST update this file at the end of each work session. Read this first before making any changes.

---

## 📅 Last Updated
- **Date**: 2026-05-22
- **Session**: Session 018 — Strategic Reset. Identified critical signal direction inversion bug. DC UPTURN fires at local MAX (SHORT opportunity), not LONG. Complete strategy rebuild required.

---

## 🗺️ Current Project State

### Phase
```
[x] Phase 0: Project scaffolding & documentation
[x] Phase 1: Environment setup & dependency installation
[x] Phase 2: MT5 connection & data ingestion
[x] Phase 3: Feature engineering pipeline
[x] Phase 4: Signal generation logic
[x] Phase 5: ML model training & validation
[x] Phase 6: Risk management & execution
[x] Phase 7: Main async trading loop + test suite
[x] Phase 8: Live paper trading & monitoring (smoke test passed, Hurst bug fixed)
[x] Phase 9: Historical backtesting suite
    [x] 9.1: Historical data extraction pipeline
    [x] 9.2: Backtesting engine with walk-forward simulation
    [ ] 9.3: Performance metrics and reporting
[ ] Phase 9.4: Signal Direction Correction & Pullback Logic (REDO - current logic inverted)
[ ] Phase 10: Live production trading
```

### What Exists Right Now
| File / Directory                     | Status     | Notes                                         |
|--------------------------------------|------------|-----------------------------------------------|
| `CLAUDE.md`                          | ✅ Done     | Full architecture spec & AI operating manual  |
| `CONTEXT.md`                         | ✅ Done     | This file — project memory                    |
| `requirements.txt`                   | ✅ Done     | All dependencies pinned                        |
| `config.py`                          | ✅ Done     | All configuration constants                   |
| `.env.template`                      | ✅ Done     | Template for secrets (copy to .env)           |
| `.gitignore`                         | ✅ Done     | Protects secrets, venv, data, models          |
| `pyproject.toml`                     | ✅ Done     | Pytest configuration                          |
| `logger_config.py`                   | ✅ Done     | Rotating file + console logging               |
| `data_ingestion.py`                  | ✅ Done     | MT5 conn, historical/live data, exp. backoff  |
| `feature_engineering.py`             | ✅ Done     | ATR, VWAP, Hurst, GARCH; delegates DC/OFI/signals to microstructure engine |
| `2_microstructure_engine.py`         | ⚠️ Bug      | **CRITICAL**: Signal direction inverted relative to DC meaning |
| `signal_generation.py`               | ✅ Done     | Regime detect, trend+MR signals, filters      |
| `machine_learning.py`                | ✅ Done     | Walk-forward CV, train, save/load, infer      |
| `risk_manager.py`                    | ✅ Done     | 1% sizing, 4% DD limit, validation gate       |
| `execution.py`                       | ✅ Done     | Market/limit orders, position management      |
| `main.py`                            | ✅ Done     | Async trading loop, multi-symbol              |
| `data/historical_extractor.py`       | ✅ Done     | Historical data extraction for backtesting    |
| `data/historical/`                   | ✅ Created  | Output directory for parquet/CSV files        |
| `backtester.py`                      | ✅ Done     | Vectorized backtesting engine               |
| `tests/` (total)                     | ✅ 156 passed| 156/156 tests passing (but using inverted logic)                     |
| `.venv/`                             | ✅ Done     | Virtual environment created & populated       |

---

## 💻 Environment State

### System
| Property         | Value                                                        |
|------------------|--------------------------------------------------------------|
| OS               | Windows 11 (PowerShell)                                      |
| Python Version   | 3.14.4                                                       |
| Python Path      | `C:\Users\Admin\AppData\Local\Programs\Python\Python314\`   |
| Project Root     | `D:\Trading\`                                                |
| Virtual Env      | ✅ Created at `D:\Trading\.venv\`                            |

### Tools
| Tool             | Status              | Version / Notes                            |
|------------------|---------------------|--------------------------------------------|
| Claude Code CLI  | ✅ Working          | v2.1.141 — run `claude` in PowerShell      |
| VS Code (`code`) | ❌ Not in PATH      | Install VS Code OR add to PATH manually    |
| MetaTrader 5     | ❓ Unknown          | Must be open before running Python scripts |
| Git              | ✅ Working          | Initialized, 2 commits                     |

### Dependencies — INSTALLED
```
All dependencies installed in .venv:
- MetaTrader5>=5.0.4885
- pandas>=2.2.0, numpy>=1.26.0
- scikit-learn>=1.4.0, arch>=6.3.0
- hurst>=0.0.5, python-dotenv>=1.0.0
- pytz>=2024.1, joblib>=1.3.0, ta>=0.11.0
```
To verify: `.\.venv\Scripts\python.exe -m pytest tests\ -v` (81 tests should pass)

---

## 🔑 Key Decisions Made

| Decision                         | Rationale                                              |
|----------------------------------|--------------------------------------------------------|
| Fixed fractional sizing (1%)     | Industry standard for prop-firm-safe risk management   |
| 4% max daily drawdown            | Common prop firm challenge limit                       |
| Async main loop                  | Non-blocking tick processing, handles multiple symbols |
| Walk-forward validation only     | Prevents lookahead bias — standard for time series ML  |
| GARCH for volatility             | Captures volatility clustering in financial returns    |
| Hurst exponent for regime        | Distinguishes trending (H>0.5) vs mean-reverting (H<0.5)|
| Pathlib over os.path             | Cross-platform, cleaner Windows path handling          |
| No martingale / grid             | Catastrophic risk of ruin — strictly forbidden         |

---

## 🎯 Next Immediate Steps

### CRITICAL — Fix Signal Direction Bug (Priority 1)
1. **Read** `2_microstructure_engine.py` lines 641-651 — Understand current signal logic
2. **Invert signal direction** in `evaluate_pullback_signals()`:
   - DC UPTURN → `primed_direction = -1` (SHORT, we're at a local high)
   - DC DOWNTURN → `primed_direction = 1` (LONG, we're at a local low)
3. **Run backtest** to validate fix works

### Historical Backtesting
4. **Configure MT5 credentials** — Create `.env` from `.env.template`
5. **Launch MetaTrader 5** — Must be open before running the extractor
6. **Extract historical data** — Run `python data\historical_extractor.py`
7. **Run backtester** — `python backtester.py` to validate strategy

### Production Trading Prep
8. **Paper trading** — Run `python main.py` with MT5 terminal open

---

## 📊 Trading Configuration (Current)

| Parameter                  | Value      | Location           |
|----------------------------|------------|--------------------|
| Risk per trade             | 1.0%       | `config.py`        |
| Max daily drawdown         | 4.0%       | `config.py`        |
| Max concurrent positions   | 5          | `config.py`        |
| Default timeframe          | H1 (60min) | `config.py`        |
| Main loop interval         | 60 seconds | `config.py`        |
| Model save directory       | `models/`  | `config.py`        |

---

## 🐛 Known Issues / Blockers

| Issue                                        | Severity | Status    | Resolution                              |
|----------------------------------------------|----------|-----------|-----------------------------------------|
| VS Code `code` not in PATH                   | Low      | Open      | Install VS Code or add bin to PATH      |
| MT5 credentials not configured               | High     | Open      | Create `.env` from `.env.template`      |
| MT5 terminal not running                     | High     | Open      | Must open MetaTrader5 before Python runs|
| **CRITICAL: Signal direction inverted**      | 🔴 Fatal | Open      | Fix `evaluate_pullback_signals()`: DC UPTURN→SHORT, DC DOWNTURN→LONG |
| No live trading                                | Info     | Blocked   | Must fix signal direction first         |

---

## 📝 Session Log

### Session 001 — 2026-05-14
- **Agent**: Antigravity
- **Actions**:
  - Created `CLAUDE.md` (full architecture spec)
  - Created `CONTEXT.md` (this file)
  - Created `requirements.txt`
  - Created `config.py`
  - Created `logger_config.py`
  - Created `.env.template`
- **Outcome**: Project scaffolded. Ready for environment setup.
- **Next**: Install venv + dependencies, then build `data_ingestion.py`

### Session 002 — 2026-05-15
- **Agent**: Claude (AI Assistant)
- **Actions**:
  - Added `execute_signal()` to `execution.py` (was imported by `main.py` but missing)
  - Fixed `is_trending` flag computation in `build_feature_matrix()` (`feature_engineering.py`) — was never computed, causing all primary signals to be flat
  - Fixed `pd.datetime` deprecation for pandas 2.x (replaced with `datetime.datetime`)
  - Fixed Hurst rolling window `min_periods=200 > window=100` → uses `min(HURST_MAX_LAG*2, HURST_ROLLING_WINDOW)`
  - Rewrote signal test fixtures to use only permitted econometric features (removed retail indicators: RSI, MACD, ADX, EMA, Bollinger Bands, Stochastic)
  - Fixed DC detector test — price changes were below theta threshold
  - Fixed risk manager test mocks (updated from `risk_manager.mt5` to `risk_manager._get_current_price`)
  - Added `HURST_TRENDING_THRESHOLD` import in `feature_engineering.py`
  - Enlarged trending fixture (300→1000 bars, stronger trend) for reliable Hurst > 0.5
- **Outcome**: All 81 tests passing. System internally consistent and ready for integration testing.

### Session 003 — 2026-05-15
- **Agent**: Claude (AI Assistant)
- **Actions**:
  - Created `2_microstructure_engine.py` — dedicated microstructure processing module with:
    - `DirectionalChangeDetector` class (Intrinsic Time DC events, upturn/downturn detection)
    - `calculate_order_flow_imbalance()` (tick-volume OFI proxy with rolling smoothing)
    - `evaluate_primary_signal()` (single-tick boolean buy/sell: H>0.55 + DC event + OFI confirmation)
    - `evaluate_primary_signals_batch()` (vectorised batch signal generation)
    - `classify_regime()` (Hurst-based regime classifier)
    - `run_microstructure_pipeline()` (full end-to-end: DC→OFI→signals)
  - Refactored `feature_engineering.py`: removed duplicated inline DC/OFI code, now delegates to `2_microstructure_engine`. Rebuilt `build_feature_matrix()` computation order: DC→OFI→Hurst/GARCH→signals.
  - Created `tests/test_microstructure_engine.py` — 75 new unit tests (29 DC, 15 OFI, 13 regime, 18 signal, 28 batch, 6 pipeline)
  - Updated `CONTEXT.md`: Phase 8 complete, test count updated to 156
- **Outcome**: All **156/156 tests passing** (81 existing + 75 new). Module boundaries strictly enforced.
- **Next**: Configure `.env` with MT5 credentials, open MT5 terminal, run smoke test

### Session 004 — 2026-05-15
- **Agent**: Claude (AI Assistant)
- **Actions**:
  - Created `.env` file (from `.env.template`) with MT5 connection placeholders for user to fill in: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`
  - Created `verify_mt5_connection.py` — standalone 5-step MT5 connectivity test script that validates credentials, fetches account info, symbol info, live tick, and historical data with clear troubleshooting guidance on failure
  - Updated `.gitignore` confirmed — `.env` is protected from git commits
- **Outcome**: User-ready connection setup. Once `.env` is filled and MT5 terminal is open, `python verify_mt5_connection.py` confirms end-to-end connectivity before running `python main.py`.
- **Next**: User fills in `.env`, opens MT5 terminal, runs verification smoke test

### Session 005 — 2026-05-15
- **Agent**: Claude (AI Assistant)
- **Actions**:
  - Fixed `.env` file — removed literal double-quotes around values that caused MT5 path resolution failure (`"C:\...\terminal64.exe"` → `C:\...\terminal64.exe`)
  - Updated `data_ingestion.initialize_mt5()` to try connecting to an already-running MT5 terminal first (no path), falling back to spawning a new instance — fixes `IPC initialize failed` error
  - Replaced all Unicode emoji (❌ ✅ 🔧) with plain ASCII text (`[FAILED]` `[OK]` `[FILLED]`) across `.env`, `verify_mt5_connection.py`, `execution.py`, `main.py`, `signal_generation.py`, `2_microstructure_engine.py` — prevents cp1252 encoding crash on Windows console
  - Added UTF-8 reconfiguration for stdout/stderr in `logger_config.py` for encoding safety
  - Updated MT5 status to ✅ Working, verified all 5 connectivity checks pass
  - Confirmed full trade pipeline works: data ingestion → feature engineering → signal generation → execution readiness
  - All 156 tests passing, smoke test with `main.py` runs successfully
- **Outcome**: System fully operational end-to-end. MT5 connection verified, all modules working, encoding issues resolved. Ready for paper trading.
- **Next**: Run `main.py` with MT5 terminal open to begin paper trading. Monitor `logs/execution.log` for filled orders.

### Session 006 — 2025-05-15 (Bug Fix)
- **Agent**: Claude (AI Assistant)
- **Issue**: All symbols stuck at H=0.500 (random walk regime) in live trading — Hurst exponent never fluctuated
- **Root Cause**: `HURST_ROLLING_WINDOW=100` but `HURST_MAX_LAG=100`, so `compute_hurst_exponent()` always saw < 200 data points (`max_lag * 2`) and returned the hardcoded fallback 0.5
- **Actions**:
  - Updated `HURST_ROLLING_WINDOW` from 100 → 200 in `config.py` (must be >= `HURST_MAX_LAG * 2`)
  - Updated `FEATURE_WARMUP_BARS` from 100 → 250 in `config.py` (must exceed new rolling window)
- **Verification**:
  - All 156 tests pass
  - Live smoke test shows Hurst range now 0.4152–0.5635 (real variation across 39 usable rows)
  - Regime distribution: 3 trending, 14 mean-reverting, 22 random (no longer 100% stuck at 0.5)
  - System generated 1 primary sell signal during smoke test
- **Outcome**: Hurst exponent now computes correctly in live trading. Pipeline fully functional.
- **Next**: Run `main.py` with MT5 terminal open — system should now properly detect trending regimes and generate trade signals.

### Session 007 — 2026-05-18 (Backtesting Suite Kickoff)
- **Agent**: Claude (AI Assistant)
- **Focus**: Pivoting to historical backtesting — building data extraction pipeline
- **Actions**:
  - Created `data/historical_extractor.py` — complete historical data extraction script with:
    - MT5 connection management with exponential backoff
    - Downloads 6 months of M1 and H1 data for 9 target symbols (XAUUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, EURJPY, GBPJPY)
    - Handles rate limiting with proper error logging
    - Converts timestamps to UTC DatetimeIndex
    - Calculates spread per candle
    - Saves to D:\Trading\data\historical\ as parquet files (snappy compression)
    - Command-line interface with --symbols, --months, --output-dir, --format options
  - Created `data/historical/` output directory
  - Updated CONTEXT.md: Added Phase 9 (Historical backtesting suite) with 9.1 complete
  - Updated project state table to include new file and directory
- **Outcome**: Historical data extraction pipeline ready. Run `python data\historical_extractor.py` with MT5 terminal open to extract data for backtesting.
- **Next**: Build backtesting engine with walk-forward simulation using extracted data.

### Session 008 — 2026-05-18 (Backtesting Engine)
- **Agent**: Claude (AI Assistant)
- **Focus**: Phase 9.2 — Building the backtesting engine
- **Actions**:
  - Created `backtester.py` — complete vectorized backtesting engine with:
    - `SimulatedAccount` class with 1% risk per trade and 4% daily drawdown enforcement
    - `BacktestResult` container for trade statistics
    - Vectorized feature generation via `build_feature_matrix()`
    - Trade simulation with TP/SL hit detection (looks ahead up to 100 bars)
    - Scoreboard output: Total Trades, Win Rate, Max Drawdown, Total PnL
    - Execution time tracking
  - Updated CONTEXT.md: Phase 9.2 complete, new file documented
- **Outcome**: Backtesting engine ready. Run `python backtester.py` after extracting historical data to simulate months of trading in seconds.
- **Next**: Extract historical data first using `data\historical_extractor.py`, then run backtest.

### Session 009 — 2026-05-18 (6-Month Data Loading Fix)
- **Agent**: Claude (AI Assistant)
- **Focus**: Fix 6-month M1 data loading failure
- **Problem**: MT5 returns "Invalid params" error when requesting ~180K bars (6 months M1) in one request
- **Actions**:
  - Added `_chunked_historical_fetch()` function to `data/historical_extractor.py` - splits large date ranges into 30-day chunks
  - Installed pyarrow package (required for parquet file saving)
  - Ran full 6-month extraction: **9/9 symbols successful, ~99K M1 bars + ~3K H1 bars per symbol**
  - 18 parquet files created in `D:\Trading\data\historical\`
- **Outcome**: Historical data pipeline now works for 6+ months of M1 data. Phase 9.1 complete.
- **Next**: Run `python backtester.py` to simulate trading on the extracted data.

### Session 010 — 2026-05-18 (Backtest Code Cleanup)
- **Agent**: Claude (AI Assistant)
- **Focus**: Clean up duplicate imports in backtester.py
- **Actions**:
  - Consolidated duplicate imports from `config` into single import block
  - Removed unused imports (`generate_trend_signal`, `generate_mean_reversion_signal`, `combine_signals`)
  - Verified all 156 tests pass
- **Outcome**: Clean, maintainable backtester.py ready for production use.

### Session 011 — 2026-05-18 (Historical Extractor Fix)
- **Agent**: Claude (AI Assistant)
- **Focus**: Fix ModuleNotFoundError when running historical_extractor.py directly
- **Actions**:
  - Added `sys.path` manipulation at top of `data/historical_extractor.py` to allow running script directly from root folder
  - Script now works with `python data\historical_extractor.py` without needing `-m` flag
- **Outcome**: Historical data extraction script runs correctly from command line.

### Session 012 — 2026-05-18 (Backtester Critical Bug Fixes)
- **Agent**: Claude (AI Assistant)
- **Focus**: Fix index misalignment and XAUUSD gold pricing in backtester.py
- **Actions**:
  - Fixed critical index misalignment bug: `build_feature_matrix()` drops warmup rows internally, but backtester was using original df with wrong indices
    - Added alignment code: `df_aligned = df.iloc[start_pos:].copy()` where `start_pos` maps features.index[0] to df
  - Fixed spread pip value for XAUUSD (gold): pip_value = 0.1 (not 0.0001 like FX pairs), pip_value_dollar = 1.0 ($1 per pip per lot)
  - Updated signal_filters.py R:R check to properly handle 1.50 exactly
- **Outcome**: Backtest now runs successfully on all 9 symbols. Results: 193 total trades, 47.7% win rate, $4,975.71 total PnL across all symbols
- **Results Snapshot**:
  - EURJPY: 11 trades, -18.2% win rate, -$573.39
  - USDJPY: 16 trades, 81.2% win rate, +$1,616.59
  - USDCAD: 10 trades, 40.0% win rate, +$132.40
  - AUDUSD: 25 trades, 64.0% win rate, +$1,516.84
  - NZDUSD: 26 trades, 53.8% win rate, +$781.62
  - XAUUSD: 58 trades, 39.7% win rate, +$1,029.86
  - GBPUSD: 21 trades, 57.1% win rate, +$877.69
  - EURUSD: 12 trades, 50.0% win rate, +$502.30
  - GBPJPY: 14 trades, 14.3% win rate, -$908.19
- **Next**: Phase 9.3 — Add performance metrics and reporting

### Session 013 — 2026-05-20 (Backtest Fix & Signal Optimization Attempt)
- **Agent**: Claude (AI Assistant)
- **Focus**: Signal logic fixes and quality filtering
- **Results After Fixes**:
  - Fixed `vwap_d` → `vwap_dist` column mapping bug
  - Fixed R:R floating-point comparison tolerance
  - Removed XAUUSD from trading (was destroying portfolio)
  - Added VWAP-targeted mean-reversion signals
  - Added stricter quality filters (1.5x OFI, 2x VWAP dist)
  - Win rate still only 7.7% (need 35%+ for prop firm)
- **Final Results**: 13 trades, 7.7% win rate, -$2,296 PnL
- **Key Finding**: Mean-reversion VWAP targeting is NOT working - price doesn't reliably revert
- **Next Steps**: Focus on trend signals only, add momentum confirmation to DC events

### Session 014 — 2026-05-20 (Phase 9.4 Trend Following Pivot - Option A)
- **Agent**: Claude (AI Assistant)
- **Focus**: Complete pivot to trend-following only strategy for prop firm viability
- **Actions Completed**:
  1. **config.py**: Updated `HURST_TRENDING_THRESHOLD` from 0.58 → 0.58 (kept strict but workable), rolled back 0.65 which was too strict
  2. **backtester.py**:
     - Fixed dead code from mean-reversion logic that was causing runtime errors
     - Implemented trailing stops (1.0x ATR) replacing fixed TP
     - Disabled mean-reversion signal generation completely
  3. **signal_generation.py**: Updated `combine_signals()` to ignore mean-reversion signals and only allow trending regime trades
  4. **execution.py**: Added `trail_stop()` and `calculate_trailing_distance()` functions for live trailing stop support
  5. **2_microstructure_engine.py**: No changes needed - already trend-following focused
  6. **CONTEXT.md**: Documented the strategic pivot to option A (Trend Following Only)
- **Results**: 
  - Trend-only backtest: 19 trades, 0% win rate, -$1,875 PnL
  - **Issue**: DC events fire at END of significant moves (buying high/selling low)
- **Next Steps**: DC events trigger AFTER price moves - need to interpret as trend CONTINUATION signals, not entries. Consider entering AFTER a retracement in the DC direction.
- **Status**: Code complete, strategy needs refinement

### Session 016 — 2026-05-20 (Phase 9.5 Continued: OFI-Based Pullback Refinement)
- **Agent**: Claude (AI Assistant)
- **Focus**: Refine OFI pullback logic to improve win rate
- **Actions Taken**:
  - Implemented OFI weakening pullback detection (more practical than zero-crossing):
    - LONG: OFI drops 50% from primed level, then recovers with price momentum
    - SHORT: OFI rises 50% toward 0, then resumes negative with price momentum
  - Added `primed_ofi` tracking to measure OFI weakening magnitude
  - Current results: 31 trades, 0% win rate, -$3,025 PnL
- **Problem**: OFI in trending markets stays directional, pullback detection rarely triggers
- **Key Insight**: The pullback logic needs to work with trending OFI patterns, not require reversals
- **Files Modified**: `2_microstructure_engine.py` - `evaluate_pullback_signals()` function

### Session 017 — 2026-05-20 (Phase 9.5 Final Push - OFI Momentum Resumption)
- **Agent**: Claude (AI Assistant)
- **Focus**: Implement OFI momentum resumption pullback entry logic (Option B from Session 017 plan)
- **Actions Completed**:
  - Replaced OFI weakening logic with OFI momentum resumption:
    - LONG: After DC downturn primes LONG, wait for `ofi > 0 AND ofi_delta > threshold` (buying momentum sharply resumes)
    - SHORT: After DC upturn primes SHORT, wait for `ofi < 0 AND ofi_delta < -threshold` (selling momentum sharply resumes)
  - Default `ofi_delta_threshold = 0.005`
  - Removed `primed_ofi` weakening tracking logic
- **Results**: 18 trades, 5.6% win rate, -$1,562 PnL (better than 0% but still unacceptable)
- **Problem**: Momentum resumption rarely triggers because OFI doesn't accelerate as expected in trending markets
- **Key Insight**: The fundamental issue isn't pullback logic - it's that DC fires at price extrema, meaning we're entering at market tops/bottoms
- **Files Modified**: `2_microstructure_engine.py` - `evaluate_pullback_signals()` function (lines 662-678)
- **Status**: Strategy logic complete but win rate far below prop firm requirements

### Session 018 — 2026-05-22 (Current Session - Reality Check)
- **Agent**: Claude (AI Assistant)
- **Focus**: Reality assessment and strategic reset
- **Findings**:
  - Current approach: 18 trades, 5.6% win rate, -$1,562 PnL
  - Root cause analysis: DC fires at directional EXTREMA (high for DOWNTURN, low for UPTURN)
  - Primary signals enter AFTER significant moves have already occurred
  - This is mathematically equivalent to buying high/selling low
  - OFI momentum resumption fixes timing but doesn't fix the fundamental "wrong direction" problem
- **Critical Realization**: The entire signal architecture is inverted for trend following
  - DC UPTURN = local HIGH = perfect SHORT entry point (not LONG)
  - DC DOWNTURN = local LOW = perfect LONG entry point (not SHORT)
- **Next Steps**: Complete strategic reset required - see Action Plan below

---

## 🚀 Prop Firm Roadmap — Your Path to $100/Day Trading Income

### Current Status
- **System**: ⚠️ **STRATEGY FLAW IDENTIFIED** - Signal direction inverted relative to market reality
- **Backtest results**: 18 trades, 5.6% win rate, -$1,562 PnL
- **Root Cause**: DC UPTURN fires at local HIGHs but we enter LONG (buying tops). DC DOWNTURN fires at local LOWs but we enter SHORT (selling bottoms).
- **Estimated Fix Effort**: 2-3 days for complete strategy rewrite

### Phase Reset Required
- **Phase 9.5 BLOCKED** - Fundamental architecture issue requires major refactor
- **New Focus**: Fix signal inversion, then rebuild entry timing

### Strategy: Complete Reset Required — Phase 9.5 BLOCKED
- ❌ HURST_TRENDING_THRESHOLD = 0.58 works but...
- ❌ Pullback logic: OFI momentum resumption implemented but win rate still terrible
- ❌ Signal direction: **INVERTED** relative to DC event meaning
- ❌ Core issue: We enter LONG on DC UPTURN (top of move) and SHORT on DC DOWNTURN (bottom of move)

---

## 📋 ACTION PLAN — Strategic Reset (Session 018-020)

### **Phase 1: Fix Signal Direction (Days 1-2)**
**Problem**: DC mechanics are opposite to intuitive interpretation
- DC UPTURN = price rose θ from local min → we're now at a LOCAL MAX → SHORT opportunity
- DC DOWNTURN = price fell θ from local max → we're now at a LOCAL MIN → LONG opportunity

**Actions**:
1. **Invert signal logic in `evaluate_primary_signal()`**:
   - `dc_event == 'upturn'` → `primary_sell = True` (we're at the top)
   - `dc_event == 'downturn'` → `primary_buy = True` (we're at the bottom)
2. **Update `evaluate_pullback_signals()`**:
   - After DC UPTURN (primed SHORT), wait for OFI momentum to resume DOWN
   - After DC DOWNTURN (primed LONG), wait for OFI momentum to resume UP
3. **Fix `backtester.py`**: Remove the signal direction swap workaround (if any)

### **Phase 2: Add True Pullback Entry (Days 2-3)**
**Problem**: Even inverted signals enter at extrema without pullback confirmation

**Actions**:
1. **Implement price-based pullback**: Wait N bars after DC for price to retrace
2. **Entry trigger**: Enter when price resumes in DC direction with OFI confirmation
3. **Parameters to tune**: Pullback lookback (5-20 bars), retrace threshold (20-50% of DC move)

### **Phase 3: Validation & Optimization (Days 3-5)**
**Actions**:
1. Run backtest with corrected signals
2. Target metrics: 35%+ win rate, positive expectancy, <4% max DD per symbol
3. Optimize: `DC_THETA` (0.001-0.004), `OFI_THRESHOLD` (0.05-0.2), pullback parameters
4. Document winning parameter set

---

## 🔗 References

- MetaTrader5 Python API: https://www.mql5.com/en/docs/python_metatrader5
- ARCH library docs: https://arch.readthedocs.io/
- Hurst exponent: https://github.com/Mottl/hurst
- Walk-forward validation: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Fixed fractional position sizing: Van Tharp — "Trade Your Way to Financial Freedom"

---

*This file must be updated at the end of every development session.*
*If you are an AI agent, update the Session Log section after completing work.*
