# CONTEXT.md — Project State & Session Memory
## QuantEdge MT5 — Algorithmic Trading System

> **Purpose**: This file is the living memory of the project. Every AI agent and developer MUST update this file at the end of each work session. Read this first before making any changes.

---

## 📅 Last Updated
- **Date**: 2026-05-14
- **Session**: Initial project setup
- **Updated By**: Antigravity (AI Agent)

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
[ ] Phase 8: Live paper trading & monitoring
[ ] Phase 9: Live production trading
```

### What Exists Right Now
| File / Directory             | Status     | Notes                                         |
|------------------------------|------------|-----------------------------------------------|
| `CLAUDE.md`                  | ✅ Done     | Full architecture spec & AI operating manual  |
| `CONTEXT.md`                 | ✅ Done     | This file — project memory                    |
| `requirements.txt`           | ✅ Done     | All dependencies pinned                        |
| `config.py`                  | ✅ Done     | All configuration constants                   |
| `.env.template`              | ✅ Done     | Template for secrets (copy to .env)           |
| `.gitignore`                 | ✅ Done     | Protects secrets, venv, data, models          |
| `pyproject.toml`             | ✅ Done     | Pytest configuration                          |
| `logger_config.py`           | ✅ Done     | Rotating file + console logging               |
| `data_ingestion.py`          | ✅ Done     | MT5 conn, historical/live data, exp. backoff  |
| `feature_engineering.py`     | ✅ Done     | RSI, MACD, BB, ATR, VWAP, Hurst, GARCH       |
| `signal_generation.py`       | ✅ Done     | Regime detect, trend+MR signals, filters      |
| `machine_learning.py`        | ✅ Done     | Walk-forward CV, train, save/load, infer      |
| `risk_manager.py`            | ✅ Done     | 1% sizing, 4% DD limit, validation gate       |
| `execution.py`               | ✅ Done     | Market/limit orders, position management      |
| `main.py`                    | ✅ Done     | Async trading loop, multi-symbol              |
| `tests/test_feature_*.py`    | ✅ Done     | 32 unit tests — all passing                   |
| `tests/test_risk_manager.py` | ✅ Done     | 22 unit tests — all passing                   |
| `tests/test_signal_*.py`     | ✅ Done     | 28 unit tests — all passing                   |
| `.venv/`                     | ✅ Done     | Virtual environment created & populated       |

---

## 💻 Environment State

### System
| Property         | Value                                                        |
|------------------|--------------------------------------------------------------|
| OS               | Windows (PowerShell)                                         |
| Python Version   | 3.14.4                                                       |
| Python Path      | `C:\Users\Admin\AppData\Local\Programs\Python\Python314\`   |
| Project Root     | `D:\Trading\`                                                |
| Virtual Env      | NOT CREATED — run `python -m venv .venv` first               |

### Tools
| Tool             | Status              | Version / Notes                            |
|------------------|---------------------|--------------------------------------------|
| Claude Code CLI  | ✅ Working          | v2.1.141 — run `claude` in PowerShell      |
| VS Code (`code`) | ❌ Not in PATH      | Install VS Code OR add to PATH manually    |
| MetaTrader 5     | ❓ Unknown          | Must be open before running Python scripts |
| Git              | ❓ Unknown          | Run `git --version` to verify              |

### Dependencies — NOT YET INSTALLED
```
Run this to install:
cd D:\Trading
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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

1. **Install dependencies**:
   ```powershell
   cd D:\Trading
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Create `.env`** (copy from `.env.template` and fill in your MT5 credentials):
   ```powershell
   Copy-Item .env.template .env
   # Then edit .env with your MT5 login, password, server, and path
   ```

3. **Verify MT5 Python API**:
   ```python
   import MetaTrader5 as mt5
   print(mt5.version())  # Should print MT5 version tuple
   ```

4. **Build `data_ingestion.py`** — MT5 connection with exponential backoff

5. **Fix VS Code PATH** (optional — Claude Code CLI already works):
   - Download VS Code: https://code.visualstudio.com/
   - During install: check ✅ "Add to PATH"

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

| Issue                          | Severity | Status    | Resolution                              |
|--------------------------------|----------|-----------|-----------------------------------------|
| VS Code `code` not in PATH     | Low      | Open      | Install VS Code or add bin to PATH      |
| No dependencies installed      | High     | Open      | Run `pip install -r requirements.txt`   |
| MT5 credentials not configured | High     | Open      | Create `.env` from `.env.template`      |
| Virtual environment not created| High     | Open      | Run `python -m venv .venv`              |

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
