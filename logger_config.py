# logger_config.py — QuantEdge MT5: Centralized Structured Logging
# All modules must import their logger from here. No print() statements allowed.

import logging
import logging.handlers
import sys
from pathlib import Path
from config import LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FORMAT, LOG_DATE_FORMAT


def setup_logger(name: str, log_dir: Path = LOG_DIR) -> logging.Logger:
    """
    Create and configure a named logger with rotating file + console handlers.

    Args:
        name: Logger name (use __name__ in each module).
        log_dir: Directory to write log files. Defaults to D:\\Trading\\logs.

    Returns:
        Configured logging.Logger instance.

    Usage:
        from logger_config import setup_logger
        logger = setup_logger(__name__)
        logger.info("System initialized.")
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # --- Rotating File Handler (DEBUG and above) ---
    log_file = Path(log_dir) / f"{name.replace('.', '_')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True  # Defer file opening to avoid Windows file locking on rotation
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --- Console Handler (INFO and above) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    try:
        # On Windows, sys.stdout may use cp1252 — reconfigure to UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        # Python < 3.7 or non-Windows — skip reconfigure
        pass
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve an existing logger or create a new one.
    Alias for setup_logger — use this in modules that may call before setup.
    """
    return setup_logger(name)
