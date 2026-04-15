#src/utils/logger.py
"""
Logging utilities for FaithCoT experiments.
Provides consistent, timestamped logging across all modules.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Global logger cache
_loggers = {}


def setup_logger(
    name: str = "faithcot",
    log_dir: str = None,
    level: str = "INFO",
    console: bool = True,
    file_log: bool = True,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files. If None, uses PROJECT_ROOT/logs
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to add console handler
        file_log: Whether to add file handler

    Returns:
        Configured logging.Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_log:
        if log_dir is None:
            # Default: project_root/logs
            log_dir = str(Path(__file__).resolve().parent.parent.parent / "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "faithcot") -> logging.Logger:
    """Get an existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        logging.Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)
