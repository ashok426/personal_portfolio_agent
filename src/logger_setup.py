from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


def init_logger(log_file: str = "app.log", level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("personal_portfolio_agent")
    logger.setLevel(lvl)
    # If handlers already configured, return existing logger
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5)
    fh.setLevel(lvl)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "personal_portfolio_agent")
