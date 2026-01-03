from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return environment variable or default."""
    return os.environ.get(name, default)


def get_required_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise EnvironmentError(f"Required environment variable {name} is not set")
    return val


# Convenience config accessors
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
PORTFOLIO_PATH = get_env("PORTFOLIO_PATH", "data/Grouped_Stock_PnL_Summary.xlsx")
LOG_FILE = get_env("LOG_FILE", "portfolio_agent.log")
LOG_LEVEL = get_env("LOG_LEVEL", "INFO")
TEST_STOCK_NAME = get_env("TEST_STOCK_NAME")
TEST_TICKER_SYMBOL = get_env("TEST_TICKER_SYMBOL")
