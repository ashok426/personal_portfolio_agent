from __future__ import annotations

import os
import sys
from typing import Optional

from .config import LOG_FILE, LOG_LEVEL
from .logger_setup import init_logger, get_logger
from .agent_logic import build_graph, AgentState

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def resolve_stock_name_from_ticker(ticker_symbol: str) -> Optional[str]:
    if not ticker_symbol or yf is None:
        return None
    try:
        info = yf.Ticker(ticker_symbol).info or {}
        return info.get("longName") or info.get("shortName") or info.get("name")
    except Exception:
        return None


def run(ticker: Optional[str] = None) -> None:
    # ensure logging configured
    init_logger(LOG_FILE, LOG_LEVEL)
    logger = get_logger(__name__)

    # Determine stock and ticker
    ticker_symbol = ticker or os.environ.get("TEST_TICKER_SYMBOL")

    if not ticker_symbol:
        logger.error("No ticker symbol provided")
        print("ERROR: Please provide a valid ticker symbol.")
        return

    stock_name = resolve_stock_name_from_ticker(ticker_symbol) or ticker_symbol

    if not ticker_symbol:
        # fallback: if the stock name *is* a ticker symbol already
        ticker_symbol = stock_name

    logger.info("Starting portfolio manager for %s (%s)", stock_name, ticker_symbol)

    graph = build_graph()

    initial_state: AgentState = {
        "stock_name": stock_name,
        "ticker_symbol": ticker_symbol,
    }

    try:
        final_state = graph.invoke(initial_state)

        recommendation = final_state.get("recommendation") or ""
        errors = final_state.get("errors") or []

        # Print output sections
        print("\n========== Portfolio Manager Result ==========\n")
        print(f"Stock Name: {stock_name}")
        print(f"Ticker Symbol: {ticker_symbol}\n")

        if errors:
            print("⚠️  Errors / Warnings:")
            for err in errors:
                print(f"  - {err}")
            print()

        print("📝 Analysis / Recommendation:\n")
        print(recommendation.strip())
        print("\n==============================================\n")

    except Exception as e:
        logger.exception("Graph execution failed: %s", e)
        print("ERROR: Graph execution failed:", e)


if __name__ == "__main__":
    # accept ticker from CLI args
    cli_ticker = sys.argv[1] if len(sys.argv) > 1 else None
    run(cli_ticker)
