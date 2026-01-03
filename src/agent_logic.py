from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypedDict

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .config import PORTFOLIO_PATH, OPENAI_API_KEY
from .logger_setup import get_logger
from .utils import fmt_value

logger = get_logger(__name__)


# ----------------------------
# State (latest LangGraph style)
# ----------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    stock_name: str                # e.g., "Bajaj Finance"
    ticker_symbol: str             # e.g., "BAJFINANCE.NS"
    portfolio_data: Dict[str, Any]
    market_data: Dict[str, Any]
    recommendation: str
    errors: list[str]


# ----------------------------
# Helpers
# ----------------------------
def load_portfolio_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")
    return pd.read_excel(file_path, engine="openpyxl")


def build_analysis_prompt() -> PromptTemplate:
    template = """
You are a personal portfolio analyst assisting an investor with
dynamic asset allocation. The user asked you to analyse a specific stock
in their portfolio and provide a recommendation on whether they should
buy more, hold, or sell.

Use the information provided about their holdings and the current market
fundamentals to support your reasoning.

Return:
1) Decision: BUY / HOLD / SELL
2) Confidence: Low / Medium / High
3) 2–4 bullet points with key reasoning
4) One risk note

{portfolio_block}

{market_block}

Assume moderate risk tolerance.
"""
    return PromptTemplate(
        input_variables=["portfolio_block", "market_block"],
        template=template.strip(),
    )


# ----------------------------
# Node 1: Portfolio fetch
# ----------------------------
def portfolio_node(state: AgentState) -> AgentState:
    stock_name = state.get("stock_name")
    file_path = PORTFOLIO_PATH

    logger.info("Portfolio node: %s", stock_name)

    if not stock_name:
        return {**state, "errors": (state.get("errors", []) + ["Missing stock_name in state"])}

    df = load_portfolio_file(file_path)

    # Match stock name safely
    mask = df["Stock Name"].astype(str).str.upper().str.strip() == stock_name.upper().strip()
    subset = df.loc[mask]

    if subset.empty:
        logger.warning("No portfolio rows found for %s", stock_name)
        return {**state, "portfolio_data": {}}

    portfolio_data = {
        "quantity": int(subset["Open Qty"].sum()),
        "market_value": float(subset["Total Market Value"].sum()),
        "buy_value": float(subset["Total Buy Value"].sum()),
        "unrealized_pl": float(subset["Total unrealized Profit/Loss"].sum()),
        "profit_loss_per_share": float(subset["Profit/Loss per share"].mean()),
        "market_price": float(subset["Market price"].mean()),
        "buy_price": float(subset["Buy Price"].mean()),
    }

    return {**state, "portfolio_data": portfolio_data}


# ----------------------------
# Node 2: Market fetch (yfinance)
# ----------------------------
def market_node(state: AgentState) -> AgentState:
    logger.info("Market node")

    if yf is None:
        return {**state, "market_data": {}, "errors": (state.get("errors", []) + ["yfinance not installed"])}

    ticker_symbol = state.get("ticker_symbol")
    if not ticker_symbol:
        # Fallback: try using stock_name directly (may fail)
        ticker_symbol = state.get("stock_name")

    if not ticker_symbol:
        return {**state, "market_data": {}, "errors": (state.get("errors", []) + ["Missing ticker_symbol"])}

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
    except Exception as e:
        logger.exception("Failed to fetch yfinance data: %s", e)
        return {**state, "market_data": {}, "errors": (state.get("errors", []) + [f"yfinance error: {e}"])}

    pe_ratio = info.get("trailingPE")
    eps = info.get("trailingEps")
    qoq_growth = info.get("revenueQuarterlyGrowth")

    # Compute recent revenue growth from quarterly financials (best-effort)
    recent_revenue_growth = None
    try:
        qf = getattr(ticker, "quarterly_financials", None)
        if qf is not None and not qf.empty:
            # Index naming differs sometimes: "Total Revenue" vs "TotalRevenue"
            possible_rows = ["Total Revenue", "TotalRevenue"]
            row = next((r for r in possible_rows if r in qf.index), None)
            if row:
                revenues = qf.loc[row].dropna()
                if len(revenues) >= 2:
                    latest, prior = revenues.iloc[0], revenues.iloc[1]
                    if prior != 0:
                        recent_revenue_growth = (latest - prior) / prior
    except Exception:
        pass

    market_data = {
        "ticker_symbol": ticker_symbol,
        "pe_ratio": pe_ratio,
        "eps": eps,
        "revenue_qoq_growth": qoq_growth,
        "recent_revenue_growth": recent_revenue_growth,
    }

    return {**state, "market_data": market_data}


# ----------------------------
# Node 3: LLM analysis
# ----------------------------
def analysis_node(state: AgentState) -> AgentState:
    logger.info("Analysis node")

    portfolio_data = state.get("portfolio_data") or {}
    market_data = state.get("market_data") or {}

    if not portfolio_data:
        portfolio_block = "**Portfolio**: The user does not hold any shares of this stock."
    else:
        portfolio_block = (
            f"**Portfolio**\n"
            f"- Quantity held: {portfolio_data.get('quantity')}\n"
            f"- Average buy price: {portfolio_data.get('buy_price', 0):.2f}\n"
            f"- Current market price: {portfolio_data.get('market_price', 0):.2f}\n"
            f"- Total buy value: {portfolio_data.get('buy_value', 0):.2f}\n"
            f"- Unrealised P/L: {portfolio_data.get('unrealized_pl', 0):.2f}\n"
            f"- Profit/Loss per share: {portfolio_data.get('profit_loss_per_share', 0):.2f}\n"
        )

    if not market_data:
        market_block = "**Market Fundamentals**: No market data available."
    else:
        market_block = (
            f"**Market Fundamentals**\n"
            f"- Ticker: {market_data.get('ticker_symbol')}\n"
            f"- PE ratio: {fmt_value(market_data.get('pe_ratio'))}\n"
            f"- EPS (TTM): {fmt_value(market_data.get('eps'))}\n"
            f"- Revenue QoQ growth: {fmt_value(market_data.get('revenue_qoq_growth'))}\n"
            f"- Recent revenue growth: {fmt_value(market_data.get('recent_revenue_growth'))}\n"
        )

    if not OPENAI_API_KEY:
        msg = "[OpenAI API key missing] Set OPENAI_API_KEY to enable LLM analysis."
        return {**state, "recommendation": msg, "errors": (state.get("errors", []) + [msg])}

    prompt = build_analysis_prompt()
    message = prompt.format(portfolio_block=portfolio_block, market_block=market_block)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        resp = llm.invoke(message)
        content = resp.content if hasattr(resp, "content") else str(resp)
        return {**state, "recommendation": content}
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return {**state, "recommendation": f"[LLM call failed] {e}", "errors": (state.get("errors", []) + [str(e)])}


# ----------------------------
# Build graph (latest LangGraph)
# ----------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("portfolio", portfolio_node)
    graph.add_node("market", market_node)
    graph.add_node("analysis", analysis_node)

    graph.set_entry_point("portfolio")
    graph.add_edge("portfolio", "market")
    graph.add_edge("market", "analysis")
    graph.add_edge("analysis", END)

    return graph.compile()
