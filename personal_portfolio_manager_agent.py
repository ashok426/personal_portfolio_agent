"""
personal_portfolio_manager_agent.py
=================================

This module implements a proof‑of‑concept personal portfolio manager built
with the LangChain ecosystem and orchestrated using LangGraph.  The agent
is designed to analyse a user’s stock holdings alongside real time market
data and make a recommendation (buy more, sell, or hold) based on the
combination of historical portfolio information and current fundamentals.

The workflow is intentionally modular.  Each major operation is exposed
as a LangChain tool so that they can be composed into a graph.  At a high
level the graph is:

1. **Fetch portfolio data** – Read an Excel sheet containing the user’s
   holdings (quantity, purchase price, profit/loss, etc.) and extract
   the relevant row for a particular stock.
2. **Fetch market data** – Query an external source to retrieve recent
   fundamentals (e.g., price‑earnings ratio, quarterly earnings growth)
   and news headlines for a given stock symbol.
3. **LLM analysis** – Call a large language model on the combined
   portfolio and market data, asking it to reason about the stock in
   context of the user’s goals and risk appetite.

Running the graph produces a recommendation string.  The code below is
intended as a template – you will need to supply your own API keys for
OpenAI and (if desired) a market data provider.  For real‑time data
collection the example uses the `yfinance` library because it does not
require an API key for basic metrics; however, you can replace this
implementation with a service such as AlphaVantage, Finnhub or your
preferred provider.

To execute the agent you can do something like:

```python
from personal_portfolio_manager_agent import build_graph

graph = build_graph()
inputs = {"stock_name": "BAJFINANCE.NS"}
result = graph.invoke(inputs)
print(result["result"])
```

Note that LangGraph is still evolving; this script targets a typical
version from early 2024.  Ensure `langchain`, `langgraph` and
`yfinance` are installed in your environment.
"""

from __future__ import annotations

import os
import pandas as pd
from typing import Any, Dict, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None  # will be handled gracefully in get_realtime_data

# LangChain imports.  These will only work if the relevant packages are
# installed.  If not available, the functions defined below will raise
# import errors when invoked; that is expected in this proof of concept.
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.schema import Document
import langgraph


def load_portfolio_file(file_path: str) -> pd.DataFrame:
    """Load the Excel sheet containing the portfolio.

    The portfolio file must have at least the following columns:
    - Stock Name
    - Open Qty
    - Total Market Value
    - Total Buy Value
    - Total unrealized Profit/Loss
    - Profit/Loss per share
    - Market price
    - Buy Price

    You can prepare this file by grouping your raw P&L statement by
    stock name and summing/averaging the numeric columns.  See the
    accompanying notebook or earlier steps in this conversation for
    details.

    Args:
        file_path: path to the Excel workbook.

    Returns:
        DataFrame with the portfolio records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")
    # Use engine openpyxl since xlsxwriter cannot read
    df = pd.read_excel(file_path, engine="openpyxl")
    return df


@tool
def get_portfolio_data(stock_name: str, file_path: str = "Grouped_Stock_PnL_Summary.xlsx") -> Dict[str, Any]:
    """Retrieve aggregate information about a given stock from the user's portfolio.

    This tool reads the Excel sheet containing the portfolio and
    extracts the row(s) corresponding to the provided stock name.  It
    performs a case‑insensitive match on the 'Stock Name' column.  If
    multiple rows match (e.g. due to multiple lots), they are grouped
    together by summing or averaging as appropriate.

    Args:
        stock_name: The name or symbol of the stock to look up.
        file_path: Path to the portfolio Excel file.  Defaults to
            "Grouped_Stock_PnL_Summary.xlsx" in the current working
            directory.

    Returns:
        A dictionary with keys quantity, market_value, buy_value,
        unrealized_pl, profit_loss_per_share, market_price, buy_price.
        If the stock is not found, returns an empty dictionary.
    """
    df = load_portfolio_file(file_path)

    # Normalize the stock name for case‑insensitive matching
    mask = df["Stock Name"].str.upper() == stock_name.upper()
    subset = df.loc[mask]

    if subset.empty:
        return {}

    # If there are multiple rows (e.g. multiple lots), aggregate them
    quantity = subset["Open Qty"].sum()
    market_value = subset["Total Market Value"].sum()
    buy_value = subset["Total Buy Value"].sum()
    unrealized_pl = subset["Total unrealized Profit/Loss"].sum()
    profit_loss_per_share = subset["Profit/Loss per share"].mean()
    market_price = subset["Market price"].mean()
    buy_price = subset["Buy Price"].mean()

    return {
        "quantity": int(quantity),
        "market_value": float(market_value),
        "buy_value": float(buy_value),
        "unrealized_pl": float(unrealized_pl),
        "profit_loss_per_share": float(profit_loss_per_share),
        "market_price": float(market_price),
        "buy_price": float(buy_price),
    }


@tool
def get_realtime_data(stock_name: str) -> Dict[str, Any]:
    """Fetch real‑time fundamentals and recent performance for a stock.

    This tool uses the `yfinance` library to retrieve basic metrics for
    the stock.  In a production system you may wish to substitute a
    different provider that offers richer data and news headlines.  If
    `yfinance` is not installed or the symbol cannot be loaded, the
    function returns an empty dictionary.

    Args:
        stock_name: The ticker symbol (e.g., 'BAJFINANCE.NS' for
            Bajaj Finance on NSE).  Make sure to append the exchange
            suffix recognized by yfinance.

    Returns:
        A dictionary containing the current PE ratio, trailing twelve
        months (TTM) earnings per share, and recent revenue growth
        quarter‑over‑quarter (QoQ).  Additional fields can be added
        depending on available data.  If data is missing the field is
        left as None.
    """
    if yf is None:
        # yfinance is not installed
        return {}
    try:
        ticker = yf.Ticker(stock_name)
        info = ticker.info
    except Exception:
        return {}

    # Some metrics may not be present for all tickers
    pe_ratio = info.get("trailingPE")
    eps = info.get("trailingEps")
    qoq_growth = info.get("revenueQuarterlyGrowth")

    # Quarter results (we fetch earnings history if available)
    try:
        # yfinance may provide quarterly financials.  We'll take the
        # most recent quarter's revenue growth if present.
        if hasattr(ticker, "quarterly_financials"):
            qf = ticker.quarterly_financials
            # qf columns are the last four quarters.  We'll compute growth
            # between the most recent and previous quarter for revenue.
            if not qf.empty and "Total Revenue" in qf.index:
                revenues = qf.loc["Total Revenue"].dropna()
                if len(revenues) >= 2:
                    latest, prior = revenues.iloc[0], revenues.iloc[1]
                    revenue_growth = (latest - prior) / prior
                else:
                    revenue_growth = None
            else:
                revenue_growth = None
        else:
            revenue_growth = None
    except Exception:
        revenue_growth = None

    return {
        "pe_ratio": pe_ratio,
        "eps": eps,
        "revenue_qoq_growth": qoq_growth,
        "recent_revenue_growth": revenue_growth,
    }


def build_analysis_prompt() -> PromptTemplate:
    """Create a prompt template for the analysis stage.

    The template asks an LLM to reason about the stock given the
    portfolio data and real‑time fundamentals.  It instructs the model
    to provide a concise recommendation on whether to buy more, hold
    or sell, taking into account risk tolerance and market context.
    """
    template = """
    You are a personal portfolio analyst assisting an investor with
    dynamic asset allocation.  The user has asked you to analyse a
    specific stock in their portfolio and provide a recommendation on
    whether they should buy more, hold, or sell.  Use the information
    provided about their holdings and the current market fundamentals to
    support your reasoning.  Present your conclusion in a clear,
    professional tone and include at most three bullet points summarising
    your key observations.

    {{portfolio_block}}

    {{market_block}}

    **Task**: Based on the above data and assuming a moderate risk
    tolerance, recommend whether the investor should accumulate more of
    this stock, hold their current position, or sell and realise profits.
    Justify your answer briefly.
    """
    return PromptTemplate(
        input_variables=["portfolio_block", "market_block"],
        template=template.strip(),
    )


@tool
def analyze_stock(portfolio_data: Dict[str, Any], market_data: Dict[str, Any]) -> str:
    """Run an LLM analysis on the combined portfolio and market information.

    This tool constructs a prompt from the inputs and calls a chat model
    via LangChain to produce a recommendation.  The OpenAI API key
    should be set in the environment as `OPENAI_API_KEY` prior to
    invocation.  If no API key is found or the call fails, a
    placeholder string is returned instead of raising.

    Args:
        portfolio_data: Dictionary returned by `get_portfolio_data`.
        market_data: Dictionary returned by `get_realtime_data`.

    Returns:
        A recommendation string generated by the LLM, or a fallback
        message if the model cannot be contacted.
    """
    # Build blocks of text for the prompt
    if not portfolio_data:
        portfolio_block = "**Portfolio**: The user does not hold any shares of this stock."
    else:
        portfolio_block = (
            f"**Portfolio**\n"
            f"- Quantity held: {portfolio_data['quantity']}\n"
            f"- Average buy price: {portfolio_data['buy_price']:.2f}\n"
            f"- Current market price: {portfolio_data['market_price']:.2f}\n"
            f"- Total buy value: {portfolio_data['buy_value']:.2f}\n"
            f"- Unrealised P/L: {portfolio_data['unrealized_pl']:.2f}\n"
            f"- Profit/Loss per share: {portfolio_data['profit_loss_per_share']:.2f}\n"
        )

    if not market_data:
        market_block = "**Market Fundamentals**: No market data available."
    else:
        def fmt(key, value):
            return f"{value:.2f}" if isinstance(value, (int, float)) and value is not None else str(value)
        market_block = (
            f"**Market Fundamentals**\n"
            f"- PE ratio: {fmt('pe_ratio', market_data.get('pe_ratio'))}\n"
            f"- Earnings per share (TTM): {fmt('eps', market_data.get('eps'))}\n"
            f"- Revenue QoQ growth: {fmt('revenue_qoq_growth', market_data.get('revenue_qoq_growth'))}\n"
            f"- Recent revenue growth (latest quarter vs previous): {fmt('recent_revenue_growth', market_data.get('recent_revenue_growth'))}\n"
        )

    prompt = build_analysis_prompt()
    message = prompt.format(
        portfolio_block=portfolio_block,
        market_block=market_block,
    )

    # Attempt to call the OpenAI chat model via LangChain
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "[OpenAI API key missing] Cannot call the LLM. "
            "Please set OPENAI_API_KEY in the environment to enable analysis."
        )
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        response = llm.invoke(message)
        # ChatOpenAI returns a Message object; we extract the content
        content = response.content if hasattr(response, "content") else str(response)
        return content
    except Exception as e:
        return f"[LLM call failed] {e}"


def build_graph() -> langgraph.Graph:
    """Assemble the LangGraph for the portfolio manager.

    The graph defines three steps:
    1. `get_portfolio_data` – fetch holdings for the specified stock.
    2. `get_realtime_data` – fetch real‑time fundamentals for the stock.
    3. `analyze_stock` – produce a recommendation given the outputs of the
       previous two steps.

    The graph expects input variables `stock_name` and (optionally)
    `portfolio_path` (defaults to "Grouped_Stock_PnL_Summary.xlsx").  It
    produces a dictionary with a single key `result` containing the
    analysis text.
    """
    # Define the graph
    graph = langgraph.Graph()

    # Register the tools as nodes
    graph.add_node("portfolio", get_portfolio_data)
    graph.add_node("market", get_realtime_data)
    graph.add_node("analysis", analyze_stock)

    # Specify the order of execution.  We want to run both data fetch
    # nodes concurrently and then feed their outputs into the analysis.
    # When executed, LangGraph will collect the outputs and pass them
    # into downstream nodes according to the mapping defined here.
    graph.set_entry_point("portfolio")
    # After portfolio, also run market
    graph.add_edge("portfolio", "market")
    # Both portfolio and market feed into analysis.  We use the
    # `analysis` node's input argument names to control mapping.
    graph.add_edge("portfolio", "analysis")
    graph.add_edge("market", "analysis")

    # Define how to merge the outputs into the analysis call.  The
    # analysis tool expects arguments `portfolio_data` and `market_data`.
    # LangGraph allows specifying merge functions on nodes.  Here we
    # override the default merge behaviour for the `analysis` node so
    # that it receives its arguments as a dict.
    def merge_to_analysis(portfolio_output, market_output, **kwargs):
        return {
            "portfolio_data": portfolio_output,
            "market_data": market_output,
        }

    graph.add_hook("analysis", merge_to_analysis)

    # Set the final node whose output is returned to the caller
    graph.set_exit_point("analysis")

    return graph


if __name__ == "__main__":
    # Example invocation for manual testing
    # Make sure the Excel file exists in the working directory and
    # OPENAI_API_KEY is set in your environment before running.
    graph = build_graph()
    stock = os.environ.get("TEST_STOCK", "BAJFINANCE.NS")
    result = graph.invoke({"stock_name": stock})
    print(result.get("result"))