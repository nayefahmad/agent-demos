"""
Example 9a: LlamaIndex FunctionTool Basics
==========================================
LlamaIndex's FunctionTool wraps plain Python functions as agent tools.
Compared to LangChain's @tool decorator, it uses a factory method pattern
(FunctionTool.from_defaults) instead of a decorator — more explicit, more
configurable.

Key concepts:
- FunctionTool.from_defaults():  wrap any callable
- Auto-inferred schemas:         name from fn.__name__, description from docstring
- fn_schema:                     optional Pydantic model for explicit input validation
- QueryEngineTool:               wraps a LlamaIndex index as a tool
- ToolSpec:                      generates multiple tools from an API spec

Run: python function_tool_basics.py
"""

from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool


# ---------------------------------------------------------------------------
# Method 1: Minimal — name and description inferred from function
# ---------------------------------------------------------------------------

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    weather_data = {
        "san francisco": "62°F, Foggy",
        "new york": "78°F, Partly cloudy",
        "london": "55°F, Rainy",
    }
    return weather_data.get(location.lower(), f"No data for {location}")


# Name → "get_weather", description → docstring, schema → inferred from type hints
weather_tool = FunctionTool.from_defaults(get_weather)


# ---------------------------------------------------------------------------
# Method 2: Explicit name, description, and Pydantic schema
# ---------------------------------------------------------------------------

class StockLookupInput(BaseModel):
    """Input for stock price lookup."""
    symbol: str = Field(description="Ticker symbol, e.g. 'AAPL'")
    include_history: bool = Field(default=False, description="Include 30-day price history")


def lookup_stock(symbol: str, include_history: bool = False) -> str:
    """Look up current stock price and optionally include recent history."""
    prices = {"AAPL": 227.50, "GOOGL": 191.30, "MSFT": 478.20}
    price = prices.get(symbol.upper())
    if not price:
        return f"Unknown symbol: {symbol}"
    result = f"{symbol.upper()}: ${price:.2f}"
    if include_history:
        result += " (30d range: $220-$235)"
    return result


stock_tool = FunctionTool.from_defaults(
    fn=lookup_stock,
    name="stock_lookup",
    description="Look up current stock price by ticker symbol",
    fn_schema=StockLookupInput,  # explicit Pydantic validation
)


# ---------------------------------------------------------------------------
# Method 3: With callback (post-processing hook)
# ---------------------------------------------------------------------------

def search_database(query: str, limit: int = 5) -> str:
    """Search the internal database for matching records."""
    return f"Found 3 results for '{query}' (limit={limit})"


def log_search(result: str) -> str:
    """Callback that runs after the tool executes."""
    print(f"  [CALLBACK] Search completed: {result[:50]}")
    return result  # can modify the result here


search_tool = FunctionTool.from_defaults(
    fn=search_database,
    name="db_search",
    description="Search internal database",
    callback=log_search,
)


# ---------------------------------------------------------------------------
# Method 4: Async tool
# ---------------------------------------------------------------------------

async def fetch_url(url: str) -> str:
    """Fetch content from a URL (simulated)."""
    return f"Content from {url}: <html>...</html>"


async_tool = FunctionTool.from_defaults(
    fn=None,           # no sync version
    async_fn=fetch_url,
    name="fetch_url",
    description="Fetch web page content",
)


# ---------------------------------------------------------------------------
# Comparison: LlamaIndex vs LangChain tool definition
# ---------------------------------------------------------------------------
#
# ┌──────────────────────────────────┬──────────────────────────────────────┐
# │  LlamaIndex                      │  LangChain                           │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │  from llama_index.core.tools     │  from langchain_core.tools           │
# │    import FunctionTool           │    import tool                       │
# │                                  │                                      │
# │  def get_weather(city: str):     │  @tool                               │
# │      """Get weather."""          │  def get_weather(city: str):          │
# │      return "sunny"              │      """Get weather."""               │
# │                                  │      return "sunny"                   │
# │  t = FunctionTool.from_defaults( │                                      │
# │      get_weather                 │  # tool IS the decorated function     │
# │  )                               │  # get_weather is now a Tool object   │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │  Pros:                           │  Pros:                               │
# │  + Explicit, configurable        │  + Concise, Pythonic                 │
# │  + fn_schema for validation      │  + Less boilerplate                  │
# │  + callback/async_callback       │  + Familiar decorator pattern        │
# │  + partial_params                │                                      │
# │                                  │                                      │
# │  Cons:                           │  Cons:                               │
# │  - More verbose                  │  - Less control over schema          │
# │  - Factory method less intuitive │  - No built-in callback              │
# └──────────────────────────────────┴──────────────────────────────────────┘


# ---------------------------------------------------------------------------
# Demo: Inspect tool metadata
# ---------------------------------------------------------------------------

def main():
    tools = [weather_tool, stock_tool, search_tool, async_tool]

    print("LlamaIndex FunctionTool Examples")
    print("=" * 60)

    for tool in tools:
        meta = tool.metadata
        print(f"\n  Name:        {meta.name}")
        print(f"  Description: {meta.description}")
        print(f"  Schema:      {meta.get_parameters_dict()}")
        print(f"  Return direct: {meta.return_direct}")

    # Actually call a tool
    print("\n" + "=" * 60)
    print("Tool Execution:")
    print(f"  weather_tool('San Francisco') → {weather_tool('San Francisco')}")
    print(f"  stock_tool('AAPL')            → {stock_tool(symbol='AAPL')}")
    print(f"  search_tool('agents')         →", end=" ")
    result = search_tool(query="agents")
    print(f"{result}")


if __name__ == "__main__":
    main()
