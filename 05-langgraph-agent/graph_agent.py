"""
Example 5: LangGraph Agent
===========================
LangGraph models agents as state machines (graphs). Nodes are actions,
edges are transitions. This gives you fine-grained control over agent flow,
including conditional routing, parallel execution, and human-in-the-loop.

Key concepts:
- StateGraph: A directed graph where nodes are functions and edges are transitions
- State: A TypedDict that flows through the graph (the agent's working memory)
- Conditional edges: Route to different nodes based on state
- create_react_agent: LangGraph's built-in ReAct agent helper

Architecture:
    ┌─────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐     tool_calls?     ┌──────────┐
    │  Agent   │ ──── yes ──────────► │  Tools   │
    │  (LLM)  │                      │ (Execute)│
    └────┬─────┘                      └────┬─────┘
         │ no                              │
    ┌────▼─────┐                           │
    │   END    │ ◄─────────────────────────┘
    └──────────┘

Run: python graph_agent.py
"""

from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# --- Define tools -----------------------------------------------------------

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    # Simulated data
    prices = {
        "AAPL": 227.50,
        "GOOGL": 191.30,
        "MSFT": 478.20,
        "AMZN": 225.80,
    }
    price = prices.get(symbol.upper())
    if price:
        return f"{symbol.upper()}: ${price:.2f}"
    return f"Unknown symbol: {symbol}"


@tool
def get_company_info(company: str) -> str:
    """Get basic information about a company."""
    info = {
        "apple": "Apple Inc. (AAPL) - Consumer electronics, software, services. Market cap: $3.5T",
        "google": "Alphabet Inc. (GOOGL) - Search, cloud, advertising. Market cap: $2.3T",
        "microsoft": "Microsoft Corp. (MSFT) - Software, cloud (Azure), AI. Market cap: $3.1T",
    }
    result = info.get(company.lower())
    return result or f"No info found for '{company}'"


@tool
def calculate_portfolio_value(holdings: str) -> str:
    """Calculate total portfolio value. Input: comma-separated 'SYMBOL:SHARES' pairs.

    Example: 'AAPL:10,GOOGL:5,MSFT:3'
    """
    prices = {"AAPL": 227.50, "GOOGL": 191.30, "MSFT": 478.20, "AMZN": 225.80}
    total = 0.0
    breakdown = []

    for holding in holdings.split(","):
        parts = holding.strip().split(":")
        if len(parts) != 2:
            continue
        symbol, shares = parts[0].strip().upper(), float(parts[1].strip())
        price = prices.get(symbol, 0)
        value = price * shares
        total += value
        breakdown.append(f"  {symbol}: {shares:.0f} shares × ${price:.2f} = ${value:,.2f}")

    return f"Portfolio breakdown:\n" + "\n".join(breakdown) + f"\n  Total: ${total:,.2f}"


tools = [get_stock_price, get_company_info, calculate_portfolio_value]


# --- Build the graph from scratch -------------------------------------------

# The state that flows through our graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the LLM with tools bound
llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """The agent node: calls the LLM to decide what to do."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Conditional edge: route to 'tools' if the last message has tool calls, else 'end'."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Add edges
graph.add_edge(START, "agent")               # Start → Agent
graph.add_conditional_edges("agent", should_continue)  # Agent → Tools or End
graph.add_edge("tools", "agent")             # Tools → Agent (loop back)

# Compile into a runnable
app = graph.compile()


# --- Run the agent ----------------------------------------------------------

def run_agent(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    result = app.invoke({"messages": [HumanMessage(content=question)]})

    # The final message is the agent's response
    final = result["messages"][-1]
    print(f"\nAgent: {final.content}")

    # Show the trace of all messages
    print(f"\n  (Trace: {len(result['messages'])} messages in conversation)")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        content_preview = str(msg.content)[:80]
        print(f"    {role}: {content_preview}")


# --- Demo -------------------------------------------------------------------

if __name__ == "__main__":
    run_agent("What's Apple's stock price and tell me about the company?")

    print("\n" + "=" * 60 + "\n")

    run_agent(
        "Calculate the value of my portfolio: 10 shares of AAPL, "
        "5 shares of GOOGL, and 3 shares of MSFT"
    )
