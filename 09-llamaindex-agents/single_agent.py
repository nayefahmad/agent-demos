"""
Example 9b: LlamaIndex Single Agent (AgentWorkflow)
====================================================
LlamaIndex has two agent architectures:

  Legacy:  AgentRunner + AgentWorker  (explicit step control)
  Current: AgentWorkflow + FunctionAgent/ReActAgent  (event-driven, recommended)

This example uses the current architecture. The agent loop is:
  1. User message → setup_agent → select active agent
  2. run_agent_step → LLM call (may produce tool calls)
  3. call_tool → execute tools
  4. aggregate_tool_results → feed back to LLM or return final answer

Two agent types:
  - FunctionAgent: Uses the LLM's native tool-calling API (faster, requires compatible LLM)
  - ReActAgent:    Uses ReAct prompting (works with ANY LLM, visible reasoning chain)

Run: python single_agent.py
"""

import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.llms.openai import OpenAI  # swap for llama_index.llms.anthropic etc.


# --- Tools ------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    data = {
        "san francisco": "62°F, Foggy",
        "new york": "78°F, Partly cloudy",
        "tokyo": "85°F, Humid",
    }
    return data.get(city.lower(), f"No weather data for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information."""
    kb = {
        "pricing": "Free tier: 100 req/day. Pro: $49/mo for 10K req/day. Enterprise: custom.",
        "api": "REST API at api.example.com. Auth via Bearer token. Rate limit: 100 req/min.",
        "setup": "pip install mylib && mylib init && mylib configure --key YOUR_KEY",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"No results for '{query}'"


tools = [
    FunctionTool.from_defaults(get_weather),
    FunctionTool.from_defaults(calculate),
    FunctionTool.from_defaults(search_knowledge_base),
]


# --- Option A: FunctionAgent (uses LLM's native tool calling) ---------------

async def demo_function_agent():
    """FunctionAgent delegates tool selection to the LLM's function-calling API."""
    print("=" * 60)
    print("FunctionAgent (native tool calling)")
    print("=" * 60)

    # Shortcut: AgentWorkflow.from_tools_or_functions creates a single FunctionAgent
    agent = AgentWorkflow.from_tools_or_functions(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    response = await agent.run(user_msg="What's the weather in Tokyo?")
    print(f"\nResponse: {response}")

    # Multi-turn: pass the handler to preserve conversation state
    handler = agent.run(user_msg="What's the weather in San Francisco?")
    response = await handler
    print(f"Response: {response}")

    handler = agent.run(user_msg="Which city is warmer?")
    response = await handler
    print(f"Response: {response}")


# --- Option B: ReActAgent (works with any LLM) ------------------------------

async def demo_react_agent():
    """ReActAgent uses Thought/Action/Observation prompting — works with any LLM."""
    print("\n" + "=" * 60)
    print("ReActAgent (ReAct prompting, any LLM)")
    print("=" * 60)

    react = ReActAgent(
        name="react_assistant",
        description="A helpful assistant that reasons step by step",
        system_prompt="You are a helpful assistant. Think step by step.",
        llm=OpenAI(model="gpt-4o"),
        tools=tools,
    )

    workflow = AgentWorkflow(agents=[react], root_agent="react_assistant")

    # Stream events to see the reasoning chain
    handler = workflow.run(
        user_msg="What's (15 + 7) * 3? Also check the weather in New York."
    )

    # Stream intermediate events
    async for event in handler.stream_events():
        # AgentStream events contain the LLM's reasoning tokens
        if hasattr(event, "delta") and event.delta:
            print(event.delta, end="", flush=True)

    response = await handler
    print(f"\n\nFinal: {response}")


# --- Comparison table -------------------------------------------------------
#
# ┌────────────────────┬───────────────────────┬───────────────────────┐
# │                    │ FunctionAgent          │ ReActAgent            │
# ├────────────────────┼───────────────────────┼───────────────────────┤
# │ LLM requirement    │ Must support tool API  │ Any LLM              │
# │ Mechanism          │ Native function calling│ Thought/Action/Obs   │
# │ Reasoning visible? │ No (opaque)            │ Yes (chain of thought)│
# │ Speed              │ Faster                 │ Slower (more tokens)  │
# │ Reliability        │ Higher for supported   │ May struggle with     │
# │                    │ LLMs                   │ complex schemas       │
# │ Best for           │ Production + tool-     │ Any-LLM compat,      │
# │                    │ calling LLMs           │ debugging, research   │
# └────────────────────┴───────────────────────┴───────────────────────┘


if __name__ == "__main__":
    asyncio.run(demo_function_agent())
    asyncio.run(demo_react_agent())
