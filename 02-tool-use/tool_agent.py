"""
Example 2: Tool Use / Function Calling
=======================================
LLMs become agents when they can take actions. Tool use (aka function calling)
lets the model request that your code execute specific functions, then the model
reasons over the results.

Key concepts:
- Defining tools with JSON Schema
- The tool use message flow: user → assistant (tool_use) → user (tool_result) → assistant
- Letting the model choose which tools to call (and when)

Flow:
  User asks question
       │
       ▼
  Claude reasons and returns a tool_use block
       │
       ▼
  Your code executes the function
       │
       ▼
  You send the result back as a tool_result
       │
       ▼
  Claude incorporates the result and responds

Run: python tool_agent.py
"""

import json
import anthropic

client = anthropic.Anthropic()

# --- Define tools as JSON Schema -------------------------------------------

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city. Returns temperature in Fahrenheit and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, and parentheses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(2 + 3) * 4'",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search a documentation knowledge base by keyword query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3)",
                },
            },
            "required": ["query"],
        },
    },
]


# --- Tool implementations --------------------------------------------------

FAKE_WEATHER_DATA = {
    "san francisco": {"temp": 62, "condition": "Foggy"},
    "new york": {"temp": 78, "condition": "Partly cloudy"},
    "london": {"temp": 55, "condition": "Rainy"},
    "tokyo": {"temp": 85, "condition": "Humid"},
}

FAKE_DOCS = [
    {"title": "Getting Started", "content": "Install with pip install mylib. Import with 'import mylib'."},
    {"title": "Authentication", "content": "Use mylib.auth(api_key='...') to authenticate. Keys are in the dashboard."},
    {"title": "Rate Limits", "content": "Free tier: 100 req/min. Pro tier: 10,000 req/min. Enterprise: unlimited."},
    {"title": "Error Handling", "content": "All errors raise MyLibError. Use try/except to handle gracefully."},
]


def get_weather(city: str) -> dict:
    data = FAKE_WEATHER_DATA.get(city.lower())
    if data:
        return {"city": city, **data}
    return {"error": f"No weather data for '{city}'"}


def calculate(expression: str) -> dict:
    # Only allow safe math characters
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression)  # safe because we validated the character set
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


def search_docs(query: str, max_results: int = 3) -> dict:
    query_lower = query.lower()
    results = [
        doc for doc in FAKE_DOCS if query_lower in doc["title"].lower() or query_lower in doc["content"].lower()
    ]
    return {"results": results[:max_results], "total": len(results)}


# Map tool names to functions
TOOL_DISPATCH = {
    "get_weather": lambda inp: get_weather(inp["city"]),
    "calculate": lambda inp: calculate(inp["expression"]),
    "search_docs": lambda inp: search_docs(inp["query"], inp.get("max_results", 3)),
}


# --- Agent loop with tool use ----------------------------------------------

def run_agent(user_message: str):
    """Single-turn agent that handles tool calls until the model produces a final text response."""
    print(f"\nUser: {user_message}")

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        # Check if the model wants to use tools
        if response.stop_reason == "tool_use":
            # The response may contain text + tool_use blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    print(f"  → Calling {tool_name}({json.dumps(tool_input)})")

                    # Execute the tool
                    result = TOOL_DISPATCH[tool_name](tool_input)
                    print(f"  ← Result: {json.dumps(result)}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

            # Send results back to Claude
            messages.append({"role": "user", "content": tool_results})

        else:
            # Model produced a final text response — we're done
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            print(f"\nClaude: {final_text}")
            return final_text


# --- Demo -------------------------------------------------------------------

if __name__ == "__main__":
    # Claude will choose the right tool(s) for each question
    run_agent("What's the weather in San Francisco and Tokyo?")
    print("\n" + "=" * 60 + "\n")
    run_agent("What's (15 + 7) * 3 - 10?")
    print("\n" + "=" * 60 + "\n")
    run_agent("How do I handle errors in mylib? Also what are the rate limits?")
