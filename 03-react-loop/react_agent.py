"""
Example 3: ReAct Loop (Reasoning + Acting)
===========================================
A full ReAct agent that autonomously reasons, takes actions, observes results,
and decides when to stop — all in a loop. This is the core pattern behind
tools like Claude Code, ChatGPT plugins, and most agent frameworks.

The ReAct loop:
  1. REASON  — The model thinks about what to do next
  2. ACT     — The model calls a tool
  3. OBSERVE — The tool result is added to context
  4. REPEAT  — Until the model decides it has enough info to answer

This example implements a research agent that can search the web, read pages,
and take notes to answer complex questions.

Run: python react_agent.py
"""

import json
import anthropic

client = anthropic.Anthropic()

# --- Tools for the research agent ------------------------------------------

tools = [
    {
        "name": "web_search",
        "description": "Search the web and return a list of result snippets. Use this to find information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Read the full text content of a web page by URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to read"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "notepad",
        "description": "Save a note to your scratchpad. Use this to track key findings as you research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "read"],
                    "description": "'add' to save a note, 'read' to see all notes",
                },
                "text": {
                    "type": "string",
                    "description": "Note text (required for 'add')",
                },
            },
            "required": ["action"],
        },
    },
]


# --- Simulated tool implementations ----------------------------------------
# In production, these would call real APIs.

FAKE_SEARCH_RESULTS = {
    "python async": [
        {"title": "Python asyncio docs", "url": "https://docs.python.org/3/library/asyncio.html",
         "snippet": "asyncio is a library to write concurrent code using the async/await syntax."},
        {"title": "Real Python: Async IO Guide", "url": "https://realpython.com/async-io-python/",
         "snippet": "A comprehensive guide to understanding async IO in Python 3."},
    ],
    "default": [
        {"title": "Example Result", "url": "https://example.com",
         "snippet": "This is a simulated search result for demonstration purposes."},
    ],
}

FAKE_PAGES = {
    "https://docs.python.org/3/library/asyncio.html": (
        "asyncio — Asynchronous I/O\n\n"
        "asyncio is used as a foundation for multiple Python asynchronous frameworks.\n"
        "Key concepts:\n"
        "- Coroutines: declared with async/await syntax\n"
        "- Event Loop: the core of every asyncio application, runs async tasks\n"
        "- Tasks: used to schedule coroutines concurrently via asyncio.create_task()\n"
        "- Futures: low-level awaitable objects representing eventual results\n"
        "- Streams: high-level async/await-ready primitives for network connections\n\n"
        "Example:\n"
        "  async def main():\n"
        "      await asyncio.sleep(1)\n"
        "      print('hello')\n"
        "  asyncio.run(main())\n"
    ),
    "https://realpython.com/async-io-python/": (
        "Async IO in Python: A Complete Walkthrough\n\n"
        "The 3 main types of awaitable objects:\n"
        "1. Coroutines - async def functions\n"
        "2. Tasks - wrappers that schedule coroutines on the event loop\n"
        "3. Futures - represent the result of work not yet completed\n\n"
        "When to use async IO:\n"
        "- IO-bound tasks (network requests, file reads, DB queries)\n"
        "- When you need high concurrency without threads\n"
        "- NOT for CPU-bound tasks (use multiprocessing instead)\n\n"
        "Common patterns:\n"
        "- asyncio.gather(*tasks) to run multiple coroutines concurrently\n"
        "- async with for async context managers\n"
        "- async for for async iterators\n"
    ),
}


notes: list[str] = []


def web_search(query: str) -> dict:
    for key, results in FAKE_SEARCH_RESULTS.items():
        if key in query.lower():
            return {"results": results}
    return {"results": FAKE_SEARCH_RESULTS["default"]}


def read_page(url: str) -> dict:
    content = FAKE_PAGES.get(url, "Page not found or could not be loaded.")
    return {"url": url, "content": content}


def notepad(action: str, text: str = "") -> dict:
    if action == "add":
        notes.append(text)
        return {"status": "saved", "total_notes": len(notes)}
    elif action == "read":
        return {"notes": notes}
    return {"error": "Unknown action"}


TOOL_DISPATCH = {
    "web_search": lambda inp: web_search(inp["query"]),
    "read_page": lambda inp: read_page(inp["url"]),
    "notepad": lambda inp: notepad(inp["action"], inp.get("text", "")),
}


# --- The ReAct Agent Loop --------------------------------------------------

def react_agent(question: str, max_steps: int = 10):
    """
    Run a ReAct agent loop.

    The agent will:
    1. Reason about what to do (visible in its text responses)
    2. Choose and call tools
    3. Observe the results
    4. Repeat until it has an answer
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    system_prompt = """You are a research agent. Your job is to thoroughly answer questions by searching for information, reading sources, and synthesizing findings.

Your approach:
1. THINK about what information you need
2. SEARCH for relevant sources
3. READ promising results for details
4. Take NOTES on key findings
5. SYNTHESIZE a comprehensive answer when you have enough information

Always explain your reasoning before taking an action. Be thorough — check multiple sources when possible."""

    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Print any reasoning text the model produces
        for block in response.content:
            if hasattr(block, "text"):
                print(f"Thought: {block.text[:200]}...")
                if len(block.text) <= 200:
                    print(f"Thought: {block.text}")

        # If the model is done (no more tool calls), return the answer
        if response.stop_reason == "end_turn":
            final = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            print(f"\n{'='*60}")
            print(f"Final Answer:\n{final}")
            print(f"{'='*60}")
            return final

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                print(f"Action: {block.name}({json.dumps(block.input)})")
                result = TOOL_DISPATCH[block.name](block.input)
                print(f"Observation: {json.dumps(result)[:150]}...")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        messages.append({"role": "user", "content": tool_results})
        print()

    print("Agent reached max steps without finishing.")
    return None


# --- Demo -------------------------------------------------------------------

if __name__ == "__main__":
    react_agent(
        "Explain Python's asyncio library. What are the key concepts and when should I use it?"
    )
