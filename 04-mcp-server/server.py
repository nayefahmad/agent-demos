"""
Example 4a: MCP Server
=======================
A Model Context Protocol (MCP) server that exposes tools and resources
to any MCP-compatible client (Claude Desktop, Claude Code, custom clients).

MCP is an open standard (by Anthropic) that provides a universal way for
LLMs to connect to external tools and data. Think of it as USB-C for AI.

Key MCP concepts demonstrated:
- Tools: Functions the LLM can call
- Resources: Data the LLM can read (like file contents, DB schemas)
- Prompts: Reusable prompt templates
- Transport: stdio (this example) or SSE/Streamable HTTP for remote

Install:  pip install mcp
Run:      python server.py          (starts stdio server)
          python server.py --http   (starts HTTP server on port 8000)
"""

import json
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    name="demo-server",
    version="1.0.0",
)

# --- In-memory data store (simulating a database) --------------------------

_tasks: list[dict] = [
    {"id": 1, "title": "Buy groceries", "done": False, "created": "2026-03-28"},
    {"id": 2, "title": "Review PR #42", "done": True, "created": "2026-03-27"},
    {"id": 3, "title": "Write agent guide", "done": False, "created": "2026-03-29"},
]
_next_id = 4


# --- Tools: Actions the LLM can perform ------------------------------------

@mcp.tool()
def list_tasks(include_done: bool = True) -> str:
    """List all tasks in the todo list.

    Args:
        include_done: Whether to include completed tasks (default: True)
    """
    filtered = _tasks if include_done else [t for t in _tasks if not t["done"]]
    return json.dumps(filtered, indent=2)


@mcp.tool()
def add_task(title: str) -> str:
    """Add a new task to the todo list.

    Args:
        title: The task description
    """
    global _next_id
    task = {
        "id": _next_id,
        "title": title,
        "done": False,
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    _tasks.append(task)
    _next_id += 1
    return json.dumps({"status": "created", "task": task})


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as completed.

    Args:
        task_id: The ID of the task to complete
    """
    for task in _tasks:
        if task["id"] == task_id:
            task["done"] = True
            return json.dumps({"status": "completed", "task": task})
    return json.dumps({"error": f"Task {task_id} not found"})


@mcp.tool()
def search_tasks(query: str) -> str:
    """Search tasks by keyword.

    Args:
        query: Search string to match against task titles
    """
    matches = [t for t in _tasks if query.lower() in t["title"].lower()]
    return json.dumps({"results": matches, "count": len(matches)})


# --- Resources: Data the LLM can read -------------------------------------

@mcp.resource("tasks://summary")
def task_summary() -> str:
    """A summary of the current task list state."""
    total = len(_tasks)
    done = sum(1 for t in _tasks if t["done"])
    pending = total - done
    return json.dumps({
        "total": total,
        "completed": done,
        "pending": pending,
        "completion_rate": f"{done/total*100:.0f}%" if total else "N/A",
    })


@mcp.resource("tasks://schema")
def task_schema() -> str:
    """The JSON schema for task objects."""
    return json.dumps({
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "string"},
            "done": {"type": "boolean"},
            "created": {"type": "string", "format": "date"},
        },
    })


# --- Prompts: Reusable prompt templates ------------------------------------

@mcp.prompt()
def daily_standup() -> str:
    """Generate a daily standup report from the current task list."""
    return (
        "Please review the current task list and generate a brief daily standup report. "
        "Include: what was completed, what's in progress, and any blockers. "
        "Use the list_tasks tool to get the current state."
    )


@mcp.prompt()
def prioritize() -> str:
    """Ask the AI to help prioritize pending tasks."""
    return (
        "Review all pending tasks and suggest a priority order. "
        "Consider urgency, dependencies, and effort. "
        "Use the list_tasks tool with include_done=false to see pending tasks."
    )


# --- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # Run as HTTP server (for remote access)
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    else:
        # Run as stdio server (default, for local use with Claude Desktop/Code)
        mcp.run(transport="stdio")
