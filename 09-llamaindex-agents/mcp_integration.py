"""
Example 9e: LlamaIndex + MCP Integration
=========================================
LlamaIndex can consume MCP servers as tool sources via llama-index-tools-mcp.
This connects our MCP server from Example 4 to a LlamaIndex agent.

Install: pip install llama-index-tools-mcp

Three integration modes:
1. Consume external MCP servers as tools (shown here)
2. Expose a LlamaIndex Workflow as an MCP server
3. LlamaCloud MCP for LlamaParse/LlamaExtract

Run: python mcp_integration.py
     (requires the MCP server from Example 4 to be available)
"""

import asyncio

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.llms.openai import OpenAI


async def demo_mcp_stdio():
    """Connect to a local MCP server via stdio (subprocess)."""
    print("=" * 60)
    print("LlamaIndex + MCP Server (stdio)")
    print("=" * 60)

    # Connect to our MCP server from Example 4
    mcp_client = BasicMCPClient(
        command_or_url="python",
        args=["../04-mcp-server/server.py"],
    )

    # Convert MCP tools to LlamaIndex tools
    tool_spec = McpToolSpec(client=mcp_client)
    tools = await tool_spec.to_tool_list_async()

    print(f"Discovered {len(tools)} tools from MCP server:")
    for t in tools:
        print(f"  - {t.metadata.name}: {t.metadata.description}")

    # Create an agent with the MCP tools
    agent = AgentWorkflow.from_tools_or_functions(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt="You are a task manager. Use the available tools to manage tasks.",
    )

    response = await agent.run(
        user_msg="List all pending tasks, then add a new task 'Deploy v2.0'"
    )
    print(f"\nResponse: {response}")


async def demo_mcp_http():
    """Connect to a remote MCP server via HTTP/SSE."""
    print("\n" + "=" * 60)
    print("LlamaIndex + MCP Server (HTTP)")
    print("=" * 60)

    # For a remote MCP server running on HTTP
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/mcp")

    tool_spec = McpToolSpec(
        client=mcp_client,
        # Optional: only expose specific tools
        # allowed_tools=["list_tasks", "add_task"],
    )
    tools = await tool_spec.to_tool_list_async()

    agent = AgentWorkflow.from_tools_or_functions(
        tools,
        llm=OpenAI(model="gpt-4o"),
    )

    response = await agent.run(user_msg="Show me all tasks")
    print(f"\nResponse: {response}")


# Shortcut: one-liner to get tools from an MCP URL
async def demo_shortcut():
    """Convenience function for quick MCP tool loading."""
    from llama_index.tools.mcp import aget_tools_from_mcp_url

    tools = await aget_tools_from_mcp_url("http://127.0.0.1:8000/mcp")
    print(f"Got {len(tools)} tools via shortcut")


if __name__ == "__main__":
    asyncio.run(demo_mcp_stdio())
