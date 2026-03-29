"""
Example 4b: MCP Client
=======================
A client that connects to our MCP server and lets Claude use its tools.
This demonstrates how an LLM application integrates with MCP servers.

The client:
1. Connects to the MCP server (via stdio subprocess)
2. Discovers available tools from the server
3. Passes them to Claude as tool definitions
4. Routes Claude's tool calls to the MCP server
5. Returns results back to Claude

Run: python client.py
"""

import asyncio
import json

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_mcp_agent(user_message: str):
    """Connect to MCP server and run an agent that can use its tools."""

    # Configure connection to our MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # our MCP server from Example 4a
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Discover available tools from the MCP server
            tools_response = await session.list_tools()
            print(f"Discovered {len(tools_response.tools)} tools from MCP server:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # Convert MCP tool definitions to Anthropic format
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            ]

            # Run the agent loop
            claude = anthropic.Anthropic()
            messages = [{"role": "user", "content": user_message}]

            print(f"User: {user_message}\n")

            while True:
                response = claude.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    tools=anthropic_tools,
                    messages=messages,
                )

                # If Claude wants to call a tool, route it to the MCP server
                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  → MCP call: {block.name}({json.dumps(block.input)})")

                            # Call the tool on the MCP server
                            result = await session.call_tool(
                                block.name, arguments=block.input
                            )

                            result_text = result.content[0].text if result.content else ""
                            print(f"  ← Result: {result_text[:120]}")

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text,
                                }
                            )

                    messages.append({"role": "user", "content": tool_results})

                else:
                    # Final response
                    final = next(
                        (b.text for b in response.content if hasattr(b, "text")), ""
                    )
                    print(f"\nClaude: {final}")
                    return final


# --- Demo -------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("MCP Client + Claude Agent")
    print("=" * 60)

    await run_mcp_agent(
        "What tasks do I have pending? Add a task to 'Deploy v2.0 to staging' "
        "and then show me a summary."
    )


if __name__ == "__main__":
    asyncio.run(main())
