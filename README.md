# LLM Agents: A Crash Course

A hands-on guide to building LLM agents, from simple chat to multi-agent
orchestration. Each example builds on the previous one, introducing new concepts
incrementally.

## Concepts Covered

| # | Example | Key Concepts |
|---|---------|-------------|
| 1 | [Basic Multi-Turn Chat](01-basic-chat/) | Conversation history, system prompts, message roles |
| 2 | [Tool Use / Function Calling](02-tool-use/) | Tool definitions, function calling, structured output |
| 3 | [ReAct Loop](03-react-loop/) | Reasoning + Acting loop, autonomous decision-making |
| 4 | [MCP Server](04-mcp-server/) | Model Context Protocol, tool servers, transport layers |
| 5 | [LangGraph Agent](05-langgraph-agent/) | Graph-based agents, state machines, conditional routing |
| 6 | [OpenAI Agents SDK](06-openai-agents-sdk/) | Agent handoffs, guardrails, tracing |
| 7 | [Multi-Agent Orchestration](07-multi-agent/) | Claude Agent SDK, agent delegation, parallel execution |
| 8 | [Tracing & Evaluation](08-tracing/) | Runs/Traces/Threads, trace-powered evals, observability |
| 9 | [LlamaIndex Agents](09-llamaindex-agents/) | FunctionTool, AgentWorkflow, Workflows, MCP integration |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      User / Application                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Agent Loop │  ← ReAct: Reason → Act → Observe → Repeat
                    │  (Orchestr.)│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────┐
              │            │                │
        ┌─────▼─────┐ ┌───▼────┐   ┌──────▼──────┐
        │   LLM     │ │ Memory │   │    Tools     │
        │  (Brain)  │ │ (State)│   │  (Actions)   │
        └───────────┘ └────────┘   └──────┬───────┘
                                          │
                              ┌───────────┼───────────┐
                              │           │           │
                         ┌────▼───┐ ┌─────▼────┐ ┌───▼────┐
                         │  APIs  │ │ MCP Srvrs│ │  Code  │
                         └────────┘ └──────────┘ └────────┘
```

### The ReAct Loop (Reasoning + Acting)

The core pattern behind most LLM agents:

```
while not done:
    thought   = llm.reason(observations)    # "I need to look up the weather"
    action    = llm.choose_tool(thought)     # call weather_api(city="NYC")
    result    = execute(action)              # {"temp": 72, "condition": "sunny"}
    observation = format(result)             # Add result to context
    done      = llm.should_stop(observation) # "I have the answer now"
```

### MCP (Model Context Protocol)

MCP is an open standard for connecting LLMs to external tools and data sources.
Think of it as **USB-C for AI** — a single protocol that lets any LLM talk to
any tool server.

```
┌──────────┐     stdio/SSE      ┌──────────────┐
│  LLM     │ ◄────────────────► │  MCP Server  │
│  Client  │   JSON-RPC 2.0    │  (Tools +    │
│          │                    │   Resources)  │
└──────────┘                    └──────────────┘
```

Key MCP concepts:
- **Tools**: Functions the LLM can call (e.g., `search_files`, `query_db`)
- **Resources**: Data the LLM can read (e.g., file contents, DB schemas)
- **Prompts**: Reusable prompt templates the server exposes
- **Transports**: stdio (local) or SSE/Streamable HTTP (remote)

## Multi-Agent Orchestration Patterns

Real-world agent systems combine multiple agents. Here are the main patterns:

```
Supervisor (~70% of production)     Pipeline (Sequential)
        [Supervisor]                [A] → [B] → [C] → Output
       /     |      \
  [Agent A] [Agent B] [Agent C]

Swarm (Decentralized)               Hierarchical (Large-scale)
  [A] ←→ [B]                              [Executive]
   ↕       ↕                              /         \
  [C] ←→ [D]                        [Manager A]  [Manager B]
                                     /    \        /    \
                                   [W1]  [W2]   [W3]  [W4]
```

| Pattern | When to Use | Example |
|---------|-------------|---------|
| **Supervisor** | Most tasks — one coordinator delegates to specialists | Example 7 |
| **Pipeline** | Clear sequential stages (research → code → review) | Example 7 |
| **Handoffs** | Routing to the right specialist (triage → billing) | Example 6 |
| **Swarm** | Resilient peer-to-peer collaboration | Advanced |
| **Hierarchical** | 50+ agent enterprise systems | Advanced |

## Memory Patterns

| Type | Scope | Implementation |
|------|-------|----------------|
| **Short-term** | Current conversation | Message history (Example 1) |
| **Working** | Current task | Agent scratchpad/notepad (Example 3) |
| **Long-term semantic** | Persistent knowledge | Vector DB + RAG |
| **Long-term episodic** | Past interactions | Stored conversation summaries |

## Framework Comparison

| Framework | Best For | Language |
|-----------|----------|----------|
| **Anthropic API** (raw) | Full control, learning fundamentals | Python/TS |
| **Claude Agent SDK** | Code agents with built-in tools | Python |
| **OpenAI Agents SDK** | Clean handoffs + guardrails | Python |
| **LangGraph** | Complex stateful workflows, graph control | Python |
| **CrewAI** | Role-based team prototyping | Python |
| **Google ADK** | Gemini ecosystem | Python |
| **Pydantic AI** | Type-safe agents | Python |
| **Mastra** | TypeScript-first web apps | TypeScript |

## Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Set your API keys
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # only needed for example 06
```

## How to Use This Guide

Start with Example 1 and work your way up. Each example has its own README
with detailed explanations. The examples are designed to be self-contained —
you can run each one independently.

```bash
# Run any example
cd 01-basic-chat && python chat.py
```
