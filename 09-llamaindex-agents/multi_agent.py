"""
Example 9c: LlamaIndex Multi-Agent with Handoffs
=================================================
AgentWorkflow supports multi-agent orchestration where agents hand off to
each other via `can_handoff_to`. This is comparable to:

  - OpenAI Agents SDK: `handoff()` between agents (Example 6)
  - LangGraph: conditional edges routing between agent nodes (Example 5)
  - Our manual orchestrator: sequential pipeline (Example 7)

LlamaIndex's approach: declare a graph of agents with `can_handoff_to`,
let the LLM decide when to hand off based on the agent descriptions.

Architecture:
    ┌──────────────┐
    │   Triage     │ (root_agent)
    │   Agent      │
    └──┬────────┬──┘
       │        │
       ▼        ▼
  ┌─────────┐ ┌──────────┐
  │Research │ │ Writing  │
  │ Agent   │ │  Agent   │
  └────┬────┘ └─────┬────┘
       │            │
       └──────┬─────┘
              ▼
       ┌────────────┐
       │  Review    │
       │  Agent     │
       └────────────┘

Run: python multi_agent.py
"""

import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI


# --- Shared state tools (all agents can read/write shared context) ----------

notes: list[str] = []
report_drafts: list[str] = []


def record_note(note: str) -> str:
    """Save a research note for other agents to reference."""
    notes.append(note)
    return f"Note recorded ({len(notes)} total)"


def read_notes() -> str:
    """Read all research notes collected so far."""
    if not notes:
        return "No notes yet."
    return "\n".join(f"- {n}" for n in notes)


def write_draft(content: str) -> str:
    """Write or overwrite the report draft."""
    report_drafts.clear()
    report_drafts.append(content)
    return "Draft saved."


def read_draft() -> str:
    """Read the current report draft."""
    if not report_drafts:
        return "No draft yet."
    return report_drafts[-1]


def submit_review(feedback: str, approved: bool) -> str:
    """Submit a review of the current draft.

    Args:
        feedback: Review comments and suggestions
        approved: Whether the draft is approved for publication
    """
    status = "APPROVED" if approved else "NEEDS REVISION"
    return f"Review submitted: [{status}] {feedback}"


# --- Agent definitions ------------------------------------------------------

llm = OpenAI(model="gpt-4o")

triage_agent = FunctionAgent(
    name="TriageAgent",
    description="Routes tasks to the right specialist. Start here.",
    system_prompt=(
        "You are a project manager. Analyze the user's request and hand off to "
        "the right specialist:\n"
        "- ResearchAgent: for gathering information\n"
        "- WritingAgent: for drafting content\n"
        "- ReviewAgent: for reviewing drafts\n\n"
        "For complex tasks, start with research, then writing, then review."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=["ResearchAgent", "WritingAgent", "ReviewAgent"],
)

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Gathers information and takes notes on a topic",
    system_prompt=(
        "You are a research specialist. Gather information on the given topic "
        "and record your findings as notes. When done, hand off to WritingAgent "
        "to draft a report based on your notes."
    ),
    llm=llm,
    tools=[
        FunctionTool.from_defaults(record_note),
        FunctionTool.from_defaults(read_notes),
    ],
    can_handoff_to=["WritingAgent", "TriageAgent"],
)

writing_agent = FunctionAgent(
    name="WritingAgent",
    description="Writes reports and content based on research notes",
    system_prompt=(
        "You are a technical writer. Read the research notes and write a clear, "
        "well-structured report. Save your draft, then hand off to ReviewAgent."
    ),
    llm=llm,
    tools=[
        FunctionTool.from_defaults(read_notes),
        FunctionTool.from_defaults(write_draft),
        FunctionTool.from_defaults(read_draft),
    ],
    can_handoff_to=["ReviewAgent", "TriageAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews drafts and provides feedback",
    system_prompt=(
        "You are an editor. Read the current draft, provide constructive feedback, "
        "and either approve it or send it back to WritingAgent for revision."
    ),
    llm=llm,
    tools=[
        FunctionTool.from_defaults(read_draft),
        FunctionTool.from_defaults(submit_review),
    ],
    can_handoff_to=["WritingAgent", "TriageAgent"],  # can send back for revisions
)


# --- Build the multi-agent workflow -----------------------------------------

workflow = AgentWorkflow(
    agents=[triage_agent, research_agent, writing_agent, review_agent],
    root_agent="TriageAgent",  # entry point
)


# --- Run with event streaming -----------------------------------------------

async def main():
    print("=" * 60)
    print("LlamaIndex Multi-Agent Workflow")
    print("=" * 60)

    handler = workflow.run(
        user_msg="Research the key benefits of ReAct agents vs function-calling agents, "
                 "then write a short comparison report."
    )

    # Stream events to see which agent is active and what's happening
    current_agent = None
    async for event in handler.stream_events():
        # Track agent handoffs
        if hasattr(event, "current_agent_name"):
            if event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"\n--- Handed off to: {current_agent} ---\n")

        # Stream LLM output tokens
        if hasattr(event, "delta") and event.delta:
            print(event.delta, end="", flush=True)

        # Track tool calls
        if hasattr(event, "tool_name"):
            print(f"\n  [Tool] {event.tool_name}()")

    response = await handler
    print(f"\n\n{'='*60}")
    print(f"Final response:\n{response}")

    if report_drafts:
        print(f"\n{'='*60}")
        print(f"Saved draft:\n{report_drafts[-1]}")


if __name__ == "__main__":
    asyncio.run(main())
