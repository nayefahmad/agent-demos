"""
Example 9d: LlamaIndex Workflows (Event-Driven Agent)
======================================================
Workflows are LlamaIndex's answer to LangGraph's StateGraph. Instead of
an explicit graph with nodes and edges, you define typed Events and Steps.
The framework infers the graph from function signatures.

Comparison:
  LangGraph:  graph.add_node("agent", fn) → graph.add_edge("agent", "tools")
  LlamaIndex: @step decorator + typed Event classes → implicit graph

┌──────────────────────────────────────────────────────────────┐
│  LangGraph (Explicit Graph)     LlamaIndex Workflows         │
│                                 (Event-Driven)               │
│  builder = StateGraph(State)    class MyWorkflow(Workflow):   │
│  builder.add_node("a", fn_a)     @step                       │
│  builder.add_node("b", fn_b)     async def a(self, ev:       │
│  builder.add_edge("a", "b")         StartEvent) -> EventB:   │
│  builder.add_edge("b", END)         return EventB(data=...)  │
│  graph = builder.compile()                                    │
│                                   @step                       │
│  # Explicit node + edge wiring    async def b(self, ev:       │
│  # You see the full topology         EventB) -> StopEvent:    │
│                                      return StopEvent(...)    │
│                                                               │
│                                  # Implicit: a→b inferred     │
│                                  # from Event types            │
└──────────────────────────────────────────────────────────────┘

Run: python workflow_agent.py
"""

import asyncio

from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context, Event


# ---------------------------------------------------------------------------
# Define typed events (these are the "edges" — they connect steps implicitly)
# ---------------------------------------------------------------------------

class ResearchEvent(Event):
    """Emitted when the user query needs research."""
    query: str


class AnalyzeEvent(Event):
    """Emitted when research is complete and needs analysis."""
    query: str
    raw_findings: list[str]


class ReviewEvent(Event):
    """Emitted when analysis is complete and needs review."""
    query: str
    analysis: str


# ---------------------------------------------------------------------------
# Define the workflow (steps are inferred from input/output event types)
# ---------------------------------------------------------------------------

class ResearchPipeline(Workflow):
    """
    A 4-step research pipeline:
      StartEvent → research → analyze → review → StopEvent

    The graph is inferred automatically:
      start(StartEvent) → ResearchEvent
      research(ResearchEvent) → AnalyzeEvent
      analyze(AnalyzeEvent) → ReviewEvent
      review(ReviewEvent) → StopEvent
    """

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ResearchEvent:
        """Parse the user's request and kick off research."""
        query = ev.get("query", "")
        print(f"[start] Received query: {query}")

        # Store the query in shared context
        await ctx.set("query", query)
        await ctx.set("step_count", 0)

        return ResearchEvent(query=query)

    @step
    async def research(self, ctx: Context, ev: ResearchEvent) -> AnalyzeEvent:
        """Gather information (simulated)."""
        print(f"[research] Searching for: {ev.query}")

        # Simulated research results
        findings = [
            f"Finding 1: {ev.query} has significant implications for production systems.",
            f"Finding 2: Recent benchmarks show 40% improvement in {ev.query} performance.",
            f"Finding 3: Industry adoption of {ev.query} grew 3x in 2025.",
        ]

        count = await ctx.get("step_count", 0)
        await ctx.set("step_count", count + 1)

        print(f"[research] Found {len(findings)} results")
        return AnalyzeEvent(query=ev.query, raw_findings=findings)

    @step
    async def analyze(self, ctx: Context, ev: AnalyzeEvent) -> ReviewEvent:
        """Analyze and synthesize the research findings."""
        print(f"[analyze] Synthesizing {len(ev.raw_findings)} findings")

        # In production, this would call an LLM
        analysis = (
            f"Analysis of '{ev.query}':\n"
            + "\n".join(f"  • {f}" for f in ev.raw_findings)
            + "\n\nConclusion: The evidence strongly supports further investment."
        )

        count = await ctx.get("step_count", 0)
        await ctx.set("step_count", count + 1)

        return ReviewEvent(query=ev.query, analysis=analysis)

    @step
    async def review(self, ctx: Context, ev: ReviewEvent) -> StopEvent:
        """Review the analysis and produce the final report."""
        print(f"[review] Reviewing analysis")

        count = await ctx.get("step_count", 0)
        await ctx.set("step_count", count + 1)

        report = (
            f"RESEARCH REPORT\n"
            f"{'='*40}\n"
            f"Query: {ev.query}\n"
            f"Steps completed: {count + 1}\n\n"
            f"{ev.analysis}\n\n"
            f"Status: APPROVED ✓"
        )

        return StopEvent(result=report)


# ---------------------------------------------------------------------------
# Advanced: Workflow with loops (conditional re-routing)
# ---------------------------------------------------------------------------

class DraftEvent(Event):
    content: str
    revision: int


class FeedbackEvent(Event):
    content: str
    revision: int
    approved: bool
    feedback: str


class EditingWorkflow(Workflow):
    """
    Demonstrates loops: the reviewer can send drafts back for revision.

    StartEvent → draft → review ──approved──→ StopEvent
                   ▲              │
                   └──not approved┘
    """

    @step
    async def draft(self, ctx: Context, ev: StartEvent | FeedbackEvent) -> DraftEvent:
        """Write or revise a draft."""
        if isinstance(ev, StartEvent):
            topic = ev.get("topic", "unknown")
            await ctx.set("topic", topic)
            print(f"[draft] Writing first draft about: {topic}")
            return DraftEvent(content=f"Draft about {topic}: initial version.", revision=1)
        else:
            # Revision based on feedback
            print(f"[draft] Revising (v{ev.revision + 1}): {ev.feedback}")
            return DraftEvent(
                content=f"{ev.content} [Revised: addressed '{ev.feedback}']",
                revision=ev.revision + 1,
            )

    @step
    async def review(self, ctx: Context, ev: DraftEvent) -> FeedbackEvent | StopEvent:
        """Review the draft — approve or request revision."""
        print(f"[review] Reviewing revision {ev.revision}")

        # Approve after 2 revisions (simulated)
        if ev.revision >= 2:
            print(f"[review] Approved!")
            return StopEvent(result=f"FINAL: {ev.content}")
        else:
            print(f"[review] Requesting revision...")
            return FeedbackEvent(
                content=ev.content,
                revision=ev.revision,
                approved=False,
                feedback="Add more detail and examples",
            )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main():
    # Linear pipeline
    print("=" * 60)
    print("Linear Pipeline Workflow")
    print("=" * 60)

    pipeline = ResearchPipeline(timeout=30)
    result = await pipeline.run(query="LLM agent architectures")
    print(f"\n{result}\n")

    # Workflow with loops
    print("=" * 60)
    print("Editing Workflow (with revision loop)")
    print("=" * 60)

    editor = EditingWorkflow(timeout=30)
    result = await editor.run(topic="ReAct vs Function Calling agents")
    print(f"\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
