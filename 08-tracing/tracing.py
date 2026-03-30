"""
Example 8: Agent Tracing & Evaluation
======================================
A lightweight tracing system that captures the three observability primitives
from LangChain's "Agent Observability Powers Agent Evaluation" framework:

  - Run:    A single LLM call (input, output, tool choice, latency)
  - Trace:  A complete agent execution (all runs for one task)
  - Thread: A multi-turn session (multiple traces over time)

These same traces power both debugging AND evaluation — you don't need
separate systems.

Storage: PostgreSQL with JSONB columns for flexible trace data.
Lightweight alternative: SQLite for local development (shown here).

Run: python tracing.py          (demo with SQLite)
     python tracing.py --pg     (demo with PostgreSQL, requires connection string)
"""

import json
import time
import uuid
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data model — maps 1:1 to the blog's three primitives
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """A single LLM call. The atomic unit of observability."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_run_id: str | None = None     # for nested tool calls
    run_type: str = "llm"                # "llm", "tool", "chain"
    name: str = ""                       # e.g. "claude-sonnet-4-20250514" or "get_weather"
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    error: str | None = None
    start_time: str = ""
    end_time: str = ""
    latency_ms: float = 0
    token_usage: dict = field(default_factory=dict)  # {"input": N, "output": N}
    metadata: dict = field(default_factory=dict)      # arbitrary tags


@dataclass
class Trace:
    """A complete agent execution — all runs for one task."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    input: str = ""                 # the user's original request
    output: str = ""                # the agent's final answer
    status: str = "running"         # "running", "success", "error"
    start_time: str = ""
    end_time: str = ""
    total_runs: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Thread:
    """A multi-turn session — groups traces across a conversation."""
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: str = ""
    last_active: str = ""
    total_traces: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Storage backend — SQLite for dev, PostgreSQL for production
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT,
    created_at  TEXT NOT NULL,
    last_active TEXT NOT NULL,
    total_traces INTEGER DEFAULT 0,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    thread_id       TEXT REFERENCES threads(thread_id),
    input           TEXT,
    output          TEXT,
    status          TEXT DEFAULT 'running',
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    total_runs      INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    total_latency_ms REAL DEFAULT 0,
    metadata        TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL REFERENCES traces(trace_id),
    parent_run_id   TEXT,
    run_type        TEXT NOT NULL,
    name            TEXT,
    input           TEXT,
    output          TEXT,
    error           TEXT,
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    latency_ms      REAL DEFAULT 0,
    token_usage     TEXT DEFAULT '{}',
    metadata        TEXT DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_runs_trace     ON runs(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_thread  ON traces(thread_id);
CREATE INDEX IF NOT EXISTS idx_traces_status  ON traces(status);
CREATE INDEX IF NOT EXISTS idx_runs_type      ON runs(run_type);
"""

# For PostgreSQL, swap TEXT → JSONB for the JSON columns:
SCHEMA_PG_UPGRADE = """
-- Run this on PostgreSQL for better JSON querying:
-- ALTER TABLE runs    ALTER COLUMN input       TYPE JSONB USING input::jsonb;
-- ALTER TABLE runs    ALTER COLUMN output      TYPE JSONB USING output::jsonb;
-- ALTER TABLE runs    ALTER COLUMN token_usage TYPE JSONB USING token_usage::jsonb;
-- ALTER TABLE runs    ALTER COLUMN metadata    TYPE JSONB USING metadata::jsonb;
-- ALTER TABLE traces  ALTER COLUMN metadata    TYPE JSONB USING metadata::jsonb;
--
-- Then you can query:
--   SELECT * FROM runs WHERE metadata->>'model' = 'claude-sonnet-4-20250514';
--   SELECT * FROM runs WHERE (token_usage->>'input')::int > 1000;
"""


class TraceStore:
    """SQLite-backed trace store. Swap for asyncpg/psycopg for PostgreSQL."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA_SQL)

    def save_thread(self, thread: Thread):
        self.conn.execute(
            "INSERT OR REPLACE INTO threads VALUES (?,?,?,?,?,?)",
            (thread.thread_id, thread.user_id, thread.created_at,
             thread.last_active, thread.total_traces, json.dumps(thread.metadata)),
        )
        self.conn.commit()

    def save_trace(self, trace: Trace):
        self.conn.execute(
            "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trace.trace_id, trace.thread_id, trace.input, trace.output,
             trace.status, trace.start_time, trace.end_time, trace.total_runs,
             trace.total_tokens, trace.total_latency_ms, json.dumps(trace.metadata)),
        )
        self.conn.commit()

    def save_run(self, run: Run):
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.run_id, run.trace_id, run.parent_run_id, run.run_type,
             run.name, json.dumps(run.input), json.dumps(run.output),
             run.error, run.start_time, run.end_time, run.latency_ms,
             json.dumps(run.token_usage), json.dumps(run.metadata)),
        )
        self.conn.commit()

    def get_trace_runs(self, trace_id: str) -> list[dict]:
        """Get all runs for a trace — the full execution tree."""
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE trace_id = ? ORDER BY start_time", (trace_id,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_thread_traces(self, thread_id: str) -> list[dict]:
        """Get all traces in a thread — the full conversation history."""
        cur = self.conn.execute(
            "SELECT * FROM traces WHERE thread_id = ? ORDER BY start_time", (thread_id,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_failed_traces(self, limit: int = 50) -> list[dict]:
        """Find failed traces for debugging and evaluation dataset building."""
        cur = self.conn.execute(
            "SELECT * FROM traces WHERE status = 'error' ORDER BY start_time DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_slow_runs(self, threshold_ms: float = 5000, limit: int = 50) -> list[dict]:
        """Find slow runs — useful for optimizing agent performance."""
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE latency_ms > ? ORDER BY latency_ms DESC LIMIT ?",
            (threshold_ms, limit),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Tracing context manager — instrument any agent with minimal code changes
# ---------------------------------------------------------------------------

class Tracer:
    """
    Instruments an agent with tracing. Usage:

        tracer = Tracer(store)
        with tracer.trace("What's the weather?") as t:
            with t.run("llm", "claude-sonnet") as r:
                response = call_llm(...)
                r.output = {"text": response}
            with t.run("tool", "get_weather") as r:
                result = get_weather("NYC")
                r.output = {"result": result}
    """

    def __init__(self, store: TraceStore, thread_id: str | None = None):
        self.store = store
        self.thread_id = thread_id or str(uuid.uuid4())
        # Ensure thread exists
        now = datetime.now(timezone.utc).isoformat()
        self.thread = Thread(
            thread_id=self.thread_id, created_at=now, last_active=now,
        )
        self.store.save_thread(self.thread)

    @contextmanager
    def trace(self, user_input: str, metadata: dict | None = None):
        """Context manager for a full agent execution."""
        t = Trace(
            thread_id=self.thread_id,
            input=user_input,
            start_time=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        ctx = _TraceContext(t, self.store)
        try:
            yield ctx
            t.status = "success"
        except Exception as e:
            t.status = "error"
            t.metadata["error"] = str(e)
            raise
        finally:
            t.end_time = datetime.now(timezone.utc).isoformat()
            t.total_runs = ctx.run_count
            t.total_tokens = ctx.total_tokens
            t.total_latency_ms = ctx.total_latency_ms
            self.store.save_trace(t)
            # Update thread
            self.thread.total_traces += 1
            self.thread.last_active = t.end_time
            self.store.save_thread(self.thread)


class _TraceContext:
    def __init__(self, trace: Trace, store: TraceStore):
        self.trace = trace
        self.store = store
        self.run_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def set_output(self, output: str):
        self.trace.output = output

    @contextmanager
    def run(self, run_type: str, name: str, parent_run_id: str | None = None,
            metadata: dict | None = None):
        """Context manager for a single run within a trace."""
        r = Run(
            trace_id=self.trace.trace_id,
            parent_run_id=parent_run_id,
            run_type=run_type,
            name=name,
            start_time=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        start = time.perf_counter()
        try:
            yield r
        except Exception as e:
            r.error = str(e)
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            r.latency_ms = elapsed
            r.end_time = datetime.now(timezone.utc).isoformat()
            self.store.save_run(r)
            self.run_count += 1
            self.total_latency_ms += elapsed
            input_tokens = r.token_usage.get("input", 0)
            output_tokens = r.token_usage.get("output", 0)
            self.total_tokens += input_tokens + output_tokens


# ---------------------------------------------------------------------------
# Evaluation helpers — traces power evals (the blog's core thesis)
# ---------------------------------------------------------------------------

class TraceEvaluator:
    """
    Evaluate traces at different granularities:
    - Single-step (Run level):  "Did the agent pick the right tool?"
    - Full-turn (Trace level):  "Did the agent complete the task correctly?"
    - Multi-turn (Thread level): "Did the agent maintain context?"
    """

    def __init__(self, store: TraceStore):
        self.store = store

    def eval_single_step(self, trace_id: str, step_index: int,
                         expected_tool: str) -> dict:
        """Unit test for agent reasoning: did step N call the expected tool?"""
        runs = self.store.get_trace_runs(trace_id)
        tool_runs = [r for r in runs if r["run_type"] == "tool"]
        if step_index >= len(tool_runs):
            return {"pass": False, "reason": f"Only {len(tool_runs)} tool calls, expected index {step_index}"}
        actual = tool_runs[step_index]["name"]
        passed = actual == expected_tool
        return {
            "pass": passed,
            "expected": expected_tool,
            "actual": actual,
            "reason": "Tool matched" if passed else f"Expected {expected_tool}, got {actual}",
        }

    def eval_trajectory(self, trace_id: str,
                        required_tools: list[str]) -> dict:
        """Integration test: did the trace use all required tools (in any order)?"""
        runs = self.store.get_trace_runs(trace_id)
        tools_called = {r["name"] for r in runs if r["run_type"] == "tool"}
        missing = set(required_tools) - tools_called
        return {
            "pass": len(missing) == 0,
            "required": required_tools,
            "called": sorted(tools_called),
            "missing": sorted(missing),
        }

    def eval_final_output(self, trace_id: str,
                          must_contain: list[str]) -> dict:
        """Check that the final response contains required information."""
        cur = self.store.conn.execute(
            "SELECT output FROM traces WHERE trace_id = ?", (trace_id,)
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return {"pass": False, "reason": "No output found"}
        output = row[0].lower()
        missing = [term for term in must_contain if term.lower() not in output]
        return {
            "pass": len(missing) == 0,
            "must_contain": must_contain,
            "missing": missing,
        }


# ---------------------------------------------------------------------------
# Demo: Traced agent with evaluation
# ---------------------------------------------------------------------------

def demo():
    import json

    store = TraceStore(":memory:")  # swap with "traces.db" for persistence
    tracer = Tracer(store, thread_id="session-001")
    evaluator = TraceEvaluator(store)

    # Simulate a traced agent execution
    print("=" * 60)
    print("Traced Agent Execution")
    print("=" * 60)

    with tracer.trace("What's the weather in NYC and SF?") as t:
        # Step 1: LLM decides to call tools
        with t.run("llm", "claude-sonnet-4-20250514") as r:
            r.input = {"messages": [{"role": "user", "content": "What's the weather in NYC and SF?"}]}
            r.output = {"tool_calls": ["get_weather(NYC)", "get_weather(SF)"]}
            r.token_usage = {"input": 45, "output": 30}
            time.sleep(0.05)  # simulate latency

        # Step 2: Tool call — NYC
        with t.run("tool", "get_weather", metadata={"city": "NYC"}) as r:
            r.input = {"city": "NYC"}
            r.output = {"temp": 72, "condition": "sunny"}
            time.sleep(0.02)

        # Step 3: Tool call — SF
        with t.run("tool", "get_weather", metadata={"city": "SF"}) as r:
            r.input = {"city": "SF"}
            r.output = {"temp": 58, "condition": "foggy"}
            time.sleep(0.02)

        # Step 4: LLM synthesizes final response
        with t.run("llm", "claude-sonnet-4-20250514") as r:
            r.input = {"tool_results": ["NYC: 72F sunny", "SF: 58F foggy"]}
            final = "NYC is 72F and sunny. San Francisco is 58F and foggy."
            r.output = {"text": final}
            r.token_usage = {"input": 120, "output": 25}
            t.set_output(final)
            time.sleep(0.03)

    trace_id = t.trace.trace_id

    # Show the trace
    print(f"\nTrace ID: {trace_id}")
    runs = store.get_trace_runs(trace_id)
    for i, run in enumerate(runs):
        print(f"  Step {i+1}: [{run['run_type']}] {run['name']} — {run['latency_ms']:.0f}ms")
        print(f"          in:  {run['input'][:80]}")
        print(f"          out: {run['output'][:80]}")

    # --- Evaluate the trace at different granularities ---
    print(f"\n{'='*60}")
    print("Evaluations")
    print("=" * 60)

    # Single-step: Did step 0 call get_weather?
    result = evaluator.eval_single_step(trace_id, step_index=0, expected_tool="get_weather")
    print(f"\n1. Single-step eval (first tool = get_weather?): {'PASS' if result['pass'] else 'FAIL'}")
    print(f"   {result}")

    # Trajectory: Were all required tools called?
    result = evaluator.eval_trajectory(trace_id, required_tools=["get_weather"])
    print(f"\n2. Trajectory eval (used get_weather?): {'PASS' if result['pass'] else 'FAIL'}")
    print(f"   {result}")

    # Final output: Does the response contain the expected cities?
    result = evaluator.eval_final_output(trace_id, must_contain=["NYC", "San Francisco"])
    print(f"\n3. Output eval (mentions both cities?): {'PASS' if result['pass'] else 'FAIL'}")
    print(f"   {result}")

    # --- Show thread-level view ---
    print(f"\n{'='*60}")
    print("Thread Overview (session-001)")
    print("=" * 60)
    traces = store.get_thread_traces("session-001")
    for tr in traces:
        print(f"  Trace: {tr['trace_id'][:8]}... | {tr['status']} | "
              f"{tr['total_runs']} runs | {tr['total_tokens']} tokens | "
              f"{tr['total_latency_ms']:.0f}ms")


if __name__ == "__main__":
    demo()
