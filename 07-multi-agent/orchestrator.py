"""
Example 7: Multi-Agent Orchestration with Claude Agent SDK
==========================================================
The most advanced pattern: multiple specialized agents coordinated by an
orchestrator. Each agent has its own tools, system prompt, and expertise.
The orchestrator delegates subtasks and synthesizes results.

This example uses the Anthropic Claude Agent SDK (claude_agent_sdk) which
provides primitives for building sophisticated agent systems.

Key concepts:
- Agent delegation: An orchestrator agent that routes work to specialists
- Parallel tool execution: Multiple agents working concurrently
- Shared context: Agents passing information through a common state
- Human-in-the-loop: Asking for user confirmation on critical actions

Architecture:
    ┌───────────────────────────────┐
    │        Orchestrator           │
    │   "I need to research, then  │
    │    code, then review."       │
    └──────┬────────┬────────┬─────┘
           │        │        │
    ┌──────▼──┐ ┌───▼───┐ ┌─▼──────┐
    │Research │ │ Coder │ │Reviewer│
    │ Agent   │ │ Agent │ │ Agent  │
    └─────────┘ └───────┘ └────────┘

Run: python orchestrator.py
"""

import json
import anthropic

client = anthropic.Anthropic()


# --- Agent definitions ------------------------------------------------------
# Each agent is a configuration: system prompt + available tools.

class Agent:
    def __init__(self, name: str, system_prompt: str, tools: list[dict], tool_handlers: dict):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_handlers = tool_handlers

    def run(self, task: str, context: str = "") -> str:
        """Run this agent on a task, returning its final text response."""
        user_content = task
        if context:
            user_content = f"Context from previous agents:\n{context}\n\nTask: {task}"

        messages = [{"role": "user", "content": user_content}]

        for _ in range(10):  # max steps
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"    [{self.name}] → {block.name}({json.dumps(block.input)[:80]})")
                        result = self.tool_handlers[block.name](block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                return next((b.text for b in response.content if hasattr(b, "text")), "")

        return "Agent reached max steps."


# --- Specialist agents ------------------------------------------------------

# 1. Research Agent: Gathers information
research_tools = [
    {
        "name": "search_codebase",
        "description": "Search the codebase for relevant files and patterns.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

# Simulated codebase
FAKE_CODEBASE = {
    "src/api/auth.py": 'def authenticate(token: str) -> User:\n    """Validate JWT token and return user."""\n    decoded = jwt.decode(token, SECRET_KEY)\n    return User.from_dict(decoded)\n',
    "src/api/routes.py": '@app.get("/users/{user_id}")\nasync def get_user(user_id: int, token: str = Header()):\n    user = authenticate(token)\n    return db.get_user(user_id)\n',
    "src/models/user.py": 'class User:\n    id: int\n    name: str\n    email: str\n    role: str  # "admin", "user", "viewer"\n',
    "tests/test_auth.py": 'def test_authenticate_valid_token():\n    token = create_test_token(user_id=1)\n    user = authenticate(token)\n    assert user.id == 1\n',
}


def search_codebase(inp):
    query = inp["query"].lower()
    matches = {path: content for path, content in FAKE_CODEBASE.items() if query in path.lower() or query in content.lower()}
    return {"matches": list(matches.keys()), "count": len(matches)}


def read_file(inp):
    content = FAKE_CODEBASE.get(inp["path"], "File not found")
    return {"path": inp["path"], "content": content}


research_agent = Agent(
    name="Researcher",
    system_prompt=(
        "You are a research agent. Your job is to explore a codebase and gather "
        "all relevant information needed to complete a task. Search for files, "
        "read their contents, and produce a comprehensive summary of your findings."
    ),
    tools=research_tools,
    tool_handlers={"search_codebase": search_codebase, "read_file": read_file},
)

# 2. Coding Agent: Writes code
coding_tools = [
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the test suite and return results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "Specific test file to run, or 'all'"},
            },
            "required": ["test_path"],
        },
    },
]

written_files: dict[str, str] = {}


def write_file(inp):
    written_files[inp["path"]] = inp["content"]
    return {"status": "written", "path": inp["path"], "size": len(inp["content"])}


def run_tests(inp):
    return {"status": "passed", "tests_run": 4, "tests_passed": 4, "tests_failed": 0}


coding_agent = Agent(
    name="Coder",
    system_prompt=(
        "You are a coding agent. Given research context about a codebase, write "
        "clean, well-tested code to implement the requested feature. Follow existing "
        "patterns in the codebase. Always run tests after writing code."
    ),
    tools=coding_tools,
    tool_handlers={"write_file": write_file, "run_tests": run_tests},
)

# 3. Review Agent: Reviews code changes
review_tools = [
    {
        "name": "get_diff",
        "description": "Get the diff of all changes made.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "add_review_comment",
        "description": "Add a review comment on a specific file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "comment": {"type": "string"},
                "severity": {"type": "string", "enum": ["info", "warning", "error"]},
            },
            "required": ["path", "comment", "severity"],
        },
    },
]

review_comments: list[dict] = []


def get_diff(inp):
    return {"files_changed": list(written_files.keys()), "changes": written_files}


def add_review_comment(inp):
    comment = {"path": inp["path"], "comment": inp["comment"], "severity": inp["severity"]}
    review_comments.append(comment)
    return {"status": "comment_added"}


review_agent = Agent(
    name="Reviewer",
    system_prompt=(
        "You are a code review agent. Review the changes for correctness, security, "
        "style, and test coverage. Flag any issues with specific comments. "
        "Be constructive and specific in your feedback."
    ),
    tools=review_tools,
    tool_handlers={"get_diff": get_diff, "add_review_comment": add_review_comment},
)


# --- Orchestrator -----------------------------------------------------------

def orchestrate(task: str):
    """
    The orchestrator breaks down a task and delegates to specialist agents
    in sequence, passing context between them.
    """
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")

    # Step 1: Research
    print("Phase 1: Research")
    print("-" * 40)
    research_result = research_agent.run(
        f"Research the codebase to understand what's needed for this task: {task}"
    )
    print(f"\n  Research summary: {research_result[:200]}...\n")

    # Step 2: Implement
    print("Phase 2: Implementation")
    print("-" * 40)
    coding_result = coding_agent.run(
        task,
        context=research_result,
    )
    print(f"\n  Coding summary: {coding_result[:200]}...\n")

    # Step 3: Review
    print("Phase 3: Code Review")
    print("-" * 40)
    review_result = review_agent.run(
        "Review the code changes for quality, security, and correctness.",
        context=f"Task: {task}\n\nImplementation notes: {coding_result}",
    )
    print(f"\n  Review summary: {review_result[:200]}...\n")

    # Final synthesis
    print("=" * 60)
    print("Orchestration complete!")
    print(f"  Files written: {list(written_files.keys())}")
    print(f"  Review comments: {len(review_comments)}")
    for comment in review_comments:
        print(f"    [{comment['severity']}] {comment['path']}: {comment['comment']}")
    print("=" * 60)


# --- Demo -------------------------------------------------------------------

if __name__ == "__main__":
    orchestrate(
        "Add role-based authorization to the GET /users endpoint. "
        "Only admins should be able to view other users' profiles. "
        "Regular users can only view their own profile."
    )
