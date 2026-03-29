"""
Example 6: OpenAI Agents SDK
==============================
The OpenAI Agents SDK provides a higher-level abstraction for building agents
with features like handoffs (agent-to-agent delegation), guardrails (input/output
validation), and built-in tracing.

Key concepts:
- Agent: An LLM with instructions, tools, and optional handoffs
- Handoffs: One agent can delegate to another (e.g., triage → specialist)
- Guardrails: Input/output validators that run alongside the agent
- Runner: Executes the agent loop and manages the conversation
- Tracing: Built-in observability for debugging agent behavior

Architecture (with handoffs):
    ┌──────────┐     handoff      ┌───────────────┐
    │  Triage  │ ───────────────► │  Specialist   │
    │  Agent   │                  │  Agent        │
    └──────────┘                  └───────────────┘
         │                              │
    uses tools                     uses tools
         │                              │
    ┌────▼─────┐                  ┌─────▼──────┐
    │ classify │                  │ lookup_doc │
    └──────────┘                  └────────────┘

Run: python agent.py
"""

from agents import Agent, Runner, function_tool, handoff


# --- Define tools -----------------------------------------------------------

@function_tool
def lookup_faq(question: str) -> str:
    """Look up frequently asked questions in the knowledge base."""
    faqs = {
        "refund": "Refunds are processed within 5-7 business days. Initiate via Settings > Billing > Request Refund.",
        "cancel": "To cancel your subscription, go to Settings > Billing > Cancel Plan. You'll retain access until the end of your billing period.",
        "upgrade": "Upgrade anytime at Settings > Billing > Change Plan. You'll be charged the prorated difference.",
        "api": "API keys are available at Settings > Developer > API Keys. Rate limit: 1000 req/min on Pro plan.",
    }
    for key, answer in faqs.items():
        if key in question.lower():
            return answer
    return "No FAQ match found. Escalating to human support."


@function_tool
def check_account_status(user_id: str) -> str:
    """Check the status of a user's account."""
    accounts = {
        "user_123": {"plan": "Pro", "status": "active", "billing_date": "2026-04-01"},
        "user_456": {"plan": "Free", "status": "active", "billing_date": None},
        "user_789": {"plan": "Enterprise", "status": "suspended", "billing_date": "2026-03-15"},
    }
    account = accounts.get(user_id)
    if account:
        return f"Plan: {account['plan']}, Status: {account['status']}, Next billing: {account['billing_date']}"
    return f"Account {user_id} not found"


@function_tool
def create_support_ticket(user_id: str, issue: str, priority: str = "medium") -> str:
    """Create a support ticket for issues that need human follow-up.

    Args:
        user_id: The user's ID
        issue: Description of the issue
        priority: low, medium, or high
    """
    ticket_id = "TKT-" + str(hash(f"{user_id}{issue}"))[:6]
    return f"Ticket {ticket_id} created (priority: {priority}). A human agent will follow up within 24 hours."


# --- Define specialist agents -----------------------------------------------

billing_agent = Agent(
    name="Billing Specialist",
    instructions="""You are a billing specialist. Help users with:
- Refunds, cancellations, upgrades
- Account status and billing questions
- Payment issues

Always check the user's account status first. Be empathetic and clear.
If you can't resolve the issue, create a support ticket.""",
    tools=[lookup_faq, check_account_status, create_support_ticket],
    model="claude-sonnet-4-20250514",
)

technical_agent = Agent(
    name="Technical Support",
    instructions="""You are a technical support specialist. Help users with:
- API issues, rate limits, authentication
- Integration problems
- Bug reports

Look up relevant FAQs first. If the issue is complex, create a support ticket.""",
    tools=[lookup_faq, create_support_ticket],
    model="claude-sonnet-4-20250514",
)


# --- Define the triage agent (entry point) ----------------------------------

triage_agent = Agent(
    name="Support Triage",
    instructions="""You are a support triage agent. Your job is to understand the user's issue
and route them to the right specialist:

- For billing, payment, subscription, or account questions → hand off to Billing Specialist
- For technical issues, API problems, or bugs → hand off to Technical Support

Ask clarifying questions if needed before routing. Be friendly and efficient.""",
    handoffs=[
        handoff(billing_agent, tool_name_override="transfer_to_billing"),
        handoff(technical_agent, tool_name_override="transfer_to_technical"),
    ],
    model="claude-sonnet-4-20250514",
)


# --- Run the agent ----------------------------------------------------------

async def main():
    import asyncio

    print("=" * 60)
    print("OpenAI Agents SDK — Customer Support with Handoffs")
    print("=" * 60)

    # Scenario 1: Billing question
    print("\n--- Scenario 1: Billing Question ---\n")
    result = await Runner.run(
        triage_agent,
        input="Hi, I'm user_123 and I want to cancel my subscription. Can I get a refund?",
    )
    print(f"Final response: {result.final_output}")
    print(f"Agent that responded: {result.last_agent.name}")

    # Scenario 2: Technical question
    print("\n--- Scenario 2: Technical Question ---\n")
    result = await Runner.run(
        triage_agent,
        input="I'm having trouble with the API. I keep getting rate limited and my key doesn't work.",
    )
    print(f"Final response: {result.final_output}")
    print(f"Agent that responded: {result.last_agent.name}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
