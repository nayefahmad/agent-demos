"""
Example 1: Basic Multi-Turn Chat
=================================
The foundation of all agent systems — maintaining a conversation with an LLM
across multiple turns while preserving context.

Key concepts:
- Message roles (user, assistant, system)
- Conversation history management
- System prompts to set agent behavior
- Streaming responses

Run: python chat.py
"""

import anthropic


def chat():
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    conversation_history: list[dict] = []

    system_prompt = (
        "You are a helpful coding assistant. You give concise, accurate answers. "
        "When showing code, always include brief comments explaining key lines."
    )

    print("Multi-turn chat with Claude. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # Append the user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Send the full conversation history to maintain context
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history,
        )

        assistant_message = response.content[0].text

        # Append assistant response to history for next turn
        conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        print(f"\nClaude: {assistant_message}\n")


def chat_streaming():
    """Same as above but with streaming — tokens appear as they're generated."""
    client = anthropic.Anthropic()
    conversation_history: list[dict] = []

    system_prompt = "You are a helpful coding assistant."

    print("Streaming chat with Claude. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        # Use streaming to get tokens incrementally
        print("\nClaude: ", end="", flush=True)
        full_response = ""

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print("\n")
        conversation_history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    import sys

    if "--stream" in sys.argv:
        chat_streaming()
    else:
        chat()
