"""Example demonstrating the EscalateTool functionality.

The EscalateTool is a built-in tool that allows an agent to request external
intervention and temporarily suspend autonomous execution. Calling `escalate`
transitions the conversation status to `PAUSED` and triggers a
`TaskEscalatedEvent`.

This example shows how an agent handles a situation requiring a password,
escalates to the user, gets the input, and then successfully resumes and
completes the task.
"""

import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    ConversationExecutionStatus,
)
from openhands.sdk.event import TaskEscalatedEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool


# --- Configuration & Setup ---

# Check for API key to decide whether to use real LLM or the TestLLM simulator
api_key = os.getenv("LLM_API_KEY")

if not api_key:
    print("=" * 80)
    print("Running in SIMULATOR mode (TestLLM) because LLM_API_KEY is not set.")
    print("To run with a real LLM, set LLM_API_KEY in your environment.")
    print("=" * 80)
    print()

    # Scripted responses for TestLLM to simulate the escalation flow:
    # 1. First assistant response: decides it needs DB password and calls 'escalate'
    # 2. Second assistant response: after user input, completes and calls 'finish'
    scripted_responses: list[Message | Exception] = [
        Message(
            role="assistant",
            content=[
                TextContent(
                    text=(
                        "I am checking the environment and database settings for "
                        "deployment. It seems the DB_PASSWORD is not configured. "
                        "I cannot complete the deployment without it, so I will "
                        "escalate this to the user."
                    )
                )
            ],
            tool_calls=[
                MessageToolCall(
                    id="call_escalate",
                    name="escalate",
                    arguments=(
                        '{"message": "Please provide the DB_PASSWORD to proceed '
                        'with production deployment."}'
                    ),
                    origin="completion",
                )
            ],
        ),
        Message(
            role="assistant",
            content=[
                TextContent(
                    text=(
                        "Thank you for providing the database password. I have "
                        "configured it and successfully deployed the web app "
                        "to production."
                    )
                )
            ],
            tool_calls=[
                MessageToolCall(
                    id="call_finish",
                    name="finish",
                    arguments='{"message": "Web application deployed successfully!"}',
                    origin="completion",
                )
            ],
        ),
    ]

    llm = TestLLM.from_messages(scripted_responses, model="mock-model")

else:
    # Real LLM configuration
    model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    base_url = os.getenv("LLM_BASE_URL")
    llm = LLM(
        usage_id="escalate-demo",
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

# EscalateTool and FinishTool are built-in and enabled by default.
# We will expose TerminalTool to the agent so it has standard tools.
tools = [Tool(name=TerminalTool.name)]
agent = Agent(llm=llm, tools=tools)

# List to keep track of captured escalation messages
escalations = []


def event_callback(event):
    """Callback to capture TaskEscalatedEvent during conversation run."""
    if isinstance(event, TaskEscalatedEvent):
        print("\n🔔 [Callback] Captured Escalation Event:")
        print(f"   Message: {event.message}")
        escalations.append(event.message)


# Initialize conversation
conversation = Conversation(
    agent=agent, callbacks=[event_callback], workspace=os.getcwd()
)

# =========================================================================
# Phase 1: Start Task and Trigger Escalation
# =========================================================================
print("Phase 1: Starting agent deployment task...")
print(f"Initial execution status: {conversation.state.execution_status}")

conversation.send_message("Deploy the web application to production.")

# Run the conversation. When the agent calls the escalate tool,
# conversation.run() will suspend and return.
conversation.run()

print()
print("Phase 1 Complete:")
print(f"- Current execution status: {conversation.state.execution_status}")
print(f"- Captured escalations: {escalations}")

# Verify that the status is PAUSED
assert conversation.state.execution_status == ConversationExecutionStatus.PAUSED, (
    "Conversation should be in PAUSED state."
)

# =========================================================================
# Phase 2: Handle Escalation and Resume Conversation
# =========================================================================
print("\n" + "=" * 80)
print("Phase 2: Resolving the escalation and resuming execution")
print("=" * 80)

# Simulate the user/caller providing the missing input
escalation_message = escalations[-1] if escalations else "missing info"
print(f"Caller received request: '{escalation_message}'")
print("Providing DB_PASSWORD...")

# Send a new message answering the escalation.
# This message is appended to the conversation history.
conversation.send_message("Here is the database password: super_secret_prod_pass_2026")

# Run the conversation again. The status will transition from PAUSED
# to RUNNING, and execution resumes from where it left off.
print("\nResuming agent run...")
conversation.run()

print()
print("Phase 2 Complete:")
print(f"- Final execution status: {conversation.state.execution_status}")

# Verify that the status is now FINISHED
assert conversation.state.execution_status == ConversationExecutionStatus.FINISHED, (
    "Conversation should be in FINISHED state."
)

# Report cost (required for all examples)
cost = llm.metrics.accumulated_cost
print(f"\nEXAMPLE_COST: {cost}")
