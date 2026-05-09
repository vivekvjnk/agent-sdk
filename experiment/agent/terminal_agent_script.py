import os
from pydantic import SecretStr
from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
)
from openhands.sdk.event import MessageEvent

def get_last_agent_message(conversation):
    for event in reversed(conversation.state.events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return event.llm_message.content
    return "No response from agent."

def run_experiment():
    # 0. Setup LLM
    api_key = os.getenv("LLM_API_KEY", "dummy")
    model = os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620")
    base_url = os.getenv("LLM_BASE_URL")

    llm = LLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

    # 1. Setup MCP Server Configuration
    mcp_config = {
        "mcpServers": {
            "terminal-session-server": {
                "url": "http://localhost:8801/mcp",
                "transport": "streamable-http"
            }
        }
    }

    # 2. Initialize Agent
    agent = Agent(
        agent_context=AgentContext(
            system_message_suffix="You are an agent testing terminal session persistence. Execute commands as requested."
            ),
        llm=llm,
        mcp_config=mcp_config
    )

    # 3. Create Conversation
    conversation = Conversation(agent)

    print("\n--- Step 1: Set environment variable in terminal ---")
    conversation.send_message("Run command 'export SESSION_TEST_VAR=OpenHands-Is-Awesome' and then 'echo $SESSION_TEST_VAR' to verify it is set.")
    conversation.run()
    print(f"Agent Response 1:\n{get_last_agent_message(conversation)}")

    print("\n--- Step 2: Verify environment variable persists in same conversation ---")
    conversation.send_message("In the same terminal session, run 'echo $SESSION_TEST_VAR' again. It should still be 'OpenHands-Is-Awesome'.")
    conversation.run()
    print(f"Agent Response 2:\n{get_last_agent_message(conversation)}")

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
