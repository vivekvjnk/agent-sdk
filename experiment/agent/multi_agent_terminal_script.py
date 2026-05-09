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

    # 2. Initialize Agents
    agent1 = Agent(
        agent_context=AgentContext(
            system_message_suffix="You are Agent 1. You have a dedicated terminal session."
            ),
        llm=llm,
        mcp_config=mcp_config
    )
    
    agent2 = Agent(
        agent_context=AgentContext(
            system_message_suffix="You are Agent 2. You have a dedicated terminal session."
            ),
        llm=llm,
        mcp_config=mcp_config
    )

    # 3. Create Conversations
    conv1 = Conversation(agent1)
    conv2 = Conversation(agent2)

    print("\n--- Step 1: Agent 1 sets AGENT_1_VAR ---")
    conv1.send_message("Run command 'export AGENT_1_VAR=Value-1' and verify it with echo.")
    conv1.run()
    print(f"Agent 1 Response:\n{get_last_agent_message(conv1)}")

    print("\n--- Step 2: Agent 2 sets AGENT_2_VAR ---")
    conv2.send_message("Run command 'export AGENT_2_VAR=Value-2' and verify it with echo.")
    conv2.run()
    print(f"Agent 2 Response:\n{get_last_agent_message(conv2)}")

    print("\n--- Step 3: Agent 1 verifies its own session and checks isolation ---")
    conv1.send_message("Verify if AGENT_1_VAR is still 'Value-1'. Also check if AGENT_2_VAR is visible (it should NOT be).")
    conv1.run()
    print(f"Agent 1 Response:\n{get_last_agent_message(conv1)}")

    print("\n--- Step 4: Agent 2 verifies its own session and checks isolation ---")
    conv2.send_message("Verify if AGENT_2_VAR is still 'Value-2'. Also check if AGENT_1_VAR is visible (it should NOT be).")
    conv2.run()
    print(f"Agent 2 Response:\n{get_last_agent_message(conv2)}")

    print("\n--- Experiment Finished ---")

if __name__ == "__main__":
    run_experiment()
