import os
import asyncio
from pydantic import SecretStr
from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    get_logger,
)

logger = get_logger(__name__)

# Configure LLM - using what's likely available in the environment or a placeholder
api_key = os.getenv("LLM_API_KEY", "dummy")
model = os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620")
base_url = os.getenv("LLM_BASE_URL")

llm = LLM(
    usage_id="experiment",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

mcp_config = {
    "mcpServers": {
        "experiment": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable-http"
        }
    }
}

agent = Agent(
    llm=llm,
    mcp_config=mcp_config,
)

conversation = Conversation(
    agent=agent,
    workspace=os.getcwd(),
)

print("Starting experiment...")
# We want to force two tool calls.
conversation.send_message("Please call the get_session_id tool. Then call it again. Tell me both session IDs you received.")
conversation.run()

print("Experiment finished.")
print(f"Total cost: {llm.metrics.accumulated_cost}")
