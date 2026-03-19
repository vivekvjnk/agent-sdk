import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

mcp_config = {
                "mcpServers": {
                    "VHL-Library-Terminal": {
                        "url": "http://localhost:8082/sse",
                    }
                }
            }

# Tools
cwd = os.getcwd()
tools = []

# Agent
agent = Agent(llm=llm, mcp_config=mcp_config)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

conversation.send_message(
    "tsci cli tool is available in the VHL-Library-Terminal. Use the tool to import BQ79600 component. Always use `tsci search <component_name>` to find a component."
    "To import a JLCPCB component, use `tsci import <component_name>"
    "To import component from any other library or author, use `tsci add <author>/<component_name>"
    "Note: avoid JLCPCB components as much as possible"
)
conversation.run()

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")
