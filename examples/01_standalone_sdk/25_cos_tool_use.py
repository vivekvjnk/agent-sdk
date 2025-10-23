import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
    setup_logging
)
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool

from openhands.tools.cat_on_steroids import CatOnSteroidsTool

# logger setup
logger = get_logger(__name__)

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    service_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

# Tools
cwd = os.getcwd()
register_tool("BashTool", BashTool)
register_tool("BrowserToolSet", BrowserToolSet)
register_tool("CatOnSteroidsTool", CatOnSteroidsTool)

tools = [
    Tool(name="CatOnSteroidsTool"),
    Tool(name="BashTool"),
    Tool(name="BrowserToolSet"),
]


# Agent
agent = Agent(llm=llm, tools=tools)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

conversation.send_message(
    "How many center aligned PWM timers are available in STM32F405?"
)
conversation.run()


print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")
