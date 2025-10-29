import os

from pydantic import SecretStr

# Set few environment variables before starting the experiment
os.environ["DEBUG"]="true"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LOG_TO_FILE"] = "true"
os.environ["DEBUG_LLM"] = "true"


from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger
)

from openhands.sdk.tool import Tool, register_tool
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool


from openhands.sdk.context import (
    Skill
)

from openhands.tools.cat_on_steroids import CatOnSteroidsTool


# logger setup
logger = get_logger(__name__)

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
    log_completions=True,
    log_completions_folder="logs"
)


with open("examples/04_pst/reader_no_notebook.md","r") as f:
    reader_u_agent_instr = f.read()

with open("examples/04_pst/architect.md","r") as f:
    architect_u_agent_instr = f.read()


# Tools

workspace = "/home/pst/Documents/crazy_orca/agent_workspace/"
os.chdir(workspace)

cwd = os.getcwd()
print(f"Current working directory: {cwd}")

register_tool("BashTool", BashTool)
# register_tool("BrowserToolSet", BrowserToolSet)
register_tool("CatOnSteroidsTool", CatOnSteroidsTool)
register_tool("FileEditorTool", FileEditorTool)

tools = [
    Tool(name="CatOnSteroidsTool"),
    Tool(name="BashTool"),
    Tool(name="FileEditorTool"),
    # Tool(name="BrowserToolSet"),
]




agent_context = AgentContext(
    skills=[
        Skill(
            name="reader.md",
            content=reader_u_agent_instr,
            trigger=None,
        ),
    ],
    system_message_suffix="",
    user_message_suffix="",
)

# Agent
agent = Agent(llm=llm, tools=tools, agent_context=agent_context)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

# Sample 2
# conversation.send_message(
#     "What are the DMA capabilities available in STM32F405 and TM4C1230C3PM?" 
#     "Prepare a detailed comparison report between these two products and save it as 'comparison_report.md' in the workspace."
# )

# Sample 3
conversation.send_message(
    "What are the steps to configure DMA from ethernet controller to flash memory in STM32F405?" 
    "Prepare a detailed guidance document('DMA_Ethernet_STM32F405.md') in the workspace. Incrementally capture all your findings and observations during exploration in the document."
    "Make sure you refer the stm32f405 reference manual available in workspace."
)
conversation.run()


print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")
