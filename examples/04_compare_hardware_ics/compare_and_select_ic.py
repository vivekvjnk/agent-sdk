import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
from openhands.sdk.context import Skill
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.cat_on_steroids import CatOnSteroidsTool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool


# logger setup
logger = get_logger(__name__)

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
    log_completions=True,
    log_completions_folder="logs",
)


# set workspace directory outside current project
workspace = "../workspace/"
os.chdir(workspace)
with open("skills/component_selector_skill.md") as f:
    component_selector_skill = f.read()


cwd = os.getcwd()
print(f"Current working directory: {cwd}")

register_tool("BashTool", BashTool)
register_tool("CatOnSteroidsTool", CatOnSteroidsTool)
register_tool("FileEditorTool", FileEditorTool)

tools = [
    Tool(name="CatOnSteroidsTool"),
    Tool(name="BashTool"),
    Tool(name="FileEditorTool"),
]


agent_context = AgentContext(
    skills=[
        Skill(
            name="component_selector", content=component_selector_skill, trigger=None
        ),
    ],
    system_message_suffix="",
    user_message_suffix="",
)

condenser = LLMSummarizingCondenser(
    keep_first=6, kind="LLMSummarizingCondenser", max_size=30, llm=llm
)  # No condenser for now
# Agent
agent = Agent(llm=llm, tools=tools, agent_context=agent_context, condenser=condenser)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)


bms_ic_selection_prompt = """We are building a stackable multi-purpose BMS system of 64V and 2kW spec per each module. Primary objective of this design is to achieve High Voltage BMS capabilities by stacking multiple modules of the above given spec. At the same time these modules should work very well at any multiples of its stacking. ie if we stack 2 of them, we should get 128v system, 3 for 172v system etc. We should also be able to build low voltage solutions form the same architecture without major design changes.
Essentially we are trying to build a platform for BMS solutions.
From this perspective do a detailed comparison between each one of the provided BMS ICs(datasheets are saved in the workspace in PDF format). We are looking for the baseline BMS ic for our BMS platform.
"""
conversation.send_message(bms_ic_selection_prompt)

conversation.run()


# print("=" * 100)
# print("Conversation finished. Got the following LLM messages:")
# for i, message in enumerate(llm_messages):
#     print(f"Message {i}: {str(message)[:200]}")
