import os
from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger
)
from openhands.sdk.context import (
    Skill
)
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.execute_bash import BashTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.cat_on_steroids import CatOnSteroidsTool

from openhands.sdk.context.condenser import LLMSummarizingCondenser


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
    log_completions_folder="logs"
)



# set workspace directory outside current project
workspace = "/home/pst/Documents/crazy_orca/architect_workspace/"
os.chdir(workspace)
with open("skills/component_selector_skill.md","r") as f:
    architect_skill = f.read()


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
            name="architect.md",
            content=architect_skill,
            trigger=None
        ),
    ],
    system_message_suffix="",
    user_message_suffix="",
)

condenser = LLMSummarizingCondenser(keep_first=6,kind="LLMSummarizingCondenser",max_size=30,llm=llm)  # No condenser for now
# Agent
agent = Agent(llm=llm, tools=tools, agent_context=agent_context,condenser=condenser)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

conversation.send_message(trs_agent_prompt)

conversation.run()


# print("=" * 100)
# print("Conversation finished. Got the following LLM messages:")
# for i, message in enumerate(llm_messages):
#     print(f"Message {i}: {str(message)[:200]}")
