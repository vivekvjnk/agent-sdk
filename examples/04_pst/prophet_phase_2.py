"""
Multi-agent system for end to end embedded system development.

Phase 2 Agents: 
1. Architect
2. Researcher

Phase 2 target:
Top level HDD/SDD document for the given product/system development task

Implementation plan:
1. Start from Agent delegation example from Agent SDK
2. Integrate Architect agent in delegation example
3. Prepare stub for research agent
4. Validate if delegation is working as expected 
5. Start integration of Researcher agent
    - Combine researcher and reader into one agent
    - Validate functionality of combined researcher agent
    - Integrate COS tool int pst_phase_2 branch
6. Integrate all agents and test


Agent Delegation Example
This example demonstrates the agent delegation feature where a main agent
delegates tasks to sub-agents for parallel processing.
Each sub-agent runs independently and returns its results to the main agent,
which then merges both analyses into a single consolidated report.
"""

import os
# Set few environment variables before starting the experiment
os.environ["DEBUG"]="true"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["LOG_TO_FILE"] = "true"
os.environ["DEBUG_LLM"] = "true"

workspace = "/home/pst/Documents/crazy_orca/architect_workspace/"
os.chdir(workspace)


from pydantic import SecretStr


from openhands.sdk import (
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Tool,
    get_logger,
)
from openhands.sdk.tool import register_tool
from openhands.tools.delegate import DelegateTool
from openhands.tools.cat_on_steroids import CatOnSteroidsTool

from openhands.sdk.context.skills.trigger import (
    KeywordTrigger,
    TaskTrigger,
)

from openhands.sdk.context import (
    Skill
)

from openhands.tools.preset.default import get_default_tools


logger = get_logger(__name__)

# Configure LLM and agent
# You can get an API key from https://app.all-hands.dev/settings/api-keys
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "openhands/claude-sonnet-4-5-20250929")
llm = LLM(
    model=model,
    api_key=SecretStr(api_key),
    base_url=os.environ.get("LLM_BASE_URL", None),
    usage_id="agent",
)

# Load skills 
with open("examples/04_pst/reader_no_notebook.md","r") as f:
    reader_u_agent_instr = f.read()

with open("examples/04_pst/architect.md","r") as f:
    architect_u_agent_instr = f.read()


# Set custom work directory 
workspace = "/home/pst/Documents/crazy_orca/agent_workspace/"
os.chdir(workspace)

cwd = os.getcwd()


reader_agent_context = AgentContext(
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


# Setup tools
register_tool("DelegateTool", DelegateTool)
register_tool("CatOnSteroidsTool", CatOnSteroidsTool)
tools = get_default_tools(enable_browser=False)
tools.extend(
        [
            Tool(name="DelegateTool",params={"agent_context":reader_agent_context}),
            Tool(name="CatOnSteroidsTool")
        ]
    )

main_agent_context = AgentContext(
    skills=[
        Skill(
            name="architect.md",
            content=reader_u_agent_instr,
            trigger=KeywordTrigger(
                keywords=["architect"]),
        ),
    ],
    system_message_suffix="",
    user_message_suffix="",
)

main_agent = Agent(
    llm=llm,
    tools=tools,
    agent_context=main_agent_context
)


conversation = Conversation(
    agent=main_agent,
    workspace=cwd,
    name_for_visualization="Delegator",
)


task_message = (
    "Forget about coding. Let's switch to travel planning. "
    "Let's plan a trip to London. I have two issues I need to solve: "
    "Lodging: what are the best areas to stay at while keeping budget in mind? "
    "Activities: what are the top 5 must-see attractions and hidden gems? "
    "Please use the delegation tools to handle these two tasks in parallel. "
    "Make sure the sub-agents use their own knowledge "
    "and dont rely on internet access. "
    "They should keep it short. After getting the results, merge both analyses "
    "into a single consolidated report.\n\n"
)
conversation.send_message(task_message)
conversation.run()

conversation.send_message(
    "Ask the lodging sub-agent what it thinks about Covent Garden."
)
conversation.run()

