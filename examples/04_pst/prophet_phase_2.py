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
from pydantic import SecretStr

from openhands.sdk import (
    get_logger,
    LLM,
    Agent,
    AgentContext,
    Conversation,
    Tool,
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

# Load skills 
with open("examples/04_pst/persistent_memory_skill.md","r") as f:
    persistent_memory_skill = f.read()

with open("examples/04_pst/architect_with_persistence.md","r") as f:
    architect_skill = f.read()

with open("examples/04_pst/reader_with_persistence.md","r") as f:
    reader_skill = f.read()

architect_u_agent_instr = persistent_memory_skill + "\n\n"+ architect_skill
reader_u_agent_instr =  persistent_memory_skill + "\n\n" + reader_skill

# Content of the "persistent_memory" section in persistent memory skill should be filled by loading the information from the persistent memory file. Assumption is that each sub agent will be allocated with a dedicated workspace under the delegation agent workspace. Inside the workspace of each agent, persistent memory file will be stored.
# Hence simply passing persistent memory file from the workspace will not serve the purpose. Implementation has to be done from inside Agent SDK. We will pass personas from here. Agent memory needs to be integrated deep inside. Look at the microagent/skill implementation. It may have feature to read and load instructions during runtime.
# Call .format() method over the agent instructions string to fill the "persistent_memory" section from where ever we fill the content

workspace = "/home/pst/Documents/crazy_orca/architect_workspace/"
os.chdir(workspace)

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
            name="Principal Architect",
            content=architect_u_agent_instr,
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
"""
Build 64v, 2kw stackable li ion BMS.
As a hardware architect do a thorough research, then convert the above requirement into a hardware design document. Collect all the necessary data. With this HDD, a beginner hardware engineer should be able to build the system.

Make sure you create an HDD document in markdown format with collected information under workspace directory, prepare proper folder structure if required, and organize information such that further development is structured and and scientific.

Every artifact you collect during research should be stored in the workspace under research/docs directory(make it if not present)
"""
)
conversation.send_message(task_message)
conversation.run()

# conversation.send_message(
#     "Ask the lodging sub-agent what it thinks about Covent Garden."
# )
# conversation.run()

