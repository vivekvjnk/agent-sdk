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
import uuid
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
from openhands.sdk.context import Skill
from openhands.sdk.context.skills.trigger import (
    KeywordTrigger,
)
from openhands.sdk.tool import register_tool
from openhands.tools.cat_on_steroids import CatOnSteroidsTool
from openhands.tools.memory import PersistentMemoryTool
from openhands.tools.preset.default import get_default_tools


with open("examples/04_pst/architect_with_persistence.md","r") as f:
    architect_skill = f.read()


architect_u_agent_instr =  architect_skill

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
    max_output_tokens=16000,
    disable_vision=True,
)


cwd = os.getcwd()

from openhands.sdk.context.memory import PersistentMemoryManager
from pathlib import Path
memory_manager = PersistentMemoryManager(base_path=Path(cwd),file_name="persistent_memory.yaml")

# Setup tools
register_tool("CatOnSteroidsTool", CatOnSteroidsTool)
register_tool("PersistentMemoryTool", PersistentMemoryTool)

tools = get_default_tools(enable_browser=False)
tools.extend(
        [
            Tool(name="CatOnSteroidsTool"),
            Tool(name="PersistentMemoryTool",params={"memory_manager":memory_manager}),
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
    # agent_context=main_agent_context
)

conversation_id = uuid.uuid4()
persistence_dir = "./.conversations"

conversation = Conversation(
    agent=main_agent,
    workspace=cwd,
    name_for_visualization="Delegator",
    persistence_dir=persistence_dir,
    conversation_id=conversation_id,

)


task_message = (
"""
Build a 64 V, 2 kW stackable Li-ion Battery Management System (BMS).

As a hardware architect:
1. Perform a comprehensive technical study covering all aspects of such a system — 
   cell chemistry, series/parallel configuration, current rating, efficiency targets, 
   balancing topology, protection circuitry, sensing, communication, and thermal design.
2. Summarize your findings and decisions into a structured **Hardware Design Document (HDD)** 
   in Markdown format. Save it under the `workspace/` directory.
3. Organize any collected research data, datasheets, reference designs, and notes under 
   `workspace/research/docs/` (create the directory if it does not exist).
4. The final HDD should be detailed and scientifically structured so that 
   a beginner hardware engineer could reproduce the system from it.
5. Structure your workflow methodically:
   - Begin by identifying key design challenges and performance goals.
   - Gather supporting data, standards, and reference architectures.
   - Derive your own design choices through reasoning and justification.
   - Summarize your progress regularly so the entire design process remains traceable.

**Expected outcome:**
A well-organized workspace containing:
- A complete, self-contained `Hardware_Design_Document.md` describing the full BMS design.
- Supporting technical resources under `workspace/research/docs/`.
- A clear and logical flow of reasoning visible through the artifacts created during the process.

**Evaluation criteria (for internal validation):**
- Clarity and completeness of the hardware design description.
- Correctness of engineering reasoning and data consistency.
- Evidence of progressive refinement (each phase builds on the previous).
- Document structure suitable for continuation by another engineer.
"""
)
conversation.send_message(task_message)
conversation.run()