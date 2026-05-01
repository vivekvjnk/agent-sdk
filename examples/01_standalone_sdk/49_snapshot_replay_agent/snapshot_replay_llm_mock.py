from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.agent.replay_llm import ReplayLLM
from openhands.sdk.agent.replay_callback import ObservationDriftCallback
from openhands.sdk.conversation.snapshot import SnapshotLoader
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

import os

# Get the location of this python file
script_dir = str(os.path.dirname(os.path.abspath(__file__)))

# 1. Setup Replay LLM
llm = ReplayLLM.from_persistence(f"{script_dir}/hello_world_recorded")

# 2. Setup Drift Monitoring
events = SnapshotLoader.load_events(f"{script_dir}/hello_world_recorded")
expected_obs = SnapshotLoader.extract_expected_observations(events)
drift_callback = ObservationDriftCallback(expected_obs, f"{script_dir}/drift.jsonl")

# 3. Run Standard Agent
agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
)

conversation = Conversation(
    agent=agent,
    workspace=os.getcwd(),
    persistence_dir=f"{script_dir}/hello_world_recorded",
    callbacks=[drift_callback]
)


conv = Conversation(agent=agent, callbacks=[drift_callback])
conv.send_message("Write 3 facts about the current project into FACTS.txt.")
conv.run()
print(f"Recording complete. Files created in {script_dir}/hello_world_recorded/\n")