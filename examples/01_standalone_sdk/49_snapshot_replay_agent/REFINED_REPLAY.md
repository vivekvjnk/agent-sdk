# Snapshot-Based Agent Replay Mechanism

This document outlines the architecture and approach for replaying recorded agent sessions within the OpenHands SDK. The mechanism allows for deterministic regression testing by re-injecting recorded LLM responses into a live environment while monitoring for "observation drift."

## Background

Traditional regression testing for LLM-based agents is challenging because:
1.  **Non-Determinism**: LLMs can return different responses for the same input.
2.  **Cost & Latency**: Running full agent loops against live LLMs is expensive and slow.
3.  **Environment Stability**: Tool outputs (observations) may change even if the agent's actions are identical.

The **Snapshot Replay Mechanism** solves these issues by decoupling the LLM from the agent's execution flow and providing a side-channel for observation validation.

---

## Architecture Overview

The refined replay approach uses a compositional pattern rather than a specialized agent subclass. This ensures that the production `Agent` class is tested in its exact production configuration.

### Core Components

#### 1. `SnapshotLoader`
A utility class (`openhands.sdk.conversation.snapshot`) that handles the discovery and extraction of data from persistence directories.
- **Discovery**: Automatically resolves conversation subdirectories from a base path.
- **Message Extraction**: Converts recorded `ActionEvent`s and `MessageEvent`s into LLM `Message` objects.
- **Observation Mapping**: Maps `tool_call_id`s to their recorded `ObservationEvent`s for ground-truth comparison.

#### 2. `ReplayLLM`
A specialized mock LLM (`openhands.sdk.agent.replay_llm`) that inherits from `TestLLM`.
- **Factory Method**: `from_persistence()` hydrates the LLM with the sequence of responses extracted by the `SnapshotLoader`.
- **Fidelity**: Bypasses the need for complex `@patch` decorators by providing a real `LLM` subclass that "plays back" a script.

#### 3. `ObservationDriftCallback`
A decoupled interceptor (`openhands.sdk.agent.replay_callback`) that monitors the agent's interaction with the environment.
- **Event Interception**: Listens for `ObservationEvent`s emitted by the conversation.
- **Drift Detection**: Compares the live observation against the recorded ground truth, ignoring volatile metadata (IDs, timestamps).
- **Structured Logging**: Appends drift reports to a JSONL file for automated analysis.

---

## The Replay Workflow

To replay a session, follow these steps:

1.  **Initialize `ReplayLLM`**: Point it to the recorded persistence directory.
2.  **Setup Drift Monitoring**: Load expected observations and initialize the callback.
3.  **Execute Agent**: Create a standard `Agent` with the `ReplayLLM` and run it via a `Conversation`.

### Example Implementation

```python
from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.agent.replay_llm import ReplayLLM
from openhands.sdk.agent.replay_callback import ObservationDriftCallback
from openhands.sdk.conversation.snapshot import SnapshotLoader

# 1. Setup Replay LLM from a recorded session
llm = ReplayLLM.from_persistence("path/to/recorded_session")

# 2. Setup Drift Monitoring
events = SnapshotLoader.load_events("path/to/recorded_session")
expected_obs = SnapshotLoader.extract_expected_observations(events)
drift_callback = ObservationDriftCallback(expected_obs, "drift.jsonl")

# 3. Run Standard Agent using the mock LLM and Callback
agent = Agent(llm=llm, tools=[...])
conversation = Conversation(
    agent=agent,
    callbacks=[drift_callback]
)

# Trigger the replay by sending the original starting message
conversation.send_message("Original User Message")
conversation.run()
```

---

## Benefits of the Compositional Approach

-   **High Fidelity**: Tests the actual `Agent.step()` and `_ActionBatch` logic used in production.
-   **Zero Maintenance**: No need to update a specialized `SnapshotReplayAgent` whenever the core `Agent` logic changes.
-   **Observability**: Provides clear, structured logs of exactly where and how a live run diverged from the recording.
-   **Scalability**: Easily integrated into CI pipelines by pointing the `ReplayLLM` to a library of golden snapshots.
