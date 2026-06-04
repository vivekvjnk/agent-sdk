# OpenHands SDK: FINISH Tool & Pause Functionality Analysis

This document provides a technical deep-dive into how the OpenHands SDK handles agent termination (via the `FINISH` tool) and execution suspension (via the `Pause` functionality).

---

## 1. FINISH Tool Mechanism

The `FINISH` tool is a specialized builtin tool that signals the successful completion or terminal failure of a task. Unlike other tools that return data for the agent to continue processing, the `FINISH` tool triggers a transition to a terminal state in the Agent Loop.

### Core Components
- **Implementation Path**: `openhands-sdk/openhands/sdk/tool/builtins/finish.py`
- **Action**: `FinishAction` contains a `message` field (the final report to the user).
- **Observation**: `FinishObservation` is a simple acknowledgement that the finish command was received.
- **Executor**: `FinishExecutor` simply wraps the action message into an observation.

### How it Stops the Agent Loop
The termination logic is orchestrated within `openhands-sdk/openhands/sdk/agent/agent.py` through the `_ActionBatch` class and the `Agent.step()` method.

1.  **Truncation**: In `_ActionBatch._truncate_at_finish`, the SDK scans the list of tool calls produced by the LLM. If a `FinishTool` call is detected, all subsequent tool calls in that same batch are discarded. This ensures the agent does not attempt further actions after signaling completion.
2.  **State Transition**: When the `_ActionBatch` is finalized in `_ActionBatch.finalize`, it checks if `has_finish` is true.
3.  **Terminal State**: If the agent is finishing (and no iterative refinement is required), it calls the `mark_finished` callback. This callback sets the `conversation.state.execution_status` to `ConversationExecutionStatus.FINISHED`.
4.  **Loop Termination**: The `LocalConversation.run()` loop in `local_conversation.py` monitors the execution status. Once it transitions to `FINISHED`, the `while` loop terminates, and control is returned to the user.

---

## 2. Pause Functionality

The `Pause` functionality allows a user or a hook to suspend agent execution without losing the current conversation state. This is useful for manual intervention, resource management, or human-in-the-loop workflows.

### Implementation Details
- **Status Trigger**: Execution is paused by setting `conversation.state.execution_status = ConversationExecutionStatus.PAUSED`.
- **Loop Interruption**: The `LocalConversation.run()` method (located in `local_conversation.py`) checks the status at the start of every iteration:
  ```python
  while self.state.execution_status == ConversationExecutionStatus.RUNNING:
      # ... agent step logic ...
  ```
- **Resumption**: To resume, the status must be set back to `RUNNING`, and `conversation.run()` must be called again. The conversation state (events, history, workspace changes) remains intact because it is stored in the `ConversationState` object.

---

## 3. Interaction with Confirmation Logic

The `FINISH` tool has a privileged status within the agent's security and confirmation framework (`Agent._requires_user_confirmation` in `agent.py`):

- **Auto-Approval**: A single `FinishAction` (or `ThinkAction`) is explicitly exempt from user confirmation, even if confirmation mode is enabled for the conversation. 
- **Rationale**: Since finishing is an intent to *stop* acting on the environment, it is considered inherently safe compared to destructive actions like file deletions or code execution.

---

## Summary of Findings

| Feature | Trigger | Implementation Mechanism | State Result |
| :--- | :--- | :--- | :--- |
| **FINISH Tool** | LLM calls `finish` tool | `_ActionBatch` truncation + `mark_finished` callback | `FINISHED` (Terminal) |
| **Pause** | Manual status change | `while` loop condition check in `Conversation.run()` | `PAUSED` (Recoverable) |
| **Confirmation** | `ConfirmationPolicy` | `_requires_user_confirmation` check before execution | `WAITING_FOR_CONFIRMATION` |
