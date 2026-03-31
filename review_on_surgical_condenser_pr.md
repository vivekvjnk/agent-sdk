# PR Description
## Summary

This PR introduces the LargeFileSurgicalCondenser, a specialized context management tool designed to mitigate context window bloat caused by large tool outputs (specifically images and large file reads from the file_editor tool).

Unlike standard windowing or summarization condensers, this "surgical" condenser follows a Post-Inference Cleanup strategy. It preserves raw, high-fidelity data (like base64 images) during the turn it is first viewed to ensure the agent has the necessary information for analysis. Crucially, it only replaces that data with a concise summary after the agent has responded, ensuring subsequent turns benefit from a lean context and increased KV cache efficiency without sacrificing the quality of the initial analysis.

## Key Features
- **Targeted Condensation**: Specifically monitors observations from the` file_editor` tool (configurable).
- **Post-Inference Cleanup**: Only triggers condensation when an observation is followed by an agent action/response, ensuring the data has already been "processed."
- **Smart Summaries**: Attempts to correlate observations with their preceding actions to include file paths in the condensation summary (e.g., `[Condensation] Viewed sample_image.png`).
- **Vision Optimization**: Automatically triggers for any ImageContent, which is a primary driver of context bloat in multi-modal workflows.
- **Configurable Thresholds**: Includes a default byte-size threshold (10KB) for `TextContent`.



## Changes
- Core: Added LargeFileSurgicalCondenser in `openhands/sdk/context/condenser/`.
- **Tests**: Comprehensive unit tests covering tool-name filtering, threshold logic, image detection, and sequence-based condensation triggers.
- **Example**: Added a new vision-agent example script in `examples/06_custom_examples/vision_agent/` demonstrating the condenser in a multi-turn conversation.

## Performance Impact
By surgically removing prefix-heavy data like base64 images after their first use, we significantly increase prefix-matching for KV caches in multi-turn interactions, reducing latency and token costs for long-running sessions.


## Files changed
examples/06_custom_examples/vision_agent/surgical_condenser.py
+176
Lines changed: 176 additions & 0 deletions
openhands-sdk/openhands/sdk/context/condenser/__init__.py
+4
Lines changed: 4 additions & 0 deletions
openhands-sdk/openhands/sdk/context/condenser/large_file_surgical_condenser.py
+133
Lines changed: 133 additions & 0 deletions
openhands-sdk/openhands/sdk/__init__.py
+4
Lines changed: 4 additions & 0 deletions
tests/sdk/context/condenser/test_large_file_surgical_condenser.py

## Commit messages 
1. Added surgical condenser
2. Merging with main branch
3. Added condensed file name(if any) in condensation summary
4. Fix linting and formatting for surgical condenser
5. Remove litellm from editable dependencies
6. Updated uv lock with litellm dependency


# Review comments
1. Action events should always be followed by an observation event
```openhands-sdk/openhands/sdk/context/condenser/large_file_surgical_condenser.py
                            )
                            summary = f"[Condensation]Viewed {file_info}"

                        return Condensation(
```
Collaborator
@csmith49
csmith49
last week
Unfortunately I don't think this works the way you might expect. If we see the pattern:
```
<prefix>
Action event: retrieve large file
Observation event: the large file
<suffix>
```
You probably want the resulting sequence to look like:
```
<prefix>
Action event: retrieve large file
Condensation: summary of large file
<suffix>
```

But LLM APIs expect every action event to have a matching observation event and will throw an exception if that isn't the case. We prevent these exceptions by filtering out unmatched actions and observations when constructing the View, so the actual resulting sequence is probably something like:

```
<prefix>
Condensation: summary of large file
<suffix>
```

2. Is hard condensation really required? 
```openhands-sdk/openhands/sdk/context/condenser/large_file_surgical_condenser.py
                    next_event = view.events[index + 1]
                    if next_event.source == "agent":
                        if self._should_condense_event(event):
                            return CondensationRequirement.HARD
```
Collaborator
@csmith49
csmith49
last week

A hard condensation requirement indicates that the agent cannot proceed without a condensation, and that if we can't apply one we need to take drastic measures (like a hard context reset). Not quite the case here: if we delay condensation the only harm is that we spend a little extra time/money committing to the wrong cache prefix.

3. Switch to CondenserBase class instead of RollingCondenser
```openhands-sdk/openhands/sdk/context/condenser/large_file_surgical_condenser.py
from openhands.sdk.llm import LLM, ImageContent, TextContent


class LargeFileSurgicalCondenser(RollingCondenser):
```

You probably want the CondenserBase class instead, the RollingCondenser interface is set up to make condensers with moving "attention windows" a little easier to mess with.

My recommendation is to tweak the LargeFileSurgicalCondenser to use the CondenserBase interface, which will simplify things:
```
class LargeFileSurgicalCondenser(CondenserBase):
    def condense(
        self,
        view: View,
        agent_llm: LLM | None = None
    ) -> View | Condensation:
        # The view is a list of events we want to show the LLM
        # Instead of trying to replace existing events with a
        # Condensation, just do the surgery on the events
        # directly while constructing a new view
```
No need for condensation_requirement or get_condensation, you can just always return a modified View to get the behavior you want. Doing surgery directly on the events in the view will help prevent the action/observation mismatch I noted.