from typing import Any

from pydantic import Field

from openhands.sdk.context.condenser.base import (
    CondensationRequirement,
    NoCondensationAvailableException,
    RollingCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import ObservationEvent
from openhands.sdk.llm import LLM, ImageContent, TextContent


class LargeFileSurgicalCondenser(RollingCondenser):
    """A highly specific condenser that removes bloated data from observations.

    This condenser intercepts observations (like ImageContent or large TextContent from
    the 'file_editor' tool) and replaces them with a small summary string.
    This preserves KV cache efficiency by keeping the prefix stable and forcing early
    eviction of large branches.
    param target_tool: Only condense observations from this tool (default: 'file_editor').
    param threshold_bytes: For TextContent, the byte size threshold to trigger condensation (default: 10KB).
    """

    threshold_bytes: int = Field(default=10240)  # 10KB default threshold for condensation
    target_tool: str = Field(default="file_editor")

    def _should_condense_event(self, event: ObservationEvent) -> bool:
        if event.tool_name != self.target_tool:
            return False

        current_bytes = 0
        # TODO - Observation.content is a list of TextContent/ImageContent
        # Some agents are capable of generating multiple tool calls to the FileEditorTool in one turn.
        # Hence it's possible for an ObservationEvent to contain one or more bloated content blocks.
        # our current logic simply condenses the entire ObservationEvent if first bloated content block is found, but we may want to consider a more surgical approach in the future where we only condense the bloated content block(s) and preserve the rest of the observation content if it's not bloated.
        for content in event.observation.content:
            # If ImageContent, directly trigger condensation. No need to calculate size as images are always bloated for our purposes.
            if isinstance(content, ImageContent):
                return True
            # If TextContent, check if it exceeds the byte threshold.
            elif isinstance(content, TextContent):
                current_bytes += len(content.text.encode("utf-8"))
                if current_bytes > self.threshold_bytes:
                    return True
        return False

    def condensation_requirement(
        self, view: View, agent_llm: LLM | None = None
    ) -> CondensationRequirement | None:
        """Determines if a surgical condensation is required.

        We trigger a HARD condensation if we encounter a bloated target observation
        that is *not* the last event in the view. We want to ensure that the agent
        has had at least one turn to analyze the data (inference) before we rip it out.
        """
        for index, event in enumerate(view.events):
            if isinstance(event, ObservationEvent):
                # Only operate on events that are followed by another event 
                # (which implies the agent has seen and responded to it)
                if index < len(view.events) - 1:
                    # Look ahead to see if the next event is an action from the agent
                    next_event = view.events[index + 1]
                    if next_event.source == "agent":
                        if self._should_condense_event(event):
                            return CondensationRequirement.HARD
        return None

    def get_condensation(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation:
        """Returns a Condensation object exactly matching the bloated event."""
        for index, event in enumerate(view.events):
            if isinstance(event, ObservationEvent) and index < len(view.events) - 1:
                next_event = view.events[index + 1]
                if next_event.source == "agent":
                    if self._should_condense_event(event):
                        # Determine size for the summary message
                        size_kb = 0.0
                        has_image = False
                        for content in event.observation.content:
                            if isinstance(content, ImageContent):
                                has_image = True
                            elif isinstance(content, TextContent):
                                size_kb += len(content.text.encode("utf-8")) / 1024.0

                        file_info = "image/data" if has_image else f"data ({size_kb:.2f}KB)"
                        
                        summary = f"[Surgical Condensation] Viewed {file_info}"

                        return Condensation(
                            forgotten_event_ids=[event.id],
                            summary=summary,
                            summary_offset=index,
                            llm_response_id="surgical-condenser",
                            source="environment",
                        )

        raise NoCondensationAvailableException(
            "Expected to find a surgical condensation target, but none was valid."
        )

    def hard_context_reset(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation | None:
        """This condenser only performs surgical replacements."""
        return None
