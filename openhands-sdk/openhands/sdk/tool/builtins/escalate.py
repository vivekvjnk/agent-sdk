from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from pydantic import Field
from rich.text import Text

from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


class EscalateAction(Action):
    """
    Request external intervention and
    temporarily suspend autonomous execution.
    """

    message: str = Field(
        description=(
            "Escalation message to surface "
            "to the caller."
        )
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action."""
        content = Text()
        content.append("Escalate with message:\n", style="bold red")
        content.append(self.message)
        return content


class EscalateObservation(Observation):
    """
    Observation returned after an escalation.
    """

    @property
    def visualize(self) -> Text:
        """Return an empty Text representation since the message is in the action."""
        return Text()


TOOL_DESCRIPTION = """Signals that autonomous execution cannot continue
without external intervention.

Use this tool when:
- Additional information is required
- Approval is required
- A resource is unavailable
- Another system must act before execution can continue
- Human input is needed
- Any external dependency blocks progress

This tool does NOT indicate task completion.

Calling Escalate temporarily suspends autonomous execution
and surfaces the provided message to the caller.

The caller determines how the escalation is handled.
"""


class EscalateExecutor(ToolExecutor):
    def __call__(
        self,
        action: EscalateAction,
        conversation: "BaseConversation | None" = None,  # noqa: ARG002
    ) -> EscalateObservation:
        return EscalateObservation.from_text(text=action.message)


class EscalateTool(ToolDefinition[EscalateAction, EscalateObservation]):
    """Tool for requesting external intervention and suspending execution."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,  # noqa: ARG003
        **params,
    ) -> Sequence[Self]:
        """Create EscalateTool instance.

        Args:
            conv_state: Optional conversation state (not used by EscalateTool).
            **params: Additional parameters (none supported).

        Returns:
            A sequence containing a single EscalateTool instance.

        Raises:
            ValueError: If any parameters are provided.
        """
        if params:
            raise ValueError("EscalateTool doesn't accept parameters")
        return [
            cls(
                action_type=EscalateAction,
                observation_type=EscalateObservation,
                description=TOOL_DESCRIPTION,
                executor=EscalateExecutor(),
                annotations=ToolAnnotations(
                    title="escalate",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]
