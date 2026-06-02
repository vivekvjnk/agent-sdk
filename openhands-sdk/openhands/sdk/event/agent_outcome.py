from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class TaskEscalatedEvent(Event):
    """Event indicating that the task was escalated."""

    source: SourceType = "agent"
    message: str = Field(description="Escalation message from the agent")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this event."""
        content = Text()
        content.append("Task Escalated:\n", style="bold yellow")
        content.append(self.message)
        return content

    def __str__(self) -> str:
        return f"TaskEscalatedEvent: {self.message}"
