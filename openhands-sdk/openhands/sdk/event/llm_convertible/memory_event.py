from typing import Optional

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import SourceType
from openhands.sdk.llm import Message, content_to_str


class MemoryEvent(LLMConvertibleEvent):
    """An immutable event representing a snapshot of the persistent memory.

    This event is created by reading the content from the PersistentMemoryManager
    and is guaranteed to be immutable once created.
    """

    # model_config: ClassVar[ConfigDict] = ConfigDict(
    #     extra='forbid',
    #     frozen=True,  # Enforce immutability
    # )

    source: SourceType = Field(
        default='memory', description="The source of the event, defaults to 'memory'"
    )
    
    # The immutable snapshot of the memory file content
    content: Message = Field(
        ..., description='The complete text content from the memory file wrapped in a Message object'
    )
    
    # Metadata for visualization/context
    memory_file_name: Optional[str] = Field(
        default=None, description="The name of the memory file for context/visualization"
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of the memory content."""
        # This implementation requires Rich library, which is not available in standard execution.
        # We will return a placeholder for testing.
        text_content = content_to_str(self.content.content)[0] if self.content.content else ''
        return Text(f"Snapshot of {self.memory_file_name}: {text_content[:50]}...")

    def to_llm_message(self) -> Message:
        """Returns the memory content as an LLM Message (its own content)."""
        return self.content

    def __str__(self) -> str:
        """Plain text string representation for MemoryEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        
        if self.memory_file_name:
            base_str += f" [{self.memory_file_name}]"

        try:
            text_parts = content_to_str(self.content.content)
            
            if text_parts:
                content_preview = text_parts[0]
                if len(content_preview) > N_CHAR_PREVIEW:
                    content_preview = content_preview[: N_CHAR_PREVIEW - 3] + '...'

                content_preview = content_preview.replace('\n', ' ').replace('\r', ' ')

                return f'{base_str}\n  Content: {content_preview}'
            else:
                return f'{base_str}\n  Content: [empty]'

        except Exception as e:
            error_preview = str(e)
            if len(error_preview) > N_CHAR_PREVIEW:
                error_preview = error_preview[: N_CHAR_PREVIEW - 3] + '...'
            return f'{base_str}\n  Error: {error_preview}'

