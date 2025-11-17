"""Definition file for the persistent memory tool.

This mirrors the style of openhands.tools.execute_bash.definition.py while
adapting fields and behavior to the persistent-memory concept:
- Memory file contents are inserted into the chat history automatically (3rd last message).
- Tool allows adding entries (append) and deleting entries by index.
- No explicit read API since the whole file is added to the history automatically.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Literal

from pydantic import Field
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
)
from openhands.sdk.utils import maybe_truncate

if TYPE_CHECKING:
    # Optional: conversation state might provide workspace / user info if needed later
    from openhands.sdk.conversation.state import ConversationState

# Constants controlling how much memory content we surface to the LLM in observations
MAX_MEMORY_PREVIEW_CHARS = 2000
MEMORY_SNIPPET_HEAD_CHARS = 800
MEMORY_SNIPPET_TAIL_CHARS = 800


class MemoryAction(Action):
    """Schema for memory operations.

    Allowed operations:
      - "append" : Append new entries to the end of the persistent memory list.
      - "delete" : Delete entries by index (provides list of integer indices to delete).
      - "clear_all" : (optional) Clear the entire memory file. Use with care.
    """

    operation: Literal["append", "delete", "clear_all"] = Field(
        description=(
            "The memory operation to perform. One of: "
            "`append` (add entries to the end), "
            "`delete` (remove entries by indices), "
        )
    )

    # content is required for append, ignored for delete/clear_all
    content: List[str] | None = Field(
        default=None,
        description="List of new memory entries to append to the memory file. Required for `append`.",
    )

    # delete_indices is required for delete operation, ignored for append/clear_all
    delete_indices: str | None = Field(
        default=None,
        description="""List of zero-based indices to remove from the memory list in STRING format. Required for `delete`. use python list indexing. ie "-1" deletes the last entry, ":" deletes all entries etc. NOTE: this string will be parsed as a python expression by the executor, so ensure it is a valid python slice/index expression.""",
    )

    @property
    def visualize(self) -> Text:
        """Return a small Rich Text representation for UI/CLI."""
        t = Text()
        t.append("Memory operation: ", style="bold")
        t.append(self.operation, style="cyan bold")
        if self.operation == "append" and self.content:
            t.append("\nEntries to append:\n", style="bold")
            # show up to first 3 entries as preview
            for i, entry in enumerate(self.content[:3]):
                t.append(f"  [{i}] ", style="green")
                # keep preview short
                t.append(maybe_truncate(entry, 200), style="white")
                t.append("\n")
            if len(self.content) > 3:
                t.append(f"  ... (+{len(self.content)-3} more)\n", style="yellow")
        elif self.operation == "delete" and self.delete_indices:
            t.append("\nDelete indices: ", style="bold")
            t.append(", ".join(map(str, self.delete_indices)), style="red")
        elif self.operation == "clear_all":
            t.append("\nThis will clear the entire memory file.", style="red bold")
        return t


class MemoryObservation(Observation):
    """Observation produced after a memory operation."""

    # We do not provide a read operation; this field can be used for short previews
    # or status messages. None by default.
    content_preview: str | None = Field(
        default=None,
        description="A short preview (head + tail) of the memory file after the operation."
    )

    status: str = Field(
        description="Status message summarizing the result of the memory operation."
    )

    new_size: int | None = Field(
        default=None,
        description="New number of entries in the memory file after the operation, if available."
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Return up to MAX_MEMORY_PREVIEW_CHARS of the memory file (head + tail) and the status.

        Note: the full memory is already injected into the chat history by the backend
        as the 3rd-last message; this is only a concise assistant-facing preview.
        """
        preview_text = ""
        if self.content_preview:
            preview_text = self.content_preview
        status_text = f"[Status] {self.status}"
        size_text = f"\n[Memory entries] {self.new_size}" if self.new_size is not None else ""
        combined = f"{preview_text}\n\n{status_text}{size_text}"
        return [TextContent(text=maybe_truncate(combined, MAX_MEMORY_PREVIEW_CHARS))]

    @property
    def visualize(self) -> Text:
        t = Text()
        t.append("Persistent Memory Tool Result\n", style="bold underline")
        t.append("\n")
        if self.status:
            t.append("Status: ", style="bold")
            # color success/failure
            if "success" in self.status.lower():
                t.append(self.status + "\n", style="green")
            elif "error" in self.status.lower() or "failed" in self.status.lower():
                t.append(self.status + "\n", style="red")
            else:
                t.append(self.status + "\n", style="yellow")
        if self.new_size is not None:
            t.append(f"Entries in memory: {self.new_size}\n", style="blue")
        if self.content_preview:
            t.append("\nPreview (head ... tail):\n", style="bold")
            t.append(self.content_preview, style="white")
            t.append("\n")
        return t


TOOL_DESCRIPTION = """Persistent memory tool for agent-managed compressed knowledge.

Behavior:
* A persistent memory file exists at a configurable location (backend-managed). The file contains a list of memory entries.
* The **full** memory file content is automatically injected into the agent's message history as the 3rd-last message on each turn (no separate 'read' operation is necessary).
* This tool exposes mutation operations only:
  * `append` - append one or more entries to the memory list.
  * `delete` - delete specific entries by zero-based index. use python list indexing.

Notes / Best practices:
* Use short, high-signal entries (compressed facts, summaries, checkpoints). Large verbose entries will lengthen context.
* Indices are zero-based and refer to the current ordering in the memory list.
* There is no 'read' tool; the frontend/backend ensures the memory is part of the context for the agent.
* The executor is responsible for file locking, atomic updates, index validation, and persistence guarantees.
"""

# Annotations (tuned to the operation semantics)
memory_tool = ToolDefinition(
    name="persistent_memory",
    description=TOOL_DESCRIPTION,
    action_type=MemoryAction,
    observation_type=MemoryObservation,
    annotations=ToolAnnotations(
        title="persistent_memory",
        readOnlyHint=False,         # tool performs writes
        destructiveHint=True,       # can delete/clear memory
        idempotentHint=False,       # append/delete are not necessarily idempotent
        openWorldHint=True,
    ),
)


class PersistentMemoryTool(ToolDefinition[MemoryAction, MemoryObservation]):
    """ToolDefinition subclass for persistent-memory that wires the executor."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        memory_manager,  # an object responsible for low-level memory I/O (injected by backend)
    ) -> Sequence["PersistentMemoryTool"]:
        """
        Initialize the PersistentMemoryTool with a memory manager/executor.

        Args:
            memory_manager: object or manager used by the executor to access the memory file.
                            The executor implementation (PersistentMemoryExecutor) decides the
                            exact interface; it should handle locking and validation.
        """
        # Import here to avoid circular imports and keep executor logic outside of definition
        from openhands.tools.memory.impl import PersistentMemoryExecutor

        executor = PersistentMemoryExecutor(memory_manager=memory_manager)

        return [
            cls(
                name=memory_tool.name,
                description=TOOL_DESCRIPTION,
                action_type=MemoryAction,
                observation_type=MemoryObservation,
                annotations=memory_tool.annotations,
                executor=executor,
            )
        ]
