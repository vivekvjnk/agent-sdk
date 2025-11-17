# persistent_memory_executor.py
from collections.abc import Iterable
from typing import TYPE_CHECKING, List

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation

from openhands.tools.memory.definition import MemoryAction, MemoryObservation
from openhands.sdk.context.memory import PersistentMemoryManager  # your manager module
from openhands.sdk.utils import maybe_truncate

logger = get_logger(__name__)


class PersistentMemoryExecutor(ToolExecutor[MemoryAction, MemoryObservation]):
    """
    Executor that maps MemoryAction -> PersistentMemoryManager operations and
    returns MemoryObservation objects.

    Expected manager interface:
      - get_entries() -> List[str]
      - append_entries(iterable[str]) -> new_size: int
      - delete_indices(iterable[int]) -> new_size: int
      - clear_all() -> new_size: int
      - replace_all(entries) -> new_size: int
      - get_content() -> Message(role='system', content=[TextContent(text=...)])

    The executor performs basic validation of action fields and translates results
    into MemoryObservation instances used by the tool definition.
    """

    manager: PersistentMemoryManager
    # preview config (chars)
    PREVIEW_HEAD_ENTRIES = 1
    PREVIEW_TAIL_ENTRIES = 2
    PREVIEW_MAX_CHARS = 1200

    def __init__(self, memory_manager: PersistentMemoryManager):
        """
        Initialize with a PersistentMemoryManager instance.

        Args:
            memory_manager: instance responsible for YAML-backed memory I/O and caching.
        """
        self.manager = memory_manager
        # Ensure manager has loaded content
        try:
            _ = self.manager.get_entries()
        except Exception as ex:
            logger.warning("Failed to preload memory manager content during executor init: %s", ex)
        logger.info("PersistentMemoryExecutor initialized with file: %s", str(self.manager.file_path))

    def _make_preview_from_entries(self, entries: List[str]) -> str:
        """Construct a concise preview string from the entries (head ... tail)."""
        if not entries:
            return "[memory is empty]"

        head = entries[: self.PREVIEW_HEAD_ENTRIES]
        tail = entries[-self.PREVIEW_TAIL_ENTRIES :] if len(entries) > self.PREVIEW_HEAD_ENTRIES else []
        preview_parts = []

        # Format head
        for i, item in enumerate(head):
            preview_parts.append(f"[{i}] {item}")

        # If there's a gap between head and tail, show omitted indicator
        if tail and (len(entries) > (self.PREVIEW_HEAD_ENTRIES + self.PREVIEW_TAIL_ENTRIES)):
            preview_parts.append(f"... ({len(entries) - len(head) - len(tail)} entries omitted) ...")

        # Format tail with proper indices
        if tail:
            start_idx = max(len(entries) - len(tail), len(head))
            for offset, item in enumerate(tail):
                preview_parts.append(f"[{start_idx + offset}] {item}")

        preview_text = "\n".join(preview_parts)
        # Truncate to PREVIEW_MAX_CHARS to keep it compact for LLM consumption
        return maybe_truncate(preview_text, self.PREVIEW_MAX_CHARS)

    def __call__(self, action: MemoryAction, conversation: "LocalConversation | None" = None) -> MemoryObservation:
        """
        Execute the requested memory operation and return a MemoryObservation.

        Supported operations (as defined in the MemoryAction schema):
          - "append": requires action.content (list[str]) -> appends entries
          - "delete": requires action.delete_indices (list[int]) -> deletes indices
          - "clear_all": clears the whole memory (no extra field required)

        Returns:
            MemoryObservation with content_preview (short head/tail preview), status, and new_size.
        """
        op = action.operation
        logger.debug("PersistentMemoryExecutor called with operation=%s", op)

        # Normalize operation
        if not op:
            raise ValueError("MemoryAction.operation must be set")

        op = op.lower()

        # Dispatch
        try:
            if op == "append":
                if not action.content:
                    raise ValueError("append operation requires non-empty content list")
                # ensure strings
                entries_to_append = [str(x) for x in action.content]
                new_size = self.manager.append_entries(entries_to_append)
                entries_after = self.manager.get_entries()
                preview = self._make_preview_from_entries(entries_after)
                status = f"Appended {len(entries_to_append)} entries successfully."
                return MemoryObservation(content_preview=preview, status=status, new_size=new_size)

            elif op == "delete":
                if not action.delete_indices:
                    raise ValueError("delete operation requires delete_indices")
                # coerce to ints and filter invalid inside manager
                indices = [int(i) for i in action.delete_indices]
                new_size = self.manager.delete_indices(indices)
                entries_after = self.manager.get_entries()
                preview = self._make_preview_from_entries(entries_after)
                status = f"Deleted indices: {sorted(set(indices))}. New size: {new_size}."
                return MemoryObservation(content_preview=preview, status=status, new_size=new_size)

            elif op == "clear_all":
                new_size = self.manager.clear_all()
                preview = self._make_preview_from_entries([])
                status = "Cleared all memory entries."
                return MemoryObservation(content_preview=preview, status=status, new_size=new_size)

            else:
                raise ValueError(f"Unsupported memory operation: {op}")

        except Exception as ex:
            # On any error, attempt to return a helpful observation (no crash of orchestrator)
            logger.exception("Error while performing memory operation %s: %s", op, ex)
            # Try best-effort to include a preview
            try:
                entries = self.manager.get_entries()
                preview = self._make_preview_from_entries(entries)
                size = len(entries)
            except Exception:
                preview = "[unable to load memory preview]"
                size = None
            status = f"Error performing operation '{op}': {ex}"
            return MemoryObservation(content_preview=preview, status=status, new_size=size)
