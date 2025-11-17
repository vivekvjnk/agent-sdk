# memory_manager.py
import time
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Iterable
from abc import ABC

from openhands.sdk.llm import Message, TextContent
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
)

try:
    import yaml
except Exception as e:  # pragma: no cover - fail early with helpful message
    raise ImportError(
        "PyYAML is required for PersistentMemoryManager. Install with `pip install pyyaml`."
    ) from e

# TODO: prepare persistent memory tool. Instance of PersistentMemoryManager is initialized and managed by the top level orchestrator. This allows custom specification for persistent memory location. This instance has to be passed down to the persistent memory tool and agent. On agent side, we've already implemented optional variable for dependency injection.
class PersistentMemoryManager_v0:
    """Class managing persistent memory file, caching, and change tracking.

    This manager is the single source of truth for the persistent memory file.
    It caches the content in RAM and uses file modification time to efficiently
    check for external updates.
    """

    # Instance state attributes (will be initialized in __init__)
    base_path: Path
    file_name: str
    _cached_content: Optional[Message]
    _last_modified_time: float

 
    def __init__(self, base_path: Path, file_name: str):
        """Initializes the manager (only runs once for the singleton)."""
        # Check if the instance has already been initialized to prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.base_path = base_path
        self.file_name = file_name
        self._cached_content = None
        self._last_modified_time = 0.0

        # Run setup methods
        self.initialize()
        self.reload_content()

        # Mark as initialized
        self._initialized = True

    @property
    def file_path(self) -> Path:
        """Returns the full path to the memory file."""
        return self.base_path / self.file_name

    def initialize(self):
        """Initializes the memory file. Creates the base directory and the file if they don't exist."""
        _ensure_path_exists(self.file_path)
        if not self.file_path.exists():
            self.file_path.touch()

    def _read_file_content(self) -> str:
        """Reads the complete text content from the memory file."""
        if not self.file_path.exists():
            self.initialize()
        
        # NOTE: Using 'r' mode to read content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def reload_content(self) -> None:
        """Checks the file modification time and reloads content if changed."""
        if not self.file_path.exists():
            self._cached_content = Message(role='system', content=[TextContent(text='')])
            self._last_modified_time = 0.0
            return

        current_mtime = 0.0
        try:
            current_mtime = self.file_path.stat().st_mtime
        except OSError:
            # File might be inaccessible/deleted
            pass

        # Reload if the modification time has changed OR if content has never been loaded
        if current_mtime > self._last_modified_time or self._cached_content is None:
            try:
                memory_text = self._read_file_content()
                self._cached_content = Message(
                    role='system',
                    content=[TextContent(text=memory_text)],
                )
                self._last_modified_time = current_mtime
                # print(f"[Manager] Content reloaded from disk. New mtime: {current_mtime}")
            except Exception:
                self._cached_content = Message(role='system', content=[TextContent(text='[Memory Load Failed]')])
                self._last_modified_time = current_mtime if current_mtime > 0 else time.time()
        # else:
            # print(f"[Manager] Content is cached. No reload needed.")


    def update(self, new_content: str):
        """Updates the memory file and eagerly updates the in-RAM cache."""
        _ensure_path_exists(self.file_path)
        # NOTE: Using 'w' mode to overwrite content
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        # Eagerly update the cache and mtime after writing
        current_mtime = time.time()
        try:
            current_mtime = self.file_path.stat().st_mtime
        except OSError:
            pass # Fallback to time.time() if stat fails

        self._cached_content = Message(
            role='system',
            content=[TextContent(text=new_content)],
        )
        self._last_modified_time = current_mtime
        
    def get_content(self) -> Message:
        """Gets the latest cached content, checking for file changes first."""
        self.reload_content()
        if self._cached_content is None:
            raise RuntimeError("Memory content could not be loaded or initialized.")
        return self._cached_content


def _ensure_path_exists(file_path: Path):
    """Ensures the directory for the given file_path exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

def _safe_atomic_write(target_path: Path, data: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `data` to `target_path` by writing to a temp file in the same
    directory and renaming it over the target. This avoids partial writes on crash.
    """
    _ensure_path_exists(target_path)
    dirpath = target_path.parent
    # Write to a named temporary file inside the same directory to preserve atomic rename semantics
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(dirpath), encoding=encoding) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_name = Path(tmp.name)
    # On POSIX, replace is atomic
    shutil.move(str(tmp_name), str(target_path))

class PersistentMemoryManager(DiscriminatedUnionMixin,ABC):
    """
    Manage a YAML-backed persistent memory file containing a top-level list of strings.

    - Memory file is a YAML sequence (e.g. `- entry 1\n- entry 2\n`).
    - The manager caches the list in memory and reloads when the file mtime changes.
    - Provides safe atomic writes and simple in-process locking.
    - Convenience mutation helpers: append_entries, delete_indices, clear_all.

    The agent/backend expects get_content() -> Message(role='system', content=[TextContent(text=...)]),
    where the text is the YAML dump (so it can be injected into the chat history).
    """

    base_path: Path
    file_name: str
    _cached_entries: Optional[List[str]]
    _last_modified_time: float
    _lock: threading.Lock

    def __init__(self, base_path: Path, file_name: str = "persistent_memory.yaml"):
        """Initialize manager and eagerly load/create the memory file."""
        # Avoid double-init in case orchestrator reuses instance (defensive)
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__(base_path=base_path, file_name=file_name)

        self.base_path = Path(base_path)
        self.file_name = file_name
        self._cached_entries = None
        self._last_modified_time = 0.0
        self._lock = threading.Lock()

        # create path and file if needed
        self.initialize()
        # load current content into cache
        self.reload_content()

        self._initialized = True

    @property
    def file_path(self) -> Path:
        """Full path to the persistent memory YAML file."""
        return self.base_path / self.file_name

    def initialize(self) -> None:
        """Create base directory and file if they do not exist. Ensure it contains a YAML list."""
        _ensure_path_exists(self.file_path)
        if not self.file_path.exists():
            # initialize with an empty YAML list
            initial_yaml = yaml.safe_dump([], sort_keys=False)
            _safe_atomic_write(self.file_path, initial_yaml)
            # set mtime to now
            self._last_modified_time = self.file_path.stat().st_mtime

    def _read_file_text(self) -> str:
        """Return raw file text (UTF-8). Creates file if missing."""
        if not self.file_path.exists():
            self.initialize()
        with open(self.file_path, "r", encoding="utf-8") as fh:
            return fh.read()

    def _load_entries_from_yaml(self, raw_text: str) -> List[str]:
        """Parse YAML and coerce into a list of strings.

        Accepts:
          - YAML sequence of scalars -> list[str]
          - Single scalar -> [str]
          - Empty file -> []
        """
        if not raw_text.strip():
            return []
        try:
            parsed = yaml.safe_load(raw_text)
        except yaml.YAMLError:
            # If parsing fails, fall back to line-based splitting (best-effort).
            entries = [line.rstrip("\n") for line in raw_text.splitlines() if line.strip()]
            return entries

        # If YAML is a list, attempt to coerce all items to strings
        if isinstance(parsed, list):
            coerced: List[str] = []
            for item in parsed:
                if item is None:
                    coerced.append("")
                else:
                    coerced.append(str(item))
            return coerced

        # If YAML parsed to a scalar, return single-element list
        if isinstance(parsed, (str, int, float, bool)):
            return [str(parsed)]

        # For mapping or other structures, attempt to serialize them to YAML and return as single entry
        return [yaml.safe_dump(parsed, sort_keys=False)]

    def _serialize_entries_to_yaml(self, entries: Iterable[str]) -> str:
        """Serialize a list of string entries to a YAML sequence (string)."""
        # ensure we write a simple YAML sequence of scalars
        safe_entries = [("" if e is None else e) for e in entries]
        return yaml.safe_dump(list(safe_entries), sort_keys=False, allow_unicode=True)

    def reload_content(self) -> None:
        """
        Reload entries from disk if file modification time changed or cache is empty.

        Thread-safe for in-process readers/writers via self._lock.
        """
        with self._lock:
            if not self.file_path.exists():
                # initialize file and cache
                self.initialize()
                self._cached_entries = []
                self._last_modified_time = 0.0
                return

            current_mtime = 0.0
            try:
                current_mtime = self.file_path.stat().st_mtime
            except OSError:
                # If stat fails, force a reload by setting mtime to 0
                current_mtime = 0.0

            if current_mtime > self._last_modified_time or self._cached_entries is None:
                try:
                    raw = self._read_file_text()
                    entries = self._load_entries_from_yaml(raw)
                    self._cached_entries = entries
                    self._last_modified_time = current_mtime
                    # debug: print("[PersistentMemoryManager] Reloaded from disk; entries:", len(entries))
                except Exception:
                    # On failure, set cache to safe sentinel so caller can handle it
                    self._cached_entries = []
                    self._last_modified_time = time.time()

    def _write_entries(self, entries: List[str]) -> None:
        """Atomically write `entries` to the YAML file and update cache & mtime."""
        yaml_text = self._serialize_entries_to_yaml(entries)
        # atomic write
        _safe_atomic_write(self.file_path, yaml_text)
        # update cached state
        try:
            mtime = self.file_path.stat().st_mtime
        except OSError:
            mtime = time.time()
        self._cached_entries = list(entries)
        self._last_modified_time = mtime

    # Public API: read helpers

    def get_entries(self) -> List[str]:
        """Return the current list of memory entries (reloads from disk if changed)."""
        self.reload_content()
        # return a copy to prevent callers mutating internal list
        return list(self._cached_entries or [])

    def get_content(self) -> Message:
        """
        Return a Message object representing the memory file (YAML text) suitable for
        injecting into the chat history as the system/3rd-last message.
        """
        # Ensure latest content loaded
        self.reload_content()
        entries = self._cached_entries or []
        yaml_text = self._serialize_entries_to_yaml(entries)
        return Message(role="system", content=[TextContent(text=yaml_text)])

    # Public API: mutation helpers (these eagerly update cache & file)

    def append_entries(self, new_entries: Iterable[str]) -> int:
        """
        Append new entries at the end of the memory list.

        Returns:
            new_size (int): number of entries after append.
        """
        if new_entries is None:
            return len(self.get_entries())

        with self._lock:
            current = self.get_entries()
            current.extend([str(e) for e in new_entries])
            self._write_entries(current)
            return len(current)

    def delete_indices(self, indices: Iterable[int]) -> int:
        """
        Delete entries by (zero-based) indices. Indices may be unsorted and may contain duplicates.

        Behavior:
          - Invalid indices are ignored.
          - If after deletions the list is empty, file contains empty YAML list.
        Returns:
          new_size (int): number of entries after deletions.
        """
        with self._lock:
            current = self.get_entries()
            if not current:
                return 0
            # Normalize indices: only keep valid integers within range
            valid = set(i for i in (int(i) for i in indices) if 0 <= i < len(current))
            if not valid:
                # nothing to delete
                return len(current)
            # build new list excluding the indices
            new_list = [v for idx, v in enumerate(current) if idx not in valid]
            self._write_entries(new_list)
            return len(new_list)

    def clear_all(self) -> int:
        """
        Remove all entries from memory. Returns new size (0).
        """
        with self._lock:
            self._write_entries([])
            return 0

    # Convenience: replace entire list (used if executor wants to write wholesale)
    def replace_all(self, entries: Iterable[str]) -> int:
        """Replace the entire memory list with provided entries. Returns new size."""
        with self._lock:
            new_list = [str(e) for e in entries]
            self._write_entries(new_list)
            return len(new_list)
