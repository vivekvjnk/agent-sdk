from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import EventID, SourceType, ToolCallID
from openhands.sdk.llm import Message, TextContent, content_to_str
from openhands.sdk.tool.schema import Observation


import base64
import json
from typing import Any

# If you're using Pydantic v2, BaseModel is pydantic.BaseModel
from pydantic import BaseModel

# ---------------------------
# Sanitizer utilities
# ---------------------------
def _is_bytes_like(x: Any) -> bool:
    return isinstance(x, (bytes, bytearray))

def _bytes_to_base64_str(b: bytes) -> str:
    """Return an ASCII base64 encoding safe for JSON."""
    return base64.b64encode(b).decode("ascii")

def _try_decode_utf8(b: bytes) -> str | None:
    """Try strict utf-8 decode, return str on success else None."""
    try:
        return b.decode("utf-8", errors="strict")
    except Exception:
        return None

def _sanitize_structure(
    obj: Any,
    *,
    convert_bytes_to_base64: bool = True,
    allow_replacement_decode: bool = False,
) -> Any:
    """
    Recursively convert nested bytes/bytearray into either:
      - decoded UTF-8 string if valid utf-8
      - or base64 ascii string (default) if convert_bytes_to_base64=True
    Other containers (list/tuple/set/dict) are walked recursively.
    """
    # Bytes-like
    if _is_bytes_like(obj):
        raw = bytes(obj)
        # Prefer returning text if it's valid UTF-8
        decoded = _try_decode_utf8(raw)
        if decoded is not None:
            return decoded
        if allow_replacement_decode:
            # best-effort decode with replacement characters
            return raw.decode("utf-8", errors="replace")
        if convert_bytes_to_base64:
            return _bytes_to_base64_str(raw)
        # Fail-fast if caller doesn't want conversion
        raise ValueError("Found binary bytes in structure and convert_bytes_to_base64=False")

    # dict -> sanitize values
    if isinstance(obj, dict):
        return {
            k: _sanitize_structure(v, convert_bytes_to_base64=convert_bytes_to_base64,
                                   allow_replacement_decode=allow_replacement_decode)
            for k, v in obj.items()
        }

    # list/tuple -> sanitize elements
    if isinstance(obj, list):
        return [
            _sanitize_structure(v, convert_bytes_to_base64=convert_bytes_to_base64,
                                allow_replacement_decode=allow_replacement_decode)
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _sanitize_structure(v, convert_bytes_to_base64=convert_bytes_to_base64,
                                allow_replacement_decode=allow_replacement_decode)
            for v in obj
        )
    if isinstance(obj, set):
        return {
            _sanitize_structure(v, convert_bytes_to_base64=convert_bytes_to_base64,
                                allow_replacement_decode=allow_replacement_decode)
            for v in obj
        }

    # otherwise return as-is (str, int, float, None, etc.)
    return obj

# ---------------------------
# Mixin / override implementation
# ---------------------------
class _SanitizingModelMixin(BaseModel):
    """
    Mixin to override model_dump and model_dump_json to sanitize nested bytes.
    Inherit this before BaseModel or use as a sibling base:
        class CatOnSteroidsObservation(_SanitizingModelMixin, BaseModel):
            ...
    """

    def model_dump(self, *args, convert_bytes_to_base64: bool = True, allow_replacement_decode: bool = False, **kwargs) -> Any:
        """
        Override: call BaseModel.model_dump(...) then sanitize the returned Python structure.

        Extra keywords:
          - convert_bytes_to_base64: if True, convert discovered bytes -> base64 ascii str.
          - allow_replacement_decode: if True, attempt utf-8 decode with replacement before base64.
        """
        # Obtain the raw python structure from Pydantic
        raw = super().model_dump(*args, **kwargs)
        # Sanitize recursively
        sanitized = _sanitize_structure(
            raw,
            convert_bytes_to_base64=convert_bytes_to_base64,
            allow_replacement_decode=allow_replacement_decode,
        )
        return sanitized

    def model_dump_json(self, *args, convert_bytes_to_base64: bool = True, allow_replacement_decode: bool = False, **kwargs) -> str:
        """
        Override: produce a JSON string from a sanitized model dump.

        We call our model_dump() to get a sanitized python object and then json.dumps it.
        This avoids Pydantic's internal JSON conversion from attempting to coerce bytes to text.
        Extra keywords same as in model_dump.
        """
        # Use our sanitized model_dump to get a safe python object
        sanitized_obj = self.model_dump(*args, convert_bytes_to_base64=convert_bytes_to_base64,
                                       allow_replacement_decode=allow_replacement_decode, **kwargs)
        # Now do JSON dump. Use ensure_ascii=False so unicode is preserved, but it's still UTF-8
        return json.dumps(sanitized_obj, ensure_ascii=False)



class ObservationBaseEvent(LLMConvertibleEvent):
    """Base class for anything as a response to a tool call.

    Examples include tool execution, error, user reject.
    """

    source: SourceType = "environment"
    tool_name: str = Field(
        ..., description="The tool name that this observation is responding to"
    )
    tool_call_id: ToolCallID = Field(
        ..., description="The tool call id that this observation is responding to"
    )


class ObservationEvent(_SanitizingModelMixin,ObservationBaseEvent):
    observation: Observation = Field(
        ..., description="The observation (tool call) sent to LLM"
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation event."""
        to_viz = self.observation.visualize
        content = Text()
        if to_viz.plain.strip():
            content.append("Tool: ", style="bold")
            content.append(self.tool_name)
            content.append("\nResult:\n", style="bold")
            content.append(to_viz)
        return content
    
    def to_condensed_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=self.observation.to_condensed_llm_content,
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )
    
    def to_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=self.observation.to_llm_content,
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for ObservationEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        content_str = "".join(content_to_str(self.observation.to_llm_content))
        obs_preview = (
            content_str[:N_CHAR_PREVIEW] + "..."
            if len(content_str) > N_CHAR_PREVIEW
            else content_str
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Result: {obs_preview}"


class UserRejectObservation(ObservationBaseEvent):
    """Observation when user rejects an action in confirmation mode."""

    rejection_reason: str = Field(
        default="User rejected the action",
        description="Reason for rejecting the action",
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this user rejection event."""
        content = Text()
        content.append("Tool: ", style="bold")
        content.append(self.tool_name)
        content.append("\n\nRejection Reason:\n", style="bold")
        content.append(self.rejection_reason)
        return content

    def to_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=[TextContent(text=f"Action rejected: {self.rejection_reason}")],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for UserRejectObservation."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        reason_preview = (
            self.rejection_reason[:N_CHAR_PREVIEW] + "..."
            if len(self.rejection_reason) > N_CHAR_PREVIEW
            else self.rejection_reason
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Reason: {reason_preview}"


class AgentErrorEvent(ObservationBaseEvent):
    """Error triggered by the agent.

    Note: This event should not contain model "thought" or "reasoning_content". It
    represents an error produced by the agent/scaffold, not model output.
    """

    source: SourceType = "agent"
    error: str = Field(..., description="The error message from the scaffold")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this agent error event."""
        content = Text()
        content.append("Error Details:\n", style="bold")
        content.append(self.error)
        return content

    def to_llm_message(self) -> Message:
        # Provide plain string error content; serializers handle Chat vs Responses.
        # For Responses API, output is a string; JSON is not required.
        return Message(
            role="tool",
            content=[TextContent(text=self.error)],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for AgentErrorEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        error_preview = (
            self.error[:N_CHAR_PREVIEW] + "..."
            if len(self.error) > N_CHAR_PREVIEW
            else self.error
        )
        return f"{base_str}\n  Error: {error_preview}"
