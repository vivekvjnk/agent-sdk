"""Delegate tool definitions for OpenHands agents."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field

from openhands.sdk.llm.message import ImageContent, TextContent
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
)
from openhands.sdk import AgentContext


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


CommandLiteral = Literal["spawn", "delegate"]


# class DelegateAction(Action):
#     """Schema for delegation operations."""

#     command: CommandLiteral = Field(
#         description="The commands to run. Allowed options are: `spawn`, `delegate`."
#     )
#     ids: list[str] | None = Field(
#         default=None,
#         description="Required parameter of `spawn` command. "
#         "List of identifiers to initialize sub-agents with.",
#     )
#     tasks: dict[str, str] | None = Field(
#         default=None,
#         description=(
#             "Required parameter of `delegate` command. "
#             "Dictionary mapping sub-agent identifiers to task descriptions."
#         ),
#     )

class DelegateAction(Action):
    """Schema for delegation operations."""

    command: CommandLiteral = Field(
        description=(
            """The command to execute. Allowed options are: `spawn`, `delegate`.
            **Examples:**
            - `{"command": "spawn"}`
            - `{"command": "delegate"}`
            """
        )
    )

    ids: list[str] | None = Field(
        default=None,
        description=(
            """Required parameter for the `spawn` command.
            List of unique string identifiers for sub-agents to initialize.
            **Example:**
            ```json
            {"ids": ["research", "implementation", "testing"]}
            ```
            """
        ),
    )

    tasks: dict[str, str] | None = Field(
        default=None,
        description=(
            """
            Required parameter for the `delegate` command.
            Dictionary mapping sub-agent identifiers (keys) to task descriptions (values).
            **Example:**
            ```json
            {
            "tasks": {
                "research": "Find best practices for async code",
                "implementation": "Refactor the MyClass class",
                "testing": "Write unit tests for new implementation"
                } 
            }
            ```
            """
        ),
    )


class DelegateObservation(Observation):
    """Observation from delegation operations."""

    command: CommandLiteral = Field(
        description="The command that was executed. Either `spawn` or `delegate`."
    )
    message: str = Field(description="Result message from the operation")

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Get the observation content to show to the agent."""
        return [TextContent(text=self.message)]


TOOL_DESCRIPTION = """Delegation tool for spawning sub-agents and delegating tasks to them.

This tool provides two commands:

**spawn**: Initialize sub-agents with meaningful identifiers
- Use descriptive identifiers that make sense for your use case (e.g., 'refactoring', 'run_tests', 'research')
- Each identifier creates a separate sub-agent conversation
- Example: `{{"command": "spawn", "ids": ["research", "implementation", "testing"]}}`

**delegate**: Send tasks to specific sub-agents and wait for results
- Use a dictionary mapping sub-agent identifiers to task descriptions
- This is a blocking operation - waits for all sub-agents to complete
- Returns a single observation containing results from all sub-agents
- Example: `{{"command": "delegate", "tasks": {{"research": "Find best practices for async code", "implementation": "Refactor the MyClass class"}}}}`

**Important Notes:**
- Identifiers used in delegate must match those used in spawn
- All operations are blocking and return comprehensive results
- Sub-agents work in the same workspace as the main agent: {workspace_path}
"""  # noqa

delegate_tool = ToolDefinition(
    name="delegate",
    action_type=DelegateAction,
    observation_type=DelegateObservation,
    description=TOOL_DESCRIPTION,
    annotations=ToolAnnotations(
        title="delegate",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)


class DelegateTool(ToolDefinition[DelegateAction, DelegateObservation]):
    """A ToolDefinition subclass that automatically initializes a DelegateExecutor."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        max_children: int = 5,
        agent_context: Optional[AgentContext] = None
    ) -> Sequence["DelegateTool"]:
        """Initialize DelegateTool with a DelegateExecutor.

        Args:
            conv_state: Conversation state (used to get workspace location)
            max_children: Maximum number of concurrent sub-agents (default: 5)

        Returns:
            List containing a single delegate tool definition
        """
        # Import here to avoid circular imports
        from openhands.tools.delegate.impl import DelegateExecutor

        # Create dynamic description with workspace info
        tool_description = TOOL_DESCRIPTION.format(
            workspace_path=conv_state.workspace.working_dir
        )

        # Initialize the executor without parent conversation
        # (will be set on first call)
        executor = DelegateExecutor(max_children=max_children,agent_context=agent_context)

        # Initialize the parent Tool with the executor
        return [
            cls(
                name=delegate_tool.name,
                description=tool_description,
                action_type=DelegateAction,
                observation_type=DelegateObservation,
                annotations=delegate_tool.annotations,
                executor=executor,
            )
        ]
