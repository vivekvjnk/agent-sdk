from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from openhands.sdk.llm import TextContent
from openhands.tools.cat_on_steroids.pdf_to_dict import page_dict_to_string

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
)


class CatOnSteroidsAction(Action):
    """
    Action to view document metadata or perform structured search (COS/FOS).
    """

    doc_path: str = Field(
        description="Absolute path to the PDF document."
    )

    # --- Search Parameters (FOS) ---
    search_pattern: str | None = Field(
        default=None, description="The keyword or regex pattern to search for in file contents."
    )
    is_regex: bool = Field(
        default=False, description="Set to True if search_pattern is a full regex."
    )
    search_level: Literal[1, 2] = Field(
        default=1,
        description=(
            "Level 1: Returns only metadata (page, section, count) for an overview. All matches are returned."
            "Level 2: Returns the complete content (dictionary format) of the top N results."
        ),
    )
    n_results: int = Field(
        default=10,
        description="Number of search results to return in Level 2. Use -1 for all results.",
    )


class CatOnSteroidsObservation(Observation):
    """
    Observation containing structured document information or search results.
    """

    total_results: int = 0
    metadata_summary: list[str] = Field(default_factory=list)  # For Level 1 output
    content_results: list[dict] = Field(default_factory=list)  # For Level 2 output
    doc_metadata: dict = Field(default_factory=dict)  # For COS (viewing) output

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        # --- Handle Search Results (FOS) ---
        if self.total_results > 0:
            # Level 1 Summary
            if self.metadata_summary:
                summary_text = "\n".join(self.metadata_summary[:20])
                more = "\n..." if self.total_results > 20 else ""
                ret = (
                    f"Found {self.total_results} occurrences of the search term.\n"
                    f"Metadata Summary (Page: Section and subsection details):\n{summary_text}\n{more}\n"
                    f"To see full content, call CatOnSteroidsAction with search_level=2."
                )
                return [TextContent(text=ret)]

            # Level 2 Content
            elif self.content_results:
                # Format the list of dictionaries into a clean, LLM-digestible string
                formatted_results = "\n\n--- Result ---\n\n".join(
                    page_dict_to_string(r) for r in self.content_results
                )
                ret = (
                    f"Found {self.total_results} results. Returning {len(self.content_results)} complete pages.\n"
                    f"Content:\n{formatted_results}"
                )
                return [TextContent(text=ret)]

        # --- Handle Document Metadata (COS View) ---
        elif self.doc_metadata:
            # Simple metadata view (e.g., Table of Contents)
            formatted_meta = str(self.doc_metadata)
            return [TextContent(text=f"Document Metadata:\n{formatted_meta}")]

        return [
            TextContent(text="COS tool executed, but no relevant data was returned.")
        ]


TOOL_DESCRIPTION = """Powerful tool for navigating and searching large engineering/scientific reference documents (PDFs, Datasheets,Text books, Research reports etc). Designed as an intelligent extension of `cat` bash tool.
* Converts documents into structured, metadata-rich pages for reliable AI consumption.
* **VIEW MODE:** Used to retrieve overall document structure (Table of Contents, page count).
* **SEARCH MODE:** Used to find keywords or regex patterns, returning structured page data.
* **Search Levels:**
    * Level 1 (Default): Returns a fast summary of page and section locations (metadata). Use this first to scope your search.
    * Level 2: Returns the complete, rich dictionary content for the matched pages (full data for reasoning).
* Use this tool to efficiently extract precise information (e.g., register names, thermal limits) without reading the whole document.
"""


cos_tool = ToolDefinition(
    name="cat_on_steroids",
    description=TOOL_DESCRIPTION,
    action_type=CatOnSteroidsAction,
    observation_type=CatOnSteroidsObservation,
)


class CatOnSteroidsTool(ToolDefinition[CatOnSteroidsAction, CatOnSteroidsObservation]):
    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["CatOnSteroidsTool"]:
        from openhands.tools.cat_on_steroids.impl import CatOnSteroidsExecutor

        executor = CatOnSteroidsExecutor()

        return [
            cls(
                name=cos_tool.name,
                description=TOOL_DESCRIPTION,
                action_type=CatOnSteroidsAction,
                observation_type=CatOnSteroidsObservation,
                annotations=cos_tool.annotations,
                executor=executor,
            )
        ]
