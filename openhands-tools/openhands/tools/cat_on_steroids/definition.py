from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from textwrap import shorten

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
        default=None, description="The keyword or regex pattern to search for in file contents. **NOTE: search is CASE SENSITIVE**"
    )
    search_level: Literal[1, 2] = Field(
        default=1,
        description=(
            "Level 1: Returns only metadata (page, section, count) for an overview. All matches are returned."
            "Level 2: Returns the complete content (dictionary format) of the top N results."
        ),
    )
    pages: str = Field(
        default="",
        description=(
            "Specify one or more pages to retrieve from the parsed document. "
            "Accepts single page numbers (e.g., 3), lists of pages (e.g., [1, 3, 5]), "
            "or page ranges (e.g., '2-5'). Mixed inputs are also supported, such as ['1-3', 5, 7]. "
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
    page_results: list[dict] = Field(default_factory=list) # For retrieving pages
    doc_metadata: dict = Field(default_factory=dict)  # For COS (viewing) output

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        # --- Handle Search Results (FOS) ---
        if self.total_results > 0:
            # Level 1 Summary
            if self.metadata_summary:
                # summary_text = "\n".join(self.metadata_summary[:20])
                # more = "\n..." if self.total_results > 20 else ""
                # ret = (
                #     f"Found {self.total_results} occurrences of the search term.\n"
                #     f"Metadata Summary (Page: Section and subsection details):\n{summary_text}\n{more}\n"
                #     f"To see full content, call CatOnSteroidsAction with search_level=2."
                # )

                first_20_res = format_cos_search_output(pages=self.metadata_summary[:20]) # only return first 20 occurrences of the search result
                more = ""
                if len(self.metadata_summary) > 20:
                    truncated_results_count = len(self.metadata_summary)-20
                    more = f"\n... and more...\ntruncated {truncated_results_count} results ..."
                meta_search_results = first_20_res + more
                return [TextContent(text=meta_search_results)]

            # Level 2 Content
            elif self.content_results:
                # Format the list of dictionaries into a clean, LLM-digestible string
                formatted_results = "\n\n--- <Page Break> ---\n\n".join(
                    page_dict_to_string(r) for r in self.content_results
                )
                ret = (
                    f"Found {self.total_results} results. Returning {len(self.content_results)} complete pages.\n"
                    f"Content:\n{formatted_results}"
                )
                return [TextContent(text=ret)]
            elif self.page_results:
                # Format the list of dictionaries into a clean, LLM-digestible string
                formatted_results = "\n\n--- <Page Break> ---\n\n".join(
                    page_dict_to_string(r) for r in self.page_results
                )
                ret = (
                    f"Returning {len(self.page_results)} complete pages.\n"
                    f"Content:\n{formatted_results}"
                )
                return [TextContent(text=ret)]

        # --- Handle Document Metadata (COS View) ---
        elif self.doc_metadata:
            # Simple metadata view (e.g., Table of Contents)
            formatted_meta = format_toc(metadata=self.doc_metadata,level=-1)
            return [TextContent(text=formatted_meta)]

        return [
            TextContent(text="COS tool executed, but no relevant data was returned.")
        ]
    
    @property
    def to_condensed_llm_message(self) -> Sequence[TextContent]:
        # --- Handle Search Results (FOS) ---
        if self.total_results > 0:
            # Level 1 Summary
            if self.metadata_summary:
                # summary_text = "\n".join(self.metadata_summary[:20])
                # more = "\n..." if self.total_results > 20 else ""
                # ret = (
                #     f"Found {self.total_results} occurrences of the search term.\n"
                #     f"Metadata Summary (Page: Section and subsection details):\n{summary_text}\n{more}\n"
                #     f"To see full content, call CatOnSteroidsAction with search_level=2."
                # )
                first_20_res = format_cos_search_output(pages=self.metadata_summary[:20]) # only return first 20 occurrences of the search result
                more = ""
                if len(self.metadata_summary) > 20:
                    truncated_results_count = len(self.metadata_summary)-20
                    more = f"\n... and more...\ntruncated {truncated_results_count} results ..."
                meta_search_results = first_20_res + more
                return [TextContent(text=meta_search_results)]

            # Level 2 Content
            elif self.content_results:
                # Format the list of dictionaries into a clean, LLM-digestible string
                formatted_results = "\n\n--- <Page Break> ---\n\n".join(
                    page_dict_to_string(r) for r in self.content_results
                )
                ret = (
                    f"Found {self.total_results} results. Returning {len(self.content_results)} complete pages.\n"
                    f"Content:\n{formatted_results}"
                )
                return [TextContent(text=ret)]
            elif self.page_results:
                # Format the list of dictionaries into a clean, LLM-digestible string
                formatted_results = "\n\n--- <Page Break> ---\n\n".join(
                    page_dict_to_string(r) for r in self.page_results
                )
                ret = (
                    f"Returning {len(self.page_results)} complete pages.\n"
                    f"Content:\n{formatted_results}"
                )
                return [TextContent(text=ret)]

        # --- Handle Document Metadata (COS View) ---
        elif self.doc_metadata:
            # Simple metadata view (e.g., Table of Contents)
            formatted_meta = format_toc(metadata=self.doc_metadata,level=2)
            return [TextContent(text=formatted_meta)]

        return [
            TextContent(text="COS tool executed, but no relevant data was returned.")
        ]

def format_toc(metadata: dict, level: int | None = None) -> str:
    """
    Convert document metadata into a structured, LLM-optimized Table of Contents (TOC).
    
    Args:
        metadata (dict): Document metadata containing 'total_pages' and 'sections'.
        level (int | None): Maximum depth of TOC to include.
                            If None or -1, includes all levels.
    
    Returns:
        str: Formatted TOC text with truncation info if applicable.
    """
    total_pages = metadata.get("total_pages", "Unknown")
    sections = metadata.get("sections", [])
    if not sections:
        return "No section metadata available."

    lines = [
        "Document Metadata",
        "-" * 19,
        f"Total Pages: {total_pages}",
        "",
        "Table of Contents",
        "=================",
    ]

    max_level = None if level in (None, -1) else int(level)

    # Group sections by hierarchy for efficient truncation detection
    grouped = {}
    for idx, (sec_level, title, page) in enumerate(sections):
        grouped.setdefault(sec_level, []).append((idx, title, page))

    # Helper to count deeper subsections for a given section
    def count_subsections(start_idx: int, parent_level: int) -> int:
        count = 0
        for i in range(start_idx + 1, len(sections)):
            lvl, _, _ = sections[i]
            if lvl <= parent_level:
                break
            count += 1
        return count

    # Iterate through TOC and build formatted output
    for idx, (sec_level, title, page) in enumerate(sections):
        if max_level is not None and sec_level > max_level:
            continue

        indent = "    " * (sec_level - 1)
        lines.append(f"{indent}{title}  (p.{page})")

        # If we are truncating deeper levels, summarize them
        if max_level is not None and sec_level == max_level:
            sub_count = count_subsections(idx, sec_level)
            if sub_count > 0:
                lines.append(f"{indent}    ... {sub_count} subsections truncated")

    return "\n".join(lines)


def format_cos_search_output(pages: list[dict], content_preview_chars: int = 120) -> str:
    """
    Format a list of COS (CatOnSteroids) page items into a structured,
    human-readable output with hierarchical TOC information and
    truncated content previews.

    Args:
        pages (list[dict]): List of dictionaries with keys:
            - page_number (int)
            - sections (list[list[int, str, int]])
            - content (str)
        content_preview_chars (int): Number of characters to show in content preview.

    Returns:
        str: Nicely formatted string summary.
    """

    lines = []
    total_pages = len(pages)
    lines.append(f"Found **{total_pages}** matching pages.\n")
    #     f"Page {m['page_number']}(section level,title,page number): {m['toc_details']}\nContent:{m["page_content"][:100]}..."
    for page in pages:
        page_num = page.get("page_number", "Unknown")
        sections = page.get("toc_details", [])
        content = (page.get("page_content") or "").strip()

        lines.append(f"\n📄 **Page {page_num}**")

        if sections:
            # Group subsections under their respective hierarchy levels
            prev_level = 0
            for idx, (level, title, pnum) in enumerate(sections):
                indent = "    " * (level - 1)
                lines.append(f"{indent}- {title}  (p.{pnum})")

                # Look ahead to count truncated subsections (if next items are deeper)
                next_idx = idx + 1
                if next_idx < len(sections):
                    next_level = sections[next_idx][0]
                    if next_level > level:
                        deeper = sum(1 for lvl, _, _ in sections[next_idx:] if lvl > level)
                        lines.append(f"{indent}    ... {deeper} subsections truncated")

        else:
            lines.append("  - [No section hierarchy found]")

        # Add truncated content preview
        preview = shorten(content.replace("\n", " "), width=content_preview_chars, placeholder="...")
        lines.append(f"  🧩 Preview: {preview}")

    lines.append("\n*To see full content, call `CatOnSteroidsAction` with `search_level=2`.*")

    return "\n".join(lines)


TOOL_DESCRIPTION_V0 = """Powerful tool for navigating and searching large engineering or scientific reference documents (PDFs, datasheets, textbooks, research reports, etc.). 
Designed as an intelligent extension of the Unix `cat` command.

* Converts documents into structured, metadata-rich pages for reliable AI consumption.
* **VIEW MODE:** Retrieve overall document structure (Table of Contents, page count, metadata).
* **SEARCH MODE:** Find keywords or regex patterns and return structured page-level data.
* **PAGE RETRIEVAL:** Directly fetch specific pages or ranges without performing a search. 
    * Accepts single pages (e.g., 5), lists (e.g., [1, 3, 7]), or ranges (e.g., '2-5'). 
    * Mixed inputs are supported, such as ['1-3', 5, 8].
    * Ignored if a search_pattern is provided.
* **Search Levels:**
    * Level 1 (default): Returns a concise summary of page and section metadata where matches occur — ideal for quick scoping.
    * Level 2: Returns the full, rich dictionary data for each matched page — ideal for reasoning or downstream AI processing.
* Use this tool to efficiently extract precise information (e.g., register definitions, thermal limits, or circuit parameters) without scanning entire documents manually.
"""

TOOL_DESCRIPTION = """Powerful tool for navigating and searching large engineering or scientific reference documents (PDFs, datasheets, textbooks, research reports, etc.). 
Designed as an intelligent extension of the Unix `cat` command.
* Converts documents into structured, metadata-rich pages for reliable AI consumption.
* **VIEW MODE:** Retrieve overall document structure (Table of Contents, page count, metadata).
* **SEARCH MODE:** Find keywords or regex patterns and return structured page-level data. **Search is CASE SENSITIVE**
* **PAGE RETRIEVAL:** Directly fetch specific pages or ranges without performing a search. 
    * Accepts single pages (e.g., 5), lists (e.g., [1, 3, 7]), or ranges (e.g., '2-5'). 
    * Mixed inputs are supported, such as ['1-3', 5, 8].
    * Ignored if a search_pattern is provided.
* **Search Levels:**
    * Level 1 (default): Returns a concise summary of page and section metadata where matches occur — ideal for quick scoping.
    * Level 2: Returns the full, rich dictionary data for each matched page — ideal for reasoning or downstream AI processing.
* Use this tool to efficiently extract precise information (e.g., register definitions, thermal limits, or circuit parameters) without scanning entire documents manually.

**CRITICAL PAGE LIMIT RESTRICTIONS:**
⚠️ NEVER request results that span more than 10 pages in a single invocation. This causes context overload and system failure.
* **PAGE RETRIEVAL MODE:** Request maximum 10 pages total (e.g., pages [1,2,3,4,5,6,7,8,9,10] or range '1-10' is acceptable; '1-15' or [1,5,8,12,15,18,20,22,25,30] is NOT).
* **SEARCH MODE:** If search results exceed 10 pages:
    - Use Level 1 first to scope matches
    - Narrow your search pattern to be more specific
    - Split into multiple targeted queries focusing on different sections/keywords
    - Retrieve only the most relevant 10 pages maximum per call
* **Best Practice:** Always start with VIEW MODE or Level 1 search to understand document structure, then make focused queries for specific pages.
* **If you need information across >10 pages:** Make multiple sequential tool calls, each retrieving ≤10 pages.
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
