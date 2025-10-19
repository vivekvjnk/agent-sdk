from openhands.sdk.tool import ToolExecutor

from .definition import CatOnSteroidsAction, CatOnSteroidsObservation
from .preprocessor import DocumentPreprocessor, PageDict  # Assuming a path


# Dictionary to cache parsed documents (performance optimization)
_DOC_CACHE: dict[str, DocumentPreprocessor] = {}


class CatOnSteroidsExecutor(
    ToolExecutor[CatOnSteroidsAction, CatOnSteroidsObservation]
):
    # No hierarchical tools needed for core logic, but we can pass a dummy executor
    # if required by the OpenHands framework structure.
    # We use a placeholder here as the logic is fully internal Python.
    # def __init__(self, bash_executor: BashExecutor):
    #     pass

    def _get_preprocessor(self, doc_path: str) -> DocumentPreprocessor:
        """Handles document loading and caching."""
        if doc_path not in _DOC_CACHE:
            _DOC_CACHE[doc_path] = DocumentPreprocessor(doc_path)
        return _DOC_CACHE[doc_path]

    def _run_search(
        self, preprocessor: DocumentPreprocessor, pattern: str, is_regex: bool
    ) -> list[PageDict]:
        """Core FOS logic: Search full_text and map results back to page dictionaries."""
        import re

        if not is_regex:
            # Convert simple keyword to a case-insensitive, word boundary regex
            pattern = r"\b" + re.escape(pattern) + r"\b"

        # Search the full text document
        full_text = preprocessor.full_text
        matches: list[PageDict] = []
        matched_pages: set = set()

        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            # Map the match index back to the structured PageDict
            page_dict = preprocessor.map_index_to_page_dict(match.start())

            # Use the page_num as a unique identifier to ensure we only include each page once
            if page_dict["page_num"] not in matched_pages:
                matches.append(page_dict)
                matched_pages.add(page_dict["page_num"])

        return matches

    def __call__(self, action: CatOnSteroidsAction) -> CatOnSteroidsObservation:
        try:
            preprocessor = self._get_preprocessor(action.doc_path)
        except Exception as e:
            return CatOnSteroidsObservation(
                total_results=0, metadata_summary=[f"Error loading document: {e}"]
            )

        # --- Search Logic (FOS) ---
        if action.search_pattern:
            all_matches = self._run_search(
                preprocessor, action.search_pattern, action.is_regex
            )

            total_count = len(all_matches)

            # Level 1: Metadata Summary
            if action.search_level == 1:
                metadata_summary = [
                    f"Page {m['page_num']}: {m.get('section_title', 'N/A')}"
                    for m in all_matches
                ]
                return CatOnSteroidsObservation(
                    total_results=total_count, metadata_summary=metadata_summary
                )

            # Level 2: Complete Content
            elif action.search_level == 2:
                # Apply N limit (-1 means all)
                limit = action.n_results if action.n_results != -1 else total_count
                content_results = all_matches[:limit]

                return CatOnSteroidsObservation(
                    total_results=total_count, content_results=content_results
                )
            else:
                return CatOnSteroidsAction(
                    total_results=0, content_results="arcion.search_level is missing!!"
                )

        # --- Viewing Logic (COS) ---
        else:
            # Assume no search pattern means the user wants the global document metadata (ToC)
            doc_metadata = {
                "total_pages": len(preprocessor.parsed_pages),
                "sections": preprocessor.parsed_pages[0].get(
                    "section_title", "N/A"
                ),  # Example: Return first page section list
                "full_text_length": len(preprocessor.full_text),
            }
            # For a true COS view, you might want to return the ToC structure explicitly here

            return CatOnSteroidsObservation(doc_metadata=doc_metadata)
