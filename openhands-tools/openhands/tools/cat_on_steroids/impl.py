import logging
import re

from openhands.sdk.logger import DEBUG, get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.cat_on_steroids.definition import (
    CatOnSteroidsAction,
    CatOnSteroidsObservation,
)
from openhands.tools.cat_on_steroids.preprocessor import (
    DocumentPreprocessor,
)  # Assuming a path
from openhands.tools.cat_on_steroids.utils import (
    PageDict,
    validate_and_expand_pages_json_only,
)


# Suppress browser-use logging for cleaner integration
if DEBUG:
    logging.getLogger("browser_use").setLevel(logging.DEBUG)
else:
    logging.getLogger("browser_use").setLevel(logging.WARNING)

logger = get_logger(__name__)

# Dictionary to cache parsed documents (performance optimization)
_DOC_CACHE: dict[str, DocumentPreprocessor] = {}


class CatOnSteroidsExecutor(
    ToolExecutor[CatOnSteroidsAction, CatOnSteroidsObservation]
):
    def _get_preprocessor(self, doc_path: str) -> DocumentPreprocessor:
        """Handles document loading and caching."""
        if doc_path not in _DOC_CACHE:
            _DOC_CACHE[doc_path] = DocumentPreprocessor(doc_path)
        return _DOC_CACHE[doc_path]

    def _run_search(
        self, preprocessor: DocumentPreprocessor, pattern: str
    ) -> list[PageDict]:
        """Core FOS logic: case-sensitive regex-only search.
        If the provided pattern contains no regex-special characters, treat it
        as a simple word/phrase and automatically escape it and add word
        boundaries so common simple searches behave as expected.
        """

        # Characters that indicate the user likely provided a regex
        _REGEX_SPECIALS = set(r".^$*+?{}[]\|()")

        # If there are no regex-special characters, escape and wrap with \b
        if not any(ch in _REGEX_SPECIALS for ch in pattern):
            pattern = r"\b" + re.escape(pattern) + r"\b"

        # Compile as case-sensitive regex (no re.IGNORECASE)
        try:
            prog = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        full_text = preprocessor.full_text
        matches: list[PageDict] = []
        matched_pages: set[int] = set()

        for match in prog.finditer(full_text):
            # Map the match back to the structured PageDict.
            # Keep original behavior of map_index_to_page; it may accept a match object.
            page_dict = preprocessor.map_index_to_page(match)[0]

            # Ensure each page is included only once
            if page_dict["page_number"] not in matched_pages:
                matches.append(page_dict)
                matched_pages.add(page_dict["page_number"])

        return matches

    def _retrieve_pages(
        self, preprocessor: DocumentPreprocessor, pages: str
    ) -> list[PageDict]:
        """
        Retrieve one or more pages (by number or range) from the parsed document.

        Args:
            preprocessor (DocumentPreprocessor): The preprocessed document object.
            pages (str): A string representing pages or ranges.
                Examples:
                    "5"              -> [5]
                    "1,3,5"          -> [1, 3, 5]
                    "2-4"            -> [2, 3, 4]
                    "1-3,5,7-8"      -> [1, 2, 3, 5, 7, 8]
                    '["1-10"]'       -> [1,2,...,10]   (JSON-like quoted range)
                    "[1,2,3]"        -> [1,2,3]        (JSON-like numbers)

        Returns:
            list[dict]: List of page contents for all requested pages.
        """
        # Split comma-separated list, expand each spec, flatten all results
        all_pages = validate_and_expand_pages_json_only(pages=pages)
        # Deduplicate and sort
        page_numbers = sorted(set(all_pages))

        # Retrieve content
        page_contents: list[PageDict] = [
            preprocessor.get_page(page_number=p) for p in page_numbers
        ]
        return page_contents

    def __call__(self, action: CatOnSteroidsAction, conv) -> CatOnSteroidsObservation:
        try:
            preprocessor = self._get_preprocessor(action.doc_path)
            logger.info(f"Document cache: {preprocessor}")
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise e
            return CatOnSteroidsObservation(
                total_results=0,
                metadata_summary=[{"error": f"Error loading document: {e}"}],
            )
        # --- Page retrieval logic ---
        if action.pages:
            try:
                pages = self._retrieve_pages(
                    preprocessor=preprocessor, pages=action.pages
                )
            except IndexError:
                return CatOnSteroidsObservation(
                    total_results=0,
                    content_results=[
                        {"error": "Page number doesn't exist in the document"}
                    ],
                )

            try:
                observation = CatOnSteroidsObservation(
                    total_results=len(pages), content_results=pages
                )
                # Validate serialization
                observation.model_dump()
                return observation
            except Exception as e:
                raise Exception(f"Error creating observation: {e}")

        # --- Search Logic (FOS) ---
        elif action.search_pattern:
            all_matches = self._run_search(
                preprocessor=preprocessor, pattern=action.search_pattern
            )

            total_count = len(all_matches)

            # Level 1: Metadata Summary
            if action.search_level == "surface":
                # metadata_summary = [
                #     f"Page {m['page_number']}(section level,title,page number): {m['toc_details']}\nContent:{m["page_content"][:100]}..."
                #     for m in all_matches
                # ]
                # return CatOnSteroidsObservation(
                #     total_results=total_count, metadata_summary=all_matches
                # )
                try:
                    observation = CatOnSteroidsObservation(
                        total_results=total_count, metadata_summary=all_matches
                    )
                    # Validate serialization
                    observation.model_dump()
                    return observation
                except Exception as e:
                    raise Exception(f"Error creating observation: {e}")

            # Level 2: Complete Content
            elif action.search_level == "deep":
                # Apply N limit (-1 means all)
                limit = action.n_results if action.n_results != -1 else total_count
                content_results = all_matches[:limit]

                try:
                    observation = CatOnSteroidsObservation(
                        total_results=total_count, content_results=content_results
                    )
                    # Validate serialization
                    observation.model_dump()
                    return observation
                except Exception as e:
                    raise Exception(f"Error creating observation: {e}")
            else:
                return CatOnSteroidsObservation(
                    total_results=0, content_results="action.search_level is missing!!"
                )

        # --- Viewing Logic (COS) ---
        else:
            # Assume no search pattern means the user wants the global document metadata (ToC)
            doc_metadata = {
                "total_pages": preprocessor.page_count,
                "sections": preprocessor.toc,
                "full_text_length": len(preprocessor.full_text),
                "pdf_metadata": preprocessor.doc_metadata,
            }
            # For a true COS view, you might want to return the ToC structure explicitly here

            return CatOnSteroidsObservation(doc_metadata=doc_metadata)
