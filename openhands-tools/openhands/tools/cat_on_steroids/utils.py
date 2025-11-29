import json
import re
from typing import Union

from pydantic import BaseModel, Field


Number = Union[int, float]


class PageDict(BaseModel):
    """
    A flattened, JSON-safe representation of a page.
    All fields are strictly: None, string, or number.
    """

    # Page information
    page_number: int | None = Field(default=None)

    # Flattened page indices (instead of {"start_index": ..., "end_index": ...})
    start_index: int | None = Field(default=None)
    end_index: int | None = Field(default=None)

    # Page content as plain text
    page_content: str | None = Field(default=None)

    # Optional metadata entry (string or None)
    toc_details: str | None = Field(default=None)

    model_config = {
        "extra": "ignore",  # ignore unexpected fields
        "strict": True,  # prevents bytes, dicts, lists, etc.
    }


# Regex for a single-range like "46-48" (allow spaces around '-')
_RANGE_SPLIT_RE = re.compile(r"\s*-\s*")


def _expand_token(token: object) -> list[int]:
    """
    Expand a single parsed JSON list element:
      - If token is int (or float representing int), return [int].
      - If token is str:
          - Accept a single integer string "46"
          - OR a single range "46-48" (spaces allowed around '-')
        Reject strings that contain commas or multiple ranges.
    Returns list of ints or raises ValueError for invalid tokens.
    """
    if isinstance(token, bool):
        raise ValueError("Invalid element type: boolean is not allowed")

    # Numbers (int or float convertible to int)
    if isinstance(token, (int, float)):
        try:
            n = int(token)
        except Exception:
            raise ValueError(f"Invalid numeric page element: {token!r}")
        return [n]

    # Strings
    if isinstance(token, str):
        s = token.strip()
        if not s:
            raise ValueError("Empty string element in JSON array is not allowed")

        # Disallow comma inside an element: require separate array entries instead.
        if "," in s:
            raise ValueError(
                f"Invalid array element: found comma inside string element {s!r}. "
                'Provide multiple elements in the array instead, e.g. ["1-3", "5"]'
            )

        # Range form?
        if "-" in s:
            parts = _RANGE_SPLIT_RE.split(s, maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range element: {s!r} (expected 'start-end')")
            try:
                start, end = map(int, parts)
            except ValueError:
                raise ValueError(f"Invalid range endpoints (must be integers): {s!r}")
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))

        # Single number string
        try:
            return [int(s)]
        except ValueError:
            raise ValueError(
                f"Invalid string element: {s!r} (expected integer or 'start-end')"
            )

    # All other types are invalid
    raise ValueError(f"Invalid array element type: {type(token).__name__!r}")


def validate_and_expand_pages_json_only_temp(pages: str) -> list[int]:
    """
    Strict JSON-only validator + expander.

    - `pages` must be a JSON array (string starting with '[').
    - Each element of the JSON array must be either:
        - an integer (e.g. 46), or
        - a string for a single range (e.g. "46-48") or a single integer as string ("46")
    - Comma-separated ranges inside a string element are NOT allowed; use multiple array elements.
    - Returns sorted unique list of page numbers (ints).

    Raises ValueError with a clear message on invalid input.
    """
    if pages is None:
        return []

    pages_raw = str(pages).strip()
    if not pages_raw:
        return []

    # Must be a JSON array (no outer quoting or other formats)
    if not pages_raw.startswith("["):
        raise ValueError(
            'pages parameter must be a JSON array. Example: ["1-3","5"] or [1, 11, 12, 27]. '
            "Do not add extra outer quotes."
        )

    try:
        parsed = json.loads(pages_raw)
    except json.JSONDecodeError as e:
        # Give a friendly, actionable message
        raise ValueError(
            f"Invalid JSON for pages parameter: \n Pages:{pages} \n Stripped content:{pages_raw}\n Provide a valid JSON array, "
            'for example: ["1-3","5"] or [1,11,12].'
        ) from e

    if not isinstance(parsed, list):
        raise ValueError('pages JSON must be an array (e.g. ["1-3","5"]).')

    all_pages = []
    for idx, item in enumerate(parsed):
        try:
            expanded = _expand_token(item)
            all_pages.extend(expanded)
        except ValueError as e:
            # Provide index context for easier debugging
            raise ValueError(
                f"Invalid element at index {idx} in pages array: {e}"
            ) from e

    # Deduplicate and sort
    page_numbers = sorted(set(all_pages))
    return page_numbers


def validate_and_expand_pages_json_only(pages: str) -> list[int]:
    """
    Strict JSON-only validator + expander.

    - `pages` may be a JSON array (string starting with '[') OR a single JSON value:
        - integer (e.g. 46),
        - a string for a single range (e.g. "46-48") or a single integer as string ("46")
    - Comma-separated ranges inside a string element are NOT allowed; use multiple array elements.
    - Returns sorted unique list of page numbers (ints).

    Raises ValueError with a clear message on invalid input.
    """
    if pages is None:
        return []

    pages_raw = str(pages).strip()
    if not pages_raw:
        return []

    # Try to parse as JSON. Accept either a JSON array or a single JSON scalar (number or string).
    try:
        parsed = json.loads(pages_raw)
    # except json.JSONDecodeError as e:
    #     raise ValueError(
    #         f"Invalid JSON for pages parameter: \n Pages:{pages} \n Stripped content:{pages_raw}\n "
    #         "Provide a valid JSON array (e.g. [\"1-3\",\"5\"]) or a single JSON value like 1 or \"1-3\"."
    #     ) from e
    except json.JSONDecodeError:
        # 2. Fallback: Handle "lazy" lists like "[14-19]" or simple strings "14-19"
        #    which fail JSON parsing because of missing quotes around the range.
        cleaned = pages_raw.strip()

        # Remove surrounding brackets if they exist
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]

        if not cleaned.strip():
            # Handle empty brackets "[]"
            parsed = []
        else:
            # Split by comma to create a list of strings
            # e.g., "14-19, 22" becomes ["14-19", " 22"]
            parsed = [item.strip() for item in cleaned.split(",")]

    # Handle the case where someone passed an extra-quoted JSON array, e.g. "\"[... ]\""
    if isinstance(parsed, str) and parsed.strip().startswith("["):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            raise ValueError(
                "Invalid inner JSON array in pages parameter. Provide a proper JSON array or a single JSON value."
            )

    if not isinstance(parsed, list):
        # Accept a single scalar JSON value by treating it as a one-element list.
        if isinstance(parsed, (int, float, str)):
            parsed = [parsed]
        else:
            raise ValueError(
                'pages JSON must be an array (e.g. ["1-3","5"]) or a single value (e.g. 1 or "1-3").'
            )

    all_pages = []
    for idx, item in enumerate(parsed):
        try:
            expanded = _expand_token(item)
            all_pages.extend(expanded)
        except ValueError as e:
            # Provide index context for easier debugging
            raise ValueError(
                f"Invalid element at index {idx} in pages array: {e}"
            ) from e

    # Deduplicate and sort
    page_numbers = sorted(set(all_pages))
    return page_numbers
