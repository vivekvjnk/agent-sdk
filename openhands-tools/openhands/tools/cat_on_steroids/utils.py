from pydantic import BaseModel, Field
from typing import Optional, Union

Number = Union[int, float]

class PageDict(BaseModel):
    """
    A flattened, JSON-safe representation of a page.
    All fields are strictly: None, string, or number.
    """

    # Page information
    page_number: Optional[int] = Field(default=None)

    # Flattened page indices (instead of {"start_index": ..., "end_index": ...})
    start_index: Optional[int] = Field(default=None)
    end_index: Optional[int] = Field(default=None)

    # Page content as plain text
    page_content: Optional[str] = Field(default=None)

    # Page blocks (also plain text)
    page_blocks: Optional[str] = Field(default=None)

    # Optional metadata entry (string or None)
    toc_details: Optional[str] = Field(default=None)

    model_config = {
        "extra": "ignore",  # ignore unexpected fields
        "strict": True,     # prevents bytes, dicts, lists, etc.
    }