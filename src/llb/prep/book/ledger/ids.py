"""Block ids are a function of the PDF page and the in-page ordinal, nothing else.

`page_id` already encodes the PDF page (and the document's page-count width, so every id in one
book has the same shape). The ordinal is the block's place in reading order after a stable sort,
so engine completion time never enters the id.
"""


def block_id(page_id: str, ordinal: int) -> str:
    """`p0007-b0003` for ordinal 3 on the page whose directory is `p0007`."""
    if ordinal < 0:
        raise ValueError(f"block ordinal must be >= 0, got {ordinal}")
    return f"{page_id}-b{ordinal:04d}"
