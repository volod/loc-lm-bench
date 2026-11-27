"""The figure captions the vault prints, read back for the resolved score.

The vault associates a caption with a drawing by the number the page prints, then by an
unambiguous nearest caption, and never by reading order alone. The score asks the same
selection, across every resolved page because a repeated ornament is a cross-page call, so
a caption the vault left unattached is not credited and one it attached is.
"""

from collections.abc import Sequence

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.contracts.data_prep.book_resolve import PageResolvedText
from llb.core.contracts.data_prep.book_structure import DocumentStructure


def emitted_captions(
    pages: Sequence[tuple[PageBlockLedger, PageResolvedText]],
    structure: DocumentStructure | None,
) -> dict[str, str]:
    """Figure block id to the caption block id the vault prints beneath it."""
    # Imported here: the asset selection and the render block import the renderer, which
    # the metrics package must not load at import time.
    from llb.prep.book.assets.extract import caption_links
    from llb.prep.book.render.types import RenderBlock

    blocks: list[RenderBlock] = []
    for ledger, resolved in pages:
        by_id = {block.block_id: block for block in resolved.blocks}
        for block in ledger.blocks:
            adopted = by_id.get(block.block_id)
            if adopted is None:
                continue
            blocks.append(
                RenderBlock(
                    pdf_page=block.pdf_page,
                    page_id=ledger.page_id,
                    ordinal=block.ordinal,
                    reading_order=block.reading_order,
                    kind=block.kind,
                    role=block.kind,
                    block_id=block.block_id,
                    digest=adopted.anchor,
                    text=adopted.text,
                    region=block.region,
                    resolved=adopted,
                )
            )
    inventory = structure.inventory if structure is not None else []
    return caption_links(blocks, inventory)
