"""The anchor that keeps an edit on the block it was made for.

Block ids follow geometry, so a rebuilt ledger can hand an id to a different region. The anchor
is the digest of the region and the raw readings stored on the block. An edit applies only when
both the id and this digest still match.
"""

from llb.core.contracts.data_prep.book_blocks import Block
from llb.prep.book.cache import digest_json


def block_anchor(block: Block) -> str:
    """SHA-256 of the block's region and raw readings, in the order the ledger stores them."""
    return digest_json(
        {
            "region": block.region.model_dump(mode="json"),
            "readings": [reading.model_dump(mode="json") for reading in block.readings],
        }
    )
