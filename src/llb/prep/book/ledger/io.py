"""Read and publish `pages/<pNNNN>/blocks.json`.

The `blocks` stage commits its copy inside `pages/<pNNNN>/blocks/`, which is what the stage cache
hashes. The page-level file is that copy under the path every later stage is told to open, written
only after the stage commit succeeds.
"""

import json
from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.fsutil import atomic_write_text
from llb.prep.book.cache import page_dir
from llb.prep.book.constants import BLOCKS_FILENAME


def ledger_path(run_dir: Path, page_id: str) -> Path:
    """The published block ledger of one page."""
    return page_dir(run_dir, page_id) / BLOCKS_FILENAME


def write_ledger(path: Path, ledger: PageBlockLedger) -> None:
    """Validate by construction (the caller passes the model) and write it atomically."""
    atomic_write_text(path, _json(ledger))


def _json(ledger: PageBlockLedger) -> str:
    payload = ledger.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"


def read_ledger(path: Path) -> PageBlockLedger:
    """Read a block ledger, refusing a file that is not the current contract."""
    return PageBlockLedger.model_validate_json(path.read_text(encoding="utf-8"))


def publish_page_ledger(stage_directory: Path) -> None:
    """Copy a committed `blocks` stage's ledger up to `pages/<pNNNN>/blocks.json`."""
    source = stage_directory / BLOCKS_FILENAME
    if not source.is_file():
        return
    target = stage_directory.parent / BLOCKS_FILENAME
    text = source.read_text(encoding="utf-8")
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return
    # Re-validate so a hand-edited stage file cannot publish a ledger the model refuses.
    PageBlockLedger.model_validate(json.loads(text))
    atomic_write_text(target, text)
