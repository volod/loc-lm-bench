"""A reused ledger keeps the page it was built for.

Two identical rasters share a recognition cache key. The ledger names the page in its block ids,
so copying one page's ledger onto the other would attach those ids to the wrong page.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from tests.llb.prep.book._scan_factory import make_duplicate_scan


def test_reuse_keeps_each_duplicate_page_its_own_block_ids(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    scan = make_duplicate_scan(library / "scan.pdf")
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake"}),
        BookStageSpec(name="blocks"),
    )
    registry = StageRegistry((RenderStage(), *recognition_stages(AdapterRegistry((FakeEngine(),)))))
    out = tmp_path / "runs"
    first = start_run(
        RunRequest(source=scan, pages="1-2", batch_pages=2, stages=stages), out, registry
    )
    second = start_run(
        RunRequest(
            source=scan, pages="1-2", batch_pages=2, stages=stages, reuse_from=first.run_dir
        ),
        out,
        registry,
    )
    readings = []
    for page_id, pdf_page in (("p0001", 1), ("p0002", 2)):
        ledger = read_ledger(second.run_dir / "pages" / page_id / "blocks.json")
        assert ledger.page_id == page_id
        assert ledger.pdf_page == pdf_page
        assert ledger.blocks
        assert all(block.block_id.startswith(f"{page_id}-") for block in ledger.blocks)
        readings.append([reading.text for block in ledger.blocks for reading in block.readings])
    assert readings[0] == readings[1]
