"""Intra-document repeated blocks: the census, both handling modes, and what follows the rewrite.

Pure unit tests over the committed `samples/corpora/intra_document_repeats_uk_v1/` fixture (one
converted-PDF-shaped manual repeating its own boilerplate, plus a second document sharing one
block) and hand-built documents: no PDF parser, no embedder, no GPU.
"""

from pathlib import Path

from typer.testing import CliRunner

from llb.goldset.schema import (
    GoldItem,
    SourceSpan,
)

RUNNER = CliRunner()

FIXTURE = Path("samples/corpora/intra_document_repeats_uk_v1/corpus")

REPEATED_DOC = "nastanova-oblik.md"

FIXTURE_BLOCKS, FIXTURE_GROUPS, FIXTURE_LARGEST = 18, 2, 3

FIXTURE_DROPPED_BLOCKS, FIXTURE_ANCHORED_BLOCKS = 4, 6

FIXTURE_CHARS, FIXTURE_DROP_CHARS, FIXTURE_ANCHOR_CHARS = 1957, 1440, 2137

FIXTURE_INTRA_GROUPS, FIXTURE_CROSS_GROUPS = 2, 1

PROCEDURE = "Порядок збереження документа"

SUPPORT = "**Служба підтримки:**"


def fixture_text() -> str:
    return (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")


def _corpus_docs(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*.md"))]


def _item(item_id: str, text: str, start: int, length: int) -> GoldItem:
    return GoldItem(
        id=item_id,
        question="Як зберегти документ?",
        reference_answer="Натисніть кнопку Зберегти.",
        source_doc_id=REPEATED_DOC,
        source_spans=[
            SourceSpan(
                doc_id=REPEATED_DOC,
                char_start=start,
                char_end=start + length,
                text=text[start : start + length],
            )
        ],
        provenance="human-authored",
        verified=True,
        split="final",
    )
