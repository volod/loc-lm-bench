"""A planted two-lane book with six missing and six surviving evidence spans."""

import re
from pathlib import Path

import fitz
import numpy as np

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.prep.book.project.spans import Fragment, assemble
from llb.prep.book.project.write import ProjectedDocument, write_document
from llb.prep.book.source import sha256_file

N_ITEMS = 12
N_MISSING = N_ITEMS // 2
DECOY_PAGE = N_ITEMS + 1
MDE = 0.25
MARKER = "{{llb-marker:unresolved:p0013-b0000:0-1}}"
_QUERY = re.compile(r"\bquery([0-9]{2})\b")


class PlantedEmbedder:
    """One vector dimension per planted locator, shared by both stores."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode_passages(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _encode(self, texts):
        vectors = np.zeros((len(texts), N_ITEMS), dtype=np.float32)
        for row, text in enumerate(texts):
            for match in _QUERY.finditer(text):
                vectors[row, int(match[1]) - 1] = 1
        return vectors


def query(page: int) -> str:
    return f"query{page:02}"


def evidence(page: int) -> str:
    return f"evidence{page:02} 42"


def write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    """Generate a synthetic PDF, plant a projection, and label its source spans."""
    pdf = _write_pdf(root / "pdf" / "planted.pdf")
    projection = root / "projection"
    doc_id = "projected.md"
    fragments = [
        _fragment(page, f"{query(page)} {evidence(page)}" if page <= N_MISSING else evidence(page))
        for page in range(1, N_ITEMS + 1)
    ]
    fragments.append(
        _fragment(DECOY_PAGE, " ".join(query(page) for page in range(N_MISSING + 1, DECOY_PAGE)))
    )
    fragments.append(_fragment(DECOY_PAGE, MARKER))
    rendered = assemble(fragments)
    projected = ProjectedDocument("planted", rendered.text, rendered.pages, "fixture", "root")
    projection.mkdir(parents=True)
    write_document(
        projection,
        projected,
        doc_id=doc_id,
        revision=None,
        licence="redistributable",
        language="en",
        source="another-path/planted.pdf",
        digest=sha256_file(pdf),
        run_id="planted",
        captured="2000-01-01T00:00:00Z",
        references="{}",
    )
    gold = root / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id=f"item-{page:02}",
                question=query(page),
                reference_answer="42",
                source_doc_id=doc_id,
                source_spans=[source_span(rendered.text, doc_id, evidence(page))],
                provenance="sample-generated",
                verified=True,
                split="final",
                lang="en",
            )
            for page in range(1, N_ITEMS + 1)
        ],
        gold,
    )
    config = root / "config.yaml"
    config.write_text(
        f"data_dir: {root}\nstrategy: page\nchunk_size: 256\nchunk_overlap: 0\n"
        "embedding_model: planted-locator\n",
        encoding="utf-8",
    )
    return pdf.parent, projection, gold, config


def source_span(text: str, doc_id: str, label: str) -> SourceSpan:
    start = text.index(label)
    return SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(label), text=label)


def _fragment(page: int, text: str) -> Fragment:
    return Fragment(page, text, f"p{page:04}-b0000", None, "anchor", "fixture.md", "\n\n")


def _write_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    with fitz.open() as document:
        for page in range(1, N_ITEMS + 1):
            text = f"{query(page)} {evidence(page)}"
            if page <= N_MISSING:
                text += "0"
            document.new_page().insert_text((72, 100), text)
        document.new_page().insert_text((72, 100), evidence(1))
        document.save(path)
    return path
