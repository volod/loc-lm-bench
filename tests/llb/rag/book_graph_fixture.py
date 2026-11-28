"""Planted questions over the recovered synthetic book, with a CPU-only embedder."""

from pathlib import Path

import numpy as np

from llb.goldset.schema import GoldItem, dump_goldset
from llb.prep.book.project.build import project_corpus
from llb.prep.book.structure.build import build_structure
from tests.llb.prep.book.synthetic_book import begin, make_registry, write_scan
from tests.llb.rag.book_fixture import source_span

CITING = "The ignition load is documented in Table 2-1."
GRID = "| Tank | 1200 |"
HEADING = "1.1 Orbit Window"
MDE = 0.25
CUTOFF = 8
RESAMPLES = 256
N_PER_SLICE = 6


class BookGraphEmbedder:
    """Favor query cues in prose and give the table grid no query affinity."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode_passages(self, texts):
        return np.array(
            [
                [
                    2.0 if "ignition" in text else float("|" not in text),
                    2.0 if HEADING in text else 0.0,
                ]
                for text in texts
            ],
            dtype=np.float32,
        )

    def encode_queries(self, texts):
        return np.array(
            [[1.0, 0.0] if "ignition" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def write_graph_inputs(root: Path) -> tuple[Path, Path, Path]:
    scan = write_scan(root / "library" / "scan.pdf")
    run = begin(scan, root / "book-ocr", make_registry(kill=False))
    build_structure(run)
    projection = root / "projection"
    project_corpus(run, projection, "redistributable", language="en", slug="scan")

    source = {}
    for path in projection.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for phrase in (CITING, GRID, HEADING):
            if phrase in text:
                source[phrase] = source_span(text, path.name, phrase)
    goldset = root / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id=f"{kind}-{i}",
                lang="en",
                question=f"{question} ({i})",
                reference_answer=answer,
                source_doc_id=source[phrases[0]].doc_id,
                source_spans=[source[phrase] for phrase in phrases],
                provenance="sample-generated",
                split="final",
                verified=True,
            )
            for kind, question, answer, phrases in (
                ("table", "What ignition load is documented?", "1200", (CITING, GRID)),
                ("text", "What is the Orbit Window section called?", HEADING, (HEADING,)),
            )
            for i in range(1, N_PER_SLICE + 1)
        ],
        goldset,
    )
    config = root / "config.yaml"
    config.write_text(
        f"data_dir: {root / 'stores'}\ncorpus_root: {projection}\n"
        "strategy: fixed\nchunk_size: 40\nchunk_overlap: 0\n"
        "embedding_model: planted-book-graph\ngraph_weight: 0.75\n",
        encoding="utf-8",
    )
    return projection, goldset, config
