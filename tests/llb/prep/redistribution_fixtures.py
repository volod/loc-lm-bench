"""The committed acquired projection, staged for redistribution-gate tests.

`samples/corpora/acquired_projection_v1/` plants exactly one document whose `licence` is
`local-only`. Every egress-path test stages that fixture through real ingestion, so the licence
reaches each path by the field path production uses (sidecar -> manifest -> chunk metadata or
bundle provenance) rather than by a hand-written dict. The CONTROL stages the same fixture with
that one sidecar rewritten to `redistributable`: when the control passes and the refusal fails,
the recorded licence -- and nothing else -- is what refused the export.
"""

import json
import shutil
from pathlib import Path

from llb.goldset.schema import GoldItem, SourceSpan
from llb.prep.corpus.ingest import ingest_corpus
from llb.prep.corpus.redistribution import REDISTRIBUTABLE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = PROJECT_ROOT / "samples" / "corpora" / "acquired_projection_v1"
LOCAL_ONLY_DOC = "fixture-doc-07.md"
PERMITTED_DOC = "fixture-doc-08.md"
LOCAL_ONLY_PHRASE = "may not be\nredistributed"
PERMITTED_PHRASE = "transfer to the archive every April"


def staged_projection(tmp_path: Path, *, permit_local_only: bool = False) -> Path:
    """Ingest the committed projection into `tmp_path`; optionally lift the one refusal."""
    source = tmp_path / "projection"
    shutil.copytree(FIXTURE_ROOT, source)
    if permit_local_only:
        sidecar = source / f"{LOCAL_ONLY_DOC}.metadata.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["licence"] = REDISTRIBUTABLE
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    ingest_corpus(source, corpus, min_chars=1)
    return corpus


def span_item(
    corpus: Path, doc_id: str, phrase: str, *, item_id: str, split: str = "tuning"
) -> GoldItem:
    """A verified gold item whose one span is `phrase` inside the staged `doc_id`."""
    text = (corpus / doc_id).read_text(encoding="utf-8")
    start = text.index(phrase)
    return GoldItem(
        id=item_id,
        question=f"What does {doc_id} say?",
        reference_answer=phrase,
        source_doc_id=doc_id,
        source_spans=[
            SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(phrase), text=phrase)
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def fixture_items(corpus: Path, *, split: str = "tuning") -> list[GoldItem]:
    """One item on the local-only document and one on a redistributable neighbour."""
    return [
        span_item(corpus, LOCAL_ONLY_DOC, LOCAL_ONLY_PHRASE, item_id="local-only-1", split=split),
        span_item(corpus, PERMITTED_DOC, PERMITTED_PHRASE, item_id="permitted-1", split=split),
    ]
