"""Shared fixtures and fakes for the corpus-conflict tests.

The committed fixture corpus at `samples/corpora/conflicts_uk_v1/` plants one instance of every
relation the tiers are supposed to find, so each tier can be asserted against a known answer:

  regulation-2021.md / regulation-2021-copy.md        byte-identical      -> hash, raw
  regulation-2021.md / regulation-2021-reformatted.md same after tokens   -> hash, normalized
  e-appeals-note.md  inside regulation-2024.md        contained whole     -> lexical, containment
  regulation-2021.md / regulation-2024.md Section 2   thirty vs fifteen   -> claim, superseded_by
  regulation-2021.md / regulation-2024.md Section 1,3 restated unchanged  -> claim, duplicate
  deadline-note.md   / regulation-2024.md Section 2   vague vs specific   -> claim, subsumed_by
  archive-policy.md                                   unrelated           -> no finding

The embedder is the project's hashed bag-of-words fake, so "semantically close" means "high token
overlap". That makes the vague-versus-specific pair sit near cosine 0.86 rather than the 0.9+ a
real multilingual encoder would give it, which is why the tests pass an explicit lower threshold
instead of the production default -- the tier logic is what is under test here, not the encoder.
"""

import zlib
from pathlib import Path

import pytest

from llb.conflicts.store_access import StoreView
from llb.conflicts.vectorops import VectorSet
from llb.core.paths import PROJECT_ROOT
from llb.rag.chunking.corpus import chunk_corpus

FIXTURE_CORPUS = PROJECT_ROOT / "samples" / "corpora" / "conflicts_uk_v1" / "corpus"

# Calibrated for the hashed-BoW fake (see the module docstring).
FAKE_COS_THRESHOLD = 0.85
DIM = 64

DOC_2021 = "regulation-2021.md"
DOC_2021_COPY = "regulation-2021-copy.md"
DOC_2021_REFORMATTED = "regulation-2021-reformatted.md"
DOC_2024 = "regulation-2024.md"
DOC_EAPPEALS = "e-appeals-note.md"
DOC_DEADLINE = "deadline-note.md"
DOC_ARCHIVE = "archive-policy.md"


def bow_vector(text: str) -> list[float]:
    """Deterministic hashed bag-of-words unit vector (the curation/refresh test pattern)."""
    vector = [0.0] * DIM
    for token in text.casefold().split():
        vector[zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
    return vector


def chunk_fixture(corpus_root: Path | str = FIXTURE_CORPUS, size: int = 600):
    """Chunk a corpus by heading, so each numbered section is its own claim-bearing chunk."""
    return chunk_corpus(Path(corpus_root), "heading", size, 0)


def fake_store_view(corpus_root: Path | str = FIXTURE_CORPUS, size: int = 600) -> StoreView:
    """A StoreView over real chunk records and fake BoW vectors -- no FAISS, no encoder."""
    chunks = chunk_fixture(corpus_root, size)
    vectors = VectorSet([bow_vector(chunk["text"]) for chunk in chunks])
    return StoreView(
        index_dir=Path(corpus_root),
        chunks=chunks,
        vectors=vectors,
        meta={"embedding_model": "fake-hashed-bow", "corpus_fingerprint": "fixture"},
    )


def ordinal_of(chunks, doc_id: str, needle: str) -> int:
    """The ordinal of `doc_id`'s chunk containing `needle` (fails loudly when absent)."""
    for ordinal, chunk in enumerate(chunks):
        if chunk["doc_id"] == doc_id and needle in chunk["text"]:
            return ordinal
    raise AssertionError(f"no chunk of {doc_id} contains {needle!r}")


def relation_for(findings, doc_a: str, doc_b: str) -> set[str]:
    """Every relation reported for the given unordered document pair."""
    wanted = tuple(sorted([doc_a, doc_b]))
    return {finding.relation for finding in findings if finding.doc_pair() == wanted}


def scripted_completer(script):
    """A fake `LLMComplete` returning the next scripted completion per call.

    `script` maps a substring that must appear in the prompt to the JSON completion to return, so
    a test states "when the model is shown the fifteen-days passage, it answers contradicts"
    without depending on adjudication order.
    """

    def complete(prompt: str) -> str:
        for marker, response in script.items():
            if marker in prompt:
                return response
        return '{"relation": "complementary", "confidence": 0.1, "claim_a": "", "claim_b": ""}'

    return complete


def probe_aware(base, *, correct: bool = True):
    """Wrap an adjudicator so the frozen calibration probe is answered from its own labels.

    `correct=True` is an adjudicator that is calibrated by construction; `correct=False` calls
    every complementary probe pair a duplicate, which is the failure mode the gate exists to
    catch -- it inflates precision on a corpus with nothing to find.
    """
    from llb.conflicts.claim_calibration import load_calibration_probe

    labels = {
        (pair.left_text, pair.right_text): pair.relation
        if correct or pair.actionable
        else "duplicate"
        for pair in load_calibration_probe().pairs
    }

    def complete(prompt: str) -> str:
        # The fixture documents restate each other verbatim, so the two passages must be read out
        # of the prompt's own delimiters rather than searched for -- a substring probe binds the
        # wrong pair whenever two sections carry identical text.
        quoted = prompt.split('"""')
        passages = (quoted[1].strip(), quoted[3].strip()) if len(quoted) > 4 else ("", "")
        if passages not in labels:
            return base(prompt)
        return (
            f'{{"relation": "{labels[passages]}", "confidence": 0.9,'
            ' "claim_a": "", "claim_b": "", "rationale": "probe"}'
        )

    return complete


def adjudicated_rows(flags, *, left_keys=None, right_keys=None, parsed=None):
    """Claim-tier rows carrying the given actionable flags, ranked highest cosine first."""
    from llb.conflicts.claim_precision import AdjudicatedRow
    from llb.conflicts.constants import REL_COMPLEMENTARY, REL_DUPLICATE

    left_keys = left_keys or [f"left#{index}" for index in range(len(flags))]
    right_keys = right_keys or [f"right#{index}" for index in range(len(flags))]
    parsed = [True] * len(flags) if parsed is None else parsed
    return [
        AdjudicatedRow(
            rank=index + 1,
            left_key=left_keys[index],
            right_key=right_keys[index],
            score=1.0 - index / 1000,
            relation=(REL_DUPLICATE if flag else REL_COMPLEMENTARY) if ok else None,
            parsed=ok,
        )
        for index, (flag, ok) in enumerate(zip(flags, parsed))
    ]


def calibrated_stub(accuracy_lcb: float = 0.9) -> dict:
    """A calibration payload that clears the gate, so precision tests isolate the precision."""
    from llb.conflicts.claim_calibration import MIN_ADJUDICATOR_ACCURACY_LCB

    return {
        "probe_id": "test",
        "parsed_pairs": 24,
        "agreements": 23,
        "accuracy": 0.958333,
        "accuracy_wilson_95": [accuracy_lcb, 1.0],
        "min_accuracy_lcb": MIN_ADJUDICATOR_ACCURACY_LCB,
        "calibrated": True,
    }


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS
