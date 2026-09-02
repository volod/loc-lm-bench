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

import json
import zlib
from pathlib import Path

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.bundle.record import (
    CANDIDATES_KEY,
    CHUNKS_KEY,
    DOC_ID_KEY,
    DOCUMENTS_KEY,
    EXCLUSIONS_KEY,
    SCHEMA_KEY,
    naming_of,
)
from llb.conflicts.bundle.candidate_record import ENTRIES_KEY
from llb.conflicts.constants import TIER_SEMANTIC
from llb.conflicts.bundle.document_affix import PREFIX_KEY, SUFFIX_KEY, IdAffix
from llb.conflicts.bundle.document_chunks import COUNT_KEYS as CHUNK_COUNT_KEYS
from llb.conflicts.bundle.document_exclusions import REASON_NAMES
from llb.conflicts.bundle.document_index import COUNT_DEFAULT_KEY, EXTRA_IDS_KEY, DocumentNaming
from llb.conflicts.models import AuditResult
from llb.conflicts.store_access import StoreView
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.paths import PROJECT_ROOT
from llb.rag.chunking.corpus import chunk_corpus

FIXTURE_CORPUS = PROJECT_ROOT / "samples" / "corpora" / "conflicts_uk_v1" / "corpus"

# The exclusion maps that are per-document counts, so the ones a recorded default can fold.
# `recovery_floor` is deliberately absent: it declines the fold (`document_exclusions.py`).
REASON_KEYS = tuple(REASON_NAMES)

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


# --- throwaway dated corpora (the stage attribution and its replay) ----------------------------

# Enough tokens to clear `MIN_CLAIM_TOKENS`, and few enough to chunk as one heading-sized unit.
BODY_TOKENS = 40


def dated_body(start: int, count: int = BODY_TOKENS) -> str:
    """A body drawn from one token window. Two disjoint windows are unrelated to cosine alike."""
    return " ".join(f"пункт{index:04d}" for index in range(start, start + count))


def dated_corpus(root: Path, documents: dict[str, tuple[str, str]]) -> Path:
    """`{name: (effective_date, body)}` written as a corpus of dated Markdown documents.

    An EMPTY date writes the body with no front matter at all, which is the undated document every
    corpus on this host is made of -- the case the record carries as a bare id.

    A name may carry directories, since a doc_id is a corpus-relative PATH and the shape of that
    path is what the id fold is priced on.
    """
    root.mkdir(parents=True, exist_ok=True)
    for name, (date, body) in documents.items():
        front = f"---\neffective_date: {date}\n---\n" if date else ""
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{front}{body}\n", encoding="utf-8")
    return root


def store_over(root: Path, *, without: str = "") -> StoreView:
    """A StoreView over the corpus's real chunks, optionally missing one document's entirely."""
    chunks = [chunk for chunk in chunk_fixture(root) if chunk["doc_id"] != without]
    return StoreView(
        index_dir=root,
        chunks=chunks,
        vectors=VectorSet([bow_vector(chunk["text"]) for chunk in chunks]),
        meta={"embedding_model": "fake-hashed-bow", "corpus_fingerprint": "stage"},
    )


# The corpora the bundle-record tests audit, and the one audit both test modules run over them.
# Shared because the record's two subjects -- what it ANSWERS (`test_bundle_record.py`) and what
# SHAPE it answers from (`test_bundle_id_table.py`) -- ask the same corpora different questions, and
# a second copy of a corpus is a second thing to keep in step.

SHORT_BODY = "коротка примітка"

# Two documents a store holds and the comparison drops, for the two DIFFERENT reasons the floor
# knob can and cannot move: one is under the floor, the other is nothing but front matter.
BELOW_THE_FLOOR = {
    "old.md": ("2021-01-01", dated_body(0)),
    "new.md": ("2024-01-01", SHORT_BODY),
}

# Three dated documents whose candidate ranking is a chain: `a` matches `b` strongly and `c` weakly,
# and `b` and `c` share nothing. Every pair is orderable, so which one the attribution names is
# decided purely by how deep into the ranking the budget reaches.
RANKED = {
    "a-first.md": ("2021-01-01", f"{dated_body(0, 30)} {dated_body(100, 10)}"),
    "b-second.md": ("2022-01-01", f"{dated_body(0, 30)} {dated_body(200, 10)}"),
    "c-third.md": ("2023-01-01", f"{dated_body(100, 10)} {dated_body(300, 30)}"),
}
# Between the b/c pair's cosine and the a/c pair's, so the ranking has a tail the budget cuts.
RANKED_COS = 0.89

# The same ranking with one document carrying no ordering field, which is what every corpus on this
# host looks like: the record holds a labelled entry and a bare id in the same table.
RANKED_WITH_UNDATED = {**RANKED, "d-undated.md": ("", dated_body(0, 30))}

# Nothing to order on anywhere -- the corpus the bare-id form is priced on.
UNDATED = {name: ("", body) for name, (_, body) in RANKED.items()}

# The two id SHAPES the fold is priced between, carrying the SAME documents either way so the id is
# the only thing that differs. `PATH_SHAPED` is what a real corpus looks like -- one directory and
# one file type, so every id repeats `docs/uk/` and `.txt`; `SCATTERED` is the corpus that shares
# nothing to fold, its documents in different directories under different extensions. The stems run
# 00..39 rather than 0000..0039 so they share no leading digit of their own, which would otherwise
# make even the scattered corpus foldable and stop it testing what it is here to test.
FOLD_BODIES = {f"{index:02d}": ("", dated_body(index * 5, 40)) for index in range(40)}
PATH_SHAPED = {f"docs/uk/{stem}.txt": value for stem, value in FOLD_BODIES.items()}
SCATTERED = {
    f"{chr(ord('a') + int(stem) % 20)}{stem}/doc{stem}.{'txt' if int(stem) % 2 else 'md'}": value
    for stem, value in FOLD_BODIES.items()
}


def audit_over(
    root: Path,
    documents: dict[str, tuple[str, str]],
    *,
    effort: str = TIER_SEMANTIC,
    cos_threshold: float = FAKE_COS_THRESHOLD,
    min_claim_tokens: int | None = None,
    candidate_budget: int | None = None,
    record_cap: int | None = None,
) -> AuditResult:
    """One real audit over a throwaway dated corpus, with a store only from the semantic tier up."""
    corpus = dated_corpus(root, documents)
    params = AuditParams(
        effort=effort,
        cos_threshold=cos_threshold,
        max_candidate_pairs=candidate_budget,
        max_candidate_record_pairs=record_cap,
    )
    if min_claim_tokens is not None:
        params.min_claim_tokens = min_claim_tokens
    store = store_over(corpus) if effort == TIER_SEMANTIC else None
    return run_audit(corpus, params, store=store)


def bundle_of(result: AuditResult) -> tuple[dict, list]:
    """The two files a replay reads, through JSON exactly as `write_audit` puts them on disk."""
    return (
        json.loads(json.dumps(result.summary(), ensure_ascii=False)),
        json.loads(json.dumps(result.rows(), ensure_ascii=False)),
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


def probe_aware(base, *, correct: bool = True, wrong_tiers: tuple[str, ...] = ()):
    """Wrap an adjudicator so the frozen calibration probe is answered from its own labels.

    `correct=True` is an adjudicator that is calibrated by construction; `correct=False` calls
    every complementary probe pair a duplicate, which is the failure mode the gate exists to
    catch -- it inflates precision on a corpus with nothing to find. `wrong_tiers` restricts that
    failure to the named tiers, which is how a model that holds the floor and loses the separator
    is written down.
    """
    from llb.conflicts.claim.probe import load_calibration_probe

    def answered(pair) -> str:
        wrong = not correct or pair.tier in wrong_tiers
        return pair.relation if pair.actionable or not wrong else "duplicate"

    labels = {
        (pair.left_text, pair.right_text): answered(pair) for pair in load_calibration_probe().pairs
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
    from llb.conflicts.claim.precision import AdjudicatedRow
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
    from llb.conflicts.claim.calibration import MIN_ADJUDICATOR_ACCURACY_LCB
    from llb.conflicts.claim.probe import BASE_TIER, HARD_TIER

    def tier(pairs: int, agreements: int, gate: float | None) -> dict:
        return {
            "probe_pairs": pairs,
            "parsed_pairs": pairs,
            "unparsed_pairs": 0,
            "agreements": agreements,
            "accuracy": round(agreements / pairs, 6),
            "accuracy_wilson_95": [accuracy_lcb, 1.0],
            "labelled_actionable": pairs // 2,
            "labelled_complementary": pairs // 2,
            "recall_on_actionable": 1.0,
            "specificity_on_complementary": round((agreements - pairs // 2) / (pairs // 2), 6),
            "min_accuracy_lcb": gate,
            "gates": gate is not None,
            "passed": True if gate is not None else None,
        }

    return {
        "probe_id": "test",
        "probe_corpora": {BASE_TIER: "fixture", HARD_TIER: "fixture-hard"},
        "parsed_pairs": 40,
        "agreements": 38,
        "accuracy": 0.95,
        "accuracy_wilson_95": [accuracy_lcb, 1.0],
        "min_accuracy_lcb": MIN_ADJUDICATOR_ACCURACY_LCB,
        "tiers": {
            BASE_TIER: tier(24, 23, MIN_ADJUDICATOR_ACCURACY_LCB),
            HARD_TIER: tier(16, 15, None),
        },
        "tier_separation": {
            "floor_tier": BASE_TIER,
            "floor_accuracy": 0.958333,
            "scored_tiers": {
                HARD_TIER: {
                    "accuracy": 0.9375,
                    "delta_from_floor": -0.020833,
                    "recall_on_actionable": 1.0,
                    "specificity_on_complementary": 0.875,
                }
            },
        },
        "calibrated": True,
        "gate_failures": [],
    }


def _map_by_id(value: dict, naming: DocumentNaming) -> dict:
    """One recorded map back in id-keyed form, whatever its values are."""
    return {
        naming.name(str(key)): (
            [naming.name(str(item)) for item in inner] if isinstance(inner, list) else inner
        )
        for key, inner in value.items()
    }


def _part_by_id(part: dict, naming: DocumentNaming) -> dict:
    """A record part that holds maps (`chunks`, `exclusions`) with every map keyed by id again.

    `min_claim_tokens` sits beside those maps and is a plain integer, so it passes through.
    """
    return {
        name: _map_by_id(inner, naming) if isinstance(inner, dict) else inner
        for name, inner in part.items()
    }


def _candidates_by_id(candidates: dict, naming: DocumentNaming) -> dict:
    """The ranked prefix with each row naming its two documents by id instead of by position."""
    return {
        **candidates,
        ENTRIES_KEY: [
            [rank, naming.name(str(left)), naming.name(str(right)), cosine]
            for rank, left, right, cosine in candidates[ENTRIES_KEY]
        ],
    }


def _plain_counts(recorded: dict, corpus_size: int) -> dict:
    """One count map with its recorded default written back out per document, as schema 6 did.

    A map that carries no default folded nothing and comes through untouched. The expansion mirrors
    `named_counts`: the default first so an explicit entry wins, then the zeros dropped, because a
    zero under a fold is the absence the plain map expresses.
    """
    default = recorded.get(COUNT_DEFAULT_KEY)
    if not isinstance(default, int):
        return recorded
    expanded = {
        **{str(position): default for position in range(corpus_size)},
        **{key: value for key, value in recorded.items() if key != COUNT_DEFAULT_KEY},
    }
    return {key: value for key, value in expanded.items() if value}


def unfolded_counts(record: dict, *, schema: int = 6) -> dict:
    """The same bundle record with every count map written per document, as schema 6 did.

    Schema 7 records the count most corpus documents share once and lists only the documents that
    differ. Only the maps under `chunks` and `exclusions` can carry a default, and `recovery_floor`
    declines the fold outright, so it is not among them.
    """
    corpus_size = len(record[DOCUMENTS_KEY])
    unfolded = {**record, SCHEMA_KEY: schema}
    for part, foldable in ((CHUNKS_KEY, CHUNK_COUNT_KEYS), (EXCLUSIONS_KEY, REASON_KEYS)):
        if not isinstance(unfolded.get(part), dict):
            continue
        unfolded[part] = {
            name: _plain_counts(value, corpus_size)
            if name in foldable and isinstance(value, dict)
            else value
            for name, value in unfolded[part].items()
        }
    return unfolded


def unfolded_documents(record: dict, *, schema: int = 5) -> dict:
    """The same bundle record with every document entry spelling out its WHOLE id, as schema 5 did.

    Schema 6 records the head and tail every id shares once and leaves each entry its stem. Undoing
    that is the affix put back and the two keys dropped, so a record folding nothing comes through
    unchanged apart from its version. The count fold above it is undone first, since schema 5 knew
    neither.
    """
    record = unfolded_counts(record)
    affix = IdAffix.from_record(record)
    unfolded = {key: value for key, value in record.items() if key not in (PREFIX_KEY, SUFFIX_KEY)}
    unfolded[SCHEMA_KEY] = schema
    unfolded[DOCUMENTS_KEY] = [
        affix.expand(entry)
        if isinstance(entry, str)
        else {**entry, DOC_ID_KEY: affix.expand(str(entry[DOC_ID_KEY]))}
        for entry in record[DOCUMENTS_KEY]
    ]
    return unfolded


def labelled_documents(record: dict, *, schema: int = 4) -> dict:
    """The same bundle record with every document entry LABELLED, as schemas 1-4 wrote them.

    Schema 5 records a document with no ordering field as the bare id; before it, that document was
    a one-key object. The fold is undone first, since schemas 1-4 wrote whole ids -- so what is left
    changing is only the entries.
    """
    flat = unfolded_documents(record)
    return {
        **flat,
        SCHEMA_KEY: schema,
        DOCUMENTS_KEY: [
            {DOC_ID_KEY: entry} if isinstance(entry, str) else entry
            for entry in flat[DOCUMENTS_KEY]
        ],
    }


def keyed_by_id(record: dict, *, schema: int) -> dict:
    """The same bundle record in the PRE-INTERNING form every bundle on disk before schema 4 uses.

    Schema 4 replaced each document id outside `documents` with its corpus position, schema 5
    dropped the label from a document with nothing to label, and schema 6 folded the shared head and
    tail out of the ids -- the only three changes the record has ever made to a shape rather than
    adding to it. Every reading has to replay identically through all four forms, so the tests need
    the older ones -- rewritten here from a fresh record rather than committed as a frozen blob, so
    a new field cannot quietly go untested on the older side.
    """
    naming = naming_of(record)
    legacy = {
        key: value for key, value in labelled_documents(record).items() if key != EXTRA_IDS_KEY
    }
    legacy[SCHEMA_KEY] = schema
    for key in (CHUNKS_KEY, EXCLUSIONS_KEY):
        if key in legacy:
            legacy[key] = _part_by_id(legacy[key], naming)
    if CANDIDATES_KEY in legacy:
        legacy[CANDIDATES_KEY] = _candidates_by_id(legacy[CANDIDATES_KEY], naming)
    return legacy


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS
