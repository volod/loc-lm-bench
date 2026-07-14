"""Shared curation primitives: lenient loading, normalization, dedup engine, and the report.

External services (Claude Projects, NotebookLM/Gemini, ChatGPT Projects) return artifacts in
batches, per service, per session -- so the raw material for one benchmark is typically MANY files
of one artifact kind, with overlapping content, occasional paraphrase duplicates, and a tail of
invalid rows (paraphrased "verbatim" quotes, truncated JSON, prompt-echo prose around the code
block). Curation merges those files into ONE importable artifact while maximizing kept quality:
repair what is mechanically repairable (re-ground a near-verbatim quote to the exact corpus text),
drop what is invalid or flabby, and deduplicate exact and near-duplicate questions across services.

Everything here is deterministic and dependency-light; the semantic near-dup step takes any
`QuestionEmbedder` (the pinned-E5 adapter in production, a fake in tests) and is skipped with a
warning when no embedder is available.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from llb.prep.ontology.dedup import QuestionEmbedder, Vector, _cosine

_LOG = logging.getLogger(__name__)

# --- thresholds and limits (single source; CLI flags override where noted) -------------------

# Cosine threshold above which two questions are near-duplicates (matches the ontology
# drafting default so "duplicate" means the same thing in both lanes; CLI: --dedup-threshold).

# A question shorter than this (chars / word tokens) cannot carry a retrieval needle.
# A context passage shorter than this gives retrieval nothing to find (CLI: --min-context-chars).
# An answer span longer than this, or covering more than this fraction of its context, is a
# whole-paragraph answer that defeats span scoring.

# Questions that reference the document structure instead of asking a natural domain question
# ("according to the passage ...") test prompt reading, not retrieval. UA + EN stems.


# --- lenient artifact loading ----------------------------------------------------------------


# --- curation report --------------------------------------------------------------------------


@dataclass
class CurationReport:
    """Counts and per-item reasons for one curation run (written beside the output artifact)."""

    kind: str
    sources: dict[str, int] = field(default_factory=dict)  # input file -> rows loaded
    loaded: int = 0
    invalid: list[dict[str, Any]] = field(default_factory=list)  # unusable rows
    flabby: list[dict[str, Any]] = field(default_factory=list)  # low-value rows
    repaired: list[dict[str, Any]] = field(default_factory=list)  # re-grounded / re-snapped
    exact_duplicates: list[dict[str, Any]] = field(default_factory=list)
    near_duplicates: list[dict[str, Any]] = field(default_factory=list)
    id_rewrites: list[dict[str, Any]] = field(default_factory=list)
    kept: int = 0
    notes: list[str] = field(default_factory=list)

    def reject_invalid(self, item_id: str, source: str, reason: str) -> None:
        self.invalid.append({"id": item_id, "source": source, "reason": reason})

    def reject_flabby(self, item_id: str, source: str, reason: str) -> None:
        self.flabby.append({"id": item_id, "source": source, "reason": reason})

    def note_repair(self, item_id: str, source: str, what: str) -> None:
        self.repaired.append({"id": item_id, "source": source, "repair": what})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sources": self.sources,
            "loaded": self.loaded,
            "kept": self.kept,
            "counts": {
                "invalid": len(self.invalid),
                "flabby": len(self.flabby),
                "repaired": len(self.repaired),
                "exact_duplicates": len(self.exact_duplicates),
                "near_duplicates": len(self.near_duplicates),
                "id_rewrites": len(self.id_rewrites),
            },
            "invalid": self.invalid,
            "flabby": self.flabby,
            "repaired": self.repaired,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "id_rewrites": self.id_rewrites,
            "notes": self.notes,
        }


# --- duplicate detection ----------------------------------------------------------------------


def drop_exact_duplicates(
    keys: list[str], report: CurationReport, ids: list[str], sources: list[str]
) -> list[int]:
    """Return indices to KEEP after first-wins exact-key dedup; log dropped rows to the report."""
    seen: dict[str, str] = {}
    kept: list[int] = []
    for i, key in enumerate(keys):
        if key in seen:
            report.exact_duplicates.append(
                {"id": ids[i], "source": sources[i], "duplicate_of": seen[key]}
            )
            continue
        seen[key] = ids[i]
        kept.append(i)
    return kept


def _note_near_duplicate(
    report: CurationReport, item_id: str, source: str, duplicate_of: str, similarity: float
) -> None:
    report.near_duplicates.append(
        {
            "id": item_id,
            "source": source,
            "duplicate_of": duplicate_of,
            "similarity": round(similarity, 4),
        }
    )


def _best_kept_match(
    vector: Vector,
    group: str,
    groups: list[str],
    kept: list[int],
    kept_vectors: list[Vector],
    ids: list[str],
    threshold: float,
) -> tuple[str | None, float]:
    """The most similar already-kept row at or above `threshold` (skipping protected twins)."""
    duplicate_of: str | None = None
    best = 0.0
    for j, kept_vector in zip(kept, kept_vectors):
        if group and group == groups[j]:
            continue  # intentional twins (bias pair / cross-language group)
        sim = _cosine(vector, kept_vector)
        if sim >= threshold and sim > best:
            best = sim
            duplicate_of = ids[j]
    return duplicate_of, best


def drop_near_duplicates(
    texts: list[str],
    embedder: QuestionEmbedder | None,
    threshold: float,
    report: CurationReport,
    ids: list[str],
    sources: list[str],
    *,
    protected_groups: list[str] | None = None,
    prior_texts: list[str] | None = None,
) -> list[int]:
    """Greedy first-wins near-duplicate suppression; returns indices to KEEP.

    Two rows sharing a non-empty `protected_groups` key (a bias pair id, a cross-language group)
    are never deduplicated against each other -- their similarity is intentional. `prior_texts`
    (questions from earlier accepted bundles) suppress re-drafts of already-covered needles.
    Without an embedder the semantic step is skipped and every index is kept.
    """
    if embedder is None or not texts:
        if embedder is None and texts:
            report.notes.append("semantic dedup skipped: no embedder available")
        return list(range(len(texts)))
    vectors = embedder.embed(texts)
    prior_vectors: list[Vector] = embedder.embed(prior_texts) if prior_texts else []
    groups = protected_groups or [""] * len(texts)
    kept: list[int] = []
    kept_vectors: list[Vector] = []
    for i, vector in enumerate(vectors):
        prior_best = max((_cosine(vector, pv) for pv in prior_vectors), default=0.0)
        if prior_best >= threshold:
            _note_near_duplicate(report, ids[i], sources[i], "prior-bundle", prior_best)
            continue
        duplicate_of, best = _best_kept_match(
            vector, groups[i], groups, kept, kept_vectors, ids, threshold
        )
        if duplicate_of is not None:
            _note_near_duplicate(report, ids[i], sources[i], duplicate_of, best)
            continue
        kept.append(i)
        kept_vectors.append(vector)
    return kept


def resolve_embedder(semantic: bool) -> QuestionEmbedder | None:
    """The pinned-E5 embedder when requested and installed; None (with a warning) otherwise."""
    if not semantic:
        return None
    try:
        from llb.prep.ontology.dedup import E5QuestionEmbedder

        return E5QuestionEmbedder()
    except Exception as exc:  # sentence-transformers absent or model unavailable
        _LOG.warning("[curate] semantic dedup unavailable (%s); exact dedup only", exc)
        return None


def unique_ids(ids: list[str], report: CurationReport, sources: list[str]) -> list[str]:
    """Rewrite colliding ids deterministically (`<id>-r2`, `-r3`, ...) so merges stay importable."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for i, item_id in enumerate(ids):
        count = seen.get(item_id, 0) + 1
        seen[item_id] = count
        if count == 1:
            out.append(item_id)
            continue
        rewritten = f"{item_id}-r{count}"
        report.id_rewrites.append({"id": item_id, "rewritten": rewritten, "source": sources[i]})
        out.append(rewritten)
    return out
