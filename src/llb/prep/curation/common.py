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

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.prep.frontier import parse_json_block
from llb.prep.ontology.constants import NEAR_DUP_COSINE_THRESHOLD
from llb.prep.ontology.dedup import QuestionEmbedder, Vector, _cosine

_LOG = logging.getLogger(__name__)

# --- thresholds and limits (single source; CLI flags override where noted) -------------------

# Cosine threshold above which two questions are near-duplicates (matches the ontology
# drafting default so "duplicate" means the same thing in both lanes; CLI: --dedup-threshold).
DEFAULT_DEDUP_THRESHOLD = NEAR_DUP_COSINE_THRESHOLD

# A question shorter than this (chars / word tokens) cannot carry a retrieval needle.
MIN_QUESTION_CHARS = 12
MIN_QUESTION_WORDS = 3
# A context passage shorter than this gives retrieval nothing to find (CLI: --min-context-chars).
DEFAULT_MIN_CONTEXT_CHARS = 80
# An answer span longer than this, or covering more than this fraction of its context, is a
# whole-paragraph answer that defeats span scoring.
MAX_ANSWER_CHARS = 400
MAX_ANSWER_CONTEXT_FRACTION = 0.6

# Questions that reference the document structure instead of asking a natural domain question
# ("according to the passage ...") test prompt reading, not retrieval. UA + EN stems.
_STRUCTURE_REFERENCE_STEMS = (
    "у наведеному тексті",
    "в наведеному тексті",
    "у цьому документі",
    "в цьому документі",
    "у цьому тексті",
    "в цьому тексті",
    "у документі",
    "в документі",
    "в уривку",
    "в уривці",
    "у фрагменті",
    "згідно з текстом",
    "згідно з наведеним",
    "у параграфі",
    "в абзаці",
    "according to the text",
    "according to the passage",
    "according to the document",
    "in this document",
    "in this passage",
    "in this excerpt",
)


def normalize_text(text: str) -> str:
    """Casefold + collapse whitespace: the key used for exact-duplicate detection."""
    return " ".join(text.split()).casefold()


def references_document_structure(question: str) -> bool:
    """True when the question points at the document/passage instead of asking naturally."""
    q = normalize_text(question)
    return any(stem in q for stem in _STRUCTURE_REFERENCE_STEMS)


def question_too_vague(question: str) -> bool:
    """True for questions too short to identify a needle."""
    q = question.strip()
    return len(q) < MIN_QUESTION_CHARS or len(q.split()) < MIN_QUESTION_WORDS


# --- lenient artifact loading ----------------------------------------------------------------


def load_json_documents(path: Path) -> list[Any]:
    """Load every JSON value in a file: raw JSON, one or more ``` fenced blocks, or JSONL.

    Service replies are exported by hand; a file may hold one clean JSON document, several fenced
    code blocks (one per batch), or JSON Lines. Returns the parsed values in file order and raises
    on a file with no parseable JSON at all (silent emptiness would hide an export mistake).
    """
    text = path.read_text(encoding="utf-8")
    fenced = re.findall(r"```(?:json[l5]?|jsonl)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return [_parse_lenient(block, source=f"{path}#fence") for block in fenced]
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path}: empty artifact file")
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass
    lines = [line for line in stripped.splitlines() if line.strip()]
    parsed: list[Any] = []
    for line_no, line in enumerate(lines, 1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            parsed = []
            break
    if parsed:
        return parsed
    return [_parse_lenient(stripped, source=str(path))]


def _parse_lenient(block: str, *, source: str) -> Any:
    try:
        return parse_json_block(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: unparseable JSON ({exc})") from exc


def load_jsonl_rows(values: list[Any]) -> list[Any]:
    """Flatten loaded JSON values into rows: arrays are splatted, objects pass through."""
    rows: list[Any] = []
    for value in values:
        if isinstance(value, list):
            rows.extend(value)
        else:
            rows.append(value)
    return rows


def load_corpus_texts(corpus_root: Path) -> dict[str, str]:
    """Read every .md/.txt under `corpus_root` keyed by its relative path (the doc id)."""
    texts: dict[str, str] = {}
    for path in sorted(corpus_root.rglob("*")):
        if path.suffix.lower() in (".md", ".txt") and path.is_file():
            texts[str(path.relative_to(corpus_root))] = path.read_text(encoding="utf-8")
    if not texts:
        raise SystemExit(f"[curate] no .md/.txt corpus documents under {corpus_root}")
    return texts


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
            report.near_duplicates.append(
                {
                    "id": ids[i],
                    "source": sources[i],
                    "duplicate_of": "prior-bundle",
                    "similarity": round(prior_best, 4),
                }
            )
            continue
        duplicate_of: str | None = None
        best = 0.0
        for j, kept_vector in zip(kept, kept_vectors):
            if groups[i] and groups[i] == groups[j]:
                continue  # intentional twins (bias pair / cross-language group)
            sim = _cosine(vector, kept_vector)
            if sim >= threshold and sim > best:
                best = sim
                duplicate_of = ids[j]
        if duplicate_of is not None:
            report.near_duplicates.append(
                {
                    "id": ids[i],
                    "source": sources[i],
                    "duplicate_of": duplicate_of,
                    "similarity": round(best, 4),
                }
            )
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
