"""Draft-time ambiguous-evidence guard: how many times a gold span's text repeats in the corpus.

A drafted gold span can land inside a passage the corpus repeats verbatim (page furniture, a
boilerplate clause, a table row that recurs). Such an item is ambiguous by construction: the answer
text exists in several places, the retrieval metric credits any of them, and a reviewer reading the
worksheet cannot tell that the span they are accepting is not unique.

This module counts, for each gold item, how many times its primary span's text occurs across the
WHOLE corpus (its own labeled place plus every other). A count of one is unique; a count above
`OCCURRENCE_FLAG_THRESHOLD` is flagged so the verification worksheet and review card can show
"this evidence appears in N places" before a reviewer accepts it. Counting is a direct corpus scan
(non-overlapping substring occurrences), which -- unlike chunk collapse -- catches a span wherever
it sits, including inside a chunk that carries other text.

The guard only annotates: it never rejects an item or changes the retrieval metric. A bundle whose
spans are all unique writes no sidecar and keeps its worksheet byte-for-byte.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem
from llb.prep.pdf.model import PDF_CITATION_SUFFIX, PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY

# Additive worksheet column and bundle sidecar. Both stay absent when nothing is flagged, so an
# all-unique corpus is unchanged.
SPAN_OCCURRENCES_COL = "span_occurrences"
OCCURRENCES_SIDECAR = "span_occurrences.jsonl"

# Flag an item when its span text occurs strictly MORE than this many times in the corpus. One is
# the span's own place, so the guard fires at two occurrences and up.
OCCURRENCE_FLAG_THRESHOLD = 1

# Corpus-directory files that are metadata, not source documents: the span text is not counted in
# them. The PDF lane copies these beside the `.txt`/`.md` docs into a bundle's `corpus/`.
_NON_DOCUMENT_NAMES = frozenset({PDF_CORPUS_MANIFEST, PDF_CORPUS_QUALITY})


def count_span_occurrences(corpus_texts: Iterable[str], span_text: str) -> int:
    """Total non-overlapping occurrences of `span_text` across every corpus document."""
    if not span_text:
        return 0
    return sum(text.count(span_text) for text in corpus_texts)


def span_occurrence_counts(
    items: Sequence[GoldItem], corpus_texts: Mapping[str, str]
) -> dict[str, int]:
    """Occurrence count of each item's PRIMARY span text across the corpus, keyed by item id.

    The primary span is `source_spans[0]` -- the span the worksheet renders -- so the flagged count
    matches the evidence a reviewer sees. Counting the whole corpus (not just the item's own
    document) is deliberate: a passage repeated across documents is exactly the ambiguous case.
    """
    texts = list(corpus_texts.values())
    return {item.id: count_span_occurrences(texts, item.source_spans[0].text) for item in items}


def flagged_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Keep only the items whose span repeats above the guard threshold."""
    return {item_id: n for item_id, n in counts.items() if n > OCCURRENCE_FLAG_THRESHOLD}


def load_corpus_texts(corpus_root: Path) -> dict[str, str]:
    """Read every source document under `corpus_root`, skipping PDF citation/quality sidecars."""
    root = Path(corpus_root)
    texts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(PDF_CITATION_SUFFIX) or path.name in _NON_DOCUMENT_NAMES:
            continue
        try:
            texts[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return texts


def write_occurrences_sidecar(bundle: Path, counts: Mapping[str, int]) -> int:
    """Write the flagged-item sidecar into `bundle`; skip the write entirely when none are flagged.

    Returns the number of flagged rows written (0 leaves no file, so an unflagged bundle is
    byte-for-byte unchanged). Rows are sorted by id for a deterministic artifact.
    """
    flagged = flagged_counts(counts)
    if not flagged:
        return 0
    lines = [
        json.dumps({"id": item_id, SPAN_OCCURRENCES_COL: flagged[item_id]}, ensure_ascii=False)
        for item_id in sorted(flagged)
    ]
    atomic_write_text(Path(bundle) / OCCURRENCES_SIDECAR, "\n".join(lines) + "\n")
    return len(flagged)


def load_occurrences_sidecar(bundle: Path) -> dict[str, int]:
    """Read the flagged-item occurrence sidecar from a bundle (empty when it is absent)."""
    path = Path(bundle) / OCCURRENCES_SIDECAR
    if not path.is_file():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        item_id = row.get("id")
        value = row.get(SPAN_OCCURRENCES_COL)
        if item_id and isinstance(value, int) and value > OCCURRENCE_FLAG_THRESHOLD:
            counts[str(item_id)] = value
    return counts


def worksheet_occurrences(
    bundle: Path, sample: Sequence[GoldItem], corpus_root: Path
) -> dict[str, int]:
    """Flagged occurrence counts for the sampled items: the sidecar when present, else a scan.

    A drafted bundle carries the sidecar (written at draft time from the authoritative in-memory
    corpus). A bundle without one -- an older draft, or a fixture -- is scanned directly so the
    worksheet still surfaces ambiguous evidence. Only flagged items (count above the threshold)
    are returned; every other item stays blank in the worksheet.
    """
    sidecar = load_occurrences_sidecar(bundle)
    sample_ids = {item.id for item in sample}
    if sidecar:
        return {item_id: n for item_id, n in sidecar.items() if item_id in sample_ids}
    counts = span_occurrence_counts(sample, load_corpus_texts(corpus_root))
    return flagged_counts(counts)
