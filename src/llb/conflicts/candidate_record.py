"""The ranked candidate list, recorded so a finished bundle can be re-read at a smaller budget.

The claim tier's budget is a RANK CUTOFF into the store's own cosine ordering: `detect_semantic_pairs`
returns its pairs sorted by descending similarity, `--max-claim-pairs` keeps a prefix of that list,
and `--max-candidate-pairs` picks the cosine threshold that makes the list about that long. So "what
would a smaller budget have returned" is answerable exactly -- from a PREFIX -- and needs no store,
no vectors, and no re-adjudication.

It does need the ranking, and the ranking is the one thing `findings.jsonl` loses. Every candidate
pair does reach that file (pairs past `--max-claim-pairs` land there as provisional `duplicate`
rows), but the rows are written in report order -- actionable first, then by the ADJUDICATOR's
score, which for a claim-tier row is not the cosine that ranked it. A bundle therefore holds the
candidate set and not the candidate ORDER, and a budget replay needs the order.

What is recorded is the ranked list collapsed to DOCUMENT pairs, each with the rank it first
appears at:

- collapsed, because every reading built on it is a document-pair reading (the stage attribution,
  the governance coverage), and the first rank a document pair appears at is exactly what decides
  whether a budget returns it;
- capped, because the uncollapsed list is quadratic in CHUNKS and the collapsed one is quadratic in
  DOCUMENTS, and neither is a size a record may grow to. The cap is the run's own
  `--max-candidate-pairs` when it set one, and a constant otherwise; `covered_to_rank` says how far
  the record can answer, and a budget past it is REFUSED rather than answered from a truncated list.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

# How many document pairs a run with no budget of its own records. Two hundred rows of
# `[rank, doc, doc, cosine]` is ~12 KiB, which is the same order as the per-document maps beside it
# and far below the bundle's own rows; a corpus whose candidate list is longer than that is a corpus
# whose operator set a budget.
DEFAULT_CANDIDATE_RECORD_PAIRS = 200

ENTRIES_KEY = "document_pairs"
TOTAL_KEY = "total_pairs"
COVERED_KEY = "covered_to_rank"
COSINE_DIGITS = 4


@dataclass(frozen=True)
class CandidateRecord:
    """Document pairs in candidate-rank order, with how far down the ranking the record reaches."""

    # (rank, left doc, right doc, cosine) with rank 1-based and the pair sorted by document id.
    entries: tuple[tuple[int, str, str, float], ...] = ()
    total_pairs: int = 0
    covered_to_rank: int = 0

    @classmethod
    def of(
        cls,
        pairs: Sequence[tuple[int, int, float]],
        chunks: Sequence[ChunkRecord],
        *,
        limit: int = DEFAULT_CANDIDATE_RECORD_PAIRS,
    ) -> "CandidateRecord":
        """Collapse the ranked chunk-pair list to first-appearance document pairs, capped at `limit`."""
        entries: list[tuple[int, str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        # The deepest rank the record can speak for. It advances past a rank whose document pair is
        # ALREADY recorded even after the cap is reached -- such a rank adds nothing to answer with
        # -- and stops at the first rank that would have added a pair the cap refuses to keep.
        covered = 0
        for rank, (left, right, similarity) in enumerate(pairs, start=1):
            left_doc, right_doc = sorted(
                [str(chunks[left]["doc_id"]), str(chunks[right]["doc_id"])]
            )
            doc_pair = (left_doc, right_doc)
            if doc_pair not in seen:
                if len(entries) >= max(limit, 0):
                    break
                seen.add(doc_pair)
                entries.append((rank, left_doc, right_doc, round(float(similarity), COSINE_DIGITS)))
            covered = rank
        return cls(entries=tuple(entries), total_pairs=len(pairs), covered_to_rank=covered)

    def payload(self) -> JsonObject:
        return {
            ENTRIES_KEY: [list(entry) for entry in self.entries],
            TOTAL_KEY: self.total_pairs,
            COVERED_KEY: self.covered_to_rank,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateRecord":
        """Read back what `payload` wrote, dropping any entry that is not a well-formed row."""
        rows = payload.get(ENTRIES_KEY)
        entries = [
            (int(row[0]), str(row[1]), str(row[2]), float(row[3]))
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, list) and len(row) == 4
        ]
        total = payload.get(TOTAL_KEY)
        covered = payload.get(COVERED_KEY)
        return cls(
            entries=tuple(entries),
            total_pairs=total if isinstance(total, int) else len(entries),
            covered_to_rank=covered if isinstance(covered, int) else 0,
        )

    def covers(self, budget: int) -> bool:
        """Whether a prefix this long is fully recorded -- a budget past it has no honest answer."""
        return min(budget, self.total_pairs) <= self.covered_to_rank

    def doc_pairs_within(self, budget: int) -> set[tuple[str, str]]:
        """The document pairs the candidate list would have returned at this budget."""
        return {(left, right) for rank, left, right, _ in self.entries if rank <= budget}
