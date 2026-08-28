"""Deterministic multi-query search tasks whose scored fact is a MIDDLE observation's hit count.

The memory-chain workloads answer "does a fact survive the fold". This shape answers the other
half: does a machine-computed AGGREGATE survive it. Every search observation is a fat hit list, so
the folded transcript is far larger than the live prompt (the summarizer is handed RAW observations
while the prompt shows trimmed ones), and the fold reliably overflows the summarize-input cap.

The scored query sits in the MIDDLE of the search order on purpose. That is the position a
whole-transcript head-and-tail trim drops first, so if the aggregate header were ever carried by
the free-text summary rather than by `fold_aggregate_headers`, this task set is where the loss
would show. Under the shipped machine preservation it must not, whichever trim strategy runs.
"""

from typing import Any

from llb.bench.agentic.model import ASSERT_ANSWER_CONTAINS

# One search observation per query, each a hit list over the planted corpus. Defaults sized so a
# handful of searches crosses a several-thousand-char compaction trigger.
DEFAULT_N_QUERIES = 5
DEFAULT_N_DOCS = 14
DEFAULT_DOC_CHARS = 900

_PAD_UNIT = "довідка про стан комунальної інфраструктури громади "


def _document(term_line: str, doc_chars: int) -> str:
    repeats = (doc_chars // len(_PAD_UNIT)) + 1
    return f"{term_line}\n{(_PAD_UNIT * repeats)[:doc_chars]}"


def _planted_corpus(
    index: int, terms: list[str], *, n_docs: int, doc_chars: int
) -> tuple[dict[str, str], dict[str, int]]:
    """A corpus where each term's document frequency is decided here, not derived from prose."""
    corpus: dict[str, str] = {}
    hits = dict.fromkeys(terms, 0)
    for doc in range(n_docs):
        # Every term lands in its own residue class plus one dedicated document, so the frequencies
        # differ between terms and no two queries share an answer.
        mentioned = [
            term
            for position, term in enumerate(terms)
            if doc % len(terms) == position or doc == position
        ]
        for term in mentioned:
            hits[term] += 1
        corpus[f"doc-{index:03d}-{doc:02d}.txt"] = _document(
            " ".join(mentioned) if mentioned else "без термінів", doc_chars
        )
    return corpus, hits


def aggregate_search_task(
    index: int,
    *,
    n_queries: int = DEFAULT_N_QUERIES,
    n_docs: int = DEFAULT_N_DOCS,
    doc_chars: int = DEFAULT_DOC_CHARS,
) -> dict[str, Any]:
    """Build one task: search every planted term in order, then report the MIDDLE term's count."""
    if n_queries < 3 or n_docs < n_queries:
        raise ValueError("an aggregate-search task needs at least three queries and one doc each")
    terms = [f"термін-{index:03d}-{query}" for query in range(n_queries)]
    corpus, hits = _planted_corpus(index, terms, n_docs=n_docs, doc_chars=doc_chars)
    scored = terms[n_queries // 2]
    order = ", ".join(f'"{term}"' for term in terms)
    return {
        # `search-count` so the shipped task-kind tables score this in the count slice.
        "id": f"search-count-{index:03d}-q{n_queries}",
        "family": "aggregate-search",
        "prompt": (
            f"Case {index:03d}. Виконай search по кожному з термінів РІВНО в такому порядку: "
            f"{order}. Після всіх пошуків виклич finish і передай ЛИШЕ число -- скільки "
            f'документів згадують "{scored}".'
        ),
        "setup": {"corpus": corpus},
        "success": [{"kind": ASSERT_ANSWER_CONTAINS, "value": str(hits[scored])}],
        "queries": terms,
        "scored_query": scored,
        "answer": str(hits[scored]),
    }


def build_aggregate_search_tasks(
    *,
    n_tasks: int,
    n_queries: int = DEFAULT_N_QUERIES,
    n_docs: int = DEFAULT_N_DOCS,
    doc_chars: int = DEFAULT_DOC_CHARS,
) -> list[dict[str, Any]]:
    """Build a deterministic aggregate-search set with a stable shared digest."""
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    return [
        aggregate_search_task(index, n_queries=n_queries, n_docs=n_docs, doc_chars=doc_chars)
        for index in range(n_tasks)
    ]
