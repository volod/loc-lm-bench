"""agentic benchmark real-UA-corpus agentic search tasks -- generated, deterministic, no human authoring.

The committed agentic seed (`samples/agentic_tasks_uk.json`) is small and hand-authored. This module
GROWS the task set from a REAL corpus (the category expansion text-analysis corpus is the natural source): it plants
search tasks whose success assertion is computed PURELY from the corpus, so the answer is objective
and needs no human gold authoring to BUILD (the human verification gate sample-verify still gates headline use):

  * count  -- "how many documents mention X?" -> the answer is the document frequency of X;
  * locate -- "which document mentions X?"     -> the answer is the single doc id that contains X
              (only terms appearing in EXACTLY ONE doc are used, so the answer is unambiguous).

The agent solves these with the sandbox `search` tool (substring over `setup.corpus`) + `finish`, so
each task drops straight into the existing `bench-agentic` loop and `check_success`. Query terms are
DERIVED from the corpus by document frequency (UA-stopword filtered) or supplied from planted labels,
so the generator is pure and unit-tested with no model, network, or GPU.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.bench.agentic import ASSERT_ANSWER_CONTAINS

_LOG = logging.getLogger(__name__)

KIND_COUNT = "count"
KIND_LOCATE = "locate"

_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_MIN_TERM_LEN = 5

# A small UA/EN stopword set so derived query terms are content words (not function words).
STOPWORDS = frozenset(
    {
        "який",
        "яка",
        "яке",
        "які",
        "цей",
        "того",
        "щоб",
        "після",
        "перед",
        "через",
        "також",
        "більше",
        "менше",
        "тому",
        "коли",
        "тоді",
        "проте",
        "однак",
        "разом",
        "цього",
        "цьому",
        "своїх",
        "цими",
        "лише",
        "було",
        "буде",
        "their",
        "there",
        "which",
        "about",
        "would",
        "could",
    }
)


def _doc_frequency(corpus: Mapping[str, str]) -> dict[str, set[str]]:
    """Map each content term (len >= 5, not a stopword) to the set of doc ids that contain it."""
    df: dict[str, set[str]] = {}
    for doc_id, text in corpus.items():
        seen = {
            w.casefold()
            for w in _WORD.findall(text)
            if len(w) >= _MIN_TERM_LEN and w.casefold() not in STOPWORDS
        }
        for term in seen:
            df.setdefault(term, set()).add(doc_id)
    return df


def _doc_count(corpus: Mapping[str, str], query: str) -> int:
    low = query.casefold()
    return sum(1 for text in corpus.values() if low in text.casefold())


def _docs_containing(corpus: Mapping[str, str], query: str) -> list[str]:
    low = query.casefold()
    return [doc_id for doc_id, text in corpus.items() if low in text.casefold()]


def search_count_task(corpus: Mapping[str, str], query: str, index: int) -> dict[str, Any]:
    """A 'how many docs mention X' task; the answer is the document frequency (corpus-derived)."""
    count = _doc_count(corpus, query)
    return {
        "id": f"search-count-{index:03d}",
        "prompt": (
            f"Скориставшись інструментом пошуку, з'ясуй, у скількох документах корпусу згадується "
            f"«{query}». Повідом лише число."
        ),
        "setup": {"corpus": dict(corpus)},
        "success": [{"kind": ASSERT_ANSWER_CONTAINS, "value": str(count)}],
    }


def search_locate_task(corpus: Mapping[str, str], query: str, index: int) -> dict[str, Any] | None:
    """A 'which doc mentions X' task; only built when EXACTLY ONE doc contains the term."""
    hits = _docs_containing(corpus, query)
    if len(hits) != 1:
        return None
    return {
        "id": f"search-locate-{index:03d}",
        "prompt": (
            f"Скориставшись інструментом пошуку, знайди документ, у якому згадується «{query}», "
            f"і повідом його ідентифікатор."
        ),
        "setup": {"corpus": dict(corpus)},
        "success": [{"kind": ASSERT_ANSWER_CONTAINS, "value": hits[0]}],
    }


def derive_queries(corpus: Mapping[str, str], *, top_k: int = 8) -> tuple[list[str], list[str]]:
    """Derive (count_terms, locate_terms) from the corpus by document frequency.

    count_terms are the most frequent content terms (appear in >= 1 doc); locate_terms are content
    terms appearing in EXACTLY ONE doc (so the located doc id is unambiguous).
    """
    df = _doc_frequency(corpus)
    by_freq = sorted(df.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    count_terms = [term for term, docs in by_freq if len(docs) >= 1][:top_k]
    locate_terms = sorted(term for term, docs in df.items() if len(docs) == 1)[:top_k]
    return count_terms, locate_terms


def build_search_tasks(
    corpus: Mapping[str, str],
    *,
    queries: Sequence[str] | None = None,
    top_k: int = 8,
    kinds: Sequence[str] = (KIND_COUNT, KIND_LOCATE),
) -> list[dict[str, Any]]:
    """Build deterministic real-corpus search tasks (count + locate). `queries` overrides derivation
    (e.g. planted-label surfaces); otherwise terms are derived from the corpus by document frequency."""
    if not corpus:
        return []
    count_terms = list(queries) if queries is not None else []
    locate_terms = list(queries) if queries is not None else []
    if queries is None:
        count_terms, locate_terms = derive_queries(corpus, top_k=top_k)

    tasks: list[dict[str, Any]] = []
    if KIND_COUNT in kinds:
        for i, term in enumerate(count_terms):
            tasks.append(search_count_task(corpus, term, i))
    if KIND_LOCATE in kinds:
        for i, term in enumerate(locate_terms):
            task = search_locate_task(corpus, term, i)
            if task is not None:
                tasks.append(task)
    _LOG.info("[agentic-tasks] built %d search tasks over %d docs", len(tasks), len(corpus))
    return tasks


def build_from_corpus(
    corpus_root: Path | str, *, top_k: int = 8, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load a corpus dir (`.md`/`.txt`) and build deterministic real-corpus search tasks."""
    from llb.rag.chunking import iter_docs

    corpus = dict(iter_docs(Path(corpus_root)))
    if not corpus:
        raise ValueError(f"no .md/.txt documents under {corpus_root}")
    if limit is not None:
        corpus = dict(list(corpus.items())[:limit])
    return build_search_tasks(corpus, top_k=top_k)
