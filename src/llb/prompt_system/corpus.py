"""M7.3 corpus preparation -- turn a supplied text corpus into a compact prompt-ready package.

Deterministic + dependency-free: a caller-provided corpus is read, split into paragraph passages
(with exact source-span offsets preserved), the most salient passages are selected into an
ANTHOLOGY, per-document METADATA is summarized, and a knowledge-graph-to-RAG MAPPING (salient term
-> the passage ids that ground it) is emitted. These three artifacts feed the prompt-template
generator; keeping selection deterministic makes every prompt-system run reproducible and
manifest-addressable (no model or network here).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import TypedDict

# A small, lowercase UA + generic stopword set -- enough to keep salient TERMS meaningful without a
# heavy NLP dependency (the selection only needs to be stable, not linguistically perfect).
_STOPWORDS = {
    "and",
    "the",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "are",
    "was",
    "не",
    "що",
    "як",
    "це",
    "для",
    "або",
    "так",
    "при",
    "над",
    "під",
    "the",
    "які",
    "яка",
    "який",
    "теж",
    "був",
    "було",
    "вона",
    "вони",
    "його",
    "її",
    "цей",
    "цього",
    "усі",
    "все",
    "всі",
    "між",
    "тому",
    "щоб",
    "має",
    "бути",
    "лише",
    "також",
    "при",
    "про",
    "своїх",
    "своє",
    "свою",
}
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЇїІіЄєҐґ']+", re.UNICODE)
_MIN_TERM_LEN = 3

CORPUS_EXTENSIONS = (".md", ".txt")


class Passage(TypedDict):
    passage_id: str
    doc_id: str
    char_start: int
    char_end: int
    text: str


class DocMetadata(TypedDict):
    doc_id: str
    title: str
    n_chars: int
    n_paragraphs: int
    top_terms: list[str]


@dataclass(slots=True)
class CorpusPackage:
    """The prepared, model-independent corpus inputs for prompt-system generation."""

    anthology: list[Passage]
    metadata: list[DocMetadata]
    graph_rag_mapping: dict[str, list[str]]  # salient term -> passage ids grounding it
    salient_terms: list[str] = field(default_factory=list)


def read_corpus(corpus_root: Path | str) -> dict[str, str]:
    """Read every `.md` / `.txt` file under `corpus_root` into {relative_doc_id: text}."""
    root = Path(corpus_root)
    docs: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in CORPUS_EXTENSIONS:
            docs[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return docs


def tokenize_terms(text: str) -> list[str]:
    """Lowercase word tokens that look like content terms (>= 3 chars, not a stopword)."""
    return [
        token
        for raw in _TOKEN_RE.findall(text.casefold())
        if len(token := raw.strip("'")) >= _MIN_TERM_LEN and token not in _STOPWORDS
    ]


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Split into paragraphs on blank lines, preserving exact (char_start, char_end) offsets."""
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
        chunk = match.group()
        spans.append((match.start(), match.end(), chunk.strip()))
    return [(s, e, t) for s, e, t in spans if t]


def salient_terms(docs: dict[str, str], top_k: int) -> list[str]:
    """The `top_k` most frequent content terms across the whole corpus (stable tie-break)."""
    counts: Counter[str] = Counter()
    for text in docs.values():
        counts.update(tokenize_terms(text))
    return [term for term, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def _score_paragraph(text: str, corpus_freq: Counter[str]) -> float:
    """Salience: summed corpus frequency of the paragraph's terms, length-normalized so a few
    information-dense sentences outrank a long, repetitive block."""
    terms = tokenize_terms(text)
    if not terms:
        return 0.0
    return float(sum(corpus_freq[t] for t in terms)) / math.sqrt(len(terms))


def select_anthology(docs: dict[str, str], *, max_passages: int, min_chars: int) -> list[Passage]:
    """Pick the most salient paragraph passages across the corpus (source spans preserved)."""
    corpus_freq: Counter[str] = Counter()
    for text in docs.values():
        corpus_freq.update(tokenize_terms(text))
    scored: list[tuple[float, Passage]] = []
    for doc_id, text in docs.items():
        for index, (start, end, para) in enumerate(split_paragraphs(text)):
            if len(para) < min_chars:
                continue
            passage: Passage = {
                "passage_id": f"{doc_id}::p{index:03d}",
                "doc_id": doc_id,
                "char_start": start,
                "char_end": end,
                "text": para,
            }
            scored.append((_score_paragraph(para, corpus_freq), passage))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["passage_id"]))
    return [passage for _score, passage in scored[:max_passages]]


def doc_metadata(docs: dict[str, str], *, top_terms_k: int) -> list[DocMetadata]:
    """Per-document metadata: title (first non-empty line), size, paragraph count, top terms."""
    out: list[DocMetadata] = []
    for doc_id, text in docs.items():
        first_line = next((ln.strip(" #") for ln in text.splitlines() if ln.strip()), doc_id)
        counts = Counter(tokenize_terms(text))
        top = [t for t, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_terms_k]]
        out.append(
            {
                "doc_id": doc_id,
                "title": first_line,
                "n_chars": len(text),
                "n_paragraphs": len(split_paragraphs(text)),
                "top_terms": top,
            }
        )
    return out


def graph_rag_mapping(passages: list[Passage], terms: list[str]) -> dict[str, list[str]]:
    """Map each salient term (a graph node) to the anthology passage ids that ground it.

    This is the knowledge-graph-to-RAG bridge: a prompt template can cite, per concept, exactly
    which selected passages support it -- only terms with at least one grounding passage are kept."""
    mapping: dict[str, list[str]] = {}
    for term in terms:
        grounded = [p["passage_id"] for p in passages if term in tokenize_terms(p["text"])]
        if grounded:
            mapping[term] = grounded
    return mapping


def build_corpus_package(
    docs: dict[str, str],
    *,
    max_passages: int = 12,
    min_passage_chars: int = 120,
    top_terms_k: int = 12,
) -> CorpusPackage:
    """Assemble the anthology + metadata + graph/RAG mapping from an in-memory corpus."""
    if not docs:
        raise ValueError("empty corpus: no .md/.txt documents to prepare")
    terms = salient_terms(docs, top_terms_k)
    anthology = select_anthology(docs, max_passages=max_passages, min_chars=min_passage_chars)
    metadata = doc_metadata(docs, top_terms_k=top_terms_k)
    mapping = graph_rag_mapping(anthology, terms)
    return CorpusPackage(
        anthology=anthology,
        metadata=metadata,
        graph_rag_mapping=mapping,
        salient_terms=terms,
    )
