"""Traceable claim-bearing transformations for hard-control research."""

from dataclasses import dataclass
from hashlib import sha256
import re

from llb.conflicts.null_research.geometry import CorpusGeometry, EmbedTexts
from llb.conflicts.semantic_tree.vector_math import dot
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.contracts.common import JsonObject

_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_NEGATION = "\u043d\u0435"
_MODAL_WORDS = (
    "\u043c\u0430\u0454",
    "\u043c\u043e\u0436\u0435",
    "\u043f\u043e\u0432\u0438\u043d\u0435\u043d",
    "\u043f\u043e\u0432\u0438\u043d\u043d\u0430",
    "\u043f\u043e\u0432\u0438\u043d\u043d\u0456",
    "\u0434\u043e\u0437\u0432\u043e\u043b\u0435\u043d\u043e",
    "\u0437\u0430\u0431\u043e\u0440\u043e\u043d\u0435\u043d\u043e",
)
_MODAL = re.compile(r"\b(" + "|".join(re.escape(word) for word in _MODAL_WORDS) + r")\b", re.I)


@dataclass(frozen=True)
class _Transformation:
    source: int
    edit_type: str
    transformed: str
    start: int
    end: int
    before: str
    after: str


@dataclass(frozen=True)
class CounterfactualControls:
    """Original-to-transformed scores and their construction provenance."""

    scores: list[float]
    source_scores: dict[int, list[float]]
    effective_units: int
    diagnostics: JsonObject
    traces: list[JsonObject]

    def cluster_maxima(self) -> list[float]:
        return sorted(max(values) for values in self.source_scores.values())


def _replace(text: str, match: re.Match[str], replacement: str) -> str:
    return text[: match.start()] + replacement + text[match.end() :]


def _changed_number(value: str) -> str:
    digits = list(value)
    for index in range(len(digits) - 1, -1, -1):
        if digits[index].isdigit():
            digits[index] = str((int(digits[index]) + 1) % 10)
            break
    return "".join(digits)


def _content_start(text: str) -> int:
    """Skip YAML front matter when a first recursive chunk still carries the document header."""
    if not text.startswith("---"):
        return 0
    closing = text.find("\n---", 3)
    if closing < 0:
        return 0
    return closing + len("\n---")


def _capitalized_pool(target: CorpusGeometry) -> list[str]:
    entities = {
        match.group()
        for ordinal in target.allowed
        for match in _WORD.finditer(
            target.chunks[ordinal]["text"],
            _content_start(target.chunks[ordinal]["text"]),
        )
        if len(match.group()) > 3 and match.group()[0].isupper() and not match.group().isupper()
    }
    return sorted(entities, key=lambda value: (value.casefold(), value))


def _capitalized_argument_transformation(
    target: CorpusGeometry, source: int, pool: list[str]
) -> _Transformation | None:
    text = target.chunks[source]["text"]
    content_start = _content_start(text)
    candidates = [
        match
        for match in _WORD.finditer(text, content_start)
        if match.start() > content_start
        and len(match.group()) > 3
        and match.group()[0].isupper()
        and not match.group().isupper()
    ]
    if not candidates or len(pool) < 2:
        return None
    match = candidates[0]
    replacements = [word for word in pool if word.casefold() != match.group().casefold()]
    if not replacements:
        return None
    replacement = replacements[source % len(replacements)]
    return _Transformation(
        source=source,
        edit_type="capitalized_argument_swap",
        transformed=_replace(text, match, replacement),
        start=match.start(),
        end=match.end(),
        before=match.group(),
        after=replacement,
    )


def _transformations(target: CorpusGeometry) -> list[_Transformation]:
    pool = _capitalized_pool(target)
    out: list[_Transformation] = []
    for source in target.allowed:
        text = target.chunks[source]["text"]
        content_start = _content_start(text)
        number = _NUMBER.search(text, content_start)
        if number is not None:
            replacement = _changed_number(number.group())
            out.append(
                _Transformation(
                    source,
                    "quantity_or_date_swap",
                    _replace(text, number, replacement),
                    number.start(),
                    number.end(),
                    number.group(),
                    replacement,
                )
            )
        argument = _capitalized_argument_transformation(target, source, pool)
        if argument is not None:
            out.append(argument)
        modal = _MODAL.search(text, content_start)
        if modal is not None:
            prefix = text[max(0, modal.start() - len(_NEGATION) - 1) : modal.start()].casefold()
            if prefix == _NEGATION + " ":
                start = modal.start() - len(_NEGATION) - 1
                end = modal.start()
                before = text[start:end]
                after = ""
            else:
                start = end = modal.start()
                before = ""
                after = _NEGATION + " "
            out.append(
                _Transformation(
                    source,
                    "modality_flip",
                    text[:start] + after + text[end:],
                    start,
                    end,
                    before,
                    after,
                )
            )
    return out


def _trace(target: CorpusGeometry, item: _Transformation) -> JsonObject:
    chunk = target.chunks[item.source]
    reconstructed = chunk["text"][: item.start] + item.after + chunk["text"][item.end :]
    return {
        "dataset": target.name,
        "source_ordinal": item.source,
        "doc_id": chunk["doc_id"],
        "chunk_id": chunk.get("chunk_id", ""),
        "edit_type": item.edit_type,
        "char_start": item.start,
        "char_end": item.end,
        "before": item.before,
        "after": item.after,
        "source_sha256": sha256(chunk["text"].encode()).hexdigest(),
        "transformed_sha256": sha256(item.transformed.encode()).hexdigest(),
        "trace_verified": reconstructed == item.transformed
        and chunk["text"][item.start : item.end] == item.before,
    }


def counterfactual_traces(target: CorpusGeometry) -> list[JsonObject]:
    """Construction provenance for every deterministic edit, without embedding any of them."""
    return [_trace(target, item) for item in _transformations(target)]


def build_counterfactual_controls(
    target: CorpusGeometry, embed: EmbedTexts
) -> CounterfactualControls:
    """Embed deterministic counterfactuals and retain an exact edit trace for every row."""
    transformations = _transformations(target)
    if not transformations:
        raise ValueError(f"{target.name} has no transformable claim-bearing chunks")
    transformed = VectorSet.from_any(embed([item.transformed for item in transformations]))
    if transformed.dim != target.vectors.dim:
        raise ValueError("counterfactual embedder dimension differs from the target store")
    if target.center_mean is not None:
        transformed = transformed.centered(target.center_mean)
    source_scores: dict[int, list[float]] = {}
    traces: list[JsonObject] = []
    edit_counts: dict[str, int] = {}
    for position, item in enumerate(transformations):
        score = dot(target.vectors.row(item.source), transformed.row(position))
        source_scores.setdefault(item.source, []).append(score)
        edit_counts[item.edit_type] = edit_counts.get(item.edit_type, 0) + 1
        traces.append(_trace(target, item))
    unique_source_texts = len({target.chunks[source]["text"] for source in source_scores})
    return CounterfactualControls(
        scores=sorted(score for values in source_scores.values() for score in values),
        source_scores=source_scores,
        effective_units=unique_source_texts,
        diagnostics={
            "control_rows": len(transformations),
            "source_clusters": len(source_scores),
            "unique_source_texts": unique_source_texts,
            "transformation_counts": edit_counts,
            "all_traces_verified": all(bool(trace["trace_verified"]) for trace in traces),
            "independent_semantic_verification": False,
        },
        traces=traces,
    )
