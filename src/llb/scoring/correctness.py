"""Reference-based answer correctness (objective, pure Python).

This is the axis that RANKS generation models (design: reference-based answer
correctness, NOT retrieval recall, which is constant under pinned retrieval). Three
complementary signals over Unicode-normalized text so Ukrainian morphology and casing do
not break matching:

  - exact:    normalized strings are identical (strict, sparse)
  - token_f1: SQuAD-style token overlap F1 (graded, the headline objective signal)
  - contains: all reference tokens appear in the prediction (recall-ish, lenient)

`score` is token_f1 -- a single graded number for ranking. Semantic-embedding similarity
is an optional later signal (needs the pinned embedder); the objective axis stays
dependency-free here.
"""

import unicodedata
from typing import Any

from llb.contracts import CorrectnessScores


def normalize(text: str) -> str:
    """Lowercase, drop punctuation/marks, collapse whitespace (Unicode-aware)."""
    if not text:
        return ""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(ch)
        if category[0] in ("P", "S"):  # punctuation, symbols
            out.append(" ")
        elif category[0] == "C":  # control
            continue
        else:
            out.append(ch.lower())
    return " ".join("".join(out).split())


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) and normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 over normalized tokens."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    pred_counts: dict[str, int] = {}
    for tok in pred:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    overlap = sum(min(count, ref_counts.get(tok, 0)) for tok, count in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains(prediction: str, reference: str) -> float:
    """1.0 if every reference token appears somewhere in the prediction."""
    pred = set(_tokens(prediction))
    ref = _tokens(reference)
    if not ref:
        return 0.0
    return 1.0 if all(tok in pred for tok in ref) else 0.0


def semantic_similarity(prediction: str, reference: str, embedder: Any) -> float:
    """Cosine similarity of prediction vs reference via the PINNED embedder.

    The objective axis includes a semantic match (design: exact / semantic / structured
    match) for paraphrases that token overlap misses under Ukrainian morphology. The
    `embedder` is injected (duck-typed `encode_queries(list[str]) -> normalized vectors`),
    so this stays dependency-free and unit-testable with a fake. Returns 0.0 for empty
    inputs; clamped to [0, 1].
    """
    if not prediction.strip() or not reference.strip():
        return 0.0
    vectors = embedder.encode_queries([prediction, reference])
    cosine = sum(float(a) * float(b) for a, b in zip(vectors[0], vectors[1]))
    return max(0.0, min(1.0, cosine))


def answer_correctness(prediction: str, reference: str, embedder: Any = None) -> CorrectnessScores:
    """All objective signals plus the headline `score` (token_f1).

    `score` stays token_f1 -- a stable, dependency-free ranking number. When an `embedder`
    is supplied, a `semantic` signal is added (recorded, not yet blended into `score`;
    blending weights are a tuning decision).
    """
    f1 = token_f1(prediction, reference)
    out: CorrectnessScores = {
        "exact": exact_match(prediction, reference),
        "token_f1": f1,
        "contains": contains(prediction, reference),
        "score": f1,
    }
    if embedder is not None:
        out["semantic"] = semantic_similarity(prediction, reference, embedder)
    return out
