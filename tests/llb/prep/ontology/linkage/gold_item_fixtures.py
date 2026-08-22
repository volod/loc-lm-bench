"""Deterministic inputs for the gold-item linkage lane: a fixture batch and a fake embedder.

The prior bundle is a slice of the committed post-edited fixture; the drafted batch is built from
it, so the duplicates are real repeats of real Ukrainian questions rather than invented strings.
No GPU and no model server: the embedder is a hashed bag of words, which gives the graded
similarity a single-basis fake embedder cannot.
"""

import hashlib
import math
import re

from llb.core.paths import PROJECT_ROOT
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset
from llb.prep.ontology.drafting.question_types import classify_question_type
from llb.prep.ontology.models import ItemLabels

PRIOR_GOLDSET = PROJECT_ROOT / "samples" / "goldsets" / "ua_squad_postedited_v1" / "goldset.jsonl"
OTHER_GOLDSET = PROJECT_ROOT / "samples" / "goldsets" / "ip_regulation_uk" / "goldset.jsonl"

N_PRIOR = 60
N_COPIES = 6
N_PARAPHRASES = 5
_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingBagEmbedder:
    """A hashed bag of words: shared tokens raise the cosine, and the width never changes."""

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        rows: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimension
            for token in _TOKEN.findall(text.casefold()):
                vector[hashlib.md5(token.encode("utf-8")).digest()[0] % self._dimension] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            rows.append([value / norm for value in vector])
        return rows


def write_prior_bundle(root) -> tuple[object, list[GoldItem]]:
    """A prior draft bundle holding the first `N_PRIOR` items of the committed fixture."""
    prior = load_goldset(PRIOR_GOLDSET)[:N_PRIOR]
    bundle = root / "prior"
    bundle.mkdir(parents=True, exist_ok=True)
    dump_goldset(prior, bundle / "goldset.jsonl")
    return bundle, prior


def _paraphrase(item: GoldItem) -> str:
    """Same content words, a different opening -- a repeat the question cosine alone can miss."""
    words = item.question.split()
    return " ".join(["Отже,", *words[1:]]) if len(words) > 3 else f"{item.question} саме"


def draft_batch(prior: list[GoldItem]) -> list[GoldItem]:
    """Exact repeats, near-paraphrases, one intra-batch repeat, and genuinely new questions."""
    copies = [
        item.model_copy(update={"id": f"copy-{n}"}) for n, item in enumerate(prior[:N_COPIES])
    ]
    paraphrases = [
        item.model_copy(update={"id": f"para-{n}", "question": _paraphrase(item)})
        for n, item in enumerate(prior[20 : 20 + N_PARAPHRASES])
    ]
    fresh = [
        item.model_copy(update={"id": f"new-{n}"})
        for n, item in enumerate(load_goldset(OTHER_GOLDSET))
    ]
    return [*copies, *paraphrases, *fresh, fresh[-1].model_copy(update={"id": "repeat-1"})]


def draft_labels(items: list[GoldItem]) -> dict[str, ItemLabels]:
    return {
        item.id: ItemLabels(
            question_type=classify_question_type(item.question, item.reference_answer),
            difficulty="medium",
        )
        for item in items
    }
