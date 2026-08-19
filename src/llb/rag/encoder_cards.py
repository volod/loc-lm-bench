"""Per-encoder card references: the numbers a candidate must reproduce before it can be ranked.

Each entry is the model card's OWN retrieval example -- its queries, its passages, and the
similarities it prints -- run through the query/passage convention this repo registered for that
family (`llb.rag.embedding_families`). That makes the check cover both halves of a row's
readability at once: the weights load correctly AND the format we score them under is the format
the card documents. A candidate scored under a guessed prefix and a candidate whose remote code is
broken both fail here, and the report says which.

An id with no entry is scored ungated and says so on its row (`no_reference_declared`) -- several
cards publish no reference numbers at all, and "nobody checked" must not read as "it reproduces".

`jinaai/jina-embeddings-v3` is the one candidate whose card publishes a runnable snippet but no
numbers, so its reference is that snippet: the card's own
`encode(texts, task=..., prompt_name=...)` call, which lets the MODEL apply the prompt its
repository declares, against this registry's copy of that prompt applied by hand. What the
comparison isolates is therefore the format rather than the weights -- a registry entry that has
drifted from the repo's own prompt understates the encoder exactly the way a wrong prefix does, and
no published number could have caught it.

Pure apart from the injected encoders: the tables and the arithmetic have no torch, no download.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llb.rag.card_parity import (
    CardExpectation,
    CardParityResult,
    compare_to_card,
    probe_error_result,
    unpublished_result,
)
from llb.rag.embedding_families import resolve_convention

# The `encode()` keyword a task-adapter family selects its LoRA adapter with. The card's own
# reference call takes the same keyword, so the reference is made with the adapter the row is.
TASK_KWARG = "task"

# (texts) -> one embedding per text. Bound to `Embedder.encode_queries` / `encode_passages`.
EncodeTexts = Callable[[list[str]], Sequence[Sequence[float]]]
# (already-convention-applied texts, task) -> one embedding per text, via the card's own snippet.
ReferenceEncode = Callable[[list[str], str], Sequence[Sequence[float]]]

_E5_INSTRUCT_DOC_EN = (
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is "
    "46 grams per day. But, as you can see from this chart, you'll need to increase that if "
    "you're expecting or training for a marathon. Check out the chart below to see how much "
    "protein you should be eating each day."
)
_E5_INSTRUCT_DOC_ZH = (
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮"
    "去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟"
    "左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、"
    "将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不"
    "时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
)


@dataclass(frozen=True)
class EncoderCardReference:
    """One encoder card's documented retrieval example plus what it publishes about the result.

    `queries` are encoded under the family's QUERY convention and `passages` under its PASSAGE
    convention, so the observed matrix is row-major cosine(queries x passages) -- the same order
    every card prints. `reference_task` is set only for a card that publishes a snippet instead of
    numbers; the task keyword that snippet passes is the family's own (`query_kwargs` /
    `passage_kwargs`), so the reference cannot drift from the call the bake-off makes.
    """

    model: str
    source: str
    queries: tuple[str, ...]
    passages: tuple[str, ...]
    expectation: CardExpectation = field(default_factory=CardExpectation)
    reference_implementation: bool = False


ENCODER_CARD_REFERENCES: dict[str, EncoderCardReference] = {
    "Alibaba-NLP/gte-multilingual-base": EncoderCardReference(
        model="Alibaba-NLP/gte-multilingual-base",
        source="https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
        # The card's sentence-transformers block: one query against three texts, encoded bare on
        # both sides (this family's registered convention is symmetric).
        queries=("what is the capital of China?",),
        passages=(
            "how to implement quick sort in python?",
            "北京",
            "快排算法介绍",
        ),
        # Printed as `(embeddings[:1] @ embeddings[1:].T) * 100`, but the values the card shows
        # beneath that line are the unscaled cosines -- so the published numbers, not the formula,
        # are what is reproduced here.
        expectation=CardExpectation(values=(0.3016997, 0.7503870, 0.3203085)),
    ),
    "intfloat/multilingual-e5-large-instruct": EncoderCardReference(
        model="intfloat/multilingual-e5-large-instruct",
        source="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
        queries=("how much protein should a female eat", "南瓜的家常做法"),
        passages=(_E5_INSTRUCT_DOC_EN, _E5_INSTRUCT_DOC_ZH),
        # The card prints `(embeddings[:2] @ embeddings[2:].T) * 100`, and its numbers follow that
        # formula -- hence the scale. The queries are bare here because the registered `e5-instruct`
        # convention builds the card's own `Instruct: <task>\nQuery: ` prefix, which is the half of
        # the format this check is here to verify.
        expectation=CardExpectation(values=(91.92854, 67.58030, 70.38143, 92.13307), scale=100.0),
    ),
    "Qwen/Qwen3-Embedding-0.6B": EncoderCardReference(
        model="Qwen/Qwen3-Embedding-0.6B",
        source="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        queries=("What is the capital of China?", "Explain gravity"),
        passages=(
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to "
            "physical objects and is responsible for the movement of planets around the sun.",
        ),
        expectation=CardExpectation(values=(0.7646, 0.1414, 0.1355, 0.6000)),
    ),
    "jinaai/jina-embeddings-v3": EncoderCardReference(
        model="jinaai/jina-embeddings-v3",
        source="https://huggingface.co/jinaai/jina-embeddings-v3",
        # The card publishes a snippet and no numbers, so the reference is the snippet: its
        # `encode(..., task=t, prompt_name=t)` call, which applies the repo's own prompt for `t`.
        queries=("Follow the white rabbit.",),
        passages=(
            "Sigue al conejo blanco.",
            "Suis le lapin blanc.",
            "Folge dem weißen Kaninchen.",
        ),
        # Both sides run the same weights through the same library, so the only thing that can
        # move this number is the prompt string -- the default tolerance is already generous.
        expectation=CardExpectation(),
        reference_implementation=True,
    ),
}


def card_reference(model_name: str) -> EncoderCardReference | None:
    """The declared card reference for an encoder id (None when nobody recorded one)."""
    return ENCODER_CARD_REFERENCES.get(model_name)


def _normalized(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    rows: list[list[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        rows.append([value / norm for value in values])
    return rows


def similarity_matrix(
    queries: Sequence[Sequence[float]], passages: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """Row-major cosine(queries x passages) -- the order every encoder card prints."""
    left, right = _normalized(queries), _normalized(passages)
    return tuple(sum(a * b for a, b in zip(query, passage)) for query in left for passage in right)


def check_encoder_card(
    model_name: str,
    *,
    encode_queries: EncodeTexts,
    encode_passages: EncodeTexts,
    reference_encode: ReferenceEncode | None = None,
) -> CardParityResult:
    """Run this encoder's declared card example and say whether it reproduced the card.

    The encoders are injected, so the whole gate is testable over fakes. A probe that raises is a
    verdict (`probe_failed`), not an exception the bake-off has to survive: a candidate that cannot
    even run its card example has told us what we needed to know.
    """
    reference = card_reference(model_name)
    if reference is None:
        return unpublished_result(model_name)
    try:
        observed = similarity_matrix(
            encode_queries(list(reference.queries)), encode_passages(list(reference.passages))
        )
        expected = _reference_values(reference, reference_encode)
    except Exception as exc:  # a broken remote-code load raises here, which IS the verdict
        return probe_error_result(
            model_name, reference.source, f"card probe failed: {type(exc).__name__}: {exc}"
        )
    return compare_to_card(
        model_name, reference.source, reference.expectation, observed, expected=expected
    )


def _reference_values(
    reference: EncoderCardReference, reference_encode: ReferenceEncode | None
) -> tuple[float, ...] | None:
    """The expected side: published values, or the card's own snippet run here.

    The snippet receives the SAME convention-applied strings the scored path uses, so the prefixes
    cancel and what the comparison isolates is the load itself.
    """
    if not reference.reference_implementation:
        return None
    if reference_encode is None:
        raise RuntimeError(
            f"{reference.model} publishes no reference numbers; its card's own reference "
            "implementation is required to check parity and none was provided"
        )
    convention = resolve_convention(reference.model)
    # RAW texts on the reference side: the card's own call applies the repo's declared prompt for
    # the task, which is exactly the string this registry claims to have copied. The scored side
    # applies the registry's copy. The two matrices agreeing IS the claim being checked.
    return similarity_matrix(
        reference_encode(list(reference.queries), str(convention.query_kwargs.get(TASK_KWARG, ""))),
        reference_encode(
            list(reference.passages), str(convention.passage_kwargs.get(TASK_KWARG, ""))
        ),
    )
