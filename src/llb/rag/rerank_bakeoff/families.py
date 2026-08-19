"""Registry of per-model INPUT conventions for cross-encoder reranker candidates.

Same failure mode the embedder registry exists to prevent (`llb.rag.embedding_families`), one stage
later: a reranker scored without the instruction its card documents silently loses rank quality, and
the loss is invisible in a bake-off row -- the candidate simply looks worse than it is. So the
roster is a declared REGISTRY, not a chain of substring guesses, and every entry names the card it
was read from.

Two facts per candidate decide how it is loaded:

  - **the query-side instruction.** A reranker either scores the bare (question, passage) pair or
    puts a task instruction in front of the query. Every registered candidate that needs one ships
    it in its own `config_sentence_transformers.json` (`default_prompt_name`), so the loader lets
    sentence-transformers apply the model's own prompt rather than re-typing it here -- what this
    registry records is WHICH instruction that is, so a report reader can see the format a row was
    scored under without opening the repo.
  - **repository-supplied modelling code** (`trust_remote_code`). That executes code downloaded with
    the weights, which is an operator decision; the convention records the requirement and
    `llb.rag.rerank_bakeoff.roster` declines the candidate unless it was explicitly opted into.

Pure and dependency-free: no torch, no network.
"""

from dataclasses import dataclass

from llb.rag.embedding_families import RETRIEVAL_TASK
from llb.rag.model_stack import REQUIRED_TRANSFORMERS_MAJOR_LEGACY

FAMILY_BGE_RERANKER = "bge-reranker"
FAMILY_JINA_RERANKER_V2 = "jina-reranker-v2"
FAMILY_GTE_RERANKER = "gte-multilingual-reranker"
FAMILY_MXBAI_RERANK_V2 = "mxbai-rerank-v2"
FAMILY_QWEN3_RERANKER = "qwen3-reranker"
# Not a convention: the absence of one. An id nobody registered is REFUSED rather than scored
# under a guessed format (see `llb.rag.rerank_bakeoff.roster`).
FAMILY_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RerankConvention:
    """One reranker family's documented input format plus what loading it costs.

    `default_prompt` is the query-side instruction the model's OWN sentence-transformers config
    applies (None when the pair is scored bare); it is recorded for the report and for the operator,
    never re-typed into the call, so a card change cannot leave this table quietly disagreeing with
    the weights. `source` is the card the entry was read from.

    `requires_transformers_major` is the transformers major that repository code targets, when it
    targets one this repo does not pin -- a PACKAGING fact that routes the row to the legacy
    scoring pass instead of failing the run (`llb.rag.model_stack`).
    """

    family: str
    source: str
    trust_remote_code: bool = False
    default_prompt: str | None = None
    requires_transformers_major: int | None = None


# family id -> its documented convention. Every entry cites the card it was read from.
CONVENTIONS: dict[str, RerankConvention] = {
    FAMILY_BGE_RERANKER: RerankConvention(
        family=FAMILY_BGE_RERANKER,
        source="https://huggingface.co/BAAI/bge-reranker-v2-m3",
        # Plain XLM-R sequence classifier: the card's usage scores bare (query, passage) pairs.
    ),
    FAMILY_JINA_RERANKER_V2: RerankConvention(
        family=FAMILY_JINA_RERANKER_V2,
        source="https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual",
        # Ships its own XLM-RoBERTa flash implementation as repository code.
        trust_remote_code=True,
        # That code imports `create_position_ids_from_input_ids`, which transformers 5.x removed.
        requires_transformers_major=REQUIRED_TRANSFORMERS_MAJOR_LEGACY,
    ),
    FAMILY_GTE_RERANKER: RerankConvention(
        family=FAMILY_GTE_RERANKER,
        source="https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base",
        # `auto_map` points at Alibaba-NLP/new-impl modelling code.
        trust_remote_code=True,
        # Same uninitialized `position_ids` buffer as the gte encoder: an out-of-bounds rope
        # gather that reports IndexError on CPU and a device-side assert on CUDA.
        requires_transformers_major=REQUIRED_TRANSFORMERS_MAJOR_LEGACY,
    ),
    FAMILY_MXBAI_RERANK_V2: RerankConvention(
        family=FAMILY_MXBAI_RERANK_V2,
        source="https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2",
        # `config_sentence_transformers.json` declares `prompts: {}`: bare pairs, no instruction.
    ),
    FAMILY_QWEN3_RERANKER: RerankConvention(
        family=FAMILY_QWEN3_RERANKER,
        source="https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
        # `default_prompt_name: "query"` -- the same retrieval task sentence the Qwen3 embedding
        # line puts in front of a query, applied by the model's own sentence-transformers config.
        default_prompt=RETRIEVAL_TASK,
    ),
    FAMILY_UNKNOWN: RerankConvention(family=FAMILY_UNKNOWN, source=""),
}

# Ordered resolution: the FIRST row whose every substring appears in the lowercased model id wins.
# Order matters -- `bge-reranker-v2-m3` carries `bge` and `m3`, so the reranker rows are matched on
# the `rerank` token that no embedding id has.
_MATCH_TABLE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("bge-reranker",), FAMILY_BGE_RERANKER),
    (("jina-reranker",), FAMILY_JINA_RERANKER_V2),
    (("gte-multilingual-reranker",), FAMILY_GTE_RERANKER),
    (("mxbai-rerank",), FAMILY_MXBAI_RERANK_V2),
    (("qwen3-reranker",), FAMILY_QWEN3_RERANKER),
)


def rerank_family(model_name: str) -> str:
    """Resolve the input convention family for a reranker id (case-insensitive).

    Returns `FAMILY_UNKNOWN` for an id no row matches -- deliberately not a "bare pairs" default,
    because a caller that would score an unread card must be able to see it is guessing.
    """
    name = model_name.lower()
    for required, family in _MATCH_TABLE:
        if all(part in name for part in required):
            return family
    return FAMILY_UNKNOWN


def resolve_convention(model_name: str) -> RerankConvention:
    """The documented convention for a reranker id (the `unknown` record when nothing matches)."""
    return CONVENTIONS[rerank_family(model_name)]


def is_registered(model_name: str) -> bool:
    """Whether this reranker id has a declared convention (False means nobody read its card)."""
    return rerank_family(model_name) != FAMILY_UNKNOWN


def registered_families() -> tuple[str, ...]:
    """The family ids a new candidate can be registered under, for error messages."""
    return tuple(family for family in CONVENTIONS if family != FAMILY_UNKNOWN)
