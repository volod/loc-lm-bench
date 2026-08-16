"""Registry of per-model query/passage conventions for the RAG embedder.

A retrieval-tuned encoder scored with the WRONG instruction silently loses recall, and the loss is
invisible in a bake-off row -- the candidate simply looks worse than it is. That is what this table
exists to prevent, so it is a declared REGISTRY rather than a chain of substring tests:

  - resolution is ordered and explicit, so `multilingual-e5-large-instruct` resolves to the
    INSTRUCT convention (`Instruct: <task>\\nQuery: <q>` on the query side, NO prefix on the
    passage side) instead of falling into plain `e5` and being scored with `query:` / `passage:`;
  - an id nobody registered resolves to `unknown`, NOT to `plain`, because "symmetric, no prefix"
    is a documented property of the paraphrase/STS line and not a safe default for an encoder whose
    card nobody read (`llb.rag.embedding_bakeoff_roster` refuses an unregistered candidate);
  - every convention names the model card that documents it, so a row's format is auditable
    without re-reading this module's history.

Some encoders need more than a prefix: `jina-embeddings-v3` selects a task LoRA adapter through an
`encode()` keyword, so a convention carries per-side encode kwargs beside its prefixes. Some ship
their forward pass as repository code (`trust_remote_code`), which is an execution decision an
operator makes, not one a bake-off makes for them -- the convention records the requirement and the
loader refuses it unless explicitly opted into.
"""

from dataclasses import dataclass, field
from typing import Mapping

# The retrieval task description both instruct-style families put in front of a query. Same
# sentence on both model cards, so it is one constant rather than two copies that can drift.
RETRIEVAL_TASK = "Given a web search query, retrieve relevant passages that answer the query"

FAMILY_E5 = "e5"
FAMILY_E5_INSTRUCT = "e5-instruct"
FAMILY_BGE_M3 = "bge-m3"
FAMILY_BGE = "bge"
FAMILY_GTE_MULTILINGUAL = "gte-multilingual"
FAMILY_JINA_V3 = "jina-v3"
FAMILY_QWEN3_EMBEDDING = "qwen3-embedding"
FAMILY_PLAIN = "plain"
# Not a convention: the absence of one. Kept distinct from `plain` so an unregistered id is a
# decision the caller has to make rather than a silent no-instruction pass.
FAMILY_UNKNOWN = "unknown"

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
# BGE v1.5 / bge-large retrieval instruction (English line). BGE-M3 needs NO instruction on
# either side, so it resolves to FAMILY_BGE_M3 and is deliberately excluded here.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbeddingConvention:
    """One family's documented query/passage input format plus what loading it costs.

    `query_kwargs` / `passage_kwargs` are extra `SentenceTransformer.encode()` keywords the card
    documents (jina-v3's task adapter); an empty mapping is the common case. `source` is the model
    card the prefixes were read off, and `trust_remote_code` records that the family ships its
    forward pass as repository code.
    """

    family: str
    source: str
    query_prefix: str = ""
    passage_prefix: str = ""
    query_kwargs: Mapping[str, str] = field(default_factory=dict)
    passage_kwargs: Mapping[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False

    @property
    def symmetric(self) -> bool:
        """True when queries and passages are encoded identically (no instruction on either side)."""
        return not (self.query_prefix or self.passage_prefix or self.query_kwargs)


# family id -> its documented convention. Every entry cites the card its prefixes were read from.
CONVENTIONS: dict[str, EmbeddingConvention] = {
    FAMILY_E5: EmbeddingConvention(
        family=FAMILY_E5,
        source="https://huggingface.co/intfloat/multilingual-e5-base",
        query_prefix=E5_QUERY_PREFIX,
        passage_prefix=E5_PASSAGE_PREFIX,
    ),
    FAMILY_E5_INSTRUCT: EmbeddingConvention(
        family=FAMILY_E5_INSTRUCT,
        source="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
        # Card: `f'Instruct: {task_description}\nQuery: {query}'`; "No need to add instruction
        # for retrieval documents" -- so the passage side is BARE, not `passage: `.
        query_prefix=f"Instruct: {RETRIEVAL_TASK}\nQuery: ",
    ),
    FAMILY_BGE_M3: EmbeddingConvention(
        family=FAMILY_BGE_M3,
        source="https://huggingface.co/BAAI/bge-m3",
    ),
    FAMILY_BGE: EmbeddingConvention(
        family=FAMILY_BGE,
        source="https://huggingface.co/BAAI/bge-large-en-v1.5",
        query_prefix=BGE_QUERY_INSTRUCTION,
    ),
    FAMILY_GTE_MULTILINGUAL: EmbeddingConvention(
        family=FAMILY_GTE_MULTILINGUAL,
        source="https://huggingface.co/Alibaba-NLP/gte-multilingual-base",
        # Card's sentence-transformers usage encodes queries and documents in ONE bare
        # `model.encode(input_texts)` call: no instruction on either side.
        trust_remote_code=True,
    ),
    FAMILY_JINA_V3: EmbeddingConvention(
        family=FAMILY_JINA_V3,
        source="https://huggingface.co/jinaai/jina-embeddings-v3",
        # Prefixes are the repo's own `config_sentence_transformers.json` prompts for the
        # `retrieval.query` / `retrieval.passage` tasks; `task=` additionally selects the LoRA
        # adapter, which the prompt text alone does NOT do.
        query_prefix="Represent the query for retrieving evidence documents: ",
        passage_prefix="Represent the document for retrieval: ",
        query_kwargs={"task": "retrieval.query"},
        passage_kwargs={"task": "retrieval.passage"},
        trust_remote_code=True,
    ),
    FAMILY_QWEN3_EMBEDDING: EmbeddingConvention(
        family=FAMILY_QWEN3_EMBEDDING,
        source="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        # The repo's `config_sentence_transformers.json` "query" prompt, verbatim -- it ends at
        # `Query:` with NO trailing space (unlike e5-instruct), and the "document" prompt is "".
        query_prefix=f"Instruct: {RETRIEVAL_TASK}\nQuery:",
    ),
    FAMILY_PLAIN: EmbeddingConvention(
        family=FAMILY_PLAIN,
        source="https://huggingface.co/lang-uk/ukr-paraphrase-multilingual-mpnet-base",
    ),
    FAMILY_UNKNOWN: EmbeddingConvention(
        family=FAMILY_UNKNOWN,
        source="",
    ),
}

# Ordered resolution: the FIRST row whose every substring appears in the lowercased model id wins.
# Order is what makes this correct -- an instruct-tuned E5 carries `e5` in its id and would match
# the plain E5 row, so it is tested first. Same for BGE-M3 ahead of the BGE v1.5 line.
_MATCH_TABLE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("e5", "instruct"), FAMILY_E5_INSTRUCT),
    (("e5",), FAMILY_E5),
    (("bge-m3",), FAMILY_BGE_M3),
    (("bge_m3",), FAMILY_BGE_M3),
    (("bge",), FAMILY_BGE),
    (("gte-multilingual",), FAMILY_GTE_MULTILINGUAL),
    (("jina-embeddings-v3",), FAMILY_JINA_V3),
    (("qwen3-embedding",), FAMILY_QWEN3_EMBEDDING),
    # The symmetric paraphrase/STS line: no instruction is a DOCUMENTED property of these models,
    # which is why they are registered rather than left to fall through.
    (("paraphrase",), FAMILY_PLAIN),
    (("labse",), FAMILY_PLAIN),
    (("all-minilm",), FAMILY_PLAIN),
    (("all-mpnet",), FAMILY_PLAIN),
)


def embedding_family(model_name: str) -> str:
    """Resolve the query/passage convention family for a model id (case-insensitive).

    Returns `FAMILY_UNKNOWN` for an id no row matches. That is deliberately NOT `plain`: callers
    that would encode with no instruction at all must be able to see they are guessing.
    """
    name = model_name.lower()
    for required, family in _MATCH_TABLE:
        if all(part in name for part in required):
            return family
    return FAMILY_UNKNOWN


def resolve_convention(model_name: str) -> EmbeddingConvention:
    """The documented convention for a model id (the `unknown` record when nothing matches)."""
    return CONVENTIONS[embedding_family(model_name)]


def is_registered(model_name: str) -> bool:
    """Whether this model id has a declared convention (False means nobody read its card)."""
    return embedding_family(model_name) != FAMILY_UNKNOWN


def registered_models() -> tuple[str, ...]:
    """The family ids an operator can register a new candidate under, for error messages."""
    return tuple(family for family in CONVENTIONS if family != FAMILY_UNKNOWN)


def apply_query_convention(model_name: str, texts: list[str]) -> list[str]:
    """Prefix `texts` as QUERIES per the model family (no-op for symmetric families)."""
    prefix = resolve_convention(model_name).query_prefix
    return [prefix + t for t in texts] if prefix else list(texts)


def apply_passage_convention(model_name: str, texts: list[str]) -> list[str]:
    """Prefix `texts` as PASSAGES per the model family (no-op for families that tag neither side)."""
    prefix = resolve_convention(model_name).passage_prefix
    return [prefix + t for t in texts] if prefix else list(texts)
