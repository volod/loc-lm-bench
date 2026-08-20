"""Declared load precision for encoder candidates: what the publisher shipped vs what we measure.

Warm chunks/s is read as a MODEL property -- "this encoder is 3.4x faster than that one" -- and on
a mixed roster it is not one. The published checkpoints differ in PRECISION: `multilingual-e5-base`
ships float32, `multilingual-e5-large-instruct` float16, `Qwen3-Embedding-0.6B` bfloat16, so a
throughput column measured at "whatever each publisher uploaded" compares checkpoints, not
architectures, and a half-precision row wins a race the operator never entered it in.

So precision is a DECLARED knob rather than an inherited one:

  - `auto` (the default) keeps each checkpoint's own dtype. It is what every recorded reading was
    taken at, so it is what reproduces them;
  - an explicit dtype (`float32` / `float16` / `bfloat16`) loads EVERY candidate at that precision,
    which is what makes the throughput column a recommendation.

`PUBLISHED_CHECKPOINT_DTYPE` records what each roster repository actually uploaded (read from its
`config.json`), so a report can state the requested dtype, the effective one, and the published one
side by side -- a row measured at `auto` still says which precision that turned out to be.

Pure and dependency-free: no torch import, no download. `torch_dtype` strings are handed to
transformers, which resolves them.
"""

from typing import Mapping

# Keep each checkpoint's uploaded precision. The default, because it is what the recorded
# encoder readings were taken at.
DTYPE_AUTO = "auto"

DTYPE_FLOAT32 = "float32"
DTYPE_FLOAT16 = "float16"
DTYPE_BFLOAT16 = "bfloat16"

# What `--encoder-dtype` accepts. `auto` is not a precision; it is the absence of a declared one.
SUPPORTED_DTYPES: tuple[str, ...] = (DTYPE_AUTO, DTYPE_FLOAT32, DTYPE_FLOAT16, DTYPE_BFLOAT16)

# The precision to declare when an operator wants the throughput column to be a MODEL comparison:
# every incumbent already ships float32, so it is the one value that changes the fewest rows while
# putting the half-precision newcomers on the same footing.
CONTROLLED_DTYPE = DTYPE_FLOAT32

# model id -> the `torch_dtype` its published `config.json` declares. Recorded so the throughput
# column can be read with the checkpoint precision beside it even on an `auto` run.
PUBLISHED_CHECKPOINT_DTYPE: Mapping[str, str] = {
    "intfloat/multilingual-e5-small": DTYPE_FLOAT32,
    "intfloat/multilingual-e5-base": DTYPE_FLOAT32,
    "intfloat/multilingual-e5-large": DTYPE_FLOAT32,
    "intfloat/multilingual-e5-large-instruct": DTYPE_FLOAT16,
    "BAAI/bge-m3": DTYPE_FLOAT32,
    "Alibaba-NLP/gte-multilingual-base": DTYPE_FLOAT16,
    "jinaai/jina-embeddings-v3": DTYPE_BFLOAT16,
    "Qwen/Qwen3-Embedding-0.6B": DTYPE_BFLOAT16,
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base": DTYPE_FLOAT32,
    "BAAI/bge-reranker-v2-m3": DTYPE_FLOAT32,
    "jinaai/jina-reranker-v2-base-multilingual": DTYPE_BFLOAT16,
    "Alibaba-NLP/gte-multilingual-reranker-base": DTYPE_FLOAT16,
    "mixedbread-ai/mxbai-rerank-base-v2": DTYPE_BFLOAT16,
    "Qwen/Qwen3-Reranker-0.6B": DTYPE_BFLOAT16,
}


class UnsupportedDtypeError(ValueError):
    """A requested load precision this lane does not accept."""


def normalize_dtype(requested: str | None) -> str:
    """Validate a requested load precision, defaulting to `auto` for an empty request."""
    value = (requested or "").strip().lower() or DTYPE_AUTO
    if value not in SUPPORTED_DTYPES:
        raise UnsupportedDtypeError(
            f"unsupported encoder dtype {requested!r}; choose one of {', '.join(SUPPORTED_DTYPES)}"
        )
    return value


def published_dtype(model_name: str) -> str | None:
    """The precision this repository uploaded, or None for an id nobody recorded."""
    return PUBLISHED_CHECKPOINT_DTYPE.get(model_name)


def load_model_kwargs(dtype: str) -> dict[str, str]:
    """`model_kwargs` that pin a load precision (empty for `auto`, which inherits the checkpoint).

    `torch_dtype` rather than the newer `dtype` alias: it is honored by both the pinned
    transformers 5.x stack and the transformers 4.x legacy pass (`llb.rag.encoders.model_stack`), so one
    string works on both sides of the comparison.
    """
    return {} if dtype == DTYPE_AUTO else {"torch_dtype": dtype}
