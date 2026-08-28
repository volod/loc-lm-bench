"""Pinned text embedder (sentence-transformers, lazy-loaded) with per-family conventions.

The embedding model is validated separately and PINNED (Premise 4): a weak Ukrainian
embedder silently caps every generation model's RAG score. This wraps one SentenceTransformer
behind a tiny interface and applies each model FAMILY's required query/passage convention from
the registry in `llb.rag.encoders.families`, because a retrieval-tuned encoder scored with the
WRONG convention silently loses recall -- exactly the failure the embedding bake-off
(`llb compare-embeddings`, `src/llb/rag/embedding_bakeoff/run.py`) must never introduce.

An id with no registered convention resolves to `unknown` and is encoded with no instruction, so
this module WARNS rather than passing silently; the bake-off refuses such a candidate outright
(`llb.rag.embedding_bakeoff.roster`). A locally fine-tuned directory has no convention readable
from its path, so it resolves through the BASE model its manifest records
(`llb.rag.encoders.tuned`) -- fine-tuning changes the weights, not the format they expect.

Heavy imports (`sentence_transformers`, `numpy`) are deferred to first use so the module
imports fine in the base install; the real embedding path needs the `[rag]` extra.
"""

import logging
import os
from typing import Any

from llb.core import env
from llb.core.config_validation import DEFAULT_EMBEDDING_MODEL
from llb.rag.encoders.precision import (
    DTYPE_AUTO,
    load_model_kwargs,
    normalize_dtype,
)
from llb.rag.encoders.families import (
    FAMILY_UNKNOWN,
    EmbeddingConvention,
    apply_passage_convention,
    apply_query_convention,
    embedding_family,
    resolve_convention,
)
from llb.rag.encoders.tuned import convention_id

_LOG = logging.getLogger(__name__)

# `LLB_TRUST_REMOTE_CODE` values that opt into executing a model repo's own modelling code.
TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def remote_code_opt_in() -> bool:
    """Whether the process opted into `trust_remote_code` (`LLB_TRUST_REMOTE_CODE`)."""
    return os.environ.get(env.LLB_TRUST_REMOTE_CODE, "").strip().lower() in TRUTHY_ENV_VALUES


class Embedder:
    """Lazy wrapper over a SentenceTransformer; normalizes vectors for cosine/IP search."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        *,
        trust_remote_code: bool | None = None,
        dtype: str | None = None,
    ):
        self.model_name = model_name
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._dtype = dtype
        self._model = None
        # A locally fine-tuned directory carries its base model's convention, not one readable from
        # its path (`llb.rag.encoders.tuned`). Resolved once here so every encode call is a lookup.
        self._convention_id = convention_id(model_name)

    @property
    def family(self) -> str:
        """The query/passage convention family this model belongs to."""
        return embedding_family(self._convention_id)

    @property
    def convention(self) -> EmbeddingConvention:
        """The documented query/passage format this model is encoded under."""
        return resolve_convention(self._convention_id)

    def _resolve_device(self) -> str | None:
        """Device for the SentenceTransformer: explicit constructor arg wins, else the
        `LLB_EMBED_DEVICE` env knob, else `None` (sentence-transformers auto-selects)."""
        return self._device or os.environ.get(env.LLB_EMBED_DEVICE) or None

    def _resolve_dtype(self) -> str:
        """Declared load precision: constructor arg wins, else the `LLB_EMBED_DTYPE` env knob.

        `auto` (the default) inherits each checkpoint's uploaded precision, which is what every
        recorded reading was taken at; a declared value makes precision a controlled variable
        across a mixed roster (`llb.rag.encoders.precision`).
        """
        return normalize_dtype(self._dtype or os.environ.get(env.LLB_EMBED_DTYPE))

    def effective_dtype(self) -> str | None:
        """The precision the loaded weights actually hold (None before the model is loaded)."""
        if self._model is None:
            return None
        try:
            return str(next(self._model.parameters()).dtype).removeprefix("torch.")
        except (StopIteration, AttributeError):  # a fake or parameter-free embedder in tests
            return None

    def _resolve_remote_code(self) -> bool:
        """Whether repo-supplied modelling code may run: constructor arg wins, else the env knob."""
        if self._trust_remote_code is not None:
            return self._trust_remote_code
        return remote_code_opt_in()

    def _load_kwargs(self) -> dict[str, Any]:
        """SentenceTransformer constructor kwargs for this model's family.

        A family that ships its forward pass as repository code is an EXECUTION decision, so an
        un-opted-in load is refused here rather than silently running downloaded code.
        """
        convention = self.convention
        dtype = self._resolve_dtype()
        kwargs: dict[str, Any] = {}
        if dtype != DTYPE_AUTO:
            kwargs["model_kwargs"] = load_model_kwargs(dtype)
        if convention.family == FAMILY_UNKNOWN:
            _LOG.warning(
                "[embedding] no registered query/passage convention for %s -- encoding with NO "
                "instruction, which caps recall for a retrieval-tuned encoder. Register it in "
                "llb.rag.encoders.families.",
                self.model_name,
            )
        if not convention.trust_remote_code:
            return kwargs
        if not self._resolve_remote_code():
            raise SystemExit(
                f"ERROR: {self.model_name} ships its own modelling code and needs "
                f"trust_remote_code. It is opt-in: set {env.LLB_TRUST_REMOTE_CODE}=1 (or pass "
                "--allow-remote-code to compare-embeddings) after reviewing "
                f"{convention.source}."
            )
        _LOG.warning(
            "[embedding] %s runs repository-supplied modelling code (trust_remote_code=True), "
            "opted in explicitly",
            self.model_name,
        )
        kwargs["trust_remote_code"] = True
        return kwargs

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from transformers.utils.logging import disable_progress_bar
            except ImportError as exc:
                raise SystemExit(
                    'ERROR: embeddings need the [rag] extra. Run: uv pip install -e ".[rag]"'
                ) from exc
            # Persisted CLI logs must remain line-oriented ASCII, not contain tqdm control output.
            disable_progress_bar()
            self._model = SentenceTransformer(
                self.model_name, device=self._resolve_device(), **self._load_kwargs()
            )
        return self._model

    def loaded_model(self) -> Any:
        """The loaded SentenceTransformer, for a caller that must make the model's OWN call.

        The card-parity probe (`llb.rag.encoders.card_probe`) is the reason this exists: for a card
        that publishes a runnable snippet instead of numbers, the reference is that snippet run on
        these weights, and the snippet calls sentence-transformers directly.
        """
        return self._load()

    def encode_passages(self, texts: list[str]) -> Any:
        """Embed corpus chunks. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            apply_passage_convention(self._convention_id, texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            **self.convention.passage_kwargs,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_queries(self, texts: list[str]) -> Any:
        """Embed questions. Returns a float32 (n, dim) numpy array, L2-normalized."""
        import numpy as np

        model = self._load()
        vectors = model.encode(
            apply_query_convention(self._convention_id, texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            **self.convention.query_kwargs,
        )
        return np.asarray(vectors, dtype="float32")

    def release(self) -> None:
        """Drop loaded weights and free CUDA cache so the next candidate does not stack VRAM."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # optional path; never fail a bake-off for cache cleanup
            pass

    # --- token-level passage hooks (late chunking, `llb.rag.late_encoding`) ---

    def max_seq_tokens(self) -> int:
        """The encoder's window in tokens (late chunking sizes its document windows by it)."""
        return int(self._load().get_max_seq_length() or 512)

    def passage_token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Char span of every token of raw `text` (no special tokens, no truncation).

        `verbose=False` silences the tokenizer's over-max-length warning: this untruncated pass
        only extracts offsets for late-chunking windowing -- the model never sees the sequence.
        """
        encoded = self._load().tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return [(start, end) for start, end in encoded["offset_mapping"] if end > start]

    def encode_passage_tokens(self, text: str) -> tuple[list[tuple[int, int]], list[list[float]]]:
        """Per-token char spans + embeddings for ONE passage window (<= `max_seq_tokens`).

        The window is encoded under the family's PASSAGE convention (prefix included), and
        prefix/special tokens are dropped so every returned span indexes into raw `text`.
        """
        model = self._load()
        prefixed = apply_passage_convention(self._convention_id, [text])[0]
        shift = len(prefixed) - len(text)
        token_vectors = model.encode(
            prefixed,
            output_value="token_embeddings",
            show_progress_bar=False,
            **self.convention.passage_kwargs,
        ).tolist()
        offsets = model.tokenizer(
            prefixed,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_tokens(),
        )["offset_mapping"]
        spans: list[tuple[int, int]] = []
        vectors: list[list[float]] = []
        for (start, end), vector in zip(offsets, token_vectors):
            if end <= max(start, shift):  # special tokens (0,0) and the passage prefix
                continue
            spans.append((max(0, start - shift), end - shift))
            vectors.append([float(x) for x in vector])
        return spans, vectors
