"""Resolution for a LOCALLY FINE-TUNED embedder directory (`llb finetune-embedder`).

A tuned encoder is a DIRECTORY on this host, not a hub id, so the two things every encoder lane
reads off an id cannot be read off this string:

  - **which query/passage convention it is encoded under.** Fine-tuning does not change the format
    the weights were trained to expect, so a tuned E5 is still `query: ` / `passage: `. The
    convention therefore resolves through the BASE model recorded in the tuned manifest, and a
    tuned directory whose base nobody registered is refused exactly like any unregistered id
    (`llb.rag.encoders.families`).
  - **which encoder a store was built with.** A path is not an identity: training again into the
    same directory produces DIFFERENT weights under the same string, and a store built by the first
    one and queried by the second silently retrieves against vectors no encoder in the process
    produced. So the identity is the base model plus the tuned digest, and that fingerprint is what
    the store records and the query path checks
    (`llb.rag.vector_store.validation.store_embedder_mismatch`).

Pure apart from reading one small JSON manifest: no torch, no network, no store.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from llb.rag.encoders.families import (
    EmbeddingConvention,
    embedding_family,
    is_registered,
    resolve_convention,
)

# Written by `llb finetune-embedder` into the tuned model directory.
TUNED_EMBEDDER_MANIFEST = "embedder_manifest.json"
MANIFEST_KIND = "llb.finetune.embedder"

# Characters of the tuned digest carried in a fingerprint -- enough to separate two trainings of
# the same base on the same host, short enough to read in a store meta or an error message.
DIGEST_SHORT_CHARS = 12
FINGERPRINT_PREFIX = "tuned"


@dataclass(frozen=True)
class TunedEmbedder:
    """What a tuned directory declares about itself: where it came from and which training it is."""

    path: str
    base_model: str
    tuned_digest: str

    @property
    def fingerprint(self) -> str:
        """The encoder identity a store records: base model + which training produced it."""
        return f"{FINGERPRINT_PREFIX}:{self.base_model}:{self.tuned_digest[:DIGEST_SHORT_CHARS]}"


def load_tuned_embedder(model: str) -> TunedEmbedder | None:
    """The tuned record for `model`, or None when it is an ordinary (hub or local) encoder id.

    A directory carrying a manifest of another kind is NOT a tuned embedder, so it reads as an
    ordinary id rather than being half-resolved from fields it does not have.
    """
    if not model:
        return None
    manifest_path = Path(model) / TUNED_EMBEDDER_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("kind") != MANIFEST_KIND:
        return None
    base_model = str(payload.get("base_model") or "")
    digest = str(payload.get("tuned_digest") or "")
    if not base_model or not digest:
        return None
    return TunedEmbedder(path=model, base_model=base_model, tuned_digest=digest)


def convention_id(model: str) -> str:
    """The id whose declared convention governs `model` (its base, for a tuned directory)."""
    tuned = load_tuned_embedder(model)
    return tuned.base_model if tuned is not None else model


def resolved_convention(model: str) -> EmbeddingConvention:
    """`resolve_convention` that sees through a tuned directory to its base model."""
    return resolve_convention(convention_id(model))


def resolved_family(model: str) -> str:
    """`embedding_family` that sees through a tuned directory to its base model."""
    return embedding_family(convention_id(model))


def convention_registered(model: str) -> bool:
    """`is_registered` that sees through a tuned directory to its base model."""
    return is_registered(convention_id(model))


def embedder_fingerprint(model: str) -> str:
    """The identity a store records for `model`: its fingerprint when tuned, else the id itself."""
    tuned = load_tuned_embedder(model)
    return tuned.fingerprint if tuned is not None else model
