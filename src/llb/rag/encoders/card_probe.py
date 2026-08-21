"""Run an encoder candidate's declared card example on this host, before it is allowed a row.

The tables and the arithmetic are pure (`llb.rag.encoders.cards`); this is the half that touches a
model. It loads the candidate exactly the way the bake-off will -- same convention, same
`trust_remote_code` decision, same declared precision -- runs its card's own example through
`Embedder`, and releases the weights. So a candidate cleared here is cleared on the call path its
row is measured on, and a candidate that fails is refused BEFORE a store is built rather than after.

For a card that publishes a snippet instead of numbers, the reference side is that snippet: the
card's own `encode(texts, task=..., prompt_name=...)` call, made on the SAME loaded weights. What
that isolates is the half a published number could not settle anyway -- whether the query/passage
format this repo registered is the one the model's own configuration declares. A registry that has
drifted from the repo's prompts silently understates the encoder, and no card number would catch it.

Heavy imports are deferred to first use, as in `llb.rag.encoders.embedder`.
"""

import logging
from collections.abc import Sequence
from typing import Any

from llb.rag.encoders.card_parity import CardParityResult
from llb.rag.encoders.embedder import Embedder
from llb.rag.encoders.cards import card_reference, check_encoder_card

_LOG = logging.getLogger(__name__)


def probe_encoder_card(
    model_name: str,
    *,
    device: str | None = None,
    trust_remote_code: bool | None = None,
    dtype: str | None = None,
) -> CardParityResult:
    """Verdict on whether this encoder reproduces its own model card on this host.

    An id with no declared reference returns the `no_reference_declared` verdict without loading
    anything: there is nothing to check, and the row says so rather than claiming a check happened.
    """
    reference = card_reference(model_name)
    if reference is None:
        return check_encoder_card(
            model_name,
            encode_queries=lambda texts: [],
            encode_passages=lambda texts: [],
        )
    embedder = Embedder(model_name, device=device, trust_remote_code=trust_remote_code, dtype=dtype)

    def reference_encode(texts: list[str], task: str) -> Sequence[Sequence[float]]:
        """The card's own call: let the MODEL apply its declared prompt for `task`."""
        model: Any = embedder.loaded_model()
        vectors = model.encode(texts, task=task, prompt_name=task, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    try:
        result = check_encoder_card(
            model_name,
            encode_queries=embedder.encode_queries,
            encode_passages=embedder.encode_passages,
            reference_encode=reference_encode if reference.reference_implementation else None,
        )
    finally:
        embedder.release()
    _LOG.info("[card-parity] %s: %s -- %s", model_name, result["status"], result["detail"])
    return result
