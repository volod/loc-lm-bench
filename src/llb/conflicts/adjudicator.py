"""Build the injectable prompt -> text callable the model-adjudicated lanes share.

The claim tier and the null-research precision/role lanes ask one local model the same narrow
relation question, so they resolve their endpoint the same way: local by default, never egressing
corpus text unless the operator points the backend somewhere else on purpose.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from llb.prep.frontier_telemetry import LLMComplete


def build_adjudicator(
    model: Optional[str], backend: str, base_url: Optional[str]
) -> "LLMComplete | None":
    """Resolve a local completion endpoint, or `None` when no model was requested."""
    if not model:
        return None
    from llb.prep.frontier_telemetry import ProvenanceLog
    from llb.prep.ontology.endpoint import build_complete
    from llb.prep.ontology.endpoint_config import (
        DEFAULT_LOCAL_BASE_URL,
        ENDPOINT_LOCAL,
        EndpointConfig,
    )

    config = EndpointConfig(
        kind=ENDPOINT_LOCAL,
        model=model,
        backend=backend,
        base_url=base_url or DEFAULT_LOCAL_BASE_URL,
    )
    return build_complete(config, ProvenanceLog())
