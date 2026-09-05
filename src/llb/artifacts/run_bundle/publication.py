"""The gate a staged run bundle passes before it becomes a published one.

Publication is a rename, and a rename is not reversible from the outside: the moment a staging
directory becomes `$DATA_DIR/<method>/<run>/`, a board may read it, a study may cite it, and an
external consumer may validate it. So every member is read back from the STAGED bytes first --
the manifest through its contract, the score rows through what the run declared they answer to,
the retrieval sidecar through its row family, and each additional artifact through the contract or
the exemption it was published under. A refusal here costs a run that was never published, which
is the cheap end of the trade against a board reading nobody can trust.
"""

from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.run_bundle.datasets import run_bundle_manifest
from llb.artifacts.run_bundle.survey import survey_run_bundle


def validate_staged_bundle(
    staging: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> None:
    """Refuse to publish a staged bundle any of whose members does not read back.

    Every refusal is reported, not just the first: a producer fixing one member wants to see the
    others in the same run rather than one per attempt.
    """
    root = Path(staging)
    manifest = run_bundle_manifest(root, registry)
    refusals = [
        reading for reading in survey_run_bundle(root, manifest, registry) if reading.refusal
    ]
    if refusals:
        detail = "; ".join(f"{reading.member_id}: {reading.refusal}" for reading in refusals)
        raise DatasetReadError(f"{root}: staged run bundle refuses publication -- {detail}")
