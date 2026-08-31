import json

import pytest
from pydantic import ValidationError

from llb.core.contracts.robotics import ProducerVersion
from llb.robotics.evidence_models import HflowProjectionRow


def _valid_payload() -> dict[str, object]:
    return {
        "bridge_schema_version": 1,
        "hflow_release": "v0.2.3",
        "hflow_revision": "d2e0f3700f2267cfeb0db1957743bb9f5f41256b",
        "hflow_schema_version": "1",
        "pipeline_version": "pipeline-v1",
        "curation_query_digest": f"sha256:{'a' * 64}",
        "check_versions": (ProducerVersion(producer="check", version="1"),),
        "enrichment_versions": (ProducerVersion(producer="enrich", version="1"),),
        "episode_id": "0123456789abcdef",
        "mcap_uri": "episodes/0123456789abcdef.mcap",
        "mcap_sha256": f"sha256:{'0' * 64}",
        "channels": ("/joint_states",),
        "start_ns": 1,
        "end_ns": 2,
        "quality_state": "accepted",
        "quarantine_tags": (),
        "projection_id": "projection-1",
        "projection_kind": "caption",
        "authored_by": "human",
        "verified": True,
        "verification_ref": None,
        "language": "en",
        "projection_uri": "projections/projection-1.md",
        "projection_sha256": f"sha256:{'1' * 64}",
        "projection_start": 0,
        "projection_end": 3,
    }


@pytest.mark.parametrize("field", ["check_versions", "enrichment_versions"])
def test_manifest_requires_every_producer_version(field: str) -> None:
    payload = _valid_payload()
    payload[field] = ()
    with pytest.raises(ValidationError, match="at least 1 item"):
        HflowProjectionRow.model_validate(payload)


def test_manifest_refuses_unknown_fields() -> None:
    payload = _valid_payload()
    payload["unexpected"] = "discarding this would hide schema drift"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HflowProjectionRow.model_validate(payload)


def test_manifest_json_round_trip_preserves_strict_tuples() -> None:
    row = HflowProjectionRow.model_validate(_valid_payload())
    assert HflowProjectionRow.model_validate_json(json.dumps(row.model_dump(mode="json"))) == row
