"""Acceptance-gate inventory regression tests."""

from llb.quality.acceptance_gates import audit


def test_acceptance_gate_inventory_has_no_unclassified_controls():
    result = audit()

    assert result["passed"], result["findings"]
    retired = result["retired_controls"][0]
    assert retired["id"] == "ua-model-roster-long-run"
    assert retired["status"] == "absent"
