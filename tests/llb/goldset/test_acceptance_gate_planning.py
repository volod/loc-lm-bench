"""Scaling contracts for experiment-derived verification and chain gates."""

import json

import pytest

from llb.goldset.promote_chains import promote_chain_bundle
from llb.goldset.verify_sampling.planning import required_sample_size
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet
from tests.llb.goldset._verify_helpers import _bundle, _item
from tests.llb.goldset.test_promote_chains import _accepted_bundle, _chain


def test_verification_target_scales_with_population():
    small = required_sample_size(20)
    large = required_sample_size(200)
    lower_expected_variance = required_sample_size(200, expected_reject_rate=0.10)

    assert small == 17
    assert large == 66
    assert small < large
    assert lower_expected_variance < large


def test_derived_sample_and_explicit_override_are_recorded(tmp_path):
    bundle = _bundle(tmp_path, [_item(f"i{index}") for index in range(20)])
    worksheet = bundle / "verify_sample.csv"

    size, _ = build_sample_worksheet(bundle, worksheet)
    manifest = json.loads((bundle / "sample_manifest.json").read_text(encoding="utf-8"))

    assert size == 17
    assert manifest["requested"] is None
    assert manifest["acceptance_gate"]["derived_target"] == 17
    assert manifest["acceptance_gate"]["operator_override"] is None

    build_sample_worksheet(bundle, worksheet, n=5)
    manifest = json.loads((bundle / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_size"] == 5
    assert manifest["acceptance_gate"]["operator_override"] == 5
    assert manifest["acceptance_gate"]["override_meets_derived_target"] is False


def test_chain_promotion_derives_relative_retention_target(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1"), _chain("c2")])
    (bundle / "sample_manifest.json").write_text(json.dumps({"sample_size": 4}), encoding="utf-8")

    manifest = promote_chain_bundle(bundle, tmp_path / "fixture")

    gate = manifest["acceptance_gate"]
    assert manifest["minimum_required"] == 2
    assert gate["derived_target"] == 2
    assert gate["operator_override"] is None


def test_chain_promotion_needs_manifest_or_explicit_override(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1")])

    with pytest.raises(ValueError, match="sample_manifest"):
        promote_chain_bundle(bundle, tmp_path / "fixture")
