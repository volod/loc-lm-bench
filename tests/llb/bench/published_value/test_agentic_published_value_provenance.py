"""The committed aggregate, the pin over its bytes, and the two-source read across both.

The pointer walk that addresses a value inside an aggregate is exercised in
`test_agentic_published_value_pointer.py`; this file is about what makes the committed evidence
EVIDENCE rather than a self-consistent claim -- that the repo carries the cited bytes themselves, so
the pin is checked on a host that never ran the study, and that a host which still has the run root
falsifies the copy against it.
"""

import hashlib
import json
from pathlib import Path

import pytest

from llb.bench.published_value.fixture import (
    COMMITTED_AGGREGATE_DIR,
    DIGEST_ALGORITHM,
    FIXTURE_SCHEMA_VERSION,
    MAX_AGGREGATE_BYTES,
    MAX_COMMITTED_BYTES,
    PROVENANCE_FIXTURE,
    CommittedAggregate,
    artifact_digest,
    committed_aggregate_path,
    committed_evidence_bytes,
    load_provenance_fixture,
    write_provenance_fixture,
)
from llb.bench.published_value.provenance import PublishedValueResolver, provenance_pair

ARTIFACT = "study/run/analysis.json"


def _surface_cell(depth: int, guard: int, delta: float, side: str) -> dict[str, object]:
    """The minimum scored cell the production interpolation reads."""
    return {
        "depth": depth,
        "valid": True,
        "max_prompt_chars": guard,
        "measured_side": side,
        "cost_evidence": {"compact_minus_cap_total_model_input_tokens": {"mean": delta}},
    }


AGGREGATE: dict[str, object] = {
    "held_fixed": {
        "n_tasks": 7,
        "pad_chars": 1200,
        "max_steps_margin": 4,
        "observation_cap_chars": 800,
        "observation_head_share": 0.6,
    },
    "cells": [
        _surface_cell(6, 14000, -125.85714285714286, "compact_cheaper"),
        _surface_cell(6, 15500, 1054.5714285714287, "cap_cheaper"),
        _surface_cell(10, 20000, -2633.4285714285716, "compact_cheaper"),
        _surface_cell(10, 23000, 1524.857142857143, "cap_cheaper"),
    ],
    "depth_surface": [
        {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942, "bracket": [14000, 15500]},
        {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056, "bracket": [20000, 23000]},
    ],
    "depth_ladders": [{"depth": 6, "boundary": {"guard_boundary_chars": 14912, "to_fold_step": 7}}],
    "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
    "reading": "crossover_bracketed",
}
SURFACE_FIELD = "depth_surface[depth=6].crossover_max_prompt_chars"


def _raw(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _host(
    tmp_path: Path,
    *,
    committed: dict[str, object] | None = None,
    measured: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """A project root carrying the committed aggregate, beside a DATA_DIR holding the run itself."""
    data_dir = tmp_path / "data"
    write_provenance_fixture(tmp_path, {ARTIFACT: _raw(committed or AGGREGATE)})
    path = data_dir / ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(measured if measured is not None else (committed or AGGREGATE)))
    return tmp_path, data_dir


def _rewrite_manifest(root: Path, entry: object) -> None:
    """Replace the one manifest entry, so a hand-edited manifest can be driven through the load."""
    (root / PROVENANCE_FIXTURE).write_text(
        json.dumps({"schema_version": FIXTURE_SCHEMA_VERSION, "aggregates": {ARTIFACT: entry}}),
        encoding="utf-8",
    )


# --- the committed evidence ---------------------------------------------------------------------


def test_the_repo_carries_the_cited_bytes_and_a_pin_over_exactly_those_bytes(tmp_path):
    root, data_dir = _host(tmp_path)
    copy = committed_aggregate_path(root, ARTIFACT)
    assert copy.read_bytes() == (data_dir / ARTIFACT).read_bytes()
    assert load_provenance_fixture(root) == {
        ARTIFACT: CommittedAggregate(digest=artifact_digest(copy), payload=AGGREGATE)
    }


def test_the_pin_is_the_digest_of_the_artifact_s_bytes_as_written(tmp_path):
    """Over the FILE, not over a re-serialization of it: the pin has to survive a re-read."""
    root, _data_dir = _host(tmp_path)
    copy = committed_aggregate_path(root, ARTIFACT)
    expected = hashlib.sha256(copy.read_bytes()).hexdigest()
    assert artifact_digest(copy) == f"{DIGEST_ALGORITHM}:{expected}"


def test_a_missing_or_mis_versioned_fixture_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_provenance_fixture(tmp_path)
    root, _data_dir = _host(tmp_path)
    (root / PROVENANCE_FIXTURE).write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"schema_version {FIXTURE_SCHEMA_VERSION}"):
        load_provenance_fixture(root)


def test_an_entry_that_pins_nothing_is_refused(tmp_path):
    """Without the pin nothing ties the committed bytes to the run whose name they carry."""
    root, _data_dir = _host(tmp_path)
    _rewrite_manifest(root, {})
    with pytest.raises(ValueError, match="records no content digest"):
        load_provenance_fixture(root)


@pytest.mark.parametrize(
    "digest",
    ["deadbeef", "sha256:not-hex", f"{DIGEST_ALGORITHM}:{'a' * 63}", "md5:" + "a" * 64, 17],
)
def test_a_pin_that_is_not_a_content_digest_is_refused(tmp_path, digest):
    root, _data_dir = _host(tmp_path)
    _rewrite_manifest(root, {"digest": digest})
    with pytest.raises(ValueError, match="content digest"):
        load_provenance_fixture(root)


def test_a_pin_whose_bytes_the_repo_does_not_carry_is_refused_as_a_claim_about_an_absent_file(
    tmp_path,
):
    """The failure this evidence exists to close: a digest of a file no present host can read.

    A pin alone is checkable only where the run root survives. On CI, a fresh clone, or this host
    after a `.data` cleanup it stands for a file nobody can open, so the resolution proved the
    evidence agrees with itself plus a hash of something absent.
    """
    root, _data_dir = _host(tmp_path)
    committed_aggregate_path(root, ARTIFACT).unlink()
    with pytest.raises(ValueError, match="carries no copy of it"):
        load_provenance_fixture(root)


def test_a_pin_that_disagrees_with_the_committed_copy_is_refused_with_no_run_on_the_host(tmp_path):
    """The pin is now falsifiable everywhere, which is what a fabricated pair could not survive.

    Before the bytes were committed, an aggregate and a pin invented together were accepted on every
    host without the run root, because there was nothing present to contradict either one.
    """
    root, _data_dir = _host(tmp_path)
    _rewrite_manifest(root, {"digest": f"{DIGEST_ALGORITHM}:{'0' * 64}"})
    with pytest.raises(ValueError, match="the committed copy digests to"):
        load_provenance_fixture(root)


def test_a_hand_edited_committed_copy_is_refused_with_no_run_on_the_host(tmp_path):
    """The same check read the other way: the bytes moved and the pin did not."""
    root, _data_dir = _host(tmp_path)
    copy = committed_aggregate_path(root, ARTIFACT)
    copy.write_bytes(_raw({**AGGREGATE, "cap_peak_prompt_chars": {"6": 8375, "10": 11926}}))
    with pytest.raises(ValueError, match="the committed copy digests to"):
        load_provenance_fixture(root)


def test_an_artifact_key_that_escapes_the_committed_tree_is_refused(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        committed_aggregate_path(tmp_path, "../outside/analysis.json")


# --- the growth policy --------------------------------------------------------------------------


def test_regeneration_drops_the_copies_no_published_value_still_cites(tmp_path):
    """The evidence tracks the CITED artifacts, so a retired study stops costing repo bytes."""
    root, _data_dir = _host(tmp_path)
    retired = "old-study/run/analysis.json"
    write_provenance_fixture(root, {ARTIFACT: _raw(AGGREGATE), retired: _raw(AGGREGATE)})
    assert committed_aggregate_path(root, retired).is_file()

    write_provenance_fixture(root, {ARTIFACT: _raw(AGGREGATE)})
    assert set(load_provenance_fixture(root)) == {ARTIFACT}
    assert not committed_aggregate_path(root, retired).exists()
    assert not (root / COMMITTED_AGGREGATE_DIR / "old-study").exists()


def test_an_aggregate_over_the_per_artifact_cap_is_refused_rather_than_committed(tmp_path):
    oversized = b"x" * (MAX_AGGREGATE_BYTES + 1)
    with pytest.raises(ValueError, match="per-artifact cap on committed evidence"):
        write_provenance_fixture(tmp_path, {ARTIFACT: oversized})
    assert not committed_aggregate_path(tmp_path, ARTIFACT).exists()


def test_the_total_committed_evidence_stays_inside_its_budget(tmp_path):
    each = b"y" * MAX_AGGREGATE_BYTES
    over = {
        f"study-{index}/run/analysis.json": each
        for index in range(MAX_COMMITTED_BYTES // len(each) + 1)
    }
    with pytest.raises(ValueError, match="budget for committed provenance evidence"):
        write_provenance_fixture(tmp_path, over)


def test_the_evidence_size_is_readable_without_stat_ing_every_copy_by_hand(tmp_path):
    root, _data_dir = _host(tmp_path)
    assert committed_evidence_bytes(root) == len(_raw(AGGREGATE))


# --- the two-source read ------------------------------------------------------------------------


def test_a_value_resolves_from_the_committed_copy_with_no_run_on_the_host(tmp_path):
    """CI is the case with no DATA_DIR at all, so the committed bytes have to be sufficient."""
    root, _data_dir = _host(tmp_path)
    resolver = PublishedValueResolver(root=root)
    assert resolver.resolve({"artifact": ARTIFACT, "field": SURFACE_FIELD}, where="test") == (
        14159.929807575942
    )


def test_a_host_without_the_run_still_validates_rather_than_skipping(tmp_path):
    """A missing artifact is an ordinary host, not a licence to stop checking the published value."""
    root, _data_dir = _host(tmp_path)
    resolver = PublishedValueResolver(root=root, data_dir=tmp_path / "empty")
    assert resolver.resolve({"artifact": ARTIFACT, "field": "cap_peak_prompt_chars.6"}, where="t")


def test_the_run_artifact_is_read_as_a_check_on_the_copy_where_the_host_has_it(tmp_path):
    root, data_dir = _host(tmp_path)
    resolver = PublishedValueResolver(root=root, data_dir=data_dir)
    assert resolver.resolve({"artifact": ARTIFACT, "field": SURFACE_FIELD}, where="t") == (
        14159.929807575942
    )


def test_an_artifact_that_is_not_the_pinned_file_is_refused_even_where_the_value_agrees(tmp_path):
    """A copy taken from ANOTHER run of the same study reads identically without the pin.

    The cited field is left untouched here on purpose: a value comparison cannot see this, so
    without the pin the resolution proved the evidence agrees with itself and nothing more.
    """
    root, data_dir = _host(
        tmp_path, measured={**AGGREGATE, "reading": "crossover_bracketed_second_run"}
    )
    with pytest.raises(ValueError, match="came from a different run"):
        PublishedValueResolver(root=root, data_dir=data_dir).resolve(
            {"artifact": ARTIFACT, "field": SURFACE_FIELD}, where="t"
        )


def test_an_artifact_with_no_committed_copy_is_refused_as_unresolvable(tmp_path):
    """The evidence was garbage-collected or never committed; either way nothing resolves it."""
    root, _data_dir = _host(tmp_path)
    with pytest.raises(ValueError, match="no committed copy of 'other/run.json'"):
        PublishedValueResolver(root=root).resolve(
            {"artifact": "other/run.json", "field": "reading"}, where="t"
        )


def test_a_field_that_resolves_to_something_other_than_a_number_is_refused(tmp_path):
    """A published value is a number; a pointer landing on a verdict string is a mis-aimed pointer."""
    root, _data_dir = _host(tmp_path)
    with pytest.raises(ValueError, match="which is not a number"):
        PublishedValueResolver(root=root).resolve(
            {"artifact": ARTIFACT, "field": "reading"}, where="t"
        )


@pytest.mark.parametrize(
    ("provenance", "match"),
    [
        (None, "must carry a `provenance` object"),
        ({"field": "reading"}, "DATA_DIR-relative run artifact"),
        ({"artifact": "/abs/run.json", "field": "reading"}, "must stay inside DATA_DIR"),
        ({"artifact": "../run.json", "field": "reading"}, "must stay inside DATA_DIR"),
        ({"artifact": ARTIFACT}, "must be the field pointer"),
    ],
)
def test_a_provenance_pair_that_names_no_artifact_or_no_field_is_refused(provenance, match):
    with pytest.raises(ValueError, match=match):
        provenance_pair(provenance, where="test")
