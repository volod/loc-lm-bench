"""The committed slice, the artifact it pins, and the two-source read over both.

The pointer walk that addresses a value inside either source is exercised in
`test_agentic_published_value_pointer.py`; this file is about what makes a slice EVIDENCE rather
than a self-consistent copy -- the content digest of the artifact it was cut from, the refusals
around it, and what the resolver does on a host that still has that artifact.
"""

import hashlib
import json
from pathlib import Path

import pytest

from llb.bench.agentic_published_value_pointer import merge_field_slice
from llb.bench.agentic_published_value_provenance import (
    DIGEST_ALGORITHM,
    FIXTURE_SCHEMA_VERSION,
    PROVENANCE_FIXTURE,
    CommittedSlice,
    PublishedValueResolver,
    artifact_digest,
    load_provenance_fixture,
    provenance_pair,
    write_provenance_fixture,
)

ARTIFACT = "study/run/analysis.json"
AGGREGATE: dict[str, object] = {
    "depth_surface": [
        {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942, "bracket": [14000, 15500]},
        {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056, "bracket": [20000, 23000]},
    ],
    "depth_ladders": [{"depth": 6, "boundary": {"guard_boundary_chars": 14912, "to_fold_step": 7}}],
    "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
    "reading": "crossover_bracketed",
}


def _slice(*fields: str) -> dict[str, object]:
    cut: dict[str, object] = {}
    for field in fields:
        merge_field_slice(cut, AGGREGATE, field)
    return cut


def _write_artifact(data_dir: Path, payload: dict[str, object]) -> Path:
    path = data_dir / ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _host(
    tmp_path: Path, *fields: str, payload: dict[str, object] | None = None
) -> tuple[Path, Path]:
    """A project root whose committed slice pins an artifact written under a DATA_DIR beside it."""
    data_dir = tmp_path / "data"
    path = _write_artifact(data_dir, payload if payload is not None else AGGREGATE)
    (tmp_path / PROVENANCE_FIXTURE.parent).mkdir(parents=True, exist_ok=True)
    write_provenance_fixture(
        tmp_path, {ARTIFACT: CommittedSlice(digest=artifact_digest(path), payload=_slice(*fields))}
    )
    return tmp_path, data_dir


def _rewrite_fixture(root: Path, entry: object) -> None:
    """Replace the one fixture entry, so a hand-edited fixture can be driven through the load."""
    (root / PROVENANCE_FIXTURE).write_text(
        json.dumps({"schema_version": FIXTURE_SCHEMA_VERSION, "aggregates": {ARTIFACT: entry}}),
        encoding="utf-8",
    )


# --- the committed fixture --------------------------------------------------------------------


def test_the_fixture_round_trips_its_slice_its_pin_and_its_schema(tmp_path):
    root, data_dir = _host(tmp_path, "cap_peak_prompt_chars.6")
    payload = json.loads((root / PROVENANCE_FIXTURE).read_text(encoding="utf-8"))
    assert payload["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert load_provenance_fixture(root) == {
        ARTIFACT: CommittedSlice(
            digest=artifact_digest(data_dir / ARTIFACT),
            payload={"cap_peak_prompt_chars": {"6": 8374}},
        )
    }


def test_the_pin_is_the_digest_of_the_artifact_s_bytes_as_written(tmp_path):
    """Over the FILE, not over a re-serialization of it: the pin has to survive a re-read."""
    path = _write_artifact(tmp_path / "data", AGGREGATE)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact_digest(path) == f"{DIGEST_ALGORITHM}:{expected}"


def test_a_missing_or_mis_versioned_fixture_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_provenance_fixture(tmp_path)
    root, _data_dir = _host(tmp_path, "reading")
    (root / PROVENANCE_FIXTURE).write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"schema_version {FIXTURE_SCHEMA_VERSION}"):
        load_provenance_fixture(root)


def test_a_slice_that_pins_no_artifact_is_refused_as_unfalsifiable(tmp_path):
    """The whole point of the pin: without it a slice is only self-consistent, on every host."""
    root, _data_dir = _host(tmp_path, "reading")
    _rewrite_fixture(root, {"slice": _slice("cap_peak_prompt_chars.6")})
    with pytest.raises(ValueError, match="records no content digest"):
        load_provenance_fixture(root)


@pytest.mark.parametrize(
    "digest",
    ["deadbeef", "sha256:not-hex", f"{DIGEST_ALGORITHM}:{'a' * 63}", "md5:" + "a" * 64, 17],
)
def test_a_pin_that_is_not_a_content_digest_is_refused(tmp_path, digest):
    root, _data_dir = _host(tmp_path, "reading")
    _rewrite_fixture(root, {"digest": digest, "slice": _slice("reading")})
    with pytest.raises(ValueError, match="content digest"):
        load_provenance_fixture(root)


def test_a_fixture_entry_with_a_pin_but_no_slice_is_refused(tmp_path):
    root, _data_dir = _host(tmp_path, "reading")
    _rewrite_fixture(root, {"digest": f"{DIGEST_ALGORITHM}:{'a' * 64}"})
    with pytest.raises(ValueError, match="carries no `slice` object"):
        load_provenance_fixture(root)


# --- the two-source read ----------------------------------------------------------------------


def test_a_value_resolves_from_the_committed_slice_with_no_run_on_the_host(tmp_path):
    """CI is the case with no DATA_DIR at all, so the slice has to be sufficient on its own."""
    field = "depth_surface[depth=6].crossover_max_prompt_chars"
    root, _data_dir = _host(tmp_path, field)
    resolver = PublishedValueResolver(root=root)
    assert resolver.resolve({"artifact": ARTIFACT, "field": field}, where="test") == (
        14159.929807575942
    )


def test_a_host_without_the_run_still_validates_rather_than_skipping(tmp_path):
    """A missing artifact is an ordinary host, not a licence to stop checking the published value."""
    field = "cap_peak_prompt_chars.6"
    root, _data_dir = _host(tmp_path, field)
    resolver = PublishedValueResolver(root=root, data_dir=tmp_path / "empty")
    assert resolver.resolve({"artifact": ARTIFACT, "field": field}, where="t") == 8374


def test_the_run_artifact_is_read_as_a_check_on_the_slice_where_the_host_has_it(tmp_path):
    field = "depth_surface[depth=6].crossover_max_prompt_chars"
    root, data_dir = _host(tmp_path, field)
    resolver = PublishedValueResolver(root=root, data_dir=data_dir)
    assert resolver.resolve({"artifact": ARTIFACT, "field": field}, where="t") == 14159.929807575942


def test_an_artifact_that_is_not_the_pinned_file_is_refused_even_where_the_value_agrees(tmp_path):
    """A slice cut from ANOTHER run of the same study reads identically without the pin.

    The cited field is left untouched here on purpose: the value check cannot see this, so before
    the pin the resolution proved the slice agrees with itself and nothing more.
    """
    field = "depth_surface[depth=6].crossover_max_prompt_chars"
    root, data_dir = _host(tmp_path, field)
    other_run = json.loads(json.dumps(AGGREGATE))
    other_run["reading"] = "crossover_bracketed_second_run"
    _write_artifact(data_dir, other_run)
    with pytest.raises(ValueError, match="the slice was cut from a different run"):
        PublishedValueResolver(root=root, data_dir=data_dir).resolve(
            {"artifact": ARTIFACT, "field": field}, where="t"
        )


def test_a_hand_edited_slice_is_refused_on_a_host_that_still_has_the_run(tmp_path):
    """The pin says the file is the cited one; the value check says the cut was not tampered with."""
    field = "cap_peak_prompt_chars.6"
    root, data_dir = _host(tmp_path, field)
    _rewrite_fixture(
        root,
        {
            "digest": artifact_digest(data_dir / ARTIFACT),
            "slice": {"cap_peak_prompt_chars": {"6": 8375}},
        },
    )
    with pytest.raises(ValueError, match="edited by hand or cut from another field"):
        PublishedValueResolver(root=root, data_dir=data_dir).resolve(
            {"artifact": ARTIFACT, "field": field}, where="t"
        )


def test_an_artifact_with_no_committed_slice_is_refused_as_unresolvable(tmp_path):
    """The evidence was garbage-collected or never committed; either way nothing resolves it."""
    root, _data_dir = _host(tmp_path, "reading")
    with pytest.raises(ValueError, match="no committed slice of 'other/run.json'"):
        PublishedValueResolver(root=root).resolve(
            {"artifact": "other/run.json", "field": "reading"}, where="t"
        )


def test_a_field_that_resolves_to_something_other_than_a_number_is_refused(tmp_path):
    """A published value is a number; a pointer landing on a verdict string is a mis-aimed pointer."""
    root, _data_dir = _host(tmp_path, "reading")
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
