"""What a finished bundle records about the STORE it read, and the one question it answers.

The `tree` block used to carry the store's whole `{doc_id: sha256}` manifest -- on the largest
bundle here, 23.5 KiB of a 36.8 KiB `summary.json`, against the 6.6 KiB the per-document record
costs after four folds. Nothing read it per document, and nothing could usefully: `store_meta.json`
holds the authoritative map and `llb refresh-index` asks the per-document question of the store
itself. What a bundle is asked is one equality test -- is the store on disk still the store this run
read -- so it records one digest over the sorted pairs.

These tests pin the four things that decision rests on:

1. the record no longer grows with the corpus, and no longer repeats an id the record just interned;
2. a store that GENUINELY changed is still detected as changed, in all three ways a manifest can
   change (an edited document, an added one, a removed one);
3. a bundle at the OLD form -- the whole map -- returns the identical verdict, so the archive on
   disk answers this question exactly as a bundle written today does;
4. a current bundle records the store's DATA_DIR-relative location, while a replay refuses an
   invalid or vanished reference and preserves the explicit ``--store`` override.
"""

import json
from pathlib import Path

import pytest

from llb.conflicts.bundle.fold import json_bytes
from llb.conflicts.report.stage_replay import replay_report, store_line
from llb.conflicts.bundle.store_identity import (
    DIGEST_KEY,
    DOCUMENTS_KEY,
    MAP_KEY,
    NO_IDENTITY,
    StoreIdentity,
    compare_store,
    fingerprint_digest,
    identity_entry,
    identity_payload,
)
from llb.conflicts.bundle.store_location import (
    STORE_DATA_DIR_RELATIVE_KEY,
    recorded_store_location,
    resolve_store_placement,
    store_location_payload,
)
from llb.conflicts.semantic_tree.tree import SemanticPrefixTree
from llb.conflicts.semantic_tree.refresh import tree_meta
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.store_generations import GENERATIONS_DIRNAME
from llb.rag.vector_store.build import META_FILE

DIGEST_HEX_CHARS = 64
# Three corpus sizes spanning two orders of magnitude, the same ones the record's own size table
# projects onto (`conflict-bundle-record.md`).
SIZES = (10, 250, 25_000)


def manifest(documents: int, *, prefix: str = "squad/") -> dict[str, str]:
    """A store manifest the shape a real one has: a corpus-relative path and a sha256 per document."""
    return {f"{prefix}{index:012x}.txt": f"{index:064x}" for index in range(documents)}


def meta_over(fingerprints: dict[str, str]) -> dict[str, object]:
    """The `tree` block a run writes today, over a store holding `fingerprints`."""
    vectors = VectorSet([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    return tree_meta(
        SemanticPrefixTree.build(vectors, leaf_size=2),
        embedding_model="intfloat/multilingual-e5-base",
        dim=2,
        corpus_fingerprint="corpus",
        doc_fingerprints=fingerprints,
        cos_threshold=0.9,
    )


def legacy_meta(fingerprints: dict[str, str]) -> dict[str, object]:
    """The same block as the bundles on disk carry it: the whole manifest, copied verbatim."""
    return {
        **{
            k: v for k, v in meta_over(fingerprints).items() if k not in (DIGEST_KEY, DOCUMENTS_KEY)
        },
        MAP_KEY: dict(fingerprints),
    }


def test_the_store_manifest_is_recorded_as_a_digest_rather_than_a_second_copy_of_it():
    fingerprints = manifest(250)
    block = meta_over(fingerprints)
    assert MAP_KEY not in block
    assert len(str(block[DIGEST_KEY])) == DIGEST_HEX_CHARS
    assert block[DOCUMENTS_KEY] == 250
    # The point of the change: no document id and no per-document digest survives into the bundle.
    encoded = json.dumps(block, ensure_ascii=False)
    assert not any(doc_id in encoded for doc_id in fingerprints)


def test_the_store_location_is_recorded_relative_to_data_dir_and_never_as_an_absolute_path(
    tmp_path,
):
    data_dir = tmp_path / "data"
    store_dir = data_dir / "llb" / "rag" / GENERATIONS_DIRNAME / "one"
    store_dir.mkdir(parents=True)

    payload = store_location_payload(store_dir, data_dir=data_dir)
    recorded = recorded_store_location(payload, data_dir=data_dir)

    assert payload == {STORE_DATA_DIR_RELATIVE_KEY: "llb/rag/generations/one"}
    assert recorded is not None
    assert recorded.path == store_dir.resolve()
    assert recorded.display == "$DATA_DIR/llb/rag/generations/one"
    assert str(data_dir) not in json.dumps(payload)
    assert store_location_payload(tmp_path / "outside", data_dir=data_dir) == {}


@pytest.mark.parametrize("location", ["../outside", "/absolute/store", ""])
def test_a_recorded_store_location_must_be_a_safe_nonempty_relative_path(tmp_path, location):
    with pytest.raises(ValueError, match=STORE_DATA_DIR_RELATIVE_KEY):
        recorded_store_location({STORE_DATA_DIR_RELATIVE_KEY: location}, data_dir=tmp_path / "data")


def test_the_identity_costs_the_same_whatever_the_corpus_size_is():
    """The whole saving: a fixed answer where the copied map grew ~94 bytes per document."""
    sizes = {documents: json_bytes(identity_payload(manifest(documents))) for documents in SIZES}
    # Everything but the digits of the count itself is a constant, and the map it replaces is not.
    assert {size - len(str(documents)) for documents, size in sizes.items()} == {
        sizes[10] - len(str(10))
    }
    assert json_bytes(manifest(25_000)) == 100 * json_bytes(manifest(250))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: {**m, "squad/000000000003.txt": "f" * 64}, id="edited"),
        pytest.param(lambda m: {**m, "squad/00000000ffff.txt": "a" * 64}, id="added"),
        pytest.param(
            lambda m: {k: v for k, v in m.items() if k != "squad/000000000003.txt"}, id="removed"
        ),
    ],
)
def test_a_store_that_genuinely_changed_is_detected_as_changed(mutate):
    fingerprints = manifest(20)
    verdict = compare_store(meta_over(fingerprints), mutate(fingerprints))
    assert verdict.comparable and verdict.changed
    assert "NOT the one this run read" in verdict.detail


def test_the_store_that_did_not_change_reads_as_the_same_store():
    fingerprints = manifest(20)
    verdict = compare_store(meta_over(fingerprints), dict(fingerprints))
    assert verdict.comparable and not verdict.changed
    assert "20 documents" in verdict.detail


def test_the_digest_is_a_property_of_the_mapping_and_not_of_the_order_it_was_written_in():
    """A rebuild that visits the corpus in another order over identical content is the same store."""
    fingerprints = manifest(20)
    reversed_order = dict(reversed(list(fingerprints.items())))
    assert list(reversed_order) != list(fingerprints)
    assert fingerprint_digest(reversed_order) == fingerprint_digest(fingerprints)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: dict(m), id="unchanged"),
        pytest.param(lambda m: {**m, "squad/000000000003.txt": "f" * 64}, id="changed"),
    ],
)
def test_a_bundle_at_either_form_returns_the_identical_verdict(mutate):
    """The archive on disk carries the map; a bundle written today carries the digest."""
    fingerprints = manifest(20)
    current = mutate(fingerprints)
    assert compare_store(legacy_meta(fingerprints), current) == compare_store(
        meta_over(fingerprints), current
    )
    assert StoreIdentity.of(legacy_meta(fingerprints)).from_recorded_map
    assert not StoreIdentity.of(meta_over(fingerprints)).from_recorded_map


def test_a_store_with_no_fingerprints_records_no_identity_rather_than_the_digest_of_nothing():
    """A silence and a claim: "identical to every other fingerprintless store" is the latter."""
    block = meta_over({})
    assert DIGEST_KEY not in block and DOCUMENTS_KEY not in block and MAP_KEY not in block
    assert StoreIdentity.of(block) is None


def test_a_bundle_that_read_no_store_is_not_placed_against_one():
    verdict = compare_store({}, manifest(5))
    assert not verdict.comparable and not verdict.changed
    assert verdict.detail == NO_IDENTITY


def test_a_store_that_cannot_identify_itself_is_refused_rather_than_called_changed():
    verdict = compare_store(meta_over(manifest(5)), {})
    assert not verdict.comparable
    assert "records no per-document fingerprints" in verdict.detail


def _write_store(path: Path, fingerprints: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / META_FILE).write_text(json.dumps({"doc_fingerprints": fingerprints}), encoding="utf-8")


def _summary_at(store_dir: Path, data_dir: Path, fingerprints: dict[str, str]) -> dict[str, object]:
    return {
        "tree": {
            **meta_over(fingerprints),
            **store_location_payload(store_dir, data_dir=data_dir),
        }
    }


def test_a_re_read_resolves_each_bundle_own_recorded_store_and_keeps_the_explicit_override(
    tmp_path,
):
    data_dir = tmp_path / "data"
    first_fingerprints = manifest(3, prefix="first/")
    second_fingerprints = manifest(4, prefix="second/")
    first_store, second_store = data_dir / "stores" / "first", data_dir / "stores" / "second"
    _write_store(first_store, first_fingerprints)
    _write_store(second_store, second_fingerprints)
    cache: dict[tuple[str, str], dict[str, str] | None] = {}

    first = resolve_store_placement(
        _summary_at(first_store, data_dir, first_fingerprints),
        fallback_store=None,
        data_dir=data_dir,
        cache=cache,
    )
    second = resolve_store_placement(
        _summary_at(second_store, data_dir, second_fingerprints),
        fallback_store=None,
        data_dir=data_dir,
        cache=cache,
    )
    legacy = resolve_store_placement(
        {"tree": meta_over(second_fingerprints)},
        fallback_store=second_store,
        data_dir=data_dir,
        cache=cache,
    )
    overridden = resolve_store_placement(
        _summary_at(first_store, data_dir, first_fingerprints),
        fallback_store=second_store,
        data_dir=data_dir,
        cache=cache,
    )

    assert first is not None and first["comparable"] and not first["changed"]
    assert first["reference"] == "$DATA_DIR/stores/first"
    assert first["reference_source"] == "bundle"
    assert second is not None and second["comparable"] and not second["changed"]
    assert second["reference"] == "$DATA_DIR/stores/second"
    assert legacy is not None and legacy["comparable"] and not legacy["changed"]
    assert legacy["reference"] == "--store"
    assert legacy["reference_source"] == "explicit"
    assert overridden is not None and overridden["comparable"] and overridden["changed"]
    assert overridden["reference_source"] == "explicit"


def test_a_recorded_store_location_reads_that_exact_directory_not_a_newer_live_generation(tmp_path):
    data_dir = tmp_path / "data"
    base = data_dir / "stores" / "versioned"
    recorded_fingerprints = manifest(3)
    _write_store(base, recorded_fingerprints)
    _write_store(base / GENERATIONS_DIRNAME / "newer", manifest(4, prefix="newer/"))

    placement = resolve_store_placement(
        _summary_at(base, data_dir, recorded_fingerprints),
        fallback_store=None,
        data_dir=data_dir,
        cache={},
    )

    assert placement is not None and placement["comparable"] and not placement["changed"]


def test_a_gone_recorded_location_is_reported_as_unavailable_not_as_an_identity_mismatch(tmp_path):
    data_dir = tmp_path / "data"
    gone = data_dir / "stores" / "gone"
    recorded_fingerprints = manifest(3)

    placement = resolve_store_placement(
        _summary_at(gone, data_dir, recorded_fingerprints),
        fallback_store=None,
        data_dir=data_dir,
        cache={},
    )

    assert placement is not None and not placement["comparable"]
    assert placement["changed"] is None
    assert placement["reference_source"] == "bundle"
    assert "is gone" in placement["detail"]
    assert "no identity comparison was made" in placement["detail"]
    assert "NOT the one" not in placement["detail"]


def _entry(label: str, summary: dict, fingerprints: dict[str, str]) -> dict:
    return {
        "label": label,
        "source": f"{label}/summary.json",
        "recomputable": False,
        "reason": "no per-document record",
        "recorded": None,
        "recomputed": None,
        "agrees": None,
        "readings": [],
        "store_identity": identity_entry(summary, fingerprints),
    }


def test_the_stage_re_read_places_every_bundle_against_the_store_it_is_pointed_at():
    """The consumer that asks the question: an archive sweep pointed at one store on disk."""
    fingerprints = manifest(20)
    entries = [
        _entry("same", {"tree": meta_over(fingerprints)}, fingerprints),
        _entry("legacy", {"tree": legacy_meta(fingerprints)}, fingerprints),
        _entry("other", {"tree": meta_over(manifest(20, prefix="other/"))}, fingerprints),
        _entry("no-store", {}, fingerprints),
    ]
    report = replay_report(entries)
    assert "## The store these bundles read" in report
    assert "- resolved store matches the bundle identity: 2 of 4" in report
    assert "- resolved store differs from the bundle identity: 1 of 4" in report
    assert "- not comparable (location or identity unavailable): 1 of 4" in report
    assert "NOT the one this run read" in store_line(entries[2])


def test_a_sweep_without_a_store_carries_no_store_reading_at_all():
    """The store question is opt-in, so a bundle re-read without `--store` is what it always was."""
    entries = [{**_entry("same", {"tree": meta_over(manifest(3))}, manifest(3))}]
    del entries[0]["store_identity"]
    assert "## The store these bundles read" not in replay_report(entries)
