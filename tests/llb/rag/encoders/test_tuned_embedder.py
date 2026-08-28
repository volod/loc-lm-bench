"""A locally fine-tuned encoder directory as a first-class candidate (`llb.rag.encoders.tuned`).

Pure: the tuned directory here is a manifest and nothing else, which is exactly what the resolution
and fingerprint rules read. No model is loaded, so no GPU and no download.
"""

import json
from pathlib import Path

import pytest

from llb.rag.embedding_bakeoff.roster import UnregisteredCandidateError, screen_candidates
from llb.rag.encoders.embedder import Embedder
from llb.rag.encoders.families import E5_QUERY_PREFIX, FAMILY_E5
from llb.rag.encoders.tuned import (
    MANIFEST_KIND,
    TUNED_EMBEDDER_MANIFEST,
    convention_registered,
    embedder_fingerprint,
    load_tuned_embedder,
    resolved_convention,
)
from llb.rag.vector_store.validation import store_embedder_mismatch

BASE_MODEL = "intfloat/multilingual-e5-base"


def _tuned_dir(root: Path, *, base: str = BASE_MODEL, digest: str = "a" * 64, **patch) -> str:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": MANIFEST_KIND, "base_model": base, "tuned_digest": digest}
    manifest.update(patch)
    (root / TUNED_EMBEDDER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return str(root)


def test_a_tuned_directory_inherits_its_base_model_convention(tmp_path: Path):
    """Fine-tuning changes the weights, not the input format the weights expect."""
    tuned = _tuned_dir(tmp_path / "tuned")

    assert resolved_convention(tuned).family == FAMILY_E5
    assert convention_registered(tuned)
    assert Embedder(tuned).family == FAMILY_E5
    assert Embedder(tuned).convention.query_prefix == E5_QUERY_PREFIX


def test_an_ordinary_id_is_untouched_by_the_tuned_lane(tmp_path: Path):
    assert load_tuned_embedder(BASE_MODEL) is None
    assert load_tuned_embedder(str(tmp_path)) is None
    assert embedder_fingerprint(BASE_MODEL) == BASE_MODEL


def test_a_directory_carrying_another_kind_of_manifest_is_not_a_tuned_encoder(tmp_path: Path):
    """Half-resolving a foreign manifest would invent a base model nobody declared."""
    root = tmp_path / "adapter"
    root.mkdir()
    (root / TUNED_EMBEDDER_MANIFEST).write_text(
        json.dumps({"kind": "llb.finetune.dataset", "base_model": BASE_MODEL}), encoding="utf-8"
    )

    assert load_tuned_embedder(str(root)) is None


def test_the_bakeoff_screens_a_tuned_directory_beside_its_base(tmp_path: Path):
    tuned = _tuned_dir(tmp_path / "tuned")

    runnable, skipped = screen_candidates([BASE_MODEL, tuned], transformers_major=5)

    assert runnable == [BASE_MODEL, tuned]
    assert skipped == []


def test_the_bakeoff_refuses_a_tuned_directory_whose_base_nobody_registered(tmp_path: Path):
    """An unregistered base is an unregistered candidate; the tuned wrapper cannot launder it."""
    tuned = _tuned_dir(tmp_path / "tuned", base="acme/unknown-encoder")

    with pytest.raises(UnregisteredCandidateError):
        screen_candidates([tuned], transformers_major=5)


def test_the_store_guard_refuses_a_retrain_at_the_same_path(tmp_path: Path):
    """A path is not an identity: same directory, different training, different vectors."""
    tuned = _tuned_dir(tmp_path / "tuned", digest="b" * 64)
    built = {"embedding_model": tuned, "embedder_fingerprint": f"tuned:{BASE_MODEL}:{'a' * 12}"}

    mismatch = store_embedder_mismatch(built, tuned)

    assert mismatch is not None
    assert "a" * 12 in mismatch


def test_the_store_guard_accepts_the_encoder_that_built_it(tmp_path: Path):
    tuned = _tuned_dir(tmp_path / "tuned", digest="b" * 64)
    built = {"embedding_model": tuned, "embedder_fingerprint": embedder_fingerprint(tuned)}

    assert store_embedder_mismatch(built, tuned) is None


def test_a_tuned_store_is_still_refused_to_a_different_encoder(tmp_path: Path):
    tuned = _tuned_dir(tmp_path / "tuned")
    built = {"embedding_model": tuned, "embedder_fingerprint": embedder_fingerprint(tuned)}

    assert store_embedder_mismatch(built, BASE_MODEL) == tuned
