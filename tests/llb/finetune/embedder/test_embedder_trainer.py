"""The embedder training seam and its split guard (`llb.finetune.embedder.trainer`).

Pure: the `fake` trainer validates the exported rows and writes the manifests without loading a
model, so the guard, the convention prefixing, and the tuned identity are all testable with no GPU
and no download.
"""

import json
from pathlib import Path

import pytest

from llb.finetune.embedder.manifest import PAIRS_MANIFEST, TUNED_MANIFEST
from llb.finetune.embedder.pairs import PAIRS_FILENAME, export_contrastive_pairs
from llb.finetune.embedder.trainer import (
    FAKE_MARKER,
    LOSS_CACHED_MNRL,
    TRAINER_FAKE,
    train_embedder,
    training_rows,
)
from llb.rag.encoders.families import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX
from llb.rag.encoders.tuned import load_tuned_embedder
from tests.llb.finetune.embedder._embedder_finetune_helpers import write_corpus, write_goldset

BASE_MODEL = "intfloat/multilingual-e5-base"


def _pairs(tmp_path: Path) -> tuple[Path, Path]:
    """Export a pair set from the fixture corpus; return (pairs_dir, goldset_path)."""
    corpus = write_corpus(tmp_path)
    goldset = write_goldset(tmp_path)
    pairs_dir = tmp_path / "pairs"
    export_contrastive_pairs(
        goldset_path=goldset, corpus_root=corpus, out_dir=pairs_dir, size=200, overlap=40
    )
    return pairs_dir, goldset


def _train(tmp_path: Path, pairs_dir: Path, goldset: Path | None = None) -> dict:
    return train_embedder(
        pairs_dir=pairs_dir,
        base_model=BASE_MODEL,
        out_dir=tmp_path / "model",
        trainer=TRAINER_FAKE,
        goldset_path=goldset,
    )


def test_the_tuned_manifest_records_what_was_trained_on(tmp_path: Path):
    pairs_dir, goldset = _pairs(tmp_path)

    manifest = _train(tmp_path, pairs_dir, goldset)

    assert manifest["base_model"] == BASE_MODEL
    assert manifest["convention_family"] == "e5"
    assert manifest["split_counts"] == {"tuning": 2}
    assert manifest["item_ids"] == ["tuning-1", "tuning-2"]
    assert manifest["trainer"] == TRAINER_FAKE
    assert (tmp_path / "model" / TUNED_MANIFEST).is_file()
    assert (tmp_path / "model" / FAKE_MARKER).is_file()


def test_the_tuned_directory_reads_back_as_a_tuned_embedder(tmp_path: Path):
    """The manifest is what makes a directory an encoder with a known convention and identity."""
    pairs_dir, goldset = _pairs(tmp_path)
    manifest = _train(tmp_path, pairs_dir, goldset)

    tuned = load_tuned_embedder(str(tmp_path / "model"))

    assert tuned is not None
    assert tuned.base_model == BASE_MODEL
    assert tuned.tuned_digest == manifest["tuned_digest"]
    assert tuned.fingerprint.startswith(f"tuned:{BASE_MODEL}:")


def test_retraining_the_same_data_and_seed_is_the_same_encoder(tmp_path: Path):
    pairs_dir, goldset = _pairs(tmp_path)

    first = _train(tmp_path / "a", pairs_dir, goldset)
    second = _train(tmp_path / "b", pairs_dir, goldset)

    assert first["tuned_digest"] == second["tuned_digest"]


def test_a_different_configuration_is_a_different_encoder(tmp_path: Path):
    """The digest is the identity a store records, so a knob that changes weights must change it."""
    pairs_dir, goldset = _pairs(tmp_path)
    baseline = _train(tmp_path / "a", pairs_dir, goldset)

    tuned = train_embedder(
        pairs_dir=pairs_dir,
        base_model=BASE_MODEL,
        out_dir=tmp_path / "b" / "model",
        trainer=TRAINER_FAKE,
        goldset_path=goldset,
        hyperparameters={"num_train_epochs": 9.0},
    )

    assert tuned["tuned_digest"] != baseline["tuned_digest"]


def test_training_refuses_a_pair_set_naming_a_protected_split(tmp_path: Path):
    pairs_dir, goldset = _pairs(tmp_path)
    _poison(pairs_dir, {"split_counts": {"tuning": 2, "final": 1}})

    with pytest.raises(SystemExit, match="non-tuning splits: final"):
        _train(tmp_path, pairs_dir, goldset)


def test_training_refuses_protected_item_ids_a_clean_manifest_hides(tmp_path: Path):
    """A pairs manifest is operator-writable, so its split counts alone are not proof."""
    pairs_dir, goldset = _pairs(tmp_path)
    _poison(pairs_dir, {"item_ids": ["tuning-1", "final-1"]})

    with pytest.raises(SystemExit, match="protected-split item ids: final-1"):
        _train(tmp_path, pairs_dir, goldset)


def test_training_rows_carry_the_base_model_query_and_passage_prefixes(tmp_path: Path):
    """A tuned E5 is still queried with `query: `, so that is what it must be trained on."""
    pairs_dir, _goldset = _pairs(tmp_path)

    rows = training_rows(pairs_dir, BASE_MODEL)

    assert all(anchor.startswith(E5_QUERY_PREFIX) for anchor in rows["anchor"])
    assert all(passage.startswith(E5_PASSAGE_PREFIX) for passage in rows["positive"])
    assert all(negative.startswith(E5_PASSAGE_PREFIX) for negative in rows["negative_1"])


def test_training_rows_refuse_a_ragged_pair_set(tmp_path: Path):
    """Truncating to the narrowest row would silently weaken the objective on every other row."""
    pairs_dir, _goldset = _pairs(tmp_path)
    rows = _read_rows(pairs_dir)
    rows[0]["negatives"] = rows[0]["negatives"][:1]
    _write_rows(pairs_dir, rows)

    with pytest.raises(ValueError, match="one fixed width"):
        training_rows(pairs_dir, BASE_MODEL)


def test_an_unknown_loss_is_refused_before_any_model_loads(tmp_path: Path):
    """The `fake` trainer rejects what the CUDA host would, so a typo costs a second."""
    pairs_dir, goldset = _pairs(tmp_path)

    with pytest.raises(SystemExit, match="unknown loss"):
        train_embedder(
            pairs_dir=pairs_dir,
            base_model=BASE_MODEL,
            out_dir=tmp_path / "model",
            trainer=TRAINER_FAKE,
            goldset_path=goldset,
            hyperparameters={"loss": "triplet"},
        )


def test_the_cached_loss_is_the_default_and_its_mini_batch_is_recorded(tmp_path: Path):
    """A 12 GiB card cannot hold the uncached loss at the default batch, so the default is cached."""
    pairs_dir, goldset = _pairs(tmp_path)

    manifest = _train(tmp_path, pairs_dir, goldset)

    assert manifest["hyperparameters"]["loss"] == LOSS_CACHED_MNRL
    assert manifest["hyperparameters"]["mini_batch_size"] >= 1


def test_an_unknown_trainer_names_the_ones_that_exist(tmp_path: Path):
    pairs_dir, _goldset = _pairs(tmp_path)

    with pytest.raises(SystemExit, match="unknown --trainer"):
        train_embedder(
            pairs_dir=pairs_dir,
            base_model=BASE_MODEL,
            out_dir=tmp_path / "model",
            trainer="lora",
        )


def _poison(pairs_dir: Path, patch: dict) -> None:
    path = pairs_dir / PAIRS_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(patch)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _read_rows(pairs_dir: Path) -> list[dict]:
    text = (pairs_dir / PAIRS_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_rows(pairs_dir: Path, rows: list[dict]) -> None:
    (pairs_dir / PAIRS_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
