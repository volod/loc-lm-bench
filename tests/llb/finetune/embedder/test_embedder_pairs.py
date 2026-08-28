"""Contrastive pair export for the embedder fine-tune (`llb.finetune.embedder.pairs`).

Pure: a three-document fixture corpus and a four-item gold set, chunked and BM25-indexed by the
same code the retrieval path uses. No encoder, no GPU, no network.
"""

import json
from pathlib import Path

import pytest

from llb.finetune.embedder.manifest import PAIRS_MANIFEST, load_pairs_manifest
from llb.finetune.embedder.pairs import PAIRS_FILENAME, export_contrastive_pairs
from tests.llb.finetune.embedder._embedder_finetune_helpers import (
    default_items,
    gold_item,
    write_corpus,
    write_goldset,
)


def _export(tmp_path: Path, **overrides) -> tuple[dict, list[dict]]:
    corpus = write_corpus(tmp_path)
    goldset = overrides.pop("goldset", None) or write_goldset(tmp_path)
    out = tmp_path / "pairs"
    manifest = export_contrastive_pairs(
        goldset_path=goldset, corpus_root=corpus, out_dir=out, size=200, overlap=40, **overrides
    )
    rows = [
        json.loads(line)
        for line in (out / PAIRS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, rows


def test_only_tuning_items_reach_the_exported_pairs(tmp_path: Path):
    """The split IS the boundary: a calibration or final id in a training row is a leak."""
    manifest, rows = _export(tmp_path)

    assert manifest["split_counts"] == {"tuning": 2}
    assert manifest["item_ids"] == ["tuning-1", "tuning-2"]
    assert {row["item_id"] for row in rows} == {"tuning-1", "tuning-2"}


def test_the_positive_carries_the_evidence_and_the_negatives_do_not(tmp_path: Path):
    """The positive is DERIVED from the labels: it is the chunk holding the span's own text."""
    _manifest, rows = _export(tmp_path)
    evidence = {item.id: item.source_spans[0].text for item in default_items()}

    for row in rows:
        span_text = evidence[row["item_id"]]
        assert span_text in row["positive"]
        assert all(span_text not in negative for negative in row["negatives"])


def test_no_negative_repeats_the_row_positive(tmp_path: Path):
    """A passage identical to the positive is not a negative, wherever else it appears."""
    _manifest, rows = _export(tmp_path)

    for row in rows:
        assert row["positive"] not in row["negatives"]


def test_every_row_carries_the_same_number_of_negatives(tmp_path: Path):
    """A ragged row set cannot be trained as one batch, so one width covers every row."""
    manifest, rows = _export(tmp_path, negatives=2)

    assert manifest["negatives_per_pair"] == 2
    assert {len(row["negatives"]) for row in rows} == {2}


def test_a_thin_corpus_narrows_the_width_instead_of_going_ragged(tmp_path: Path):
    """Asking for more negatives than the corpus holds must not produce rows of two widths."""
    manifest, rows = _export(tmp_path, negatives=20)

    assert manifest["requested_negatives"] == 20
    assert manifest["negatives_per_pair"] < 20
    assert {len(row["negatives"]) for row in rows} == {manifest["negatives_per_pair"]}


def test_the_manifest_is_readable_and_digests_its_rows(tmp_path: Path):
    manifest, rows = _export(tmp_path)
    reread = load_pairs_manifest(tmp_path / "pairs")

    assert reread == manifest
    assert (tmp_path / "pairs" / PAIRS_MANIFEST).is_file()
    assert manifest["n_pairs"] == len(rows)
    assert len(manifest["dataset_digest"]) == 64


def test_the_same_corpus_and_gold_set_export_the_same_digest(tmp_path: Path):
    first, _rows = _export(tmp_path)
    second, _again = _export(tmp_path / "twin")

    assert first["dataset_digest"] == second["dataset_digest"]


def test_export_refuses_a_gold_set_with_no_tuning_items(tmp_path: Path):
    goldset = write_goldset(
        tmp_path,
        [gold_item("final-1", "ua/norm.txt", "Де?", "Нормандія розташована у Франції", "final")],
    )

    with pytest.raises(ValueError, match="no verified tuning-split gold items"):
        _export(tmp_path, goldset=goldset)


def test_export_refuses_a_gold_set_the_corpus_does_not_index(tmp_path: Path):
    """Spans pointing at documents the corpus does not hold mean there is nothing to train on."""
    stray = gold_item(
        "tuning-1", "ua/norm.txt", "Де розташована Нормандія?", "Нормандія розташована", "tuning"
    )
    stray.source_spans[0].doc_id = "ua/absent.txt"
    goldset = write_goldset(tmp_path, [stray])

    with pytest.raises(ValueError, match="do not describe the same documents"):
        _export(tmp_path, goldset=goldset)


def _doc_text(root: Path, doc_id: str) -> str:
    return (root / "corpus" / doc_id).read_text(encoding="utf-8")
