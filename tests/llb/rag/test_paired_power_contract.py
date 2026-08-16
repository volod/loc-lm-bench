"""Embedder and fusion paired-power lane contracts."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.core.config import RunConfig
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.rag.embedding_bakeoff_models import BuiltStore
from llb.rag.fusion_evidence.models import EvidenceItem

BASELINE = "intfloat/multilingual-e5-base"
CANDIDATE = "BAAI/bge-m3"
METRIC = "recall_at_k"


class _EmptyStore:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path
        self.meta = {"dim": 1, "n_indexed": 1}

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        assert self.plan_path.is_file()
        return []


def _embedding_reference(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "uncertainty": {"baseline": BASELINE},
                "paired_items": [
                    {
                        "item_id": f"item-{i}",
                        "models": {
                            BASELINE: {METRIC: 0.0},
                            CANDIDATE: {METRIC: value},
                        },
                    }
                    for i, value in enumerate((-0.2, 0.0, 0.2, 0.4))
                ],
            }
        ),
        encoding="utf-8",
    )


def test_embedder_lane_writes_the_plan_before_building_and_reports_realized_mde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset(
        [
            GoldItem(
                id=f"item-{i}",
                question=f"q{i}",
                reference_answer="x",
                source_doc_id="d",
                source_spans=[SourceSpan(doc_id="d", char_start=0, char_end=1, text="x")],
                provenance="human-authored",
                split="final",
            )
            for i in range(4)
        ],
        goldset,
    )
    reference = tmp_path / "embedding-reference.json"
    _embedding_reference(reference)
    out = tmp_path / "embedding-run" / "report.md"
    plan_path = out.parent / "power-plan.json"

    def builder(_cfg: RunConfig, _stores_dir: Path, **_kwargs):
        def build(_model: str) -> BuiltStore:
            assert plan_path.is_file()
            return BuiltStore(store=_EmptyStore(plan_path), embed_seconds=1.0, index_bytes=1)

        return build

    monkeypatch.setattr("llb.cli.rag.compare_embeddings.local_store_builder", builder)
    result = CliRunner().invoke(
        app,
        [
            "compare-embeddings",
            "--goldset",
            str(goldset),
            "--corpus-root",
            str(corpus),
            "--models",
            f"{BASELINE},{CANDIDATE}",
            "--baseline",
            BASELINE,
            "--power-reference",
            str(reference),
            "--power-candidate",
            CANDIDATE,
            "--minimum-detectable-delta",
            "0.2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["power_analysis"]["resolvable_mde"] == 0.0
    assert report["power_analysis"]["selector"]["candidate"] == CANDIDATE


def _fusion_reference(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "baseline": "vector",
                "focus_slice": "multi-hop",
                "focus_items": [
                    {
                        "rows": {
                            "vector": {METRIC: 0.0},
                            "fused/test": {METRIC: value},
                        }
                    }
                    for value in (-0.2, 0.0, 0.2, 0.4)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_fusion_lane_writes_the_plan_before_retrieval_and_reports_realized_mde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference = tmp_path / "fusion-reference.json"
    _fusion_reference(reference)
    out_dir = tmp_path / "fusion-run"
    plan_path = out_dir / "power-plan.json"
    items = [
        EvidenceItem(
            f"item-{i}",
            f"q{i}",
            [{"doc_id": "d", "char_start": 0, "char_end": 1, "text": "x"}],
            "multi-hop",
        )
        for i in range(4)
    ]
    monkeypatch.setattr("llb.cli.rag.fusion_inputs.evidence_items", lambda cfg, split: items)
    monkeypatch.setattr(
        "llb.cli.rag.fusion_inputs.load_lanes",
        lambda cfg, strategies: (_EmptyStore(plan_path), {}),
    )
    monkeypatch.setattr(
        "llb.rag.fusion_evidence.build_sweep_rows",
        lambda *args, **kwargs: {
            "vector": _EmptyStore(plan_path),
            "fused/test": _EmptyStore(plan_path),
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "compare-graph-fusion",
            "--power-reference",
            str(reference),
            "--power-row",
            "fused/test",
            "--minimum-detectable-delta",
            "0.2",
            "--out-dir",
            str(out_dir),
            "--no-routing-sidecar",
            "--resamples",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
    assert report["power_analysis"]["resolvable_mde"] == 0.0
    assert report["power_analysis"]["selector"]["candidate"] == "fused/test"
