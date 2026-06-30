from pathlib import Path

import pytest

from llb.cli import models


def test_local_backend_ready_skips_missing_vllm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda _name: None)

    ready, reason = models._local_backend_ready("vllm", tmp_path)

    assert ready is False
    assert "make build-vllm" in reason


def test_local_backend_ready_accepts_project_llamacpp_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda _name: None)
    binary = tmp_path / "llb" / "llamacpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    ready, reason = models._local_backend_ready("llamacpp", tmp_path)

    assert ready is True
    assert reason == ""


def test_expand_quant_variants_splits_multi_quant_vllm() -> None:
    # list-models-multi-quant-visibility: a multi-quant entry yields one plan row per vLLM quant,
    # priced independently, with distinct names; single-source entries pass through untouched.
    specs = [
        {
            "name": "mistral-small-3.1-24b",
            "backend": "vllm",
            "source": "org/mistral-w4a16",
            "params_b": 24,
            "quant": "w4a16",
            "sources": {
                "vllm": [
                    {"source": "org/mistral-fp8", "quant": "fp8"},
                    {"source": "org/mistral-w4a16", "quant": "w4a16"},
                ],
                "ollama": {"source": "mistral:24b", "quant": "q4_k_m"},
            },
        },
        {"name": "solo", "backend": "vllm", "source": "org/solo", "quant": "fp8"},
    ]

    out = models._expand_quant_variants(specs)

    by_name = {s["name"]: s for s in out}
    assert set(by_name) == {"mistral-small-3.1-24b-fp8", "mistral-small-3.1-24b", "solo"}
    assert by_name["mistral-small-3.1-24b-fp8"]["source"] == "org/mistral-fp8"
    assert by_name["mistral-small-3.1-24b-fp8"]["quant"] == "fp8"
    # the variant whose source matches the parent keeps the parent name (w4a16)
    assert by_name["mistral-small-3.1-24b"]["quant"] == "w4a16"
    assert by_name["solo"] is specs[1]  # single-source entry passes through unchanged


def test_prep_models_exits_nonzero_on_failed_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(models, "load_models", lambda _manifest: [])

    def fake_prepare_models(*_args, **_kwargs):
        return {
            "gpus": [],
            "results": [
                {
                    "status": "failed",
                    "backend": "ollama",
                    "name": "bad",
                    "source": "bad:1",
                    "detail": "boom",
                }
            ],
        }

    monkeypatch.setattr("llb.backends.prepare.prepare_models", fake_prepare_models)

    with pytest.raises(SystemExit) as exc:
        models.prep_models_cmd(manifest=tmp_path / "models.yaml")

    assert exc.value.code == 1


def test_prep_serving_targets_exits_nonzero_on_failed_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("llb.backends.prepare.load_serving_targets", lambda _path: [])

    def fake_prepare_models(*_args, **_kwargs):
        return {
            "gpus": [],
            "results": [
                {
                    "status": "failed",
                    "backend": "ollama",
                    "name": "bad",
                    "source": "bad:1",
                    "detail": "boom",
                }
            ],
        }

    monkeypatch.setattr("llb.backends.prepare.prepare_models", fake_prepare_models)

    with pytest.raises(SystemExit) as exc:
        models.prep_serving_targets_cmd(tier_json=tmp_path / "tier.json")

    assert exc.value.code == 1
