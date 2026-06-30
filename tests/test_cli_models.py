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
