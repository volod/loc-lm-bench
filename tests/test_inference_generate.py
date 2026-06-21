"""Tests for GPU tier detection and serving config generation."""

from pathlib import Path

import pytest
import yaml

from llb.inference.generate import (
    bucket_vram_mb_to_tier,
    generate_serving_configs,
    load_manifest,
)
from llb.paths import PROJECT_ROOT


@pytest.mark.parametrize(
    ("total_mb", "tier"),
    [
        (12288, 12),
        (16380, 16),
        (24576, 24),
        (32607, 32),
    ],
)
def test_bucket_vram_mb_to_tier(total_mb: int, tier: int) -> None:
    assert bucket_vram_mb_to_tier(total_mb) == tier


def test_manifest_has_supported_tiers() -> None:
    manifest = load_manifest()
    assert manifest["supported_tiers"] == [12, 16, 24, 32]
    for tier in (12, 16, 24, 32):
        assert tier in manifest["tiers"] or str(tier) in manifest["tiers"]
        entries = manifest["tiers"].get(tier) or manifest["tiers"].get(str(tier))
        for target in ("mamaylm", "gemma-4-31b", "qwen3.6"):
            assert target in entries


def test_generate_serving_configs_for_tier_16(tmp_path: Path) -> None:
    out = generate_serving_configs(gpu_gb=16, output_root=tmp_path / "gpu-16gb")
    assert out.is_dir()
    tier = yaml.safe_load((out / "tier.json").read_text(encoding="utf-8"))
    assert tier["tier_gb"] == 16
    assert (out / "serve_mamaylm.sh").exists()
    assert (out / "run_eval_mamaylm.yaml").exists()
    assert (out / "run_eval_mamaylm.sh").exists()
    cfg = (out / "run_eval_mamaylm.yaml").read_text(encoding="utf-8")
    assert "backend: ollama" in cfg
    assert (out / "serve_gemma_4_12b_vllm.sh").exists()
    vllm_cfg = (out / "run_eval_gemma_4_12b_vllm.yaml").read_text(encoding="utf-8")
    assert "gpu_memory_utilization: 0.85" in vllm_cfg
    assert "max_model_len: 8192" in vllm_cfg
    serve = (out / "serve_gemma_4_31b.sh").read_text(encoding="utf-8")
    assert "ollama pull" in serve


def test_generate_serving_configs_for_tier_32(tmp_path: Path) -> None:
    out = generate_serving_configs(gpu_gb=32, output_root=tmp_path / "gpu-32gb")
    mamaylm = (out / "run_eval_mamaylm.yaml").read_text(encoding="utf-8")
    assert "backend: vllm" in mamaylm
    assert "gpu_memory_utilization: 0.9" in mamaylm
    assert "max_model_len: 8192" in mamaylm
    serve = (out / "serve_mamaylm.sh").read_text(encoding="utf-8")
    assert "vllm serve" in serve
    assert "--max-model-len 8192" in serve
    rel = out / "run_eval_mamaylm.sh"
    assert "../../../.." in rel.read_text(encoding="utf-8")


def test_templates_live_under_samples() -> None:
    tmpl = PROJECT_ROOT / "samples" / "config-example" / "templates" / "vllm_serve.sh.tmpl"
    assert tmpl.is_file()
