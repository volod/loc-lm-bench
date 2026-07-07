"""Tests for GPU tier detection and serving config generation."""

from pathlib import Path

import pytest
import yaml

from llb.inference.generate import (
    bucket_vram_mb_to_tier,
    generate_serving_configs,
    load_manifest,
    select_host_gemma4_target,
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
        for target in ("mamaylm", "lapa", "gemma-4", "qwen3.6", "mistral"):
            assert target in entries


def test_generate_serving_configs_for_tier_12_uses_offloaded_long_context_12b(
    tmp_path: Path,
) -> None:
    out = generate_serving_configs(gpu_gb=12, output_root=tmp_path / "gpu-12gb")
    tier = yaml.safe_load((out / "tier.json").read_text(encoding="utf-8"))
    targets = {item["target"]: item for item in tier["targets"]}

    assert "gemma-4-12b-vllm" in targets
    assert "gemma-4-e4b-vllm" not in targets
    cfg = (out / "run_eval_gemma_4_12b_vllm.yaml").read_text(encoding="utf-8")
    assert "model: google/gemma-4-12B-it-qat-w4a16-ct" in cfg
    assert "gpu_memory_utilization: 0.9" in cfg
    assert "max_model_len: 16384" in cfg
    assert "cpu_offload_gb: 16" in cfg
    assert "kv_offloading_size_gb: 32" in cfg
    vllm_serve = (out / "serve_gemma_4_12b_vllm.sh").read_text(encoding="utf-8")
    assert "--cpu-offload-gb 16" in vllm_serve
    assert "--kv-offloading-size 32" in vllm_serve


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
    assert "MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M" in cfg
    lapa = (out / "run_eval_lapa.yaml").read_text(encoding="utf-8")
    assert "hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M" in lapa
    assert "backend: ollama" in lapa
    assert (out / "serve_gemma_4_12b_vllm.sh").exists()
    vllm_cfg = (out / "run_eval_gemma_4_12b_vllm.yaml").read_text(encoding="utf-8")
    assert "gpu_memory_utilization: 0.85" in vllm_cfg
    assert "max_model_len: 16384" in vllm_cfg
    assert "cpu_offload_gb: 16" in vllm_cfg
    assert "kv_offloading_size_gb: 32" in vllm_cfg
    vllm_serve = (out / "serve_gemma_4_12b_vllm.sh").read_text(encoding="utf-8")
    assert "--cpu-offload-gb 16" in vllm_serve
    assert "--kv-offloading-size 32" in vllm_serve
    assert "${VLLM_USE_FLASHINFER_SAMPLER:-0}" in vllm_serve
    assert "${{" not in vllm_serve
    serve = (out / "serve_gemma_4.sh").read_text(encoding="utf-8")
    assert "ollama pull" in serve
    gemma = (out / "run_eval_gemma_4.yaml").read_text(encoding="utf-8")
    assert "model: gemma4:31b" in gemma
    # Mistral is a primary family target: Ollama curated q4_k_m tag (offload) on the 16 GiB tier.
    mistral = (out / "run_eval_mistral.yaml").read_text(encoding="utf-8")
    assert "backend: ollama" in mistral
    assert "model: mistral-small3.1:24b" in mistral
    assert (out / "serve_mistral.sh").exists()


def test_select_host_gemma4_target_prefers_cuda_12b_on_16gb() -> None:
    row = select_host_gemma4_target(gpu_gb=16)
    assert row["target"] == "gemma-4-12b-vllm"
    assert row["backend"] == "vllm"
    assert row["model"] == "google/gemma-4-12B-it-qat-w4a16-ct"
    assert row["gpu_memory_utilization"] == 0.85
    assert row["max_model_len"] == 16384
    assert row["cpu_offload_gb"] == 16
    assert row["kv_offloading_size_gb"] == 32


def test_select_host_gemma4_target_uses_offloaded_12b_on_12gb_for_long_context() -> None:
    row = select_host_gemma4_target(gpu_gb=12, min_context_tokens=16384)
    assert row["target"] == "gemma-4-12b-vllm"
    assert row["backend"] == "vllm"
    assert row["model"] == "google/gemma-4-12B-it-qat-w4a16-ct"
    assert row["max_model_len"] == 16384
    assert row["cpu_offload_gb"] == 16
    assert row["kv_offloading_size_gb"] == 32


def test_select_host_gemma4_target_uses_31b_vllm_on_32gb() -> None:
    row = select_host_gemma4_target(gpu_gb=32)
    assert row["target"] == "gemma-4"
    assert row["backend"] == "vllm"
    assert row["model"] == "google/gemma-4-31B-it-qat-w4a16-ct"
    assert row["max_model_len"] == 16384


def test_generate_serving_configs_for_tier_32(tmp_path: Path) -> None:
    out = generate_serving_configs(gpu_gb=32, output_root=tmp_path / "gpu-32gb")
    mamaylm = (out / "run_eval_mamaylm.yaml").read_text(encoding="utf-8")
    assert "backend: vllm" in mamaylm
    assert "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic" in mamaylm
    assert "gpu_memory_utilization: 0.9" in mamaylm
    assert "max_model_len: 8192" in mamaylm
    serve = (out / "serve_mamaylm.sh").read_text(encoding="utf-8")
    assert "vllm serve" in serve
    assert "--max-model-len 8192" in serve
    lapa = (out / "run_eval_lapa.yaml").read_text(encoding="utf-8")
    assert "backend: vllm" in lapa
    assert "lapa-llm/lapa-v0.1.2-instruct" in lapa
    # Mistral serves via vLLM FP8 on the 32 GiB tier (w4a16 on 24 GiB, Ollama GGUF on 12/16 GiB).
    mistral = (out / "run_eval_mistral.yaml").read_text(encoding="utf-8")
    assert "backend: vllm" in mistral
    assert "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic" in mistral
    assert "max_model_len: 8192" in mistral
    serve_mistral = (out / "serve_mistral.sh").read_text(encoding="utf-8")
    assert "vllm serve" in serve_mistral
    assert "--quantization" not in serve_mistral  # fp8 compressed-tensors is auto-detected
    rel = out / "run_eval_mamaylm.sh"
    assert "../../../.." in rel.read_text(encoding="utf-8")


def test_generate_serving_configs_mistral_24_is_vllm_w4a16(tmp_path: Path) -> None:
    # gpu-tier-24-mistral-vllm: the 24 GiB tier serves the w4a16 quant GPU-resident via vLLM.
    out = generate_serving_configs(gpu_gb=24, output_root=tmp_path / "gpu-24gb")
    mistral = (out / "run_eval_mistral.yaml").read_text(encoding="utf-8")
    assert "backend: vllm" in mistral
    assert "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16" in mistral
    assert "max_model_len: 16384" in mistral
    serve = (out / "serve_mistral.sh").read_text(encoding="utf-8")
    assert "vllm serve" in serve
    assert "--quantization" not in serve  # compressed-tensors w4a16 is auto-detected


def test_templates_live_under_samples() -> None:
    tmpl = PROJECT_ROOT / "samples" / "config-example" / "templates" / "vllm_serve.sh.tmpl"
    assert tmpl.is_file()
