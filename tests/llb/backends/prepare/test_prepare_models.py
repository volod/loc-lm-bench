import pytest

from llb.backends import hardware
from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb, parse_smi
from llb.backends.prepare.base import (
    ACTION_CACHE,
    ACTION_PULL,
    ACTION_SKIP,
)
from llb.backends.prepare.fetch import _looks_gated
from llb.backends.prepare.manifest import load_manifest, load_serving_targets
from llb.backends.prepare.planning import acceptance_url, decide, plan
from llb.backends.prepare.run import prepare_models


def test_load_manifest_bad_yaml_raises_clean_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    # inconsistent indent under the list item (name at col 5, backend at col 6) -> YAML error
    bad.write_text("models:\n  - name: a\n     backend: vllm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_manifest(bad)


def test_load_manifest_non_mapping_entry_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  - just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_manifest(bad)


def test_load_serving_targets_extracts_generated_tier_json(tmp_path):
    tier_json = tmp_path / "tier.json"
    tier_json.write_text(
        """{
  "targets": [
    {
      "target": "mamaylm",
      "backend": "ollama",
      "model": "hf.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M"
    },
    {"target": "gemma-4-vllm", "backend": "vllm", "model": "org/gemma"}
  ]
}
""",
        encoding="utf-8",
    )

    models = load_serving_targets(tier_json)

    assert models == [
        {
            "name": "serving-mamaylm",
            "backend": "ollama",
            "source": "hf.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M",
            "min_vram_gb": 0,
            "notes": "generated serving-tier target",
        },
        {
            "name": "serving-gemma-4-vllm",
            "backend": "vllm",
            "source": "org/gemma",
            "min_vram_gb": 0,
            "notes": "generated serving-tier target",
        },
    ]


def test_acceptance_url_explicit_derived_and_none():
    assert acceptance_url({"license_url": "https://hf.co/x"}) == "https://hf.co/x"
    assert (
        acceptance_url({"gated": True, "backend": "vllm", "source": "org/m"})
        == "https://huggingface.co/org/m"
    )
    assert acceptance_url({"backend": "vllm", "source": "org/m"}) is None  # ungated


def test_looks_gated_detects_access_errors_not_404():
    assert _looks_gated(Exception("Access to model X is restricted (gated)"))
    assert _looks_gated(Exception("401 Client Error: Unauthorized"))
    assert not _looks_gated(Exception("404 Client Error: Not Found"))


def test_prepare_models_surfaces_license_link_for_gated():
    models = [
        {
            "name": "g",
            "backend": "vllm",
            "source": "org/Gated",
            "min_vram_gb": 4,
            "gated": True,
            "license_url": "https://huggingface.co/org/Gated",
        }
    ]
    report = prepare_models(models, dry_run=True, gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")])
    assert "huggingface.co/org/Gated" in report["results"][0]["detail"]


def test_parse_smi_reads_fields():
    out = "NVIDIA GeForce RTX 4060 Ti, 16380, 15500, 550.120\n"
    gpus = parse_smi(out)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA GeForce RTX 4060 Ti"
    assert g.total_mb == 16380 and g.free_mb == 15500 and g.driver == "550.120"
    assert max_vram_mb(gpus) == 16380


def test_parse_smi_skips_garbage_lines():
    assert parse_smi("\n  \nbad line\n") == []
    assert max_vram_mb([]) == 0


def test_detect_gpus_falls_back_to_absolute_nvidia_smi(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4060 Ti, 16380, 15000, 595.71.05\n"

    def fake_run(argv, **_kwargs):
        calls.append(argv[0])
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError
        return Result()

    monkeypatch.setattr(hardware.shutil, "which", lambda _name: None)
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    gpus = detect_gpus()
    assert calls[:2] == ["nvidia-smi", "/usr/bin/nvidia-smi"]
    assert gpus == [Gpu(0, "NVIDIA GeForce RTX 4060 Ti", 16380, 15000, "595.71.05")]


def test_decide_ollama_pulls_even_when_oversized():
    action, reason = decide("ollama", need_mb=20000, max_mb=16000, has_gpu=True, force=False)
    assert action == ACTION_PULL and "CPU" in reason


def test_decide_vllm_skips_oversized_unless_forced():
    action, _ = decide("vllm", need_mb=20000, max_mb=16000, has_gpu=True, force=False)
    assert action == ACTION_SKIP
    action, _ = decide("vllm", need_mb=20000, max_mb=16000, has_gpu=True, force=True)
    assert action == ACTION_CACHE


def test_decide_vllm_needs_gpu():
    action, reason = decide("vllm", need_mb=8000, max_mb=0, has_gpu=False, force=False)
    assert action == ACTION_SKIP and "CUDA GPU" in reason


def test_plan_filters_by_backend():
    models = [
        {"name": "a", "backend": "ollama", "source": "a:1", "min_vram_gb": 4},
        {"name": "b", "backend": "vllm", "source": "org/b", "min_vram_gb": 8},
    ]
    rows = plan(models, max_mb=16000, has_gpu=True, backend_filter="vllm", force=False)
    assert [r["name"] for r in rows] == ["b"]


def test_plan_expands_per_backend_sources():
    models = [
        {
            "name": "mamaylm-v2-12b",
            "backend": "vllm",
            "source": "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
            "min_vram_gb": 26,
            "sources": {
                "ollama": {
                    "source": "hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M",
                    "quant": "q4_k_m",
                    "min_vram_gb": 8,
                }
            },
        }
    ]

    rows = plan(models, max_mb=16000, has_gpu=True, backend_filter="all", force=False)

    by_name = {r["name"]: r for r in rows}
    assert by_name["mamaylm-v2-12b"]["action"] == ACTION_SKIP
    assert by_name["mamaylm-v2-12b-ollama"]["action"] == ACTION_PULL
    assert (
        by_name["mamaylm-v2-12b-ollama"]["source"]
        == "hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M"
    )


def test_plan_expands_multi_quant_vllm_list():
    # resolver-multi-quant-vllm: a vLLM backend mapped to a LIST of quants expands to one prep
    # artifact per quant, each priced by its own quant + min_vram, with distinct names.
    models = [
        {
            "name": "mistral",
            "backend": "vllm",
            "source": "org/mistral-w4a16",
            "params_b": 24,
            "quant": "w4a16",
            "min_vram_gb": 20,
            "sources": {
                "vllm": [
                    {"source": "org/mistral-fp8", "quant": "fp8", "min_vram_gb": 30},
                    {"source": "org/mistral-w4a16", "quant": "w4a16", "min_vram_gb": 20},
                ],
                "ollama": {"source": "mistral:24b", "quant": "q4_k_m", "min_vram_gb": 8},
            },
        }
    ]

    rows = plan(models, max_mb=24000, has_gpu=True, backend_filter="all", force=False)
    by_name = {r["name"]: r for r in rows}

    # fp8 (~30 GiB floor) is skipped on a 24 GiB card; the w4a16 parent caches; the GGUF pulls.
    assert by_name["mistral-vllm-fp8"]["action"] == ACTION_SKIP
    assert by_name["mistral-vllm-fp8"]["source"] == "org/mistral-fp8"
    assert by_name["mistral"]["action"] == ACTION_CACHE  # w4a16 parent (source matches)
    assert by_name["mistral-ollama"]["action"] == ACTION_PULL


# --- disk preflight (prevent a long download that fails for lack of space) ---------------------
