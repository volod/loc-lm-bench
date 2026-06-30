import pytest

from llb.backends import prepare
from llb.backends import hardware
from llb.backends.hardware import Gpu, detect_gpus, max_vram_mb, parse_smi


def test_load_manifest_bad_yaml_raises_clean_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    # inconsistent indent under the list item (name at col 5, backend at col 6) -> YAML error
    bad.write_text("models:\n  - name: a\n     backend: vllm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        prepare.load_manifest(bad)


def test_load_manifest_non_mapping_entry_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  - just-a-string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        prepare.load_manifest(bad)


def test_load_serving_targets_extracts_generated_tier_json(tmp_path):
    tier_json = tmp_path / "tier.json"
    tier_json.write_text(
        """{
  "targets": [
    {"target": "mamaylm", "backend": "ollama", "model": "hf.co/org/mamay:Q4_K_M"},
    {"target": "gemma-4-vllm", "backend": "vllm", "model": "org/gemma"}
  ]
}
""",
        encoding="utf-8",
    )

    models = prepare.load_serving_targets(tier_json)

    assert models == [
        {
            "name": "serving-mamaylm",
            "backend": "ollama",
            "source": "hf.co/org/mamay:Q4_K_M",
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
    assert prepare.acceptance_url({"license_url": "https://hf.co/x"}) == "https://hf.co/x"
    assert (
        prepare.acceptance_url({"gated": True, "backend": "vllm", "source": "org/m"})
        == "https://huggingface.co/org/m"
    )
    assert prepare.acceptance_url({"backend": "vllm", "source": "org/m"}) is None  # ungated


def test_looks_gated_detects_access_errors_not_404():
    assert prepare._looks_gated(Exception("Access to model X is restricted (gated)"))
    assert prepare._looks_gated(Exception("401 Client Error: Unauthorized"))
    assert not prepare._looks_gated(Exception("404 Client Error: Not Found"))


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
    report = prepare.prepare_models(
        models, dry_run=True, gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")]
    )
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
    action, reason = prepare.decide(
        "ollama", need_mb=20000, max_mb=16000, has_gpu=True, force=False
    )
    assert action == prepare.ACTION_PULL and "CPU" in reason


def test_decide_vllm_skips_oversized_unless_forced():
    action, _ = prepare.decide("vllm", need_mb=20000, max_mb=16000, has_gpu=True, force=False)
    assert action == prepare.ACTION_SKIP
    action, _ = prepare.decide("vllm", need_mb=20000, max_mb=16000, has_gpu=True, force=True)
    assert action == prepare.ACTION_CACHE


def test_decide_vllm_needs_gpu():
    action, reason = prepare.decide("vllm", need_mb=8000, max_mb=0, has_gpu=False, force=False)
    assert action == prepare.ACTION_SKIP and "CUDA GPU" in reason


def test_plan_filters_by_backend():
    models = [
        {"name": "a", "backend": "ollama", "source": "a:1", "min_vram_gb": 4},
        {"name": "b", "backend": "vllm", "source": "org/b", "min_vram_gb": 8},
    ]
    rows = prepare.plan(models, max_mb=16000, has_gpu=True, backend_filter="vllm", force=False)
    assert [r["name"] for r in rows] == ["b"]


def test_plan_expands_per_backend_sources():
    models = [
        {
            "name": "mamay",
            "backend": "vllm",
            "source": "org/mamay-bf16",
            "min_vram_gb": 26,
            "sources": {
                "ollama": {
                    "source": "hf.co/org/mamay-gguf:Q4_K_M",
                    "quant": "q4_k_m",
                    "min_vram_gb": 8,
                }
            },
        }
    ]

    rows = prepare.plan(models, max_mb=16000, has_gpu=True, backend_filter="all", force=False)

    by_name = {r["name"]: r for r in rows}
    assert by_name["mamay"]["action"] == prepare.ACTION_SKIP
    assert by_name["mamay-ollama"]["action"] == prepare.ACTION_PULL
    assert by_name["mamay-ollama"]["source"] == "hf.co/org/mamay-gguf:Q4_K_M"


def test_prepare_models_dispatches_and_skips():
    models = [
        {"name": "small", "backend": "ollama", "source": "small:1", "min_vram_gb": 4},
        {"name": "fits", "backend": "vllm", "source": "org/fits", "min_vram_gb": 8},
        {"name": "huge", "backend": "vllm", "source": "org/huge", "min_vram_gb": 80},
    ]
    pulled, cached = [], []
    report = prepare.prepare_models(
        models,
        gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")],
        ollama_pull=lambda src: pulled.append(src) or (True, "pulled"),
        hf_cache=lambda src, tok, cd: cached.append(src) or (True, "/cache/" + src),
    )
    by_name = {r["name"]: r for r in report["results"]}
    assert pulled == ["small:1"]
    assert cached == ["org/fits"]  # huge skipped (over VRAM), never cached
    assert by_name["small"]["status"] == "done"
    assert by_name["fits"]["status"] == "done"
    assert by_name["huge"]["status"] == "skipped"


def test_prepare_models_dry_run_touches_nothing():
    models = [{"name": "small", "backend": "ollama", "source": "small:1", "min_vram_gb": 4}]
    calls = []
    report = prepare.prepare_models(
        models,
        dry_run=True,
        gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")],
        ollama_pull=lambda src: calls.append(src),
    )
    assert calls == []
    assert report["results"][0]["status"] == "planned"
