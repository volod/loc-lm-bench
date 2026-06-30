import pytest

from llb.backends import prepare
from llb.backends import hardware
from llb.backends import resolver
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

    rows = prepare.plan(models, max_mb=24000, has_gpu=True, backend_filter="all", force=False)
    by_name = {r["name"]: r for r in rows}

    # fp8 (~30 GiB floor) is skipped on a 24 GiB card; the w4a16 parent caches; the GGUF pulls.
    assert by_name["mistral-vllm-fp8"]["action"] == prepare.ACTION_SKIP
    assert by_name["mistral-vllm-fp8"]["source"] == "org/mistral-fp8"
    assert by_name["mistral"]["action"] == prepare.ACTION_CACHE  # w4a16 parent (source matches)
    assert by_name["mistral-ollama"]["action"] == prepare.ACTION_PULL


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


# --- disk preflight (prevent a long download that fails for lack of space) ---------------------


def test_estimate_download_prices_fp8_embedding_and_floors():
    fp8 = {
        "name": "m",
        "backend": "vllm",
        "source": "org/m-fp8",
        "params_b": 24,
        "quant": "fp8",
        "vocab_size": 131072,
        "hidden_size": 5120,
        "tie_word_embeddings": False,
    }
    gguf = {
        "name": "m",
        "backend": "ollama",
        "source": "hf.co/org/m:Q4_K_M",
        "params_b": 24,
        "quant": "q4_k_m",
    }
    # fp8 (with bf16 embedding premium) is a much bigger download than the q4_k_m GGUF
    assert prepare.estimate_download_mb(fp8) > prepare.estimate_download_mb(gguf) > 0
    # no size hints -> falls back to the min_vram_gb floor, then the hard floor
    assert (
        prepare.estimate_download_mb(
            {"name": "x", "backend": "vllm", "source": "o/x", "min_vram_gb": 8}
        )
        == 8 * 1024
    )
    assert (
        prepare.estimate_download_mb({"name": "x", "backend": "vllm", "source": "o/x"})
        == prepare.MIN_DOWNLOAD_MB
    )


def test_disk_precheck_blocks_only_when_provably_too_small():
    ok, _ = prepare.disk_precheck(required_mb=10_000, free_mb=50_000)
    assert ok
    blocked, reason = prepare.disk_precheck(required_mb=24_000, free_mb=5_000)
    assert not blocked and "insufficient disk" in reason
    # free_mb == 0 means "unknown" (probe failed) and must never block
    assert prepare.disk_precheck(required_mb=24_000, free_mb=0)[0]


def test_store_dir_for_routes_ollama_vs_hf(tmp_path):
    assert prepare.store_dir_for("vllm", tmp_path / "hf") == tmp_path / "hf"
    assert prepare.store_dir_for("ollama", None).name == "models"


def test_ollama_present_check_scans_configured_store(tmp_path, monkeypatch):
    # Offline fallback: when the daemon is unreachable the reuse check scans the blob store, which
    # moves with the install (user home vs systemd /usr/share/ollama vs OLLAMA_MODELS).
    monkeypatch.setattr(resolver, "_make_ollama_probe", lambda _host: lambda _s: False)
    store = tmp_path / "models"
    manifest = store / "manifests" / "hf.co" / "lmstudio-community" / "Foo-GGUF"
    manifest.mkdir(parents=True)
    (manifest / "Q4_K_M").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    assert prepare._ollama_store_dir() == store
    assert prepare._default_present_check(
        {"backend": "ollama", "source": "hf.co/lmstudio-community/Foo-GGUF:Q4_K_M"}
    )
    assert not prepare._default_present_check(
        {"backend": "ollama", "source": "hf.co/lmstudio-community/Foo-GGUF:Q8_0"}  # other quant
    )


def test_ollama_present_check_trusts_running_daemon(tmp_path, monkeypatch):
    # prep-disk-present-probe: the daemon /api/tags probe is authoritative -- a tag the daemon
    # serves counts as cached even if it is not in a store path we scan, so a low-disk re-pull of
    # an already-pulled tag is never wrongly refused.
    monkeypatch.setattr(
        resolver, "_make_ollama_probe", lambda _host: lambda s: s == "mistral-small3.1:24b"
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "empty"))  # nothing on disk to scan
    assert prepare._default_present_check({"backend": "ollama", "source": "mistral-small3.1:24b"})
    # a daemon-served tag skips the precheck, so a near-full disk still reuses it
    models = [
        {"name": "m", "backend": "ollama", "source": "mistral-small3.1:24b", "min_vram_gb": 8}
    ]
    pulled = []
    report = prepare.prepare_models(
        models,
        gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")],
        ollama_pull=lambda src: pulled.append(src) or (True, "reused"),
        disk_free_reader=lambda _store: 1,  # basically no free space
    )
    row = report["results"][0]
    assert row["status"] == "done" and "cached (reuse)" in row["detail"]
    assert pulled == ["mistral-small3.1:24b"]


def test_prepare_fails_fast_when_store_too_small():
    models = [
        {
            "name": "big",
            "backend": "vllm",
            "source": "org/big",
            "min_vram_gb": 8,
            "params_b": 24,
            "quant": "fp8",
        }
    ]
    cached = []
    report = prepare.prepare_models(
        models,
        gpus=[Gpu(0, "Fake", 40000, 39000, "1.0")],  # fits VRAM, so it would otherwise cache
        hf_cache=lambda src, tok, cd: cached.append(src) or (True, "/cache"),
        disk_free_reader=lambda _store: 3_000,  # only 3 GB free -> refuse the ~27 GB pull
        present_check=lambda _spec: False,
    )
    row = report["results"][0]
    assert row["status"] == "failed" and "insufficient disk" in row["detail"]
    assert cached == []  # the long download never started


def test_prepare_reuses_cached_artifact_despite_low_disk():
    models = [
        {
            "name": "big",
            "backend": "vllm",
            "source": "org/big",
            "min_vram_gb": 8,
            "params_b": 24,
            "quant": "fp8",
        }
    ]
    cached = []
    report = prepare.prepare_models(
        models,
        gpus=[Gpu(0, "Fake", 40000, 39000, "1.0")],
        hf_cache=lambda src, tok, cd: cached.append(src) or (True, "/cache"),
        disk_free_reader=lambda _store: 1,  # basically no free space
        present_check=lambda _spec: True,  # already cached -> skip the precheck, reuse it
    )
    row = report["results"][0]
    assert row["status"] == "done" and "cached (reuse)" in row["detail"]
    assert cached == ["org/big"]


def test_prepare_dry_run_previews_disk_shortfall_without_blocking():
    models = [
        {
            "name": "big",
            "backend": "vllm",
            "source": "org/big",
            "min_vram_gb": 8,
            "params_b": 24,
            "quant": "fp8",
        }
    ]
    report = prepare.prepare_models(
        models,
        dry_run=True,
        gpus=[Gpu(0, "Fake", 40000, 39000, "1.0")],
        disk_free_reader=lambda _store: 3_000,
        present_check=lambda _spec: False,
    )
    row = report["results"][0]
    assert row["status"] == "planned"  # dry run never fails
    assert "insufficient disk" in row["detail"]
