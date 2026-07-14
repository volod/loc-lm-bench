"""Tests for prepare models execution."""

from llb.backends import resolver
from llb.backends.hardware import Gpu
from llb.backends.prepare.base import (
    MIN_DOWNLOAD_MB,
)
from llb.backends.prepare.run import prepare_models
from llb.backends.prepare.stores import (
    _default_present_check,
    _ollama_store_dir,
    disk_precheck,
    estimate_download_mb,
    store_dir_for,
)


def test_prepare_models_dispatches_and_skips():
    models = [
        {"name": "small", "backend": "ollama", "source": "small:1", "min_vram_gb": 4},
        {"name": "fits", "backend": "vllm", "source": "org/fits", "min_vram_gb": 8},
        {"name": "huge", "backend": "vllm", "source": "org/huge", "min_vram_gb": 80},
    ]
    pulled, cached = [], []
    report = prepare_models(
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
    report = prepare_models(
        models,
        dry_run=True,
        gpus=[Gpu(0, "Fake", 16000, 15000, "1.0")],
        ollama_pull=lambda src: calls.append(src),
    )
    assert calls == []
    assert report["results"][0]["status"] == "planned"


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
    assert estimate_download_mb(fp8) > estimate_download_mb(gguf) > 0
    # no size hints -> falls back to the min_vram_gb floor, then the hard floor
    assert (
        estimate_download_mb({"name": "x", "backend": "vllm", "source": "o/x", "min_vram_gb": 8})
        == 8 * 1024
    )
    assert (
        estimate_download_mb({"name": "x", "backend": "vllm", "source": "o/x"}) == MIN_DOWNLOAD_MB
    )


def test_disk_precheck_blocks_only_when_provably_too_small():
    ok, _ = disk_precheck(required_mb=10_000, free_mb=50_000)
    assert ok
    blocked, reason = disk_precheck(required_mb=24_000, free_mb=5_000)
    assert not blocked and "insufficient disk" in reason
    # free_mb == 0 means "unknown" (probe failed) and must never block
    assert disk_precheck(required_mb=24_000, free_mb=0)[0]


def test_store_dir_for_routes_ollama_vs_hf(tmp_path):
    assert store_dir_for("vllm", tmp_path / "hf") == tmp_path / "hf"
    assert store_dir_for("ollama", None).name == "models"


def test_ollama_present_check_scans_configured_store(tmp_path, monkeypatch):
    # Offline fallback: when the daemon is unreachable the reuse check scans the blob store, which
    # moves with the install (user home vs systemd /usr/share/ollama vs OLLAMA_MODELS).
    monkeypatch.setattr(resolver, "_make_ollama_probe", lambda _host: lambda _s: False)
    store = tmp_path / "models"
    manifest = store / "manifests" / "hf.co" / "lmstudio-community" / "Foo-GGUF"
    manifest.mkdir(parents=True)
    (manifest / "Q4_K_M").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    assert _ollama_store_dir() == store
    assert _default_present_check(
        {"backend": "ollama", "source": "hf.co/lmstudio-community/Foo-GGUF:Q4_K_M"}
    )
    assert not _default_present_check(
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
    assert _default_present_check({"backend": "ollama", "source": "mistral-small3.1:24b"})
    # a daemon-served tag skips the precheck, so a near-full disk still reuses it
    models = [
        {"name": "m", "backend": "ollama", "source": "mistral-small3.1:24b", "min_vram_gb": 8}
    ]
    pulled = []
    report = prepare_models(
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
    report = prepare_models(
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
    report = prepare_models(
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
    report = prepare_models(
        models,
        dry_run=True,
        gpus=[Gpu(0, "Fake", 40000, 39000, "1.0")],
        disk_free_reader=lambda _store: 3_000,
        present_check=lambda _spec: False,
    )
    row = report["results"][0]
    assert row["status"] == "planned"  # dry run never fails
    assert "insufficient disk" in row["detail"]
