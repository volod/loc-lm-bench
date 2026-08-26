"""Tests for resolver availability."""

from llb.backends.resolver import resolve
from llb.backends.resolver_probes import ResolverProbes
from tests.llb.backends.test_resolver import BIG, HOST_RAM, HOST_VRAM, SMALL


def test_resolve_marks_unavailable_source_not_runnable():
    probes = ResolverProbes(
        runtime_version=lambda _b: None,
        hf_repo=lambda _s: False,  # vLLM repo missing
        gguf=lambda _s: True,
        ollama_tag=lambda _s: True,
    )
    out = resolve(BIG, HOST_VRAM, HOST_RAM, probes=probes)
    vllm = next(c for c in out["candidates"] if c["backend"] == "vllm")
    assert vllm["available"] is False and vllm["runnable"] is False
    assert out["chosen_backend"] == "ollama"  # the available offload backend


def test_resolve_none_when_nothing_available():
    probes = ResolverProbes(
        runtime_version=lambda _b: None,
        hf_repo=lambda _s: False,
        gguf=lambda _s: False,
        ollama_tag=lambda _s: False,
    )
    out = resolve(SMALL, HOST_VRAM, HOST_RAM, probes=probes)
    assert out["chosen_backend"] is None
    assert out["verdict"] == "no"
    assert "no available backend" in out["note"]


def test_runtime_below_the_floor_is_a_named_skip_not_a_missing_source():
    """The Gemma 4 12B case: the tag exists and fits, and an old Ollama still cannot serve it."""
    spec = {
        **SMALL,
        "name": "gemma-4-12b-it-w4a16",
        "sources": {
            "ollama": {
                "source": "gemma4:12b",
                "runtime_arch": "gemma4",
                "min_runtime_version": "0.30.5",
            }
        },
    }
    probes = ResolverProbes(
        hf_repo=lambda _s: False,  # no vLLM path on this host
        gguf=lambda _s: True,
        ollama_tag=lambda _s: True,  # the tag IS there
        runtime_version=lambda _b: "0.20.6",
        artifact_requirement=lambda _b, _s: None,
    )
    out = resolve(spec, HOST_VRAM, HOST_RAM, probes=probes)  # type: ignore[arg-type]
    ollama = next(c for c in out["candidates"] if c["backend"] == "ollama")
    assert ollama["available"] is False and ollama["runnable"] is False
    assert ollama["skip"] == "runtime-version-floor"
    assert "ollama 0.20.6" in ollama["reason"] and "ollama >= 0.30.5" in ollama["reason"]
    assert "source not found" not in ollama["reason"]


def test_a_new_enough_runtime_leaves_the_source_available():
    spec = {
        **SMALL,
        "backend": "ollama",
        "source": "gemma4:12b",
        "runtime_arch": "gemma4",
        "min_runtime_version": "0.30.5",
    }
    probes = ResolverProbes(
        hf_repo=lambda _s: False,
        gguf=lambda _s: False,
        ollama_tag=lambda _s: True,
        runtime_version=lambda _b: "0.32.15",
        artifact_requirement=lambda _b, _s: None,
    )
    out = resolve(spec, HOST_VRAM, HOST_RAM, probes=probes)  # type: ignore[arg-type]
    assert out["chosen_backend"] == "ollama"
    assert all("skip" not in c for c in out["candidates"])


def test_the_version_is_read_once_for_a_whole_roster():
    reads: list[str] = []

    def version(backend: str) -> str:
        reads.append(backend)
        return "0.32.15"

    probes = ResolverProbes(
        hf_repo=lambda _s: True,
        gguf=lambda _s: True,
        ollama_tag=lambda _s: True,
        runtime_version=version,
        artifact_requirement=lambda _b, _s: None,
    )
    spec = {**SMALL, "min_runtime_version": "0.1.0", "sources": {"ollama": "t", "llamacpp": "g"}}
    resolve(spec, HOST_VRAM, HOST_RAM, probes=probes)  # type: ignore[arg-type]
    resolve(spec, HOST_VRAM, HOST_RAM, probes=probes)  # type: ignore[arg-type]
    # Two resolutions over three candidate backends: one version read each, memoized on the probes.
    assert sorted(reads) == ["llamacpp", "ollama", "vllm"]
