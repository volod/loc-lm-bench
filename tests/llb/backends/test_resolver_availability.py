"""Tests for resolver availability."""

from llb.backends.resolver import (
    ResolverProbes,
    resolve,
)
from test_resolver import BIG, HOST_RAM, HOST_VRAM, SMALL


def test_resolve_marks_unavailable_source_not_runnable():
    probes = ResolverProbes(
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
        hf_repo=lambda _s: False, gguf=lambda _s: False, ollama_tag=lambda _s: False
    )
    out = resolve(SMALL, HOST_VRAM, HOST_RAM, probes=probes)
    assert out["chosen_backend"] is None
    assert out["verdict"] == "no"
    assert "no available backend" in out["note"]
