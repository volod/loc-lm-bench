"""Committed Ukrainian roster invariants and refresh-entry resolver fixtures."""

from llb.backends.prepare.manifest import load_manifest
from llb.backends.resolver import resolve
from llb.backends.resolver_sources import candidate_sources
from llb.core.contracts.models import ModelSpec
from llb.core.paths import PROJECT_ROOT
from test_resolver import ALL_AVAILABLE


ROSTER = PROJECT_ROOT / "samples" / "configs" / "models_uk.yaml"
REFRESH_NAMES = ("gemma-4-26b-a4b", "qwen3.6-27b")


def _models() -> list[ModelSpec]:
    return load_manifest(ROSTER)


def test_roster_records_license_and_multi_backend_sources() -> None:
    models = _models()

    assert {model["name"] for model in models}.issuperset(REFRESH_NAMES)
    for model in models:
        assert model.get("license") in {"Apache-2.0", "Gemma"}
        assert str(model.get("license_url", "")).startswith("https://")
        assert {backend for backend, _ in candidate_sources(model)}.issuperset(
            {"vllm", "ollama", "llamacpp"}
        )


def test_refresh_entries_have_multi_quant_vllm_records() -> None:
    by_name = {model["name"]: model for model in _models()}

    for name in REFRESH_NAMES:
        vllm = by_name[name]["sources"]["vllm"]
        assert isinstance(vllm, list) and len(vllm) >= 2
        assert len({record["quant"] for record in vllm}) == len(vllm)


def test_refresh_entries_resolve_to_ollama_on_16gb_fixture() -> None:
    by_name = {model["name"]: model for model in _models()}

    for name in REFRESH_NAMES:
        result = resolve(by_name[name], 16380, 128 * 1024, probes=ALL_AVAILABLE)
        assert result["chosen_backend"] == "ollama"
        assert result["verdict"] in {"gpu", "offload"}


def test_gemma_26b_fp8_resolves_on_32gb_fixture() -> None:
    model = next(model for model in _models() if model["name"] == "gemma-4-26b-a4b")

    result = resolve(model, 32607, 128 * 1024, probes=ALL_AVAILABLE)

    assert result["chosen_backend"] == "vllm"
    assert result["chosen_source"] == "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
