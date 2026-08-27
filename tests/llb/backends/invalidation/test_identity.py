"""Placing a RECORDED model identity in the register, whichever way a run wrote it down."""

import pytest

from llb.backends.invalidation.identity import ModelIndex
from llb.backends.resolver_sources import candidate_sources
from llb.backends.roster import load_register
from llb.core.paths import PROJECT_ROOT

ROSTER = PROJECT_ROOT / "samples" / "configs" / "models_uk.yaml"


@pytest.fixture
def register():
    return load_register(ROSTER)


@pytest.fixture
def index(register):
    return ModelIndex(register)


def test_every_roster_entry_resolves_under_its_logical_name(register, index) -> None:
    for family in register.families:
        for generation in family.generations:
            for model in generation.models:
                resolved = index.resolve(str(model["name"]))

                assert resolved is not None
                assert (resolved.family_id, resolved.generation_id) == (family.id, generation.id)


def test_every_served_source_resolves_to_the_entry_that_serves_it(register, index) -> None:
    """A run records the artifact it launched, not the logical name, so both must land."""
    for family in register.families:
        for generation in family.generations:
            for model in generation.models:
                for _backend, record in candidate_sources(model):
                    resolved = index.resolve(str(record["source"]))

                    assert resolved is not None, record["source"]
                    assert resolved.generation_id == generation.id


def test_an_ollama_tag_and_the_logical_name_name_one_entry(index) -> None:
    by_tag = index.resolve("mistral-small3.1:24b")
    by_name = index.resolve("mistral-small-3.1-24b")

    assert by_tag is not None and by_name is not None
    assert by_tag.model_name == by_name.model_name == "mistral-small-3.1-24b"
    assert (by_tag.family_id, by_tag.generation_id) == ("mistral", "3.1")


def test_the_recorded_spelling_survives_resolution(index) -> None:
    """A report an operator cannot grep the evidence for is a report they have to re-derive."""
    resolved = index.resolve("  Qwen/Qwen3.8-27B-FP8  ")

    assert resolved is not None
    assert resolved.recorded == "Qwen/Qwen3.8-27B-FP8"
    assert resolved.model_name == "qwen3.8-27b"


def test_case_differs_between_a_manifest_and_a_hand_typed_table(index) -> None:
    folded, declared = index.resolve("qwen/qwen3.8-27b-fp8"), index.resolve("Qwen/Qwen3.8-27B-FP8")

    assert folded is not None and declared is not None
    assert folded.model_name == declared.model_name
    assert folded.generation_id == declared.generation_id


def test_a_quant_suffix_names_a_different_artifact_and_is_not_trimmed(index) -> None:
    """Trimming toward a match would resolve a row to a model that never ran."""
    assert index.resolve("Qwen/Qwen3.8-27B") is not None
    assert index.resolve("Qwen/Qwen3.8-27B-AWQ") is None


def test_a_model_the_register_does_not_carry_resolves_to_nothing(index) -> None:
    assert index.resolve("llama3.2:3b") is None
    assert index.resolve("") is None
    assert len(index) == len(index.known())
