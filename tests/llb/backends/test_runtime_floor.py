"""Runtime version floors: a source the installed runtime cannot serve is a NAMED skip.

Every probe is faked here -- no daemon, no network -- so the whole ladder is exercised: the
manifest pin, the artifact's own `requires`, the merge between them, and what each path prints.
"""

import pytest

from llb.backends.runtime_floor import (
    RUNTIME_FLOOR_SKIP,
    RuntimeRequirement,
    architecture_error,
    declared_requirement,
    floor_reason,
    parse_version,
    source_floor_reason,
    unsupported_architecture,
)

GEMMA_12B = {
    "source": "gemma4:12b",
    "runtime_arch": "gemma4",
    "min_runtime_version": "0.30.5",
}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ollama version is 0.32.15", (0, 32, 15)),
        ("v0.30.5", (0, 30, 5)),
        ("0.33.1-rc1", (0, 33, 1)),
        ("unversioned", None),
        (None, None),
    ],
)
def test_parse_version_reads_the_numeric_part(text, expected):
    assert parse_version(text) == expected


def test_unsupported_architecture_names_what_the_runtime_refused():
    assert unsupported_architecture("unknown model architecture: 'gemma4'") == "gemma4"
    assert unsupported_architecture("Model architectures ['Gemma4ForCausalLM'] are not supported")
    assert unsupported_architecture("500 Internal Server Error") is None
    assert unsupported_architecture("") is None


def test_floor_reason_names_runtime_version_and_architecture():
    requirement = declared_requirement("ollama", "gemma4:12b", GEMMA_12B)
    reason = floor_reason(requirement, "0.20.6")
    assert reason is not None
    assert "ollama 0.20.6" in reason
    assert "'gemma4'" in reason
    assert "gemma4:12b" in reason
    assert "ollama >= 0.30.5" in reason


def test_floor_reason_silent_when_the_runtime_is_new_enough():
    requirement = declared_requirement("ollama", "gemma4:12b", GEMMA_12B)
    assert floor_reason(requirement, "0.32.15") is None
    assert floor_reason(requirement, "0.30.5") is None


def test_an_unparseable_version_never_grounds_a_model():
    requirement = declared_requirement("ollama", "gemma4:12b", GEMMA_12B)
    assert floor_reason(requirement, None) is None
    assert floor_reason(requirement, "unknown") is None
    assert floor_reason(RuntimeRequirement("ollama", "x"), "0.1.0") is None


def test_the_artifacts_own_requirement_beats_a_lower_pin():
    """The manifest pin can only raise a floor: the runtime is the authority on its own artifact."""
    merged = declared_requirement("ollama", "gemma4:12b", {"min_runtime_version": "0.20.0"}).merge(
        RuntimeRequirement("ollama", "gemma4:12b", arch="gemma4", min_version="0.30.5")
    )
    assert merged.min_version == "0.30.5"
    assert merged.arch == "gemma4"


def test_a_pin_covers_an_artifact_that_declares_nothing():
    """A raw GGUF carries no `requires`, which is exactly when the manifest pin is the only signal."""
    reason = source_floor_reason(
        "ollama",
        "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0",
        GEMMA_12B,
        version_reader=lambda _b: "0.20.6",
        requirement_reader=lambda _b, _s: None,
    )
    assert reason is not None and "ollama >= 0.30.5" in reason


def test_an_unreadable_runtime_costs_no_artifact_probe():
    probed: list[str] = []

    reason = source_floor_reason(
        "ollama",
        "gemma4:12b",
        GEMMA_12B,
        version_reader=lambda _b: None,
        requirement_reader=lambda _b, s: probed.append(s),  # type: ignore[func-returns-value]
    )
    assert reason is None and probed == []


def test_architecture_error_names_the_runtime_and_the_artifact():
    message = architecture_error("ollama", "gemma4:12b", "gemma4")
    assert "ollama" in message and "gemma4:12b" in message and "'gemma4'" in message


def test_the_skip_category_is_stable():
    assert RUNTIME_FLOOR_SKIP == "runtime-version-floor"
