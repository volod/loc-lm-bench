"""Reading a generation out of an upstream artifact name -- including the names that are traps."""

from llb.backends.currency.generations import (
    compile_pattern,
    generation_key,
    read_generation,
    read_generations,
)


def _read(name: str, namespace: str, override: str | None = None) -> str | None:
    pattern = compile_pattern(namespace, override)
    assert pattern is not None
    found = read_generation(name, pattern)
    return found.id if found else None


def test_generation_key_orders_generations_numerically() -> None:
    assert generation_key("v2.0") == (2, 0)
    assert generation_key("3.8") > generation_key("3.6") > generation_key("3")
    assert generation_key("3.10") > generation_key("3.8")
    assert generation_key("") is None and generation_key("instruct") is None


def test_version_after_the_namespace_is_the_generation() -> None:
    assert _read("qwen3.8", "qwen") == "3.8"
    assert _read("qwen3.8-flash-next", "qwen") == "3.8"
    assert _read("Qwen3.8-27B-FP8", "Qwen") == "3.8"
    assert _read("gemma-4-E4B-it-qat-w4a16-ct", "gemma") == "4"
    assert _read("gemma3n", "gemma") == "3"
    assert _read("mistral-small3.2", "mistral-small") == "3.2"


def test_a_parameter_count_is_not_a_generation() -> None:
    # `gemma-7b` is Gemma 1 at 7B and `Mistral-Small-24B-Instruct-2501` is Mistral Small 1 at 24B.
    # Reading the size as a generation would report both families as decades behind.
    assert _read("gemma-7b-keras", "gemma") is None
    assert _read("gemma-2b-it", "gemma") is None
    assert _read("Mistral-Small-24B-Instruct-2501", "Mistral-Small") is None
    assert _read("Mistral-Small-3.1-24B-Instruct-2503", "Mistral-Small") == "3.1"


def test_a_name_outside_the_namespace_carries_no_generation() -> None:
    assert _read("gemma-scope-2b-pt-res", "gemma") is None
    assert _read("qwen", "qwen") is None
    assert _read("Devstral-Small-2-24B-Instruct-2512", "Mistral-Small") is None


def test_a_declared_pattern_wins_over_the_namespace_default() -> None:
    trailing = r"-v(\d+(?:\.\d+)*)"
    # The generation trails the Gemma 3 ARCHITECTURE here, so the namespace default reads nothing.
    assert _read("MamayLM-Gemma-3-27B-IT-v2.0", "MamayLM") is None
    assert _read("MamayLM-Gemma-3-27B-IT-v2.0", "MamayLM", trailing) == "2.0"
    assert _read("lapa-v0.1.3-instruct", "lapa", trailing) == "0.1.3"
    assert _read("lapa-12b-pt", "lapa", trailing) is None


def test_readings_are_newest_first_and_cite_the_plainest_artifact() -> None:
    pattern = compile_pattern("qwen")
    names = ("qwen2.5", "qwen3.8-flash-next", "qwen3", "qwen3.8", "not-a-qwen")

    found = read_generations(names, pattern)

    assert [gen.id for gen in found] == ["3.8", "3.8", "3", "2.5"]
    assert found[0].evidence == "qwen3.8"
    assert read_generations(names, None) == ()
