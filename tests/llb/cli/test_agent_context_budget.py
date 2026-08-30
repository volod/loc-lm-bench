"""The agent lanes' prompt-guard budget: what the CLI wrapper probes, and what binds it."""

import pytest

from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_FIXED,
    BUDGET_SOURCE_SERVED,
)
from llb.cli.bench._agent_context import agent_probe_host, resolve_agent_context_budget
from llb.core.config import RunConfig

OLLAMA_HOST = "http://localhost:11434"


class _FakeOllama:
    """An Ollama daemon that reports nothing until a warm request loads the model."""

    built: list["_FakeOllama"] = []

    def __init__(self, model, host=OLLAMA_HOST, pull=False, num_ctx=None, seed=None):
        self.model = model
        self.host = host
        self.num_ctx = num_ctx
        self.warmed = 0
        self.warm_timeout: float | None = None
        self.stopped = 0
        self.after_warm: int | None = None
        _FakeOllama.built.append(self)

    def start(self) -> None:
        pass

    def served_context(self) -> int | None:
        return None

    def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
        self.warmed += 1
        self.warm_timeout = timeout
        return self.after_warm

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def fake_ollama(monkeypatch):
    """Serve `served` on warm-up, and keep the declared side off real hardware."""
    _FakeOllama.built.clear()

    def install(served: int | None, *, start_error: Exception | None = None):
        class Launcher(_FakeOllama):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.after_warm = served

            def start(self) -> None:
                if start_error is not None:
                    raise start_error

        monkeypatch.setattr("llb.backends.ollama.OllamaLauncher", Launcher)
        return _FakeOllama.built

    monkeypatch.setattr("llb.backends.hardware.detect_gpus", lambda: [])
    monkeypatch.setattr("llb.backends.hardware.detect_ram_mb", lambda: 0)
    return install


def _cfg(**overrides) -> RunConfig:
    fields = {"model": "unlisted-model", "backend": "ollama", "ollama_host": OLLAMA_HOST}
    return RunConfig().with_overrides(**{**fields, **overrides})


def test_unpinned_ollama_run_is_warmed_so_the_served_window_can_bind(fake_ollama):
    """The regression this exists for: without a warm, `/api/ps` reports nothing at all on an
    unpinned run, so the guard resolves from the declared window alone."""
    built = fake_ollama(4096)
    cfg = _cfg()
    assert cfg.max_model_len is None and cfg.context_budget is None

    budget = resolve_agent_context_budget(cfg, base_url=None, max_prompt_chars=None)

    assert [launcher.warmed for launcher in built] == [1]
    assert built[0].num_ctx is None
    assert built[0].stopped == 1
    assert built[0].warm_timeout == cfg.request_timeout_s
    assert budget.budget_source == BUDGET_SOURCE_SERVED
    assert budget.served_max_model_len == 4096
    assert budget.bound_max_model_len == 4096
    assert budget.bounded


def test_a_smaller_served_window_binds_with_no_max_model_len_passed(fake_ollama):
    fake_ollama(4096)
    budget = resolve_agent_context_budget(
        _cfg(context_budget=32768), base_url=None, max_prompt_chars=None
    )
    assert budget.budget_source == BUDGET_SOURCE_SERVED
    assert budget.declared_max_model_len == 32768
    assert budget.served_max_model_len == 4096


def test_a_smaller_declared_window_still_binds_and_records_the_probe(fake_ollama):
    fake_ollama(8192)
    budget = resolve_agent_context_budget(
        _cfg(context_budget=2048), base_url=None, max_prompt_chars=None
    )
    assert budget.budget_source == BUDGET_SOURCE_DECLARED
    assert budget.bound_max_model_len == 2048
    assert budget.served_max_model_len == 8192


def test_an_unreachable_backend_falls_back_to_declared_instead_of_raising(fake_ollama):
    """A probe is telemetry about a window; the run's own launcher is the reachability gate."""
    fake_ollama(4096, start_error=RuntimeError("Ollama not reachable"))
    budget = resolve_agent_context_budget(
        _cfg(max_model_len=8192), base_url=None, max_prompt_chars=None
    )
    assert budget.budget_source == BUDGET_SOURCE_DECLARED
    assert budget.declared_max_model_len == 8192
    assert budget.served_max_model_len is None


def test_a_probe_that_finds_nothing_resident_falls_back_to_declared(fake_ollama):
    fake_ollama(None)
    budget = resolve_agent_context_budget(
        _cfg(context_budget=8192), base_url=None, max_prompt_chars=None
    )
    assert budget.budget_source == BUDGET_SOURCE_DECLARED
    assert budget.served_max_model_len is None


def test_an_explicit_prompt_char_budget_skips_the_probe_entirely(fake_ollama):
    built = fake_ollama(4096)
    budget = resolve_agent_context_budget(_cfg(), base_url=None, max_prompt_chars=1234)
    assert budget.budget_source == BUDGET_SOURCE_FIXED
    assert budget.max_prompt_chars == 1234
    assert built == []


def test_the_probe_uses_the_native_root_of_an_openai_compatible_ollama_url(fake_ollama):
    """`--base-url .../v1` and an OpenAI-compat `ollama_host` both name Ollama's native API one
    path segment up; the warm and the probe have to land there, not on `/v1/api/ps`."""
    built = fake_ollama(4096)
    cfg = _cfg(ollama_host="http://localhost:11434/v1")
    resolve_agent_context_budget(cfg, base_url="http://localhost:11434/v1", max_prompt_chars=None)
    assert built[0].host == OLLAMA_HOST


def test_agent_probe_host_overrides_only_for_an_ollama_url():
    cfg = _cfg()
    assert agent_probe_host(cfg, None) is None
    assert agent_probe_host(cfg, "http://localhost:11434/v1") == OLLAMA_HOST
    assert agent_probe_host(cfg, "http://elsewhere:8000/v1") is None
    assert agent_probe_host(cfg.with_overrides(backend="vllm"), "http://h:11434") is None
