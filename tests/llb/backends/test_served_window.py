"""Unit tests for live served-window probing and min(declared, served) budget binding."""

import json

from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_SERVED,
    BUDGET_SOURCE_UNBOUNDED,
    bind_window,
    is_ollama_base_url,
    launcher_served_window,
    parse_ollama_served_context,
    probe_served_max_model_len,
)
from llb.backends.context_budget import resolve_context_budget
from llb.core.config import RunConfig
from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS


def test_parse_ollama_served_context_reads_loaded_model_context():
    body = json.dumps(
        {
            "models": [
                {"name": "other:latest", "context": 2048},
                {"name": "mamaylm:latest", "context_length": 4096},
            ]
        }
    )
    assert parse_ollama_served_context(body, "mamaylm:latest") == 4096
    assert parse_ollama_served_context(body, "mamaylm") == 4096
    assert parse_ollama_served_context(body, "missing") is None
    assert parse_ollama_served_context("not-json", "mamaylm") is None


def test_bind_window_names_which_side_bound():
    assert bind_window(32768, 4096) == (4096, BUDGET_SOURCE_SERVED)
    assert bind_window(4096, 32768) == (4096, BUDGET_SOURCE_DECLARED)
    assert bind_window(8192, None) == (8192, BUDGET_SOURCE_DECLARED)
    assert bind_window(0, 4096) == (4096, BUDGET_SOURCE_SERVED)
    assert bind_window(0, None) == (0, BUDGET_SOURCE_UNBOUNDED)


def test_probe_served_max_model_len_dispatches_per_backend():
    def fake_get(url: str):
        if url.endswith("/api/ps"):
            return 200, json.dumps({"models": [{"name": "m:latest", "context": 4096}]})
        if url.endswith("/v1/models"):
            return 200, json.dumps({"data": [{"max_model_len": 8192}]})
        if url.endswith("/props"):
            return 200, json.dumps({"n_ctx": 2048})
        return None

    assert (
        probe_served_max_model_len("ollama", model="m", host="http://h", http_get=fake_get) == 4096
    )
    assert probe_served_max_model_len("vllm", model="m", host="http://h", http_get=fake_get) == 8192
    assert (
        probe_served_max_model_len("llamacpp", model="m", host="http://h", http_get=fake_get)
        == 2048
    )
    assert (
        probe_served_max_model_len("ollama", model="m", host="http://h", http_get=lambda _u: None)
        is None
    )


def test_resolve_context_budget_is_bound_by_a_smaller_served_window():
    config = RunConfig().with_overrides(model="unlisted-model", context_budget=32768)
    budget = resolve_context_budget(
        config, model_spec=None, vram_mib=0, ram_mib=0, served_max_model_len=4096
    )
    usable = 4096 - PROMPT_HEADROOM_TOKENS - config.max_tokens
    assert budget.budget_source == BUDGET_SOURCE_SERVED
    assert budget.served_max_model_len == 4096
    assert budget.declared_max_model_len == 32768
    assert budget.max_prompt_chars == int(usable * CHARS_PER_TOKEN)
    assert budget.fits(budget.max_prompt_chars) is True
    assert budget.fits(budget.max_prompt_chars + int(2 * CHARS_PER_TOKEN)) is False


def test_resolve_context_budget_falls_back_to_declared_when_probe_misses():
    config = RunConfig().with_overrides(model="unlisted-model", context_budget=8192)

    def miss(_url: str):
        return None

    budget = resolve_context_budget(
        config, model_spec=None, vram_mib=0, ram_mib=0, probe=True, http_get=miss
    )
    usable = 8192 - PROMPT_HEADROOM_TOKENS - config.max_tokens
    assert budget.budget_source == BUDGET_SOURCE_DECLARED
    assert budget.served_max_model_len is None
    assert budget.declared_max_model_len == 8192
    assert budget.max_prompt_chars == int(usable * CHARS_PER_TOKEN)
    assert budget.provenance()["budget_source"] == BUDGET_SOURCE_DECLARED


def test_is_ollama_base_url_matches_same_host():
    assert is_ollama_base_url("http://localhost:11434/v1", "http://localhost:11434") is True
    assert is_ollama_base_url("http://localhost:11434", "http://localhost:11434") is True
    assert is_ollama_base_url("http://localhost:11434/v1/", "http://localhost:11434/") is True
    assert is_ollama_base_url("http://other:11434/v1", "http://localhost:11434") is False
    assert is_ollama_base_url("http://localhost:8000/v1", "http://localhost:11434") is False


def test_drive_with_backend_routes_ollama_base_url_through_native_launcher(monkeypatch):
    """When backend=ollama, base_url points at the same host, and num_ctx is set,
    drive_with_backend must use OllamaLauncher (native /api/chat) not local_complete,
    so num_ctx is reliably honoured."""
    from llb.backends.base import ChatResult
    from llb.bench.common_backend import drive_with_backend
    from llb.core.config import RunConfig

    native_calls: list[dict] = []

    class _FakeLauncher:
        def __init__(self, model, host, num_ctx=None, seed=None):
            self.model = model
            self.host = host
            self.num_ctx = num_ctx
            self.seed = seed
            self._last = None
            self.meta: dict = {}

        def start(self):
            pass

        def stop(self):
            pass

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *exc):
            self.stop()

        def chat(self, messages, max_tokens, temperature, timeout):
            native_calls.append(
                {
                    "num_ctx": self.num_ctx,
                    "max_tokens": max_tokens,
                    "seed": self.seed,
                    "temperature": temperature,
                }
            )
            return ChatResult(text="native-ok")

        def served_context(self):
            return self.num_ctx

        def telemetry(self):
            return {}

    # OllamaLauncher is imported lazily inside the function; patch the module it comes from
    # so the lazy `from llb.backends.ollama import OllamaLauncher` sees _FakeLauncher.
    import llb.backends.ollama as _ollama_mod

    monkeypatch.setattr(_ollama_mod, "OllamaLauncher", _FakeLauncher)
    # Force the host-detection helper to always return True so no real network call is needed.
    import llb.bench.common_backend as _cb

    monkeypatch.setattr(_cb, "_is_ollama_base_url", lambda url, host: True)

    cfg = RunConfig().with_overrides(
        model="m", backend="ollama", max_model_len=8192, ollama_host="http://localhost:11434"
    )
    result = drive_with_backend(
        cfg,
        lambda complete: complete("ping"),
        base_url="http://localhost:11434/v1",
    )
    assert result == "native-ok"
    assert len(native_calls) == 1
    assert native_calls[0]["num_ctx"] == 8192
    assert native_calls[0]["seed"] == 13
    assert native_calls[0]["temperature"] == 0.0


def test_ollama_launcher_sends_num_ctx_in_options(monkeypatch):
    from llb.backends.ollama import OllamaLauncher

    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self):
            return json.dumps(
                {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}
            ).encode()

    def fake_urlopen(request, timeout=0):  # noqa: ARG001
        captured["url"] = getattr(request, "full_url", None) or request.get_full_url()
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("llb.backends.ollama.urllib.request.urlopen", fake_urlopen)
    launcher = OllamaLauncher("m", num_ctx=8192)
    result = launcher.chat(
        [{"role": "user", "content": "hi"}], max_tokens=16, temperature=0.0, timeout=5
    )
    assert result.text == "ok"
    assert captured["payload"]["options"]["num_ctx"] == 8192
    assert captured["payload"]["options"]["num_predict"] == 16


class _Ollama:
    """Ollama's shape: `/api/ps` reports nothing until a request has loaded the model."""

    def __init__(self, after_warm: int | None = 4096):
        self._served: int | None = None
        self._after_warm = after_warm
        self.warmed = 0

    def served_context(self) -> int | None:
        return self._served

    def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
        self.warmed += 1
        self._served = self._after_warm
        return self._served


def test_launcher_served_window_takes_a_window_the_launcher_already_knows():
    class Ready:
        def served_context(self) -> int:
            return 8192

    launcher = Ready()
    assert launcher_served_window(launcher) == 8192


def test_launcher_served_window_warms_a_backend_that_reports_nothing_yet():
    """The unpinned-Ollama case: without the warm request the probe reads "unknown" exactly when
    the 4096 default is about to truncate."""
    launcher = _Ollama(after_warm=4096)
    assert launcher_served_window(launcher) == 4096
    assert launcher.warmed == 1


def test_launcher_served_window_reports_none_when_the_warm_request_fails():
    """A warm request is best-effort telemetry; a failing one falls back to the declared window."""

    class Failing(_Ollama):
        def ensure_num_ctx(self, timeout: float = 120.0) -> int | None:
            raise RuntimeError("backend down")

    assert launcher_served_window(Failing()) is None
    assert launcher_served_window(object()) is None
