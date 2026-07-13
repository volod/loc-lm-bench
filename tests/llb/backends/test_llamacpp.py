"""llama.cpp launcher (llama.cpp launcher): command building, readiness, telemetry, resolver routing, and
the reclaim gate -- all driven by fakes (no llama.cpp / GPU / subprocess)."""

import pytest

from llb.backends.base import ChatResult
from llb.backends.llamacpp import (
    LlamaCppLauncher,
    build_llamacpp_command,
    llamacpp_source_args,
    parse_served_context,
    resolve_llama_server_binary,
)
from llb.core.contracts import ModelSpec


def test_source_args_maps_repo_path_and_ollama_style():
    # A local GGUF file loads with -m; an HF repo (incl. the Ollama-style hf.co/...:quant the
    # resolver carries) loads with -hf, sharing one source string across both GGUF backends.
    assert llamacpp_source_args("/models/m.gguf") == ["-m", "/models/m.gguf"]
    assert llamacpp_source_args("org/Repo-GGUF:Q4_K_M") == ["-hf", "org/Repo-GGUF:Q4_K_M"]
    assert llamacpp_source_args("hf.co/org/Repo-GGUF:Q4_K_M") == ["-hf", "org/Repo-GGUF:Q4_K_M"]


def test_build_command_includes_serving_flags():
    cmd = build_llamacpp_command(
        "org/Repo-GGUF:Q4_K_M", bind_host="127.0.0.1", port=8081, n_gpu_layers=20, ctx_size=4096
    )
    assert cmd[0] == "llama-server"
    assert cmd[1:3] == ["-hf", "org/Repo-GGUF:Q4_K_M"]
    assert "--port" in cmd and "8081" in cmd
    assert "-ngl" in cmd and "20" in cmd  # the GPU/CPU offload split
    assert "-c" in cmd and "4096" in cmd  # served context


def test_resolve_binary_prefers_data_dir_build(tmp_path):
    binary = tmp_path / "llb" / "llamacpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    assert resolve_llama_server_binary(tmp_path) == str(binary)


def test_parse_served_context_handles_both_shapes():
    assert parse_served_context('{"default_generation_settings": {"n_ctx": 4096}}') == 4096
    assert parse_served_context('{"n_ctx": 2048}') == 2048
    assert parse_served_context("not json") is None
    assert parse_served_context("{}") is None


def test_parse_served_context_handles_newer_props_shapes():
    # llama.cpp launcher: n_ctx has moved across llama.cpp versions -- the parser checks the known locations.
    assert parse_served_context('{"default_generation_settings": {"params": {"n_ctx": 8192}}}') == (
        8192
    )
    assert parse_served_context('{"generation_settings": {"n_ctx": 1024}}') == 1024
    assert parse_served_context('{"model": {"n_ctx": 16384}}') == 16384
    assert parse_served_context('{"props": {"n_ctx": 512}}') == 512
    # the model's TRAINED context must never be mistaken for the served context
    assert parse_served_context('{"n_ctx_train": 131072}') is None
    assert parse_served_context('{"default_generation_settings": {"n_ctx": null}}') is None


def test_run_eval_gpu_layers_flag_drives_partial_offload():
    from llb.cli.helpers import load_config as _load_config

    cfg = _load_config(None, model="m.gguf", backend="llamacpp", n_gpu_layers=20)
    assert cfg.n_gpu_layers == 20  # a partial split (< all) is drivable without a YAML
    assert _load_config(None, backend="llamacpp").n_gpu_layers == -1  # default stays all-on-GPU


class FakeProc:
    """A subprocess stand-in: stays alive (poll() -> None) unless `dead`."""

    def __init__(self, dead=False):
        self._dead = dead
        self.returncode = 1 if dead else None
        self.terminated = False

    def poll(self):
        return self.returncode if self._dead else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def make_launcher(proc, responses, **kwargs):
    """A launcher whose http probe yields `responses` in order (each: None or (status, body))."""
    seq = iter(responses)
    return LlamaCppLauncher(
        "org/Repo-GGUF:Q4_K_M",
        host="http://localhost:8080",
        n_gpu_layers=20,
        ctx_size=4096,
        startup_timeout=5,
        poll_interval=0.1,
        popen=lambda cmd, **kw: proc,
        http_get=lambda url, timeout=3.0: next(seq, None),
        sleep=lambda _s: None,
        **kwargs,
    )


def test_start_becomes_ready_and_records_served_context():
    # health: not-ready then 200; then /props reports the served context.
    proc = FakeProc()
    launcher = make_launcher(
        proc, [None, (200, "ok"), (200, '{"default_generation_settings": {"n_ctx": 4096}}')]
    )
    launcher.start()
    assert launcher.served_context() == 4096
    assert launcher.load_time_s >= 0.0
    assert launcher.meta["served_context"] == 4096


def test_served_context_falls_back_to_requested_when_props_unavailable():
    launcher = make_launcher(FakeProc(), [(200, "ok"), None])  # /props unreachable
    launcher.start()
    assert launcher.served_context() == 4096  # falls back to the requested ctx_size


def test_start_raises_when_process_dies():
    launcher = make_launcher(FakeProc(dead=True), [None])
    with pytest.raises(RuntimeError, match="exited"):
        launcher.start()


def test_start_times_out_when_never_ready():
    launcher = make_launcher(FakeProc(), [None] * 100)  # /health never returns 200
    with pytest.raises(RuntimeError, match="not ready"):
        launcher.start()


class FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                import types

                msg = types.SimpleNamespace(content="привіт")
                usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=2)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)], usage=usage
                )


def test_chat_round_trip_and_telemetry_records_offload():
    launcher = make_launcher(FakeProc(), [(200, "ok"), (200, '{"n_ctx": 4096}')])
    launcher.start()
    launcher._client = FakeClient()
    result = launcher.chat([{"role": "user", "content": "hi"}], 16, 0.0, 10)
    assert isinstance(result, ChatResult) and result.text == "привіт"
    tel = launcher.telemetry()
    assert tel["backend"] == "llamacpp"
    assert tel["n_gpu_layers"] == 20 and tel["ctx_size"] == 4096
    assert tel["tokens_per_s"] >= 0.0


def test_collect_telemetry_records_n_gpu_layers_and_served_context():
    from llb.backends.telemetry import collect_telemetry

    launcher = make_launcher(FakeProc(), [(200, "ok"), (200, '{"n_ctx": 4096}')])
    launcher.start()
    launcher._client = FakeClient()
    report = collect_telemetry(launcher, requested_context=8192, vram_reader=lambda: 4242)
    assert report["backend"] == "llamacpp"
    assert report["n_gpu_layers"] == 20
    assert report["served_context"] == 4096 and report["requested_context"] == 8192
    assert report["peak_vram_mb"] == 4242


def test_make_launcher_builds_llamacpp_from_config(tmp_path):
    from llb.core.config import RunConfig
    from llb.executor.runner_backend import _make_launcher

    cfg = RunConfig(
        model="hf.co/org/Repo-GGUF:Q4_K_M",
        backend="llamacpp",
        n_gpu_layers=20,
        max_model_len=4096,
        data_dir=tmp_path,
    )
    launcher = _make_launcher(cfg)
    assert isinstance(launcher, LlamaCppLauncher)
    assert launcher.n_gpu_layers == 20 and launcher.ctx_size == 4096
    cmd = launcher.command()
    assert cmd[0] == "llama-server" and "-ngl" in cmd and "20" in cmd and "4096" in cmd


def test_resolver_routes_gguf_only_model_to_llamacpp():
    from llb.backends.resolver import ResolverProbes, resolve

    # A 70B GGUF too big for 16 GB VRAM even at q4 -> only llama.cpp (CPU offload) can serve it.
    spec: ModelSpec = {
        "name": "big-gguf",
        "backend": "llamacpp",
        "source": "hf.co/org/Big-70B-GGUF:Q4_K_M",
        "params_b": 70.0,
        "quant": "q4_k_m",
        "n_layers": 80,
        "kv_dim": 2048,
        "max_context": 8192,
    }
    probes = ResolverProbes(
        hf_repo=lambda _s: False, gguf=lambda _s: True, ollama_tag=lambda _s: False
    )
    resolved = resolve(spec, vram_mib=16000, ram_mib=128000, probes=probes)
    assert resolved["chosen_backend"] == "llamacpp"
    assert resolved["verdict"] == "offload"


def test_reclaim_gate_runs_for_llamacpp():
    # llama.cpp owns its VRAM, so it is in GATE_BACKENDS and the isolation reclaim reclaim gate applies:
    # a NEW pid still holding VRAM after the cell is a leak -> abort (same contract as vLLM).
    from llb.executor.isolation import GATE_BACKENDS, isolate_cell
    from llb.executor.vram import VramNotReclaimed

    assert "llamacpp" in GATE_BACKENDS
    reads = iter([1000] + [9000] * 100)  # baseline 1000, then stuck high
    usage = iter([{100: 500}, {100: 500, 200: 3000}])  # a NEW pid (200) still holds VRAM
    with pytest.raises(VramNotReclaimed, match="leaked"):
        isolate_cell(
            lambda: "x",
            backend="llamacpp",
            vram_reader=lambda: next(reads),
            pid_usage_reader=lambda: next(usage),
            gpu_sampler=lambda: [
                {
                    "index": 0,
                    "temp_c": 40,
                    "power_w": 120.0,
                    "sm_clock_mhz": 2100,
                    "mem_clock_mhz": 9000,
                }
            ],
            sleep=lambda _s: None,
        )
