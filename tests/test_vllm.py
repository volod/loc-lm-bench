"""vLLM launcher: command building + readiness handling, driven by fakes (no vLLM/GPU)."""

import pytest

from llb.backends.base import ChatResult
from llb.backends.vllm import (
    VllmLauncher,
    build_vllm_command,
    launch_env,
    parse_served_context,
)


def test_build_command_includes_serving_flags():
    cmd = build_vllm_command(
        "org/Model",
        port=8001,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        cpu_offload_gb=16,
        kv_offloading_size_gb=32,
        quantization="awq",
    )
    assert cmd[:3] == ["vllm", "serve", "org/Model"]
    assert "--gpu-memory-utilization" in cmd and "0.9" in cmd
    assert "8192" in cmd and "awq" in cmd
    assert "--cpu-offload-gb" in cmd and "16" in cmd
    assert "--kv-offloading-size" in cmd and "32" in cmd


def test_parse_served_context():
    body = '{"data": [{"id": "m", "max_model_len": 4096}]}'
    assert parse_served_context(body) == 4096
    assert parse_served_context("not json") is None
    assert parse_served_context('{"data": []}') is None


def test_launch_env_gates_flashinfer_sampler_on_preflight_and_respects_override():
    key = "VLLM_USE_FLASHINFER_SAMPLER"
    # Preflight verdict native -> sampler stays off (the safe default).
    assert launch_env({"PATH": "/usr/bin"}, flashinfer_sampler=False)[key] == "0"
    # Preflight confirms the kernel builds on this host -> sampler enabled.
    assert launch_env({"PATH": "/usr/bin"}, flashinfer_sampler=True)[key] == "1"
    # An explicit caller value always wins, regardless of the preflight verdict.
    assert launch_env({key: "0"}, flashinfer_sampler=True)[key] == "0"
    assert launch_env({key: "1"}, flashinfer_sampler=False)[key] == "1"


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


def make_launcher(proc, responses):
    """A launcher whose http probe yields `responses` in order (each: None or (status, body))."""
    seq = iter(responses)
    return VllmLauncher(
        "org/Model",
        startup_timeout=5,
        poll_interval=0.1,
        popen=lambda cmd, **kw: proc,
        http_get=lambda url, timeout=3.0: next(seq, None),
        sleep=lambda _s: None,
    )


def test_start_becomes_ready_and_records_load_time():
    proc = FakeProc()
    launcher = make_launcher(proc, [None, (200, '{"data": [{"id": "m", "max_model_len": 2048}]}')])
    launcher.start()
    assert launcher.served_context() == 2048
    assert launcher.load_time_s >= 0.0
    assert launcher.meta["served_context"] == 2048


def test_start_raises_when_process_dies():
    launcher = make_launcher(FakeProc(dead=True), [None])
    with pytest.raises(RuntimeError, match="exited"):
        launcher.start()


def test_start_times_out_when_never_ready():
    launcher = make_launcher(FakeProc(), [None] * 100)  # never returns 200
    with pytest.raises(RuntimeError, match="not ready"):
        launcher.start()


def test_record_sampler_reflects_launch_env():
    """vLLM serving preflight: the launcher records which sampler the launch used, for the manifest."""
    launcher = make_launcher(FakeProc(), [(200, '{"data": []}')])
    launcher._record_sampler({"VLLM_USE_FLASHINFER_SAMPLER": "1"})
    assert launcher.meta["sampler"] == "flashinfer"
    launcher._record_sampler({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    assert launcher.meta["sampler"] == "native"


def test_chat_uses_injected_client():
    proc = FakeProc()
    launcher = make_launcher(proc, [(200, '{"data": []}')])
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

    launcher._client = FakeClient()
    result = launcher.chat([{"role": "user", "content": "hi"}], 16, 0.0, 10)
    assert isinstance(result, ChatResult) and result.text == "привіт"
    assert launcher.telemetry()["tokens_per_s"] >= 0.0


def test_chat_uses_lora_module_name_for_adapter_requests():
    launcher = VllmLauncher(
        "org/Model",
        adapter_path="/tmp/adapter",
        adapter_name="adapter",
        popen=lambda cmd, **kw: FakeProc(),
        http_get=lambda url, timeout=3.0: (200, '{"data": []}'),
        sleep=lambda _s: None,
    )
    launcher.start()
    seen = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    import types

                    seen.update(kwargs)
                    msg = types.SimpleNamespace(content="ok")
                    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
                    return types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=msg)], usage=usage
                    )

    launcher._client = FakeClient()
    result = launcher.chat([{"role": "user", "content": "hi"}], 16, 0.0, 10)
    assert result.text == "ok"
    assert seen["model"] == "adapter"
