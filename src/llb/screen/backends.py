"""Run the Tier-1 public screen against whichever endpoint a backend needs.

`run_screen` drives lm-eval against an OpenAI-compatible URL it is HANDED; getting that URL is a
per-backend concern -- Ollama already serves one, vLLM has to be launched and torn down around the
screen. That step lived in the CLI, which made it unavailable to any non-CLI caller that needs a
public screen (the roster confirmation run scores its finalists on the public tracks before the
adoption decision). It lives here so both reach the same launcher, and neither can drift into a
second notion of how a screen gets its endpoint.
"""

from collections.abc import Sequence
from pathlib import Path

from llb.core.config import RunConfig
from llb.core.contracts.screening import ScreenReport


class UnsupportedScreenBackend(ValueError):
    """Raised for a backend the public screen has no endpoint path for."""


def screen_with_backend(
    model: str,
    backend: str,
    cfg: RunConfig,
    *,
    base_url: str | None = None,
    extra_tasks: Sequence[str] = (),
    out_dir: Path,
    limit: int | None = None,
    evict: bool = False,
    wait: bool = False,
) -> ScreenReport:
    """Launch or reuse a backend endpoint, run the Tier-1 screen, return the report.

    `evict` / `wait` are the guard's two non-default ways out of a contended card, and they carry
    the same meaning as on `run-eval`: unload Ollama's resident models, or poll until the VRAM
    frees. Both stay opt-in -- the guard never frees another process's memory on its own.
    """
    from llb.screen.public import run_screen

    def do_screen(url: str) -> ScreenReport:
        return run_screen(
            model, backend, url, extra_tasks=list(extra_tasks), output_dir=out_dir, limit=limit
        )

    if base_url:
        return do_screen(base_url)
    if backend == "ollama":
        return do_screen(f"{cfg.ollama_host.rstrip('/')}/v1")
    if backend == "vllm":
        from llb.backends.vllm import VllmLauncher
        from llb.screen.public_report import safe_model_name

        # A screen that cannot launch its endpoint is recorded as a failure and QUALIFIES a verdict,
        # so it owes the operator a reason. Without a log directory vLLM's stdout goes to DEVNULL
        # and all that survives is "vLLM exited (code 1)" -- which never says whether the weights
        # were missing, the context did not fit, or another process still held the card.
        launcher = VllmLauncher(
            model,
            host=cfg.vllm_host,
            port=cfg.vllm_port,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            cpu_offload_gb=cfg.cpu_offload_gb,
            kv_offloading_size_gb=cfg.kv_offloading_size_gb,
            dtype=cfg.dtype,
            quantization=cfg.quantization,
            suppress_thinking=cfg.vllm_suppress_thinking,
            log_dir=out_dir / safe_model_name(model) / "backend",
        )
        # The screen shares the card with whatever ran before it. vLLM refuses to start unless
        # `gpu-memory-utilization x total` is actually FREE, and an Ollama finalist screened one
        # step earlier is still resident by design (keep-alive), so without the same pre-launch
        # guard `run-eval` uses, a mixed-backend screen dies on the second finalist.
        from llb.executor.runner_backend import guard_vllm_contention

        guard_vllm_contention(cfg, launcher, evict=evict, wait=wait, label="screen-public")
        with launcher:
            return do_screen(f"{cfg.vllm_host.rstrip('/')}/v1")
    raise UnsupportedScreenBackend(f"backend {backend!r} is not supported for the public screen")


__all__ = ["UnsupportedScreenBackend", "screen_with_backend"]
