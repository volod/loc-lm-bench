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
) -> ScreenReport:
    """Launch or reuse a backend endpoint, run the Tier-1 screen, return the report."""
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

        launcher = VllmLauncher(
            model,
            host=cfg.vllm_host,
            port=cfg.vllm_port,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            cpu_offload_gb=cfg.cpu_offload_gb,
            kv_offloading_size_gb=cfg.kv_offloading_size_gb,
        )
        with launcher:
            return do_screen(f"{cfg.vllm_host.rstrip('/')}/v1")
    raise UnsupportedScreenBackend(f"backend {backend!r} is not supported for the public screen")


__all__ = ["UnsupportedScreenBackend", "screen_with_backend"]
