"""The `complete` a benchmark category calls, over whichever endpoint the run chose.

Three shapes of adapter, one driver. `local_*` talks to an OpenAI-compatible endpoint somebody
else is running; `launcher_*` talks to a backend launcher this process started; `complete_all`
drives a sequence of prompts with a progress heartbeat. `drive_with_backend` picks between the
endpoint paths and, for a VRAM-owning backend, runs the whole workload under the shared isolation
contract. A `ThroughputMeter` threaded through any of them accumulates real generation rate.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.served_window import is_ollama_base_url as _is_ollama_base_url
from llb.core.config import RunConfig
from llb.core.contracts.common import ChatMessage
from llb.bench.common import LLMChat, LLMComplete, _R


@dataclass
class ThroughputMeter:
    """Accumulates REAL generation throughput across a category run's model calls.

    Each completed call contributes its `completion_tokens` and `latency_s` (both already reported
    by the backend `ChatResult`); `tokens_per_s` is the aggregate tokens/second over all successful
    calls. Errored/empty calls are skipped so a timeout does not deflate the rate. The first call
    carries the model cold-load, so the aggregate is a conservative steady-state estimate.
    """

    completion_tokens: int = 0
    generation_s: float = 0.0
    calls: int = 0

    def record(self, result: ChatResult) -> None:
        if result.error or result.completion_tokens <= 0 or result.latency_s <= 0:
            return
        self.completion_tokens += result.completion_tokens
        self.generation_s += result.latency_s
        self.calls += 1

    @property
    def tokens_per_s(self) -> float:
        return (
            round(self.completion_tokens / self.generation_s, 2) if self.generation_s > 0 else 0.0
        )


def local_chat(
    model: str,
    base_url: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    meter: ThroughputMeter | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
) -> LLMChat:
    """A typed-message completion over an already-running OpenAI-compatible endpoint. Heavy imports
    stay lazy; transport errors map to an empty string via `chat_once`'s normalized result. When a
    `meter` is given, each call's token count + latency is recorded for throughput reporting.

    `num_ctx` is forwarded as Ollama `options.num_ctx` via `extra_body` so a declared window is
    actually served instead of Ollama's 4096 default.
    """
    from llb.backends.openai_client import chat_once, make_client

    client = make_client(base_url)
    options: dict[str, object] = {}
    if num_ctx is not None and num_ctx > 0:
        options["num_ctx"] = num_ctx
    if seed is not None:
        options["seed"] = seed
    extra_body = {"options": options} if options else None

    def chat(messages: list[ChatMessage]) -> str:
        result = chat_once(
            client,
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
        )
        if meter is not None:
            meter.record(result)
        return result.text

    return chat


def local_complete(
    model: str,
    base_url: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    meter: ThroughputMeter | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
) -> LLMComplete:
    """A string-prompt adapter over :func:`local_chat`."""
    chat = local_chat(
        model,
        base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        meter=meter,
        num_ctx=num_ctx,
        seed=seed,
    )

    def complete(prompt: str) -> str:
        return chat([{"role": "user", "content": prompt}])

    return complete


def launcher_chat(
    launcher: BackendLauncher,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    meter: ThroughputMeter | None = None,
) -> LLMChat:
    """A typed-message completion over an already-started backend launcher."""

    def chat(messages: list[ChatMessage]) -> str:
        result = launcher.chat(messages, max_tokens, temperature, timeout)
        if meter is not None:
            meter.record(result)
        return result.text

    return chat


def launcher_complete(
    launcher: BackendLauncher,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    meter: ThroughputMeter | None = None,
) -> LLMComplete:
    """A string-prompt adapter over :func:`launcher_chat`."""
    chat = launcher_chat(
        launcher,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        meter=meter,
    )

    def complete(prompt: str) -> str:
        return chat([{"role": "user", "content": prompt}])

    return complete


def complete_all(
    complete: LLMComplete,
    prompts: Sequence[str],
    *,
    label: str,
    logger: logging.Logger,
) -> list[str]:
    """Run `complete` over `prompts` in order, logging a per-item heartbeat so the CLI isn't silent.

    A category run drives one (often slow, local) model call per case; without progress output the
    whole run looks hung while the model streams. This logs a `[label] i/n` line BEFORE each call
    (so the in-flight item is visible) and the elapsed time AFTER, and returns the outputs in order.
    """
    total = len(prompts)
    outputs: list[str] = []
    for i, prompt in enumerate(prompts, start=1):
        logger.info("[%s] prompting model %d/%d ...", label, i, total)
        started = time.monotonic()
        outputs.append(complete(prompt))
        logger.info("[%s] case %d/%d done (%.1fs)", label, i, total, time.monotonic() - started)
    return outputs


def _run_under(
    launcher: BackendLauncher,
    run: Callable[[LLMComplete], _R],
    cfg: RunConfig,
    *,
    max_tokens: int,
    meter: ThroughputMeter | None,
) -> _R:
    """Run the whole workload inside one launcher's lifetime -- the shape every path below shares."""
    with launcher:
        return run(
            launcher_complete(
                launcher,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
                meter=meter,
            )
        )


def drive_with_backend(
    cfg: RunConfig,
    run: Callable[[LLMComplete], _R],
    *,
    base_url: str | None = None,
    max_tokens: int = 512,
    vram_reader: Callable[[], int] | None = None,
    pid_usage_reader: Callable[[], dict[int, int]] | None = None,
    meter: ThroughputMeter | None = None,
) -> _R:
    """Build the candidate's `complete` for the chosen endpoint and execute `run(complete)`.

    A running endpoint (`base_url`) or Ollama is called directly; a VRAM-owning backend
    (vllm / llamacpp) is launched and the whole `run` executes under the shared `isolate_cell`
    contract (PID-attributed VRAM-reclaim gate + capped thermal cooldown), so every category
    honors the SAME isolation contract as the RAG sweep. When a `meter` is given it accumulates
    real generation throughput across the run's model calls (either endpoint path).
    """
    is_ollama = cfg.backend == "ollama"
    num_ctx = (cfg.max_model_len or cfg.context_budget) if is_ollama else None
    seed = cfg.seed if is_ollama else None
    if base_url is not None:
        # When the caller supplied --base-url pointing at an Ollama /v1 endpoint AND a num_ctx is
        # declared, route through the native launcher on that same host so num_ctx is reliably
        # honoured. Ollama's OpenAI-compat layer silently ignores extra_body.options.num_ctx on
        # some builds.
        if num_ctx and _is_ollama_base_url(base_url, cfg.ollama_host):
            from llb.backends.ollama import OllamaLauncher
            from llb.backends.served_window import native_root

            host = native_root(base_url)
            launcher = OllamaLauncher(cfg.model, host=host, num_ctx=num_ctx, seed=seed)
            return _run_under(launcher, run, cfg, max_tokens=max_tokens, meter=meter)
        return run(
            local_complete(
                cfg.model,
                base_url,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
                meter=meter,
                num_ctx=num_ctx,
                seed=seed,
            )
        )
    if is_ollama:
        # Native /api/chat (not OpenAI /v1): only the native options payload reliably honors
        # num_ctx, which is what stops Ollama's silent 4096 default from truncating prompts.
        from llb.backends.ollama import OllamaLauncher

        started = OllamaLauncher(cfg.model, host=cfg.ollama_host, num_ctx=num_ctx, seed=cfg.seed)
        return _run_under(started, run, cfg, max_tokens=max_tokens, meter=meter)

    # A VRAM-owning backend (vllm / llamacpp) we start ourselves, under the isolation contract.
    from llb.executor.isolation import isolate_cell
    from llb.executor.runner_backend import _make_launcher

    owned = _make_launcher(cfg, log_dir=cfg.data_dir / "llb" / "logs")
    result, _outcome = isolate_cell(
        lambda: _run_under(owned, run, cfg, max_tokens=max_tokens, meter=meter),
        backend=cfg.backend,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
    )
    return result
