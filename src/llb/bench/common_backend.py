"""Focused common backend implementation."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from llb.backends.base import BackendLauncher, ChatResult
from llb.backends.served_window import is_ollama_base_url as _is_ollama_base_url
from llb.core.config import RunConfig
from llb.core.contracts.common import ChatMessage
from llb.bench.common import LLMComplete, _R


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
    """A `complete` over an already-running OpenAI-compatible endpoint (no launch). Heavy imports
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

    def complete(prompt: str) -> str:
        msgs: list[ChatMessage] = [{"role": "user", "content": prompt}]
        result = chat_once(
            client,
            model,
            msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
        )
        if meter is not None:
            meter.record(result)
        return result.text

    return complete


def launcher_complete(
    launcher: BackendLauncher,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    meter: ThroughputMeter | None = None,
) -> LLMComplete:
    """A `complete` over an already-started `BackendLauncher` (its OpenAI-compatible chat)."""

    def complete(prompt: str) -> str:
        msgs: list[ChatMessage] = [{"role": "user", "content": prompt}]
        result = launcher.chat(msgs, max_tokens, temperature, timeout)
        if meter is not None:
            meter.record(result)
        return result.text

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
    ollama_num_ctx = (cfg.max_model_len or cfg.context_budget) if cfg.backend == "ollama" else None
    if base_url is not None:
        # When the caller supplied --base-url pointing at an Ollama /v1 endpoint AND a
        # num_ctx is declared, route through the native launcher on that same host so
        # num_ctx is reliably honoured.  Ollama's OpenAI-compat layer silently ignores
        # extra_body.options.num_ctx on some builds.
        if ollama_num_ctx and _is_ollama_base_url(base_url, cfg.ollama_host):
            from llb.backends.ollama import OllamaLauncher
            from llb.backends.served_window import native_root

            launcher = OllamaLauncher(
                cfg.model,
                host=native_root(base_url),
                num_ctx=ollama_num_ctx,
                seed=cfg.seed if cfg.backend == "ollama" else None,
            )
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
        return run(
            local_complete(
                cfg.model,
                base_url,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
                meter=meter,
                num_ctx=ollama_num_ctx,
                seed=cfg.seed if cfg.backend == "ollama" else None,
            )
        )
    if cfg.backend == "ollama":
        # Native /api/chat (not OpenAI /v1): only the native options payload reliably honors
        # num_ctx, which is what stops Ollama's silent 4096 default from truncating prompts.
        from llb.backends.ollama import OllamaLauncher

        launcher = OllamaLauncher(
            cfg.model,
            host=cfg.ollama_host,
            num_ctx=ollama_num_ctx,
            seed=cfg.seed,
        )
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

    from llb.executor.isolation import isolate_cell
    from llb.executor.runner_backend import _make_launcher

    owned = _make_launcher(cfg, log_dir=cfg.data_dir / "llb" / "logs")

    def work() -> _R:
        with owned:
            return run(
                launcher_complete(
                    owned,
                    max_tokens=max_tokens,
                    temperature=cfg.temperature,
                    timeout=cfg.request_timeout_s,
                    meter=meter,
                )
            )

    result, _outcome = isolate_cell(
        work,
        backend=cfg.backend,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
    )
    return result
