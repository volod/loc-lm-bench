"""The roster throughput protocol: one comparable decode-rate row per logical roster entry.

The full-roster baseline in the delivered docs is only readable as a table because every row was
taken the same way -- the fixed Ukrainian `telemetry.throughput` prompt set, one discarded warmup
pass, a fixed output budget, a pinned context, and a cleared GPU before each model. This module is
that protocol as code, so a row measured for a NEW roster generation is comparable to the rows
already in the table instead of being a fresh scratch script with its own knobs.

It owns the protocol constants, the per-entry measurement (launch -> `collect_telemetry` -> measured
placement -> stop), the `min/100` run-sizing derivation, and the markdown row the docs carry. What
it does NOT own: which models to measure (the caller passes them) and any quality claim -- a
throughput run generates against fixed prompts with no gold answers.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from llb.backends.base import BackendLauncher
from llb.backends.ollama import PLACEMENT_GPU
from llb.core.contracts.hardware import TelemetryReport

_LOG = logging.getLogger(__name__)

# The protocol the committed baseline rows were taken under. Changing one of these makes a new row
# incomparable to the table it would join, so they are constants, not defaults chosen per run.
PROTOCOL_MAX_NEW_TOKENS = 128
PROTOCOL_WARMUP = 1
PROTOCOL_CONTEXT = 4096
PROTOCOL_TEMPERATURE = 0.0
PROTOCOL_TIMEOUT_S = 300.0

# `min/100`: the decode-only run-sizing figure quoted beside tok/s -- minutes to answer this many
# cases at this many output tokens each, excluding load time and RAG prefill.
SIZING_CASES = 100
SIZING_OUTPUT_TOKENS = 256

# A row whose backend reported no split still records the rate; the cell says so rather than
# implying GPU residency.
PLACEMENT_UNKNOWN = "unknown"

_SECONDS_PER_MINUTE = 60.0


def minutes_per_100_cases(
    tokens_per_s: float,
    *,
    cases: int = SIZING_CASES,
    output_tokens: int = SIZING_OUTPUT_TOKENS,
) -> float:
    """Decode-only minutes for `cases` answers of `output_tokens` each at this rate (0 if unknown)."""
    if tokens_per_s <= 0:
        return 0.0
    return cases * output_tokens / tokens_per_s / _SECONDS_PER_MINUTE


@dataclass(frozen=True)
class ThroughputRow:
    """One roster entry measured under the protocol: the telemetry record plus its placement."""

    name: str
    backend: str
    served_artifact: str
    placement: str
    telemetry: TelemetryReport

    @property
    def tokens_per_s(self) -> float:
        return float(self.telemetry.get("steady_tokens_per_s") or 0.0)

    @property
    def tokens_per_char(self) -> float:
        return float(self.telemetry.get("tokens_per_char") or 0.0)

    @property
    def peak_vram_mb(self) -> int | None:
        peak = self.telemetry.get("peak_vram_mb")
        return int(peak) if isinstance(peak, int | float) else None

    @property
    def load_time_s(self) -> float | None:
        load = self.telemetry.get("load_time_s")
        return float(load) if isinstance(load, int | float) else None

    @property
    def minutes_per_100(self) -> float:
        return minutes_per_100_cases(self.tokens_per_s)

    def as_record(self) -> dict[str, Any]:
        """The per-model JSON record: the derived reading beside the raw telemetry it came from."""
        return {
            "name": self.name,
            "backend": self.backend,
            "served_artifact": self.served_artifact,
            "placement": self.placement,
            "tokens_per_s": round(self.tokens_per_s, 2),
            "tokens_per_char": round(self.tokens_per_char, 4),
            "minutes_per_100_cases": round(self.minutes_per_100, 1),
            "peak_vram_mb": self.peak_vram_mb,
            "load_time_s": self.load_time_s,
            "protocol": {
                "max_new_tokens": PROTOCOL_MAX_NEW_TOKENS,
                "warmup": PROTOCOL_WARMUP,
                "context": self.telemetry.get("requested_context"),
                "sizing_cases": SIZING_CASES,
                "sizing_output_tokens": SIZING_OUTPUT_TOKENS,
            },
            "telemetry": dict(self.telemetry),
        }


TABLE_HEADER = (
    "| model | served artifact | backend | tok/s | tok/UA-char | min/100 | peak VRAM (MB) |"
    " placement |\n| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"
)


def markdown_row(row: ThroughputRow) -> str:
    """The row as the delivered baseline table carries it."""
    peak = "" if row.peak_vram_mb is None else str(row.peak_vram_mb)
    return (
        f"| `{row.name}` | `{row.served_artifact}` | {row.backend} | {row.tokens_per_s:.2f} | "
        f"{row.tokens_per_char:.3f} | {row.minutes_per_100:.1f} | {peak} | {row.placement} |"
    )


def markdown_table(rows: list[ThroughputRow]) -> str:
    """The measured rows as a table, fastest first -- the order the baseline table is kept in."""
    ordered = sorted(rows, key=lambda r: r.tokens_per_s, reverse=True)
    return "\n".join([TABLE_HEADER, *(markdown_row(r) for r in ordered)])


def placement_reader(launcher: BackendLauncher) -> Callable[[], Any] | None:
    """The launcher's own placement probe, or None for a backend that does not report one."""
    reader = getattr(launcher, "placement", None)
    return reader if callable(reader) else None


def placement_label(launcher: BackendLauncher, backend: str, observed: Any = None) -> str:
    """Where the served weights actually sat, from the backend that knows.

    `observed` is the split sampled DURING the generations, which is the reading that describes
    the run: Ollama's scheduler can evict a runner the moment the last request returns, and on
    this host it does exactly that for a model that barely fits, leaving a post-hoc probe with
    nothing resident to report. vLLM never splits layers to CPU, so a launched vLLM model is
    GPU-resident by construction; llama.cpp reports the layer split it was given.
    """
    split = observed if observed is not None else _read_placement(launcher)
    if split is not None:
        return str(split.label)
    if backend == "vllm":
        return PLACEMENT_GPU
    layers = launcher.meta.get("n_gpu_layers") if hasattr(launcher, "meta") else None
    if isinstance(layers, int) and layers > 0:
        return f"{layers} layers on GPU"
    _LOG.warning(
        "[measure-throughput] %s reported no placement -- the row records the rate without the "
        "GPU/CPU split",
        backend,
    )
    return PLACEMENT_UNKNOWN


def _read_placement(launcher: BackendLauncher) -> Any:
    reader = placement_reader(launcher)
    return reader() if reader is not None else None


def measure_entry(
    name: str,
    backend: str,
    source: str,
    *,
    context: int = PROTOCOL_CONTEXT,
    max_new_tokens: int = PROTOCOL_MAX_NEW_TOKENS,
    warmup: int = PROTOCOL_WARMUP,
    launcher_factory: Callable[[str, str, int], BackendLauncher] | None = None,
    vram_reader: Callable[[], int] | None = None,
    power_reader: Callable[[], float | None] | None = None,
) -> ThroughputRow:
    """Launch one roster entry, measure it under the protocol, and stop it again.

    `launcher_factory(backend, source, context)` is the injected seam: the default builds the same
    launcher `run-eval` would, so a measured row is taken through the serving path a real run uses.
    """
    factory = launcher_factory or _default_launcher
    launcher = factory(backend, source, context)
    with launcher:
        from llb.backends.telemetry import collect_telemetry
        from llb.backends.telemetry_samplers import LastValueSampler

        warm_load(launcher)

        # Sampled alongside peak VRAM and for the same reason: placement is a property of the run,
        # observable only while it is running.
        with LastValueSampler(placement_reader(launcher)) as placements:
            telemetry = collect_telemetry(
                launcher,
                max_new_tokens=max_new_tokens,
                warmup=warmup,
                temperature=PROTOCOL_TEMPERATURE,
                timeout=PROTOCOL_TIMEOUT_S,
                requested_context=context,
                vram_reader=vram_reader,
                power_reader=power_reader,
            )
        placement = placement_label(launcher, backend, placements.value)
    _LOG.info(
        "[measure-throughput] %s (%s) %.2f tok/s, peak %s MB, %s",
        name,
        backend,
        float(telemetry.get("steady_tokens_per_s") or 0.0),
        telemetry.get("peak_vram_mb"),
        placement,
    )
    return ThroughputRow(
        name=name,
        backend=backend,
        served_artifact=source,
        placement=placement,
        telemetry=telemetry,
    )


def warm_load(launcher: BackendLauncher) -> None:
    """Load the model before the timed passes, timing the load and refreshing the served window.

    A launcher that owns its serving process (vLLM, llama.cpp) has already timed its own startup
    and answers its context at readiness. Ollama does neither: the daemon is up long before the
    run and loads the model on the FIRST request, so an unwarmed protocol charges a multi-second
    weight load to the first warmup generation and records neither a load time nor the window it
    actually served. One one-token request fixes both. The caller is expected to have cleared the
    backend first -- the `measure-throughput` command evicts before every cell -- so what this
    times is a cold load.
    """
    if launcher.load_time_s is not None:
        return
    from llb.backends.served_window import launcher_served_window

    started = time.monotonic()
    window = launcher_served_window(launcher)
    if window is None:  # nothing was warmed, so nothing was timed
        return
    launcher.load_time_s = time.monotonic() - started


def _default_launcher(backend: str, source: str, context: int) -> BackendLauncher:
    """The launcher `run-eval` would build for this (backend, source) at this context."""
    from llb.core.config import RunConfig
    from llb.executor.runner_backend import _make_launcher

    config = RunConfig().with_overrides(backend=backend, model=source, max_model_len=context)
    return _make_launcher(config)
