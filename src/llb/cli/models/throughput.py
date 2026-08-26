"""Roster throughput measurement command (`measure-throughput`).

The delivered baseline table is a comparison, so the thing that must be reproducible is the
PROTOCOL, not one run: this command re-measures a roster entry the way every committed row was
measured (fixed prompt set, fixed output budget, one warmup, pinned context, cleared GPU and a
thermal cooldown between models) and prints the markdown row for the docs. Use it when a family
upgrades a generation -- a swap invalidates the row it replaces.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.backends.roster_throughput import (
    PROTOCOL_CONTEXT,
    PROTOCOL_MAX_NEW_TOKENS,
    PROTOCOL_WARMUP,
    ThroughputRow,
    markdown_table,
    measure_entry,
)
from llb.cli.app import app
from llb.cli.helpers import cli_error, echo_gpus, planning_models, resolver_probes
from llb.core.contracts.models import ModelSpec

MEASURE_METHOD = "measure-throughput"


def _selected_specs(specs: list[ModelSpec], models: str) -> list[ModelSpec]:
    """The manifest entries named on the command line, in the order they were named."""
    if not models.strip():
        return specs
    by_name = {str(spec.get("name")): spec for spec in specs}
    wanted = [name.strip() for name in models.split(",") if name.strip()]
    missing = [name for name in wanted if name not in by_name]
    if missing:
        cli_error(f"not in the manifest: {', '.join(missing)} (see `make list-models`)")
    return [by_name[name] for name in wanted]


def _serving_choice(
    spec: ModelSpec, backend: Optional[str], source: Optional[str], *, offline: bool
) -> tuple[str, str] | None:
    """The (backend, source) to measure: forced on the command line, else host-resolved."""
    if backend and source:
        return backend, source
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import resolve

    gpus = detect_gpus()
    resolved = resolve(spec, max_vram_mb(gpus), detect_ram_mb(), probes=resolver_probes(offline))
    chosen_backend = backend or resolved["chosen_backend"]
    chosen_source = source or resolved["chosen_source"]
    if not chosen_backend or not chosen_source:
        return None
    return chosen_backend, chosen_source


def _measure_one(
    name: str,
    backend: str,
    source: str,
    *,
    context: int,
    max_new_tokens: int,
    warmup: int,
    ollama_host: str,
) -> ThroughputRow:
    """One protocol cell: clear the GPU, measure under the isolation contract, clear it again."""
    from llb.cli.helpers import best_effort_gpu_readers
    from llb.executor.isolation import isolate_cell
    from llb.executor.ollama_eviction import evict_ollama

    vram_reader, pid_usage_reader = best_effort_gpu_readers()
    evict_ollama(ollama_host)
    row, _outcome = isolate_cell(
        lambda: measure_entry(
            name,
            backend,
            source,
            context=context,
            max_new_tokens=max_new_tokens,
            warmup=warmup,
            vram_reader=vram_reader,
        ),
        backend=backend,
        vram_reader=vram_reader,
        pid_usage_reader=pid_usage_reader,
    )
    evict_ollama(ollama_host)
    return row


def _write_records(out: Path, rows: list[ThroughputRow]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([row.as_record() for row in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@app.command("measure-throughput")
def measure_throughput_cmd(
    manifest: Path = typer.Option(
        Path("samples/configs/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    models: str = typer.Option(
        "", help="comma-separated logical model names; empty measures every manifest entry"
    ),
    backend: Optional[str] = typer.Option(
        None, help="force this backend instead of resolving one for the host"
    ),
    source: Optional[str] = typer.Option(
        None, help="force this served artifact (tag / repo id) instead of the resolved one"
    ),
    context: int = typer.Option(PROTOCOL_CONTEXT, help="context pinned for the measurement"),
    max_new_tokens: int = typer.Option(PROTOCOL_MAX_NEW_TOKENS, help="output budget per prompt"),
    warmup: int = typer.Option(PROTOCOL_WARMUP, help="discarded warmup passes over the prompts"),
    out: Optional[Path] = typer.Option(
        None, help="write the per-model JSON records here (default: under $DATA_DIR)"
    ),
    offline: bool = typer.Option(
        False, help="skip availability probes when resolving (assume every source exists)"
    ),
) -> None:
    """Measure roster entries under the committed throughput protocol and print the doc rows.

    Each model is measured on its own: the GPU is cleared before and after, the cell runs under the
    shared isolation contract (VRAM-reclaim gate + thermal cooldown, so rates are read at like
    clocks), and the row carries the MEASURED placement -- Ollama's own GPU/CPU byte split -- beside
    tok/s, because an offloaded rate and a GPU-resident rate are not the same measurement.
    """
    from llb.bench.common import new_run_timestamp
    from llb.core.config import RunConfig

    specs = _selected_specs(planning_models(manifest), models)
    config = RunConfig()
    _, run_ts = new_run_timestamp()
    records_path = (
        out if out is not None else config.data_dir / MEASURE_METHOD / run_ts / "rows.json"
    )

    echo_gpus(MEASURE_METHOD)
    rows: list[ThroughputRow] = []
    for spec in specs:
        name = str(spec.get("name"))
        choice = _serving_choice(spec, backend, source, offline=offline)
        if choice is None:
            typer.echo(f"[{MEASURE_METHOD}] skip {name}: no backend on this host can serve it")
            continue
        chosen_backend, chosen_source = choice
        typer.echo(f"[{MEASURE_METHOD}] measuring {name} via {chosen_backend} ({chosen_source})")
        row = _measure_one(
            name,
            chosen_backend,
            chosen_source,
            context=context,
            max_new_tokens=max_new_tokens,
            warmup=warmup,
            ollama_host=config.ollama_host,
        )
        typer.echo(
            f"[{MEASURE_METHOD}] {name}: {row.tokens_per_s:.2f} tok/s, "
            f"{row.minutes_per_100:.1f} min/100, peak {row.peak_vram_mb} MB, {row.placement}"
        )
        rows.append(row)

    if not rows:
        cli_error("nothing measured: no selected model resolved to a runnable backend", code=1)
    _write_records(records_path, rows)
    typer.echo(markdown_table(rows))
    typer.echo(f"[{MEASURE_METHOD}] records -> {records_path}")
