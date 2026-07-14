#!/usr/bin/env python3
"""Answer a SQuAD-format goldset with a single closed RAG service.

Pipeline, per input line:
  1. read a goldset item (id + question + reference_answer),
  2. POST the question to the RAG service (with retries),
  3. write the item back out, enriched with `predicted_answer`.

Data contract
─────────────
INPUT — one JSON object per line (SQuAD-format goldset item).
Fields this script actually reads:
    question        (str, REQUIRED)  the user question sent to the RAG service.
    id              (str, optional)  used only for progress logging; "?" if absent.
Every other input field is passed through UNCHANGED and re-emitted, notably:
    reference_answer(str)  the gold answer — untouched here, consumed later by scoring.
    source_doc_id, source_spans, lang, provenance, verified, split, ...  carried verbatim.

OUTPUT — the same object with these fields ADDED (overwriting any pre-existing keys):
    predicted_answer(str)       the service's answer; "" on failure.
    error           (str|None)  None on success, else "<ExcType>: <message>".
    service         (str)       SERVICE_NAME, for run/service provenance.
    latency_s       (float)     wall-clock seconds for this item's request.

Developers edit: the CONFIG block (endpoint, auth, timeout, retries) and the two
wire-format seams `build_request()` / `parse_answer()` to match your service's shape.

Output is JSONL that preserves every input field and appends the answer, so it lines up
with the goldset by `id` and feeds the scoring step directly.

Usage:
    python3 rag_squad_response.py INPUT.jsonl OUTPUT.jsonl [--limit N]

Standalone by design: **Python 3.12 stdlib only**, no third-party packages, so it can
run inside an air-gapped / closed environment that only exposes one HTTP endpoint.

Topology — ONE RAG service that answers directly:
  You send it a question; the service does retrieval AND generation internally and
  returns the final answer. This script never embeds, retrieves, or prompts an LLM
  itself — `query_rag_service()` is the only network call.


"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast
from llb.standalone.rag_squad_client import (
    NOT_FOUND,
    RAG_SERVICE_URL,
    RETRY_ATTEMPTS,
    SERVICE_NAME,
    log,
    query_rag_service,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — remote operators edit here.
# Values fall back to environment variables so nothing has to be hard-coded.
# ─────────────────────────────────────────────────────────────────────────────

# URL of the RAG service that answers questions directly (retrieval + generation inside).

# Name recorded in each output row (for provenance when comparing runs / services).

# Optional auth. If the service sits behind a gateway needing a bearer token,
# set RAG_API_KEY in the environment; leave unset for an open local service.

# Per-request timeout in seconds.

# Transient-failure retries: total attempts and linear backoff base (seconds).

# Sentinel the service should return when the answer is absent from its corpus.

# Default input goldset used when no path is given on the CLI (dev convenience).
DEFAULT_INPUT = Path(
    os.environ.get(
        "GOLDSET_PATH",
        ".data/quickstart-pdf-corpus-goods-out/llb/goldset/squad_uk.jsonl",
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Instructions (Ukrainian) — optional guidance forwarded to the RAG service.
# Keep or drop depending on whether your service honours per-request instructions;
# a service with a fixed internal prompt will simply ignore this field.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Wire format — request/response shape lives here so it is easy to swap.
# ─────────────────────────────────────────────────────────────────────────────


# Errors worth retrying: network hiccups, timeouts, transient 5xx.


# ─────────────────────────────────────────────────────────────────────────────
# Logging + statistics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Stats:
    """Running tally for a reasonable end-of-run report."""

    total: int = 0  # items processed
    answered: int = 0  # got a concrete answer
    not_found: int = 0  # service returned NOT_FOUND
    errors: int = 0  # failed after retries
    latencies: list[float] = field(default_factory=list)

    def summary(self, wall_s: float) -> str:
        lat = self.latencies
        avg = sum(lat) / len(lat) if lat else 0.0
        rate = self.total / wall_s if wall_s else 0.0
        return (
            "\n──────── run summary ────────\n"
            f"  processed : {self.total}\n"
            f"  answered  : {self.answered}\n"
            f"  not found : {self.not_found}\n"
            f"  errors    : {self.errors}\n"
            f"  latency   : avg {avg:.2f}s | min {min(lat, default=0):.2f}s | max {max(lat, default=0):.2f}s\n"
            f"  wall time : {wall_s:.1f}s ({rate:.2f} items/s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# IO / main loop
# ─────────────────────────────────────────────────────────────────────────────


def _load_item(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    return cast(dict[str, Any], json.loads(line))


def _get_answer(item: dict[str, Any]) -> str:
    """Query the RAG service and record its answer on the item. Returns the answer."""
    answer = query_rag_service(item["question"])
    item["predicted_answer"] = answer
    item["error"] = None
    return answer


def _set_error(item: dict[str, Any], exc: Exception) -> None:
    # One bad item must not abort a long run; record and move on.
    item["predicted_answer"] = ""
    item["error"] = f"{type(exc).__name__}: {exc}"
    log(f"[warn] {item.get('id', '?')}: {item['error']}")


def _write_item(item: dict[str, Any], started: float, f_out: TextIO) -> float:
    """Stamp service + latency, write the JSONL line, and return the latency in seconds."""
    latency = round(time.monotonic() - started, 3)
    item["service"] = SERVICE_NAME
    item["latency_s"] = latency
    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    f_out.flush()  # stream to disk so an interrupt never loses completed work
    return latency


def process(input_path: Path, output_path: Path, limit: int | None) -> Stats:
    """Answer every item in the input JSONL and stream results to the output JSONL.

    Ctrl-C stops cleanly after the current item; everything written so far is kept.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [ln for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit is not None:
        lines = lines[:limit]
    n = len(lines)

    stats = Stats()
    started_run = time.monotonic()
    with output_path.open("w", encoding="utf-8") as f_out:
        try:
            for i, line in enumerate(lines, 1):
                item = _load_item(line)
                if item is None:
                    continue
                stats.total += 1

                started = time.monotonic()
                try:
                    answer = _get_answer(item)
                    stats.answered += 1
                    if answer.strip().upper().startswith(NOT_FOUND.upper()):
                        stats.not_found += 1
                except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                    _set_error(item, exc)
                    stats.errors += 1

                latency = _write_item(item, started, f_out)
                stats.latencies.append(latency)
                preview = (item["predicted_answer"] or "<empty>")[:80]
                log(f"[{i}/{n}] {item.get('id', '?')} ({latency:.2f}s) -> {preview!r}")
        except KeyboardInterrupt:
            log("\n[interrupted] stopping after current item; partial results kept.")

    log(stats.summary(time.monotonic() - started_run) + f"\n  output    : {output_path}")
    return stats


def _set_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Positionals are optional so a no-arg run (e.g. from an IDE) works out of the box;
    # DEFAULT_INPUT is a dev convenience — on the remote host pass explicit paths.
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input goldset JSONL (SQuAD format). Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output JSONL with predicted answers. Default: <input>.answers.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _set_parser()
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name(f"{args.input.stem}.answers.jsonl")

    log(f"Service:  {SERVICE_NAME}\nEndpoint: {RAG_SERVICE_URL}\nRetries:  {RETRY_ATTEMPTS}\n")
    process(args.input, output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
