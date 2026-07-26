"""Do two generation models agree on the scoped first-hit-rank reading, cell by cell?

The bar sweep decides `extend_bar` from ONE model's answers, but whether a first-hit-rank gain
converts into a better answer is partly a property of the MODEL: a stronger or more
instruction-following model may use an earlier-ranked span the first one ignored, or be robust
enough that a cell stops separating. This module reads two finished sweep reports (`comparison.json`
each) and states, per cell, whether the two models reach the SAME keep-or-extend reading -- so the
`extend_bar` verdict is confirmed as a corpus/configuration property or exposed as a single-model
artifact.

Pure and file-driven: the input is two `AdoptionBarReport`s, so the whole comparison is unit-tested
with dict reports -- no backend, no store, no GPU. It runs AFTER two heavy sweeps, never during one.
"""

from collections.abc import Mapping, Sequence

from typing_extensions import NotRequired, TypedDict

from llb.eval.embedder_adoption.models import (
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    CellReport,
)
from llb.rag.fusion_evidence.evidence_gate import (
    READING_INSUFFICIENT_EVIDENCE,
    reading_label,
)
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, separates

# One model's per-cell reading -- the same three outcomes `decide_bar` reasons over, but stated per
# cell rather than pooled into one verdict.
# The rank gain reached the ANSWER here (objective delta clears zero).
READING_ANSWER = "answer"
# The encoder ranks earlier here (reciprocal-rank delta clears zero) but the answer does not move.
READING_RANK_ONLY = "rank_only"
# Neither the answer nor first-hit rank separates in this cell.
READING_NEITHER = "neither"


class CellAgreement(TypedDict):
    """One cell's reading under each model and whether the two agree."""

    label: str
    readings: dict[str, str]  # model id -> READING_*
    agree: bool


class CrossModelReport(TypedDict):
    """Two models' per-cell readings, their agreement, and whether the headline verdicts match."""

    models: list[str]
    baseline: str
    candidate: str
    n: int
    verdicts: dict[str, str]  # model id -> BarVerdict decision
    verdicts_agree: bool
    cells: list[CellAgreement]
    agree_count: int
    metadata: NotRequired[dict[str, object]]


def cell_reading(cell: CellReport, confidence: float = DEFAULT_CONFIDENCE) -> str:
    """The keep-or-extend reading of ONE cell: did the rank gain reach the answer, or only rank?

    Same order as `decide_bar`: the answer is checked before rank, because a cell whose objective
    clears zero is an answer-quality result even if its rank delta also separates. A metric that
    clears zero on too few differing items ends the reading at `insufficient_evidence` -- the same
    rule and the same order `stability.reading_from_deltas` applies to the raw delta vectors.
    """
    for metric, reading in (
        (METRIC_OBJECTIVE, READING_ANSWER),
        (METRIC_RECIPROCAL_RANK, READING_RANK_ONLY),
    ):
        comparison = cell["paired"][metric]
        if comparison["delta"]["lo"] > 0.0:
            return reading if separates(comparison, confidence) else READING_INSUFFICIENT_EVIDENCE
    return READING_NEITHER


def model_id(report: AdoptionBarReport) -> str:
    """The generation model a sweep was scored under (from its metadata, else a stable fallback)."""
    meta = report.get("metadata") or {}
    model = meta.get("model")
    return str(model) if model else f"{report['candidate']}-vs-{report['baseline']}"


def compare_models(reports: Sequence[AdoptionBarReport]) -> CrossModelReport:
    """Read two finished sweeps cell by cell and state whether the models agree.

    Refuses reports that are not actually comparable -- a different encoder pair, cell grid, item
    set, or bootstrap seed would make "agreement" meaningless, so each is a hard error rather than a
    silently reconciled difference.
    """
    if len(reports) != 2:
        raise ValueError("the cross-model reading compares exactly two sweep reports")
    first, second = reports
    assert_comparable(first, second)
    models = [model_id(first), model_id(second)]
    if models[0] == models[1]:
        raise ValueError(f"both reports were scored under the same model {models[0]!r}")
    cells: list[CellAgreement] = []
    for cell_a, cell_b in zip(first["cells"], second["cells"]):
        readings = {
            models[0]: cell_reading(cell_a, first["confidence"]),
            models[1]: cell_reading(cell_b, second["confidence"]),
        }
        cells.append(
            {
                "label": cell_a["label"],
                "readings": readings,
                "agree": readings[models[0]] == readings[models[1]],
            }
        )
    verdicts = {
        models[0]: first["verdict"]["decision"],
        models[1]: second["verdict"]["decision"],
    }
    return {
        "models": models,
        "baseline": first["baseline"],
        "candidate": first["candidate"],
        "n": len(first["item_ids"]),
        "verdicts": verdicts,
        "verdicts_agree": verdicts[models[0]] == verdicts[models[1]],
        "cells": cells,
        "agree_count": sum(cell["agree"] for cell in cells),
    }


def assert_comparable(first: AdoptionBarReport, second: AdoptionBarReport) -> None:
    """Fail loudly on any axis that would make a cross-model reading dishonest.

    Shared with the roster reading, which applies it against a reference report so every model in
    a roster is comparable to every other one.
    """
    if (first["baseline"], first["candidate"]) != (second["baseline"], second["candidate"]):
        raise ValueError(
            "the two sweeps compared different encoder pairs "
            f"({first['candidate']} vs {first['baseline']} != "
            f"{second['candidate']} vs {second['baseline']})"
        )
    labels_a = [cell["label"] for cell in first["cells"]]
    labels_b = [cell["label"] for cell in second["cells"]]
    if labels_a != labels_b:
        raise ValueError(f"the two sweeps used different cell grids ({labels_a} != {labels_b})")
    if first["item_ids"] != second["item_ids"]:
        raise ValueError("the two sweeps scored different item sets")
    if first["seed"] != second["seed"]:
        raise ValueError(
            f"the two sweeps used different bootstrap seeds ({first['seed']} != {second['seed']})"
        )


def _cross_metadata(reports: Sequence[AdoptionBarReport]) -> dict[str, object]:
    """Carry the shared corpus/goldset facts so the cross-model artifact stands on its own."""
    meta = dict(reports[0].get("metadata") or {})
    return {key: meta[key] for key in ("goldset", "corpus", "split", "grounding") if key in meta}


def _headline(report: CrossModelReport) -> str:
    """One sentence: do the two models agree on the bar, and how uniformly per cell?"""
    verdicts = report["verdicts"]
    if report["verdicts_agree"]:
        shared = next(iter(set(verdicts.values())))
        return (
            f"Both models reach the SAME headline verdict (**{shared}**), agreeing on "
            f"{report['agree_count']} of {len(report['cells'])} cells: the reading is a property of "
            "the corpus and configuration, not of the one model first scored."
        )
    return (
        "The two models DISAGREE on the headline verdict ("
        + ", ".join(f"`{model}` -> {decision}" for model, decision in verdicts.items())
        + f"), agreeing on {report['agree_count']} of {len(report['cells'])} cells: the "
        "keep-or-extend reading is at least partly a property of the generation model."
    )


def format_cross_model(
    report: CrossModelReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Embedder adoption bar: do two models agree on the reading?",
) -> str:
    """The cross-model Markdown artifact: the headline, the per-cell agreement table, per-model verdicts."""
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- encoder pair: `{report['candidate']}` vs `{report['baseline']}`",
        f"- scored items: {report['n']} (identical set and seed in both sweeps)",
        "- models: " + ", ".join(f"`{model}`" for model in report["models"]),
        "",
        _headline(report),
        "",
        "### Per-cell reading agreement",
        "",
        "| cell | " + " | ".join(f"`{model}`" for model in report["models"]) + " | agree |",
        "| --- | " + " | ".join([":-:"] * len(report["models"])) + " | :-: |",
    ]
    for cell in report["cells"]:
        readings = " | ".join(reading_label(cell["readings"][model]) for model in report["models"])
        lines.append(f"| `{cell['label']}` | {readings} | {'yes' if cell['agree'] else 'NO'} |")
    lines += [
        "",
        "Each cell is one model's keep-or-extend reading: `answer` = the rank gain reached the "
        "answer (objective interval clears zero); `rank only` = the encoder ranks earlier but the "
        "answer does not move; `neither` = no separation; `insufficient evidence` = the interval "
        "clears zero on too few differing items for the reporting level to be reachable.",
        "",
        "### Headline verdict per model",
        "",
    ]
    lines += [f"- `{model}`: **{decision}**" for model, decision in report["verdicts"].items()]
    lines.append("")
    return "\n".join(lines) + "\n"


def format_cross_summary(report: CrossModelReport) -> str:
    """One-screen ASCII summary for the terminal."""
    lines = [
        f"[compare-adoption-models] n={report['n']} "
        f"candidate={report['candidate']} baseline={report['baseline']}",
        f"  {'cell'.ljust(12)} "
        + " ".join(f"{model.split('/')[-1][:20]:>20}" for model in report["models"])
        + "   agree",
    ]
    for cell in report["cells"]:
        readings = " ".join(
            f"{reading_label(cell['readings'][model]):>20}" for model in report["models"]
        )
        lines.append(f"  {cell['label'].ljust(12)} {readings}   {'yes' if cell['agree'] else 'NO'}")
    verdicts = " ".join(f"{m.split('/')[-1]}={d}" for m, d in report["verdicts"].items())
    lines.append(f"  verdicts: {verdicts} ({'AGREE' if report['verdicts_agree'] else 'DISAGREE'})")
    return "\n".join(lines)
