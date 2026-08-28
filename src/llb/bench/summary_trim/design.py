"""Predeclared geometry for the entry-aware summary-fold adoption study.

Every workload cell states, before any model runs, what its fold looks like under BOTH trim
strategies: how many times it folds, what each fold offers the summarizer, how much the
summarize-input cap elides, and how many characters the summarize prompt then costs. Those are
model-free facts (`compact_fold_input_probe` walks the world with an oracle), so a cell that no
longer produces the regime it was built for fails the design gate rather than the reading.

The declaration carries both strategies deliberately. The pair `(elided, summary_prompt_chars)` is
the whole cost side of the adoption question: where nothing is elided the two strategies render
byte-identical prompts and adoption is free by construction, and where something is elided the
declaration says whether the entry-aware trim spends MORE bytes, fewer, or exactly the same.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import (
    POLICY_COMPACT,
    SUMMARY_INPUT_CAP_WINDOW,
    SUMMARY_TRIM_HEAD_TAIL,
    SUMMARY_TRIM_PER_ENTRY_HEAD,
    SUMMARY_TRIM_STRATEGIES,
)
from llb.bench.agentic.design_fields import as_mapping, as_rows
from llb.bench.memory.boundary.probe import compact_tasks_fold_input_probe
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars
from llb.bench.policy_change.geometry import load_audited_design
from llb.bench.summary_trim.workloads import (
    TASK_BUILDERS,
    build_workload_tasks,
    workload_oracle,
    workload_tasks,
)

DESIGN_PATH = "samples/benchmarks/agentic_summary_trim_adoption_design.json"
STUDY_KIND = "summary_trim_strategy_adoption"
# The comparison's two arms, in the order a reading pairs them: shipped default first.
ARMS: tuple[str, ...] = (SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD)
# Fields every workload predeclares per arm, measured with no model.
DECLARED_FIELDS = (
    "n_compactions",
    "summary_input_chars",
    "summary_input_elided_chars",
    "compaction_prompt_chars",
)


def load_summary_trim_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed adoption design through the shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def workloads(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared workload, in design order (the order a live run executes them)."""
    return as_rows(design, "workloads")


def workload_geometry(workload: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """One workload's compact geometry: the study's held constants plus its own guard and share."""
    return {
        "max_prompt_chars": int(cast(int, workload["max_prompt_chars"])),
        "compact_share": float(cast(float, workload["compact_share"])),
        "max_steps": int(cast(int, workload["max_steps"])),
        "observation_cap_chars": int(cast(int, held["observation_cap_chars"])),
        "observation_head_share": float(cast(float, held["observation_head_share"])),
        "summary_input_cap": str(cast(str, held["summary_input_cap"])),
    }


def probe_workload_arm(
    workload: dict[str, object], held: dict[str, object], arm: str
) -> dict[str, object]:
    """Walk every task of one workload under one trim strategy, with an oracle and no model."""
    if arm not in SUMMARY_TRIM_STRATEGIES:
        raise ValueError(f"unknown trim strategy {arm!r}; choose from {SUMMARY_TRIM_STRATEGIES}")
    geometry = workload_geometry(workload, held)
    records = build_workload_tasks(workload)
    tasks = workload_tasks(workload)
    guard = int(cast(int, geometry["max_prompt_chars"]))
    share = float(cast(float, geometry["compact_share"]))
    probes = [
        compact_tasks_fold_input_probe(
            [task],
            max_steps=int(cast(int, geometry["max_steps"])),
            max_prompt_chars=guard,
            compact_share=share,
            summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
            summary_trim_strategy=arm,
            observation_cap_chars=int(cast(int, geometry["observation_cap_chars"])),
            observation_head_share=float(cast(float, geometry["observation_head_share"])),
            controller=workload_oracle(workload, record),
        )
        for record, task in zip(records, tasks, strict=True)
    ]
    return {
        **{field: max(int(cast(int, row[field])) for row in probes) for field in DECLARED_FIELDS},
        "summary_fold_input_chars": cast(list[int], probes[0]["summary_fold_input_chars"]),
        "compaction_trigger_chars": compaction_trigger_chars(guard, share),
    }


def probe_workload(workload: dict[str, object], held: dict[str, object]) -> dict[str, object]:
    """Both arms of one workload, keyed by trim strategy."""
    return {arm: probe_workload_arm(workload, held, arm) for arm in ARMS}


def validate_summary_trim_design(design: dict[str, object]) -> None:
    """Refuse a design that cannot answer the adoption question it declares."""
    held = _validate_header(design)
    declared = workloads(design)
    _validate_workload_contract(declared)
    for workload in declared:
        _validate_workload(workload, held)


def _validate_header(design: dict[str, object]) -> dict[str, object]:
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"summary-trim adoption study_kind must be {STUDY_KIND!r}")
    if int(cast(int, design.get("seed", 0))) < 1:
        raise ValueError("the adoption study needs a positive deterministic seed")
    if cast(list[str], design.get("policies", [])) != [POLICY_COMPACT]:
        raise ValueError("the adoption study must run the compact policy alone")
    if tuple(cast(list[str], design.get("arms", []))) != ARMS:
        raise ValueError(f"the adoption arms must be exactly {ARMS!r}, shipped default first")
    held = as_mapping(design, "held_fixed")
    if held.get("summary_input_cap") != SUMMARY_INPUT_CAP_WINDOW:
        raise ValueError("the adoption study must hold the shipped window summary-input bound")
    if held.get("preserve_memory_markers") is not True:
        raise ValueError("the adoption study must measure shipped typed-marker preservation")
    _validate_roster(design)
    return held


def _validate_roster(design: dict[str, object]) -> None:
    required = int(cast(int, design.get("required_qualified_families", 0)))
    roster = as_rows(design, "candidate_roster")
    families = [str(row.get("model_family", "")) for row in roster]
    models = [str(row.get("model", "")) for row in roster]
    if required != 2 or len(roster) < required:
        raise ValueError("the adoption study requires two qualified host-fit model families")
    if not all(families) or len(set(families)) != len(families):
        raise ValueError("adoption roster families must be present and unique")
    if not all(models) or len(set(models)) != len(models):
        raise ValueError("adoption roster models must be present and unique")
    if any(row.get("backend") != "ollama" for row in roster):
        raise ValueError("the adoption roster must use local Ollama models")


def _validate_workload_contract(declared: list[dict[str, object]]) -> None:
    """The workload set must span both regimes and name each family exactly once."""
    names = [str(row.get("workload", "")) for row in declared]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("every workload must carry a unique name")
    builders = {str(row.get("task_builder", "")) for row in declared}
    unknown = sorted(builders - set(TASK_BUILDERS))
    if unknown:
        raise ValueError(f"unknown workload task builder(s) {unknown}")
    eliding = [row for row in declared if int(cast(int, _expected(row)["elided_chars"])) > 0]
    fitting = [row for row in declared if int(cast(int, _expected(row)["elided_chars"])) == 0]
    if not eliding or not fitting:
        # Without both, the study cannot separate "the strategies agree" from "nothing was cut".
        raise ValueError(
            "the adoption study needs at least one eliding workload and one that elides nothing"
        )
    if not any(int(cast(int, _expected(row)["n_compactions"])) > 1 for row in declared):
        raise ValueError("the adoption study needs at least one repeatedly folding workload")


def _validate_workload(workload: dict[str, object], held: dict[str, object]) -> None:
    """Measure the declared regime, and require the arms to agree on everything but the bytes."""
    expected = _expected(workload)
    measured = probe_workload(workload, held)
    baseline = cast(dict[str, object], measured[SUMMARY_TRIM_HEAD_TAIL])
    candidate = cast(dict[str, object], measured[SUMMARY_TRIM_PER_ENTRY_HEAD])
    name = workload["workload"]
    for field, declared_key in (
        ("n_compactions", "n_compactions"),
        ("summary_input_chars", "offered_chars"),
        ("summary_input_elided_chars", "elided_chars"),
    ):
        if baseline[field] != expected[declared_key]:
            raise ValueError(
                f"workload {name!r} {declared_key} drifted: declared "
                f"{expected[declared_key]!r}, measured {baseline[field]!r}"
            )
    for arm, probe in (
        (SUMMARY_TRIM_HEAD_TAIL, baseline),
        (SUMMARY_TRIM_PER_ENTRY_HEAD, candidate),
    ):
        declared_cost = int(
            cast(int, cast(dict[str, object], expected["summary_prompt_chars"])[arm])
        )
        if int(cast(int, probe["compaction_prompt_chars"])) != declared_cost:
            raise ValueError(
                f"workload {name!r} summary prompt chars under {arm!r} drifted: declared "
                f"{declared_cost}, measured {probe['compaction_prompt_chars']}"
            )
    if baseline["summary_fold_input_chars"] != candidate["summary_fold_input_chars"]:
        # The arms are byte-identical up to and including the transcript the fold OFFERS; only what
        # the cap lets through can differ. A disagreement here means the comparison is unpaired.
        raise ValueError(f"workload {name!r} arms do not offer the summarizer the same transcript")
    if int(cast(int, expected["elided_chars"])) == 0 and baseline != candidate:
        raise ValueError(
            f"workload {name!r} elides nothing, so both trims must render the identical prompt"
        )


def _expected(workload: dict[str, object]) -> dict[str, object]:
    return as_mapping(workload, "expected")
