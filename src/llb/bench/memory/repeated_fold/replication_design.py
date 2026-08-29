"""Design contract for the two-family repeated-fold completion replication.

The completion reading it replicates was taken on one model family over two memory cases
(`agentic_compact_repeated_fold_completion_design.json`). This design holds the three fold-count
cells and the marker ablation EXACTLY as that one declares them -- what it adds is a larger
predeclared case set, a candidate roster the eligibility gate qualifies families from, and a
per-fold paired-evidence floor. Nothing here may move the geometry: the shared cell contract is
validated by the completion design's own validator so a drift shows up as one failure, not two.
"""

import hashlib
from pathlib import Path
from typing import cast

from llb.bench.agentic.design_fields import as_mapping, as_rows
from llb.bench.memory.repeated_fold.design import validate_repeated_fold_design
from llb.bench.policy_change.geometry import load_audited_design

DESIGN_PATH = "samples/benchmarks/agentic_compact_repeated_fold_replication_design.json"
STUDY_KIND = "compact_repeated_fold_replication"
# Two families is the whole point of a replication: one family cannot separate a fold-count rule
# from a property of the model that produced it.
REQUIRED_FAMILIES = 2
# The completion design this replicates ran two cases per cell; a replication that did not raise
# that number would restate the same ceiling with the same power.
MIN_REPLICATION_TASKS = 8


def load_repeated_fold_replication_design(path: Path | str | None = None) -> dict[str, object]:
    """Load the committed replication design through the shared strict JSON loader."""
    from llb.core.paths import PROJECT_ROOT

    return load_audited_design(PROJECT_ROOT / DESIGN_PATH if path is None else path)


def replication_roster(design: dict[str, object]) -> list[dict[str, object]]:
    """Candidate families in the order the eligibility gate should try them."""
    return as_rows(design, "candidate_roster")


def roster_digest(roster: list[dict[str, object]]) -> str:
    """Content digest of the families a run actually drove, in roster order."""
    payload = "|".join(f"{row['model_family']}:{row['model']}:{row['backend']}" for row in roster)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def minimum_paired_cases(design: dict[str, object]) -> int:
    """The predeclared paired-evidence floor a measured fold group must reach to be read."""
    held = as_mapping(design, "held_fixed")
    return int(cast(int, held["minimum_paired_cases_per_fold"]))


def validate_replication_design(design: dict[str, object]) -> None:
    """Require the completion cell contract, a larger case set, a roster, and an evidence floor."""
    validate_repeated_fold_design(design, study_kind=STUDY_KIND)
    _validate_case_set(design)
    _validate_roster(design)


def _validate_case_set(design: dict[str, object]) -> None:
    held = as_mapping(design, "held_fixed")
    n_tasks = int(cast(int, held.get("n_tasks", 0)))
    if n_tasks < MIN_REPLICATION_TASKS:
        raise ValueError(
            f"a repeated-fold replication needs at least {MIN_REPLICATION_TASKS} predeclared "
            f"cases, got {n_tasks}"
        )
    floor = int(cast(int, held.get("minimum_paired_cases_per_fold", 0)))
    if not 1 < floor <= n_tasks:
        raise ValueError("the paired-evidence floor must be above one and within the case set")


def _validate_roster(design: dict[str, object]) -> None:
    required = int(cast(int, design.get("required_qualified_families", 0)))
    roster = replication_roster(design)
    families = [str(row.get("model_family", "")) for row in roster]
    models = [str(row.get("model", "")) for row in roster]
    if required != REQUIRED_FAMILIES or len(roster) < required:
        raise ValueError(
            f"the repeated-fold replication requires {REQUIRED_FAMILIES} qualified model families"
        )
    if not all(families) or len(families) != len(set(families)):
        raise ValueError("every replication candidate must name a distinct model family")
    if not all(models) or len(models) != len(set(models)):
        raise ValueError("every replication candidate must name a distinct model")
    if any(row.get("backend") != "ollama" for row in roster):
        raise ValueError("the repeated-fold replication roster must use local Ollama models")
