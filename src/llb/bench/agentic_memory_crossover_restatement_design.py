"""Design contract for the crossover restatement: which published numbers are being re-checked.

The design names the shipped bound, the committed studies whose cells are audited, and every
published crossover those studies state -- each with the FORM it is stated in, because a fold-step
boundary and an interpolated guard are restated differently. Everything the design claims about the
published numbers is checkable against the in-repo design files it points at, so a restatement
cannot silently drift from the study it restates.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context_policy import SUMMARY_INPUT_CAP_WINDOW, SUMMARY_INPUT_CAPS
from llb.bench.agentic_policy_change_audit import AUDITED_KINDS
from llb.bench.agentic_policy_change_geometry import declared_geometry, load_audited_design
from llb.bench.agentic_memory_crossover_restatement_placement import (
    validate_derived_placements,
    validate_published_placement,
)
from llb.bench.agentic_memory_crossover_restatement_provenance import (
    validate_published_provenance,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    CROSSOVER_FORMS,
    FORM_PORTABLE_RATIO,
    INVARIANCE_RULE,
    REPORTING_CONFIDENCE,
    RESTATEMENT_RULE,
    STUDY_KIND,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.bench.agentic_published_value_readings import required_reading


def load_restatement_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader; restatement-specific validation runs separately."""
    return load_transfer_design(path)


def validate_restatement_design(
    design: dict[str, object], *, root: Path, data_dir: Path | None = None
) -> None:
    """Refuse a design that does not pin the shipped bound or misnames a study it restates.

    `data_dir` is the host's run root. It is optional because the resolution below is authoritative
    against the COMMITTED slice of each aggregate -- CI has no runs -- and the artifacts themselves
    are read only as a check that the slice still matches what a host that ran the study holds.
    """
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("crossover restatement design schema or study kind is invalid")
    if float(cast(float, design.get("reporting_confidence", 0.0))) != REPORTING_CONFIDENCE:
        raise ValueError(f"crossover restatement confidence must be {REPORTING_CONFIDENCE}")
    if design.get("shipped_summary_input_cap") not in SUMMARY_INPUT_CAPS:
        raise ValueError(f"the shipped summarize-input cap must be one of {SUMMARY_INPUT_CAPS}")
    if design.get("shipped_summary_input_cap") != SUMMARY_INPUT_CAP_WINDOW:
        raise ValueError(
            "the restatement must be stated against the bound the runtime ships "
            f"({SUMMARY_INPUT_CAP_WINDOW})"
        )

    rule = cast(dict[str, object], design.get("restatement_rule", {}))
    if rule.get("rule") != RESTATEMENT_RULE or rule.get("invariance") != INVARIANCE_RULE:
        raise ValueError("the restatement must predeclare its rule and its invariance criterion")

    control = cast(dict[str, object], design.get("control_recheck", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("crossover restatement control recheck contract is invalid")

    studies = cast(list[dict[str, object]], design.get("audited_studies", []))
    if not studies or len({str(study.get("study_kind")) for study in studies}) != len(studies):
        raise ValueError("the restatement needs at least one uniquely named audited study")
    for study in studies:
        _validate_study(study, root)
    crossovers = published_crossovers(design)
    validate_derived_placements(crossovers)
    # Resolution runs LAST: it is the only rule that reads outside the repo's design files, and its
    # message is about a number's evidence rather than about the design's shape, so a malformed
    # design should never reach it.
    validate_published_provenance(crossovers, root=root, data_dir=data_dir)


def audited_designs(design: dict[str, object], *, root: Path) -> dict[str, dict[str, object]]:
    """Every audited study's committed design, keyed by study kind."""
    return {
        cast(str, study["study_kind"]): load_audited_design(root / cast(str, study["design_path"]))
        for study in cast(list[dict[str, object]], design["audited_studies"])
    }


def published_crossovers(design: dict[str, object]) -> list[dict[str, object]]:
    """Every published crossover being restated, tagged with the study that published it."""
    return [
        {**crossover, "study_kind": study["study_kind"]}
        for study in cast(list[dict[str, object]], design["audited_studies"])
        for crossover in cast(list[dict[str, object]], study["published_crossovers"])
    ]


def _validate_study(study: dict[str, object], root: Path) -> None:
    kind = str(study.get("study_kind", ""))
    if kind not in AUDITED_KINDS:
        raise ValueError(f"{kind!r} is not an audited study kind; choose from {AUDITED_KINDS}")
    path = root / str(study.get("design_path", ""))
    if not path.is_file():
        raise ValueError(
            f"{kind}: the committed design {study.get('design_path')!r} does not exist"
        )
    audited = load_audited_design(path)
    if audited.get("study_kind") != kind:
        raise ValueError(
            f"{kind}: {study.get('design_path')!r} declares study kind "
            f"{audited.get('study_kind')!r}"
        )
    depths = {int(cast(int, cell["depth"])) for cell in declared_geometry(audited, kind)}
    crossovers = cast(list[dict[str, object]], study.get("published_crossovers", []))
    if not crossovers:
        raise ValueError(f"{kind}: a restated study must name the crossovers it published")
    for crossover in crossovers:
        _validate_crossover(kind, crossover, depths)
        # The annotation is checked LAST, once the crossover is known to be well formed: it reads the
        # value and the share, and a placement message about a crossover missing either would name
        # the wrong defect.
        validate_published_placement(kind, crossover, audited)


def _validate_crossover(kind: str, crossover: dict[str, object], depths: set[int]) -> None:
    form = str(crossover.get("form", ""))
    depth = int(cast(int, crossover.get("depth", 0)))
    if form not in CROSSOVER_FORMS:
        raise ValueError(f"{kind}: a published crossover form must be one of {CROSSOVER_FORMS}")
    if depth not in depths:
        raise ValueError(
            f"{kind}: a crossover is published for depth {depth}, which the committed design "
            f"does not test (it tests {sorted(depths)})"
        )
    if int(cast(int, crossover.get("fold_step", 0))) < 1:
        raise ValueError(
            f"{kind} depth {depth}: every published crossover must name the fold step it lands in, "
            "because that is what the restatement checks it still names"
        )
    # The portable ratio is a derived statement rather than a guard, so it carries no single value --
    # it carries the BAND it was published as, which is what its restatement is read against.
    if form == FORM_PORTABLE_RATIO:
        required_reading(crossover, where=f"{kind} depth {depth} {form}")
    elif crossover.get("value") is None:
        raise ValueError(f"{kind} depth {depth}: a {form} crossover must carry its published value")
