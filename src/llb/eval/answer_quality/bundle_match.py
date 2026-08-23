"""Does a recorded run bundle still describe the lane its label claims?

A re-render reconstitutes a finished comparison from the run bundles it recorded
(`llb.eval.answer_quality.bundles`), and that is only legitimate while those bundles still hold the
run the artifact says they hold. A bundle set that drifted -- a lane's directories repointed, a
knob changed under the same label, a different model or gold set -- would otherwise re-render into
a DIFFERENT comparison wearing the recorded one's provenance.

Everything checkable lives in each bundle's own `manifest.json`, which records the full `RunConfig`
it ran under. This module states what that config must still say, given the lane LABEL (which
parses back into retrieval knobs) and the comparison metadata (model, backend, gold set,
grounding, splits, retrieval budget), and lists every disagreement rather than stopping at the
first -- an operator fixing a stale bundle set wants the whole list.

Two rules keep the check honest about the ARCHIVE it reads:

- **Only the knobs the lane rides on.** A vector lane carries the fusion knobs as dead config;
  holding it to them measures nothing.
- **A field the manifest never recorded is not a mismatch by itself.** That run predates the knob,
  so it is consistent with the knob's DEFAULT and with nothing else -- which still refuses a label
  asking for a non-default value (`/ioverlap` on a run older than span identity).
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llb.eval.answer_quality.lanes import BACKEND_FUSED
from llb.eval.answer_quality.models import GROUNDING_DRAFTED, LaneSpec

MANIFEST_FILENAME = "manifest.json"

# The lane run-name prefix `run_answer_quality` stamps on every bundle it scores. It lives here
# rather than in `run.py` so a bundle's identity is checkable without importing the orchestration
# that produced it.
RUN_NAME_PREFIX = "answer-quality"


class BundleMismatch(ValueError):
    """The recorded bundles no longer reconstitute the comparison that named them."""


def refusal(path: Path, mismatches: Sequence[str]) -> str:
    """The one refusal message every drift in a bundle set is reported through."""
    return "\n".join(
        [
            f"{path}: the recorded run bundles no longer match the lanes this comparison was "
            "measured with; re-run the comparison instead of re-rendering it",
            *(f"  - {line}" for line in mismatches),
        ]
    )


def recorded_splits(metadata: Mapping[str, Any]) -> list[str]:
    """The gold splits the comparison recorded, in the order its bundles were run."""
    return [name.strip() for name in str(metadata.get("split", "")).split(",") if name.strip()]


def _manifest(run_dir: str) -> Mapping[str, Any]:
    """A bundle's recorded config, with the two identity fields the manifest keeps outside it."""
    manifest = Path(run_dir) / MANIFEST_FILENAME
    if not manifest.is_file():
        raise BundleMismatch(f"recorded run bundle {run_dir} has no {MANIFEST_FILENAME}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise BundleMismatch(f"{manifest}: no run config recorded")
    return {
        **config,
        "run_name": payload.get("run_name", config.get("run_name")),
        "split": payload.get("split"),
    }


def _same(recorded: Any, expected: Any) -> bool:
    if isinstance(expected, float) or isinstance(recorded, float):
        try:
            return math.isclose(float(recorded), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return bool(recorded == expected)


def _lane_expectations(spec: LaneSpec) -> dict[str, Any]:
    """What every bundle of this lane must still record, derived from the lane LABEL alone.

    Only the knobs the label's own retrieval RIDES on are checked. A vector or graph lane carries
    the fusion knobs as dead config -- they change nothing about what it retrieved -- and holding a
    bundle to the value today's default would have given it refuses runs that are perfectly
    reconstitutable.
    """
    expected: dict[str, Any] = {
        "run_name": f"{RUN_NAME_PREFIX}-{spec.label}",
        "retrieval_backend": spec.retrieval_backend,
        "restore_table_headers": spec.restore_table_headers,
    }
    if spec.retrieval_strategy is not None:
        expected["retrieval_strategy"] = spec.retrieval_strategy
    if spec.graph_weight is not None:
        expected["graph_weight"] = spec.graph_weight
    if spec.retrieval_backend == BACKEND_FUSED:
        expected.update(
            graph_fusion_candidates=spec.graph_fusion_candidates,
            graph_fusion_span_identity=spec.graph_fusion_span_identity,
            graph_fusion_span_merge_ratio=spec.graph_fusion_span_merge_ratio,
            graph_fusion_router=spec.graph_fusion_router,
        )
    return expected


def _metadata_expectations(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """What every bundle must record about the RUN the comparison says it was measured under.

    Grounding is asymmetric on disk: `run-eval` stamps `item_grounding` on a DRAFTED bundle and
    leaves a verified one unstamped, so a verified comparison expects the field to be absent --
    which is exactly what refuses a drafted bundle standing in for a verified lane.
    """
    fields = {"model": "model", "backend": "backend", "goldset": "goldset_path"}
    expected: dict[str, Any] = {
        field: metadata[key] for key, field in fields.items() if metadata.get(key) is not None
    }
    if metadata.get("grounding") is not None:
        expected["item_grounding"] = (
            GROUNDING_DRAFTED if metadata["grounding"] == GROUNDING_DRAFTED else None
        )
    return expected


def _recorded_top_k(spec: LaneSpec, metadata: Mapping[str, Any]) -> int | None:
    """The budget this lane claims to have been scored at, or None when nothing records one.

    A budget sweep carries it in the lane label (`vector#k50`); a single-budget run carries one
    integer in the artifact metadata. On an artifact predating both, the bundle's own `top_k` is
    taken as recorded rather than checked against nothing.
    """
    if spec.top_k is not None:
        return spec.top_k
    value = metadata.get("top_k")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# What a bundle that does NOT record a field must have run with. A bundle predating a knob cannot
# have been run with anything but the behavior that knob's DEFAULT names, so a label asking for the
# default round-trips and a label asking for anything else (`/ioverlap` on a run older than span
# identity) stays a real mismatch. Refusing on every unrecorded field instead would refuse every
# comparison older than the newest knob; a field NOT listed here is one every bundle records, and
# its absence is itself the mismatch.
_UNRECORDED = {**LaneSpec._field_defaults, "item_grounding": None}
_ALWAYS_RECORDED = object()


def _matches(config: Mapping[str, Any], field: str, value: Any) -> bool:
    recorded = config[field] if field in config else _UNRECORDED.get(field, _ALWAYS_RECORDED)
    return _same(recorded, value)


def _field_mismatches(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Per disagreeing field, what this bundle recorded -- as text, so it groups across bundles."""
    return [
        (field, repr(config[field]) if field in config else "not recorded")
        for field, value in expected.items()
        if not _matches(config, field, value)
    ]


def _field_lines(
    label: str,
    expected: Mapping[str, Any],
    found: Mapping[tuple[str, str], list[str]],
    total: int,
) -> list[str]:
    """One line per DISTINCT disagreement, not one per bundle.

    Every bundle of a lane was run under the same config, so a lane pointed at the wrong bundles
    repeats the same handful of field mismatches once per split. Grouping them keeps the refusal
    readable while still naming how many bundles carry each and where to look first.
    """
    return [
        f"lane {label!r}: {field} is {recorded}, the comparison recorded "
        f"{expected[field]!r} ({len(run_dirs)} of {total} bundle(s), e.g. {run_dirs[0]})"
        for (field, recorded), run_dirs in found.items()
    ]


def lane_budget(
    spec: LaneSpec, metadata: Mapping[str, Any], run_dirs: Sequence[str]
) -> tuple[int, list[str]]:
    """This lane's retrieval budget, plus every way its bundles disagree with the record.

    The budget is read from the bundles themselves -- it is what the coverage columns must be
    recomputed at -- and checked against the budget the lane label or the metadata claims.
    """
    label = spec.label
    expected = {**_lane_expectations(spec), **_metadata_expectations(metadata)}
    claimed = _recorded_top_k(spec, metadata)
    splits = recorded_splits(metadata)
    mismatches: list[str] = []
    if splits and len(splits) != len(run_dirs):
        mismatches.append(
            f"lane {label!r} recorded {len(run_dirs)} bundle(s) for split(s) {','.join(splits)}"
        )
        splits = []
    budgets: list[int] = []
    found: dict[tuple[str, str], list[str]] = {}
    for index, run_dir in enumerate(run_dirs):
        config = _manifest(run_dir)
        for disagreement in _field_mismatches(config, expected):
            found.setdefault(disagreement, []).append(run_dir)
        if splits and config.get("split") != splits[index]:
            mismatches.append(
                f"lane {label!r} bundle {run_dir}: split is {config.get('split')!r}, "
                f"the comparison recorded {splits[index]!r}"
            )
        budget = config.get("top_k")
        if not isinstance(budget, int):
            mismatches.append(f"lane {label!r} bundle {run_dir}: no retrieval budget recorded")
            continue
        if claimed is not None and budget != claimed:
            mismatches.append(
                f"lane {label!r} bundle {run_dir}: top_k is {budget}, "
                f"the comparison recorded {claimed}"
            )
        budgets.append(budget)
    if len(set(budgets)) > 1:
        mismatches.append(
            f"lane {label!r} bundles disagree on the retrieval budget: {sorted(set(budgets))}"
        )
    return (budgets[0] if budgets else 0), _field_lines(
        label, expected, found, len(run_dirs)
    ) + mismatches


__all__ = [
    "MANIFEST_FILENAME",
    "RUN_NAME_PREFIX",
    "BundleMismatch",
    "lane_budget",
    "recorded_splits",
    "refusal",
]
