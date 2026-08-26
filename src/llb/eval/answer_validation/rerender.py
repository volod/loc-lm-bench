"""Re-read a recorded answer-validation comparison from its own run bundles -- no model call.

A three-lane comparison costs the generations three times over, and the artifact it produced is
locked to the reading it was rendered under. The comparison itself is PURE over the per-case rows
its lanes recorded, and every lane's run bundles are named in its own `comparison.json` -- so a
CHANGED reading can reach a finished run instead of only the next one. That matters here more than
anywhere else in the repo: this work re-labels what counts as a false rejection, and a recorded
run that cannot be re-read under the new labelling could only be compared against by re-spending
the lanes.

Two refusals guard it, and both fire before anything is written:

  - a bundle that no longer describes the lane its label claims (`bundle_match`, shared with the
    answer-quality re-render so the two cannot drift into two notions of what a recorded lane is):
    a repointed directory, a different model, a lane whose contract knobs changed under the same
    name;
  - a rebuilt comparison that no longer covers the item set the artifact recorded.

Nothing here re-scores an answer or edits a recorded bundle. The run bundles stay exactly as the
generations left them, and the re-rendered artifact records which comparison it was rebuilt from,
so stripping the two added metadata keys gives back the artifact the generations produced.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.core.store_generations import generation_timestamp
from llb.eval.answer_quality.bundle_match import (
    MANIFEST_FILENAME,
    BundleMismatch,
    recorded_splits,
    refusal,
)
from llb.eval.answer_validation.constants import (
    LANE_OFF,
    LANE_PYDANTIC,
    LANE_PYDANTIC_ONTOLOGY,
    RUN_NAME_PREFIX,
)
from llb.eval.answer_validation.run import (
    AnswerValidationRun,
    default_out_dir,
    write_artifacts,
)
from llb.eval.answer_validation.study import analyze, with_references
from llb.eval.paired_cases import CaseRows, recorded_lane_rows
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

RERENDER_SOURCE_KEY = "rerendered_from"
RERENDER_TIMESTAMP_KEY = "rerendered_at"

# What each lane's bundles must still record about the answer contract they ran under. This is the
# whole of what a validation LANE is -- the retrieval knobs it shares with every other lane are
# checked through the comparison metadata instead.
LANE_CONTRACT: Mapping[str, Mapping[str, Any]] = {
    LANE_OFF: {"answer_format": "free_text", "answer_validation": "off"},
    LANE_PYDANTIC: {"answer_format": "envelope", "answer_validation": "off"},
    LANE_PYDANTIC_ONTOLOGY: {"answer_format": "envelope", "answer_validation": "ontology"},
}
# Comparison metadata key -> the config field every bundle must still record for it.
_METADATA_FIELDS: Mapping[str, str] = {
    "model": "model",
    "backend": "backend",
    "goldset": "goldset_path",
    "max_tokens": "max_tokens",
    "top_k": "top_k",
}


def read_recorded(path: Path) -> tuple[Mapping[str, Any], dict[str, list[str]]]:
    """The recorded comparison payload and each lane's run bundles, or a named refusal."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lanes = payload.get("lanes") if isinstance(payload, dict) else None
    if not isinstance(lanes, dict) or not lanes or "commonly_answered" not in payload:
        raise BundleMismatch(f"{path}: not a compare-answer-validation comparison")
    run_dirs: dict[str, list[str]] = {}
    for label, lane in lanes.items():
        if label not in LANE_CONTRACT:
            raise BundleMismatch(f"{path}: {label!r} is not a validation lane")
        recorded = [str(run_dir) for run_dir in lane.get("run_dirs") or []]
        if not recorded:
            raise BundleMismatch(f"{path}: lane {label!r} recorded no run bundle to re-read")
        run_dirs[label] = recorded
    return payload, run_dirs


def _manifest(run_dir: str) -> Mapping[str, Any]:
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


def _expected(label: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """What every bundle of this lane must still record: its contract plus the shared run."""
    expected: dict[str, Any] = {
        "run_name": f"{RUN_NAME_PREFIX}-{label}",
        **LANE_CONTRACT[label],
    }
    expected.update(
        {
            field: metadata[key]
            for key, field in _METADATA_FIELDS.items()
            if metadata.get(key) is not None
        }
    )
    return expected


def _recorded_as(config: Mapping[str, Any], field: str) -> str:
    """What this bundle says about one field, as text -- `not recorded` when it says nothing.

    Compared as text on purpose: the manifest round-trips a `Path` as a string and an `int` as an
    int, and the comparison metadata records whichever the run put there.
    """
    return "not recorded" if field not in config else str(config[field])


def lane_mismatches(label: str, run_dirs: list[str], metadata: Mapping[str, Any]) -> list[str]:
    """Every way this lane's recorded bundles disagree with what the comparison says they are."""
    expected = _expected(label, metadata)
    splits = recorded_splits(metadata)
    found: list[str] = []
    if splits and len(splits) != len(run_dirs):
        return [
            f"lane {label!r} recorded {len(run_dirs)} bundle(s) for split(s) {','.join(splits)}"
        ]
    for index, run_dir in enumerate(run_dirs):
        config = _manifest(run_dir)
        found += [
            f"lane {label!r} bundle {run_dir}: {field} is {_recorded_as(config, field)}, "
            f"the comparison recorded {value}"
            for field, value in expected.items()
            if _recorded_as(config, field) != str(value)
        ]
        if splits and str(config.get("split")) != splits[index]:
            found.append(
                f"lane {label!r} bundle {run_dir}: split is {config.get('split')!r}, "
                f"the comparison recorded {splits[index]!r}"
            )
    return found


def resolve_lane_rows(
    run_dirs: Mapping[str, list[str]], metadata: Mapping[str, Any]
) -> tuple[dict[str, CaseRows], list[str]]:
    """Every recorded lane's per-case rows, plus every way the bundle set drifted.

    Every lane is checked before any is trusted, so the refusal lists the whole drift rather than
    stopping at the first bundle an operator would have to fix.
    """
    rows: dict[str, CaseRows] = {}
    mismatches: list[str] = []
    for label, dirs in run_dirs.items():
        drift = lane_mismatches(label, dirs, metadata)
        mismatches += drift
        if drift:
            continue
        try:
            rows[label] = list(recorded_lane_rows(dirs))
        except (FileNotFoundError, ValueError) as exc:
            mismatches.append(f"lane {label!r}: {exc}")
    return rows, mismatches


def gold_references(goldset: str | None) -> dict[str, str]:
    """item id -> reference answer, from the gold set the comparison recorded.

    A gold set that has moved is not a refusal: the reference only enriches the refusal table and
    the inflection-tolerant labelling, and a re-render that loses it degrades to exactly the
    surface-token reading the recorded artifact already had.
    """
    if not goldset or not Path(goldset).is_file():
        return {}
    from llb.goldset.schema import load_goldset

    return {item.id: item.reference_answer for item in load_goldset(Path(goldset))}


def rebuild_report(payload: Mapping[str, Any], rows: Mapping[str, CaseRows]) -> dict[str, Any]:
    """The recorded comparison recomputed from its own run bundles under the CURRENT reading."""
    metadata = payload.get("metadata") or {}
    references = gold_references(str(metadata.get("goldset", "")) or None)
    settings = payload.get("settings") or {}
    gated = LANE_PYDANTIC_ONTOLOGY if LANE_PYDANTIC_ONTOLOGY in rows else None
    return analyze(
        {label: with_references(lane_rows, references) for label, lane_rows in rows.items()},
        baseline=str(payload.get("baseline", LANE_OFF)),
        run_dirs={label: list(dirs) for label, dirs in _recorded_dirs(payload).items()},
        gated_lane=gated,
        references=references,
        resamples=int(settings.get("resamples", DEFAULT_RESAMPLES)),
        confidence=float(settings.get("confidence", DEFAULT_CONFIDENCE)),
        seed=int(settings.get("seed", DEFAULT_SEED)),
    )


def _recorded_dirs(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        label: [str(run_dir) for run_dir in lane.get("run_dirs") or []]
        for label, lane in payload["lanes"].items()
    }


def rerender_metadata(
    metadata: Mapping[str, Any], source: Path, *, timestamp: str
) -> dict[str, Any]:
    """The recorded metadata, in its recorded order, plus where this re-render came from."""
    return {**metadata, RERENDER_SOURCE_KEY: str(source), RERENDER_TIMESTAMP_KEY: timestamp}


def rerender_from_bundles(
    comparison: Path,
    *,
    config: RunConfig | None = None,
    out_dir: Path | None = None,
    timestamp: str | None = None,
) -> AnswerValidationRun:
    """Re-render `comparison` from the run bundles it recorded, into a NEW artifact directory.

    The recorded artifact is never written to: a re-render is a new reading of the same
    generations, and overwriting the one the run produced would destroy the thing it is checked
    against.
    """
    source = Path(comparison)
    payload, run_dirs = read_recorded(source)
    metadata = payload.get("metadata") or {}
    rows, mismatches = resolve_lane_rows(run_dirs, metadata)
    if mismatches:
        raise BundleMismatch(refusal(source, mismatches))
    report = rebuild_report(payload, rows)
    if sorted(report["commonly_answered"]) != sorted(payload["commonly_answered"]):
        raise BundleMismatch(
            refusal(
                source,
                [
                    f"the bundles are commonly answered on {report['n_commonly_answered']} "
                    f"item(s), the comparison recorded {payload['n_commonly_answered']}"
                ],
            )
        )
    target = Path(out_dir) if out_dir is not None else default_out_dir(config or RunConfig())
    stamp = timestamp or generation_timestamp()
    paths = write_artifacts(
        report, target, metadata=rerender_metadata(metadata, source, timestamp=stamp)
    )
    return AnswerValidationRun(report, target, paths)


__all__ = [
    "LANE_CONTRACT",
    "RERENDER_SOURCE_KEY",
    "RERENDER_TIMESTAMP_KEY",
    "BundleMismatch",
    "gold_references",
    "lane_mismatches",
    "read_recorded",
    "rebuild_report",
    "rerender_from_bundles",
    "rerender_metadata",
    "resolve_lane_rows",
]
