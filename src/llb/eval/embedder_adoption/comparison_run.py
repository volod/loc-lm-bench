"""File-driven cross-model, roster, and screen-study runs."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from llb.eval.embedder_adoption.models import AdoptionBarReport

if TYPE_CHECKING:
    from llb.eval.embedder_adoption.cross_model import CrossModelReport
    from llb.eval.embedder_adoption.roster_models import ModelProfile, RosterReport
    from llb.eval.embedder_adoption.screen_models import ScreenReport

ReportT = TypeVar("ReportT")


def load_report(path: Path) -> AdoptionBarReport:
    """Load a finished sweep comparison from a JSON file or run directory."""
    target = Path(path)
    if target.is_dir():
        target = target / "comparison.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target}: cannot load an adoption-bar comparison: {exc}") from None
    if not isinstance(data, dict) or "cells" not in data or "verdict" not in data:
        raise ValueError(f"{target}: not an adoption-bar comparison.json (no cells/verdict)")
    return data  # type: ignore[return-value]


@dataclass(frozen=True)
class ComparisonRun(Generic[ReportT]):
    report: ReportT
    out_dir: Path
    paths: Mapping[str, str]


def run_cross_model_comparison(
    report_paths: Sequence[Path], *, out_dir: Path
) -> ComparisonRun["CrossModelReport"]:
    """Persist a per-cell cross-model agreement report."""
    from llb.eval.embedder_adoption.cross_model import (
        _cross_metadata,
        compare_models,
        format_cross_model,
    )

    reports = [load_report(path) for path in report_paths]
    report = compare_models(reports)
    metadata = _cross_metadata(reports)
    paths = _write_comparison(
        report,
        metadata,
        Path(out_dir),
        stem="cross_model",
        markdown=format_cross_model(report, metadata=metadata),
    )
    return ComparisonRun(report, Path(out_dir), paths)


def load_profiles(path: Path | None) -> dict[str, "ModelProfile"]:
    """Load explicitly declared per-model properties from JSON."""
    from llb.eval.embedder_adoption.roster_models import PROPERTIES

    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot load model profiles: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: model profiles must be an object keyed by model id")
    profiles: dict[str, ModelProfile] = {}
    for model, profile in data.items():
        if not isinstance(profile, dict):
            raise ValueError(f"{path}: profile for {model!r} must be an object")
        unknown = sorted(set(profile) - set(PROPERTIES))
        if unknown:
            raise ValueError(
                f"{path}: profile for {model!r} declares unknown propert(ies) "
                f"{', '.join(unknown)}; expected any of {', '.join(PROPERTIES)}"
            )
        profiles[str(model)] = profile  # type: ignore[assignment]
    return profiles


def run_roster_comparison(
    report_paths: Sequence[Path],
    *,
    out_dir: Path,
    profiles_path: Path | None = None,
    focus_cell: str | None = None,
) -> ComparisonRun["RosterReport"]:
    """Persist the roster property-separation reading."""
    from llb.eval.embedder_adoption.cross_model import _cross_metadata
    from llb.eval.embedder_adoption.models import DEFAULT_FOCUS_CELL
    from llb.eval.embedder_adoption.roster import compare_roster
    from llb.eval.embedder_adoption.roster_report import format_roster

    reports = [load_report(path) for path in report_paths]
    report = compare_roster(
        reports, load_profiles(profiles_path), focus_cell=focus_cell or DEFAULT_FOCUS_CELL
    )
    metadata = _cross_metadata(reports)
    paths = _write_comparison(
        report,
        metadata,
        Path(out_dir),
        stem="roster",
        markdown=format_roster(report, metadata=metadata),
    )
    return ComparisonRun(report, Path(out_dir), paths)


def run_screen_study_over_paths(
    report_paths: Sequence[Path], *, out_dir: Path, focus_cell: str | None = None, **kwargs: object
) -> ComparisonRun["ScreenReport"]:
    """Persist a per-model screen cost study."""
    from llb.eval.embedder_adoption.cross_model import _cross_metadata
    from llb.eval.embedder_adoption.models import DEFAULT_FOCUS_CELL
    from llb.eval.embedder_adoption.screen import run_screen_study
    from llb.eval.embedder_adoption.screen_report import format_screen

    reports = [load_report(path) for path in report_paths]
    report = run_screen_study(
        reports,
        focus_cell=focus_cell or DEFAULT_FOCUS_CELL,
        **kwargs,  # type: ignore[arg-type]
    )
    metadata = _cross_metadata(reports)
    paths = _write_comparison(
        report,
        metadata,
        Path(out_dir),
        stem="screen",
        markdown=format_screen(report, metadata=metadata),
    )
    return ComparisonRun(report, Path(out_dir), paths)


def _write_comparison(
    report: Mapping[str, object],
    metadata: Mapping[str, object],
    out_dir: Path,
    *,
    stem: str,
    markdown: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    report_path = out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps({**report, "metadata": dict(metadata)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(markdown, encoding="utf-8")
    return {"report": str(report_path), "comparison": str(json_path)}
