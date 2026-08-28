"""Score the finalists on the PUBLIC Ukrainian tasks before anything is adopted as the default.

The private gold set is one post-edited corpus. A model that wins on it and has never been read on
a public UA benchmark is a recommendation with a single point of failure, so the confirmation run
screens both finalists on the Tier-1 public tracks and carries the result into the verdict.

Two properties are preserved from the Tier-1 screen and matter here:

- the TRACK is a hard fence -- a loglikelihood accuracy and a generation exact-match are not
  comparable, so a public number only ever qualifies a candidate against a finalist on the SAME
  track, never across the two;
- COVERAGE is explicit -- a task that errored or was skipped is visible, so a partial screen
  qualifies the verdict rather than silently reading as a pass.

The screen is a lookup when a report for that model already exists under `$DATA_DIR/screen/`, so
re-running the confirmation does not re-pay for lm-eval.
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from llb.core.config import RunConfig
from llb.core.contracts.screening import ScreenReport
from llb.screen.public_report import safe_model_name

_LOG = logging.getLogger(__name__)

SCREEN_METHOD = "screen"
REPORT_SUFFIX = ".screen.json"

ScreenRunner = Callable[[str, str, Path], ScreenReport]


def screen_report_path(out_dir: Path, model: str) -> Path:
    """``$DATA_DIR/screen/<safe-model>.screen.json`` -- the parsed report, not lm-eval's raw JSON."""
    return out_dir / f"{safe_model_name(model)}{REPORT_SUFFIX}"


def read_report(out_dir: Path, model: str) -> ScreenReport | None:
    """A previously parsed report for this model, or None when it has never been screened."""
    path = screen_report_path(out_dir, model)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[joint-search] ignore unreadable screen report %s: %s", path, exc)
        return None
    if not isinstance(value, dict) or "track" not in value or "requested_tasks" not in value:
        _LOG.warning("[joint-search] ignore invalid screen report %s", path)
        return None
    return cast(ScreenReport, value)


def write_report(out_dir: Path, report: ScreenReport) -> Path:
    """Persist a parsed screen report beside lm-eval's own output."""
    path = screen_report_path(out_dir, report["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def default_screen_runner(cfg: RunConfig, *, limit: int | None) -> ScreenRunner:
    """The live runner: launch or reuse the backend endpoint and drive lm-eval through it."""

    def run(source: str, backend: str, out_dir: Path) -> ScreenReport:
        from llb.screen.backends import screen_with_backend

        return screen_with_backend(
            source,
            backend,
            cfg.with_overrides(model=source, backend=backend),
            out_dir=out_dir,
            limit=limit,
        )

    return run


def screen_finalists(
    finalists: Sequence[Mapping[str, str]],
    *,
    out_dir: Path,
    runner: ScreenRunner,
) -> dict[str, Any]:
    """Screen every finalist (model, backend, source) and summarize track + coverage.

    A finalist whose screen raises is recorded with its reason instead of aborting the run: the
    verdict can still be stated, qualified by the missing public evidence.
    """
    reports: dict[str, ScreenReport] = {}
    failures: list[dict[str, str]] = []
    for finalist in finalists:
        name, backend, source = finalist["name"], finalist["backend"], finalist["source"]
        prior = read_report(out_dir, source)
        if prior is not None:
            _LOG.info("[joint-search] public screen reuse %s (track=%s)", name, prior["track"])
            reports[name] = prior
            continue
        try:
            report = runner(source, backend, out_dir)
        except Exception as exc:  # a missing harness must not lose the private board
            _LOG.warning("[joint-search] public screen failed for %s: %s", name, exc)
            failures.append({"model": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        write_report(out_dir, report)
        reports[name] = report
    return summarize(reports, failures)


def summarize(
    reports: Mapping[str, ScreenReport], failures: Sequence[Mapping[str, str]] = ()
) -> dict[str, Any]:
    """Per-finalist public reading plus the two facts the verdict reads: track and coverage."""
    tracks = {report["track"] for report in reports.values()}
    return {
        "reports": {name: dict(report) for name, report in sorted(reports.items())},
        "tracks": sorted(tracks),
        "comparable": len(tracks) <= 1,
        "complete": {name: bool(report["complete"]) for name, report in sorted(reports.items())},
        "covered": {name: sorted(report["covered"]) for name, report in sorted(reports.items())},
        "failures": [dict(failure) for failure in failures],
    }


def public_note(summary: Mapping[str, Any], model: str | None) -> str:
    """The clause the verdict appends about `model`'s public standing (empty when it is clean)."""
    if model is None:
        return ""
    complete = summary.get("complete", {})
    if model not in complete:
        return f"; `{model}` has no public Ukrainian screen, so the public tracks do not back it"
    if not complete[model]:
        return f"; `{model}`'s public Ukrainian screen is PARTIAL, so its public coverage is incomplete"
    if not summary.get("comparable", True):
        return (
            "; the finalists were screened on different public tracks "
            f"({', '.join(summary.get('tracks', []))}), which are never cross-ranked"
        )
    return ""


__all__ = [
    "SCREEN_METHOD",
    "ScreenRunner",
    "default_screen_runner",
    "public_note",
    "read_report",
    "screen_finalists",
    "screen_report_path",
    "summarize",
    "write_report",
]
