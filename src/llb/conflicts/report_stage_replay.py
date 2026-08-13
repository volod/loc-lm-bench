"""Render a stage attribution recomputed from finished bundles, beside what each run recorded.

The table's job is the disagreement: a rule change is worth adopting only if it names a different
stage on some bundle that already exists, and worth trusting only if it names the SAME stage on the
bundles whose reading is already published. Both readings are on one row, so neither has to be
looked up in the run's own report.
"""

from llb.conflicts.governance_stage import LOST_PAIR_FIELD, STAGE_NAMES, lost_pair_sentence
from llb.core.contracts.common import JsonObject

_NOTHING_LOST = "no orderable pair lost"
_UNANSWERABLE = "not recomputable"
_AGREEMENT = {True: "yes", False: "**no**", None: "--"}


def stage_phrase(lost: JsonObject | None) -> str:
    """`CHUNKING (`a.md` + `z.md`)`, or the empty reading -- never an invented stage."""
    if not lost:
        return _NOTHING_LOST
    left, right = lost["documents"]
    return f"{STAGE_NAMES[lost['stage']]} (`{left}` + `{right}`)"


def replay_line(entry: JsonObject) -> str:
    """One bundle's answer as the command echoes it, in the operator's own vocabulary."""
    label = entry["label"]
    if not entry["recomputable"]:
        return f"[stage] {label}: {_UNANSWERABLE} -- {entry['reason']}"
    verdict = "agrees with the run" if entry["agrees"] else "DIFFERS from the run"
    lost = entry["recomputed"]
    reading = lost_pair_sentence({LOST_PAIR_FIELD: lost}) if lost else f"{_NOTHING_LOST}."
    return f"[stage] {label}: {verdict} -- {reading}"


def replay_report(entries: list[JsonObject]) -> str:
    """The archive as one table: what each run said, what today's rule says, and whether they part."""
    disagreeing = [entry for entry in entries if entry["agrees"] is False]
    unanswerable = [entry for entry in entries if not entry["recomputable"]]
    lines = [
        "# Stage attribution, recomputed from the bundles",
        "",
        "Each run's own `summary.json` record and its own `findings.jsonl` rows, re-read under this "
        "build's stage rule. No store, no corpus, and no model call -- so a bundle written on "
        "another host answers exactly as it did the day it was written, and a bundle that recorded "
        "no per-document accounting answers nothing rather than guessing from a rebuilt store.",
        "",
        f"- bundles read: {len(entries)}",
        f"- recomputed stage differs from the recorded one: {len(disagreeing)}",
        f"- not recomputable (no per-document record): {len(unanswerable)}",
        "",
        "| run | recorded | recomputed | agrees |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        recomputed = (
            _UNANSWERABLE if not entry["recomputable"] else stage_phrase(entry["recomputed"])
        )
        lines.append(
            f"| `{entry['label']}` | {stage_phrase(entry['recorded'])} | {recomputed} "
            f"| {_AGREEMENT[entry['agrees']]} |"
        )
    if unanswerable:
        lines += ["", *_refusal_section(unanswerable)]
    return "\n".join(lines) + "\n"


def _refusal_section(unanswerable: list[JsonObject]) -> list[str]:
    """Why the unanswerable bundles are left unanswered, said once and counted per reason.

    Per bundle it would be the same sentence over and over on an archive sweep, and the reading is
    about the archive: how many runs a rule change can be scored on, and how many cannot be.
    """
    reasons = sorted({str(entry["reason"]) for entry in unanswerable})
    return [
        "## Not recomputable",
        "",
        "Deliberately unanswered rather than re-derived: the stage these runs named was read from a "
        "store that has been rebuilt since, so recomputing it today would answer about today's "
        "store while looking like the run's own answer. What each run recorded is still in the "
        "table above, and re-running the audit records what a future re-read needs.",
        "",
        *(
            f"- {reason} -- {sum(1 for e in unanswerable if e['reason'] == reason)} of "
            f"{len(unanswerable)}"
            for reason in reasons
        ),
        "",
    ]
