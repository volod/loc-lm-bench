"""Render a stage attribution recomputed from finished bundles, beside what each run recorded.

The table's job is the disagreement: a rule change is worth adopting only if it names a different
stage on some bundle that already exists, and worth trusting only if it names the SAME stage on the
bundles whose reading is already published. Both readings are on one row, so neither has to be
looked up in the run's own report.
"""

from llb.conflicts.governance.stage import (
    LOST_PAIR_FIELD,
    lost_pair_counts_phrase,
    lost_pair_sentence,
)
from llb.conflicts.governance.stage_rule import STAGE_NAMES
from llb.core.contracts.common import JsonObject

_NOTHING_LOST = "no orderable pair lost"
_UNANSWERABLE = "not recomputable"
_AGREEMENT = {True: "yes", False: "**no**", None: "--"}
# A budget that MOVES the attribution is the notable outcome, so it is the emphasized one.
_MOVES = {True: "**yes**", False: "no", None: "--"}


def stage_phrase(lost: JsonObject | None) -> str:
    """`CHUNKING (`a.md` + `z.md`)`, or the empty reading -- never an invented stage."""
    if not lost:
        return _NOTHING_LOST
    left, right = lost["documents"]
    split = lost_pair_counts_phrase(lost)
    detail = f"; split: {split}" if split else ""
    return f"{STAGE_NAMES[lost['stage']]} (`{left}` + `{right}`{detail})"


def replay_line(entry: JsonObject) -> str:
    """One bundle's answer as the command echoes it, in the operator's own vocabulary."""
    label = entry["label"]
    if not entry["recomputable"]:
        return f"[stage] {label}: {_UNANSWERABLE} -- {entry['reason']}"
    verdict = "agrees with the run" if entry["agrees"] else "DIFFERS from the run"
    lost = entry["recomputed"]
    reading = lost_pair_sentence({LOST_PAIR_FIELD: lost}) if lost else f"{_NOTHING_LOST}."
    return f"[stage] {label}: {verdict} -- {reading}"


def budget_line(entry: JsonObject) -> str:
    """What the same bundle would have said at a smaller candidate budget, or why it cannot say."""
    at_budget = entry["at_budget"]
    label, budget = entry["label"], at_budget["budget"]
    if not at_budget["recomputable"]:
        return f"[stage] {label} at budget {budget}: {_UNANSWERABLE} -- {at_budget['reason']}"
    lost = at_budget["recomputed"]
    reading = lost_pair_sentence({LOST_PAIR_FIELD: lost}) if lost else f"{_NOTHING_LOST}."
    moved = (
        "MOVES the attribution" if at_budget["changes"] else "leaves the attribution where it is"
    )
    return (
        f"[stage] {label} at budget {budget}: {pairs_phrase(at_budget)}, and {moved} -- {reading}"
    )


def store_line(entry: JsonObject) -> str:
    """Whether the resolved or fallback store is the one this bundle's readings are about."""
    identity = entry["store_identity"]
    return f"[stage] {entry['label']}: {identity['detail']}"


def pairs_phrase(at_budget: JsonObject) -> str:
    """The total and orderable pair costs the named pair cannot give.

    On a corpus whose corpus-first lost pair is lost at every budget, the name stays fixed while
    the distinct document-pair count and the governance coverage's returned-pair count can move.
    """
    returned, run = at_budget["document_pairs"], at_budget["run_document_pairs"]
    if returned is None:
        return "the returned pairs are not recomputable"
    orderable, run_orderable = at_budget["orderable_pairs"], at_budget["run_orderable_pairs"]
    cost = run_orderable - orderable
    unit = "pair" if cost == 1 else "pairs"
    return (
        f"{returned} of the run's {run} document pairs return; "
        f"{orderable} of the run's {run_orderable} orderable returned pairs belong to them "
        f"(budget costs {cost} orderable returned {unit})"
    )


def replay_report(entries: list[JsonObject]) -> str:
    """The archive as one table: what each run said, what today's rule says, and whether they part."""
    disagreeing = [entry for entry in entries if entry["agrees"] is False]
    unanswerable = [entry for entry in entries if not entry["recomputable"]]
    lines = [
        "# Stage attribution, recomputed from the bundles",
        "",
        "Each run's own `summary.json` record and its own `findings.jsonl` rows, re-read under this "
        "build's stage rule. That stage reading needs no store, corpus, or model call -- so a "
        "bundle written on another host answers exactly as it did the day it was written, and a "
        "bundle that recorded no per-document accounting answers nothing rather than guessing "
        "from a rebuilt store. The separate store-placement section, when present, reads only each "
        "resolved store's metadata.",
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
    lines += ["", *_readings_section(entries)]
    if any("store_identity" in entry for entry in entries):
        lines += ["", *_store_section(entries)]
    if any("at_budget" in entry for entry in entries):
        lines += ["", *_budget_section(entries)]
    if unanswerable:
        lines += ["", *_refusal_section(unanswerable)]
    return "\n".join(lines) + "\n"


def _readings_section(entries: list[JsonObject]) -> list[str]:
    """Which of the questions asked of a finished run this archive can answer, per question.

    Counted across the archive rather than printed per bundle: the operator's question is "can I
    ask this of the runs I have", and the two standing refusals are properties of the record's
    boundary rather than of any one bundle, so they read the same on every row by construction.
    """
    questions: dict[str, tuple[str, int, set[str]]] = {}
    for entry in entries:
        for reading in entry.get("readings", []):
            question, answers, reasons = questions.setdefault(
                reading["name"], (reading["question"], 0, set())
            )
            if reading["available"]:
                answers += 1
            elif reading["detail"]:
                reasons.add(str(reading["detail"]))
            questions[reading["name"]] = (question, answers, reasons)
    if not questions:
        return []
    lines = [
        "## What a bundle can answer alone",
        "",
        "Per question, over the bundles read. A question answered by no bundle at ANY schema is a "
        "boundary the record keeps deliberately -- its input is bounded by neither DOCUMENTS nor a "
        "run parameter -- and the fix for it is to rebuild the store and re-run, not to re-read.",
        "",
        "| question | bundles that answer it | why not |",
        "| --- | --- | --- |",
    ]
    for question, answers, reasons in questions.values():
        why = "; ".join(sorted(reasons)) or "--"
        lines.append(f"| {question} | {answers} of {len(entries)} | {why} |")
    return lines


def _store_section(entries: list[JsonObject]) -> list[str]:
    """Whether each bundle's resolved store is the one its recorded identity names.

    This is the one question the recorded store manifest is asked, and it is an equality test: a
    bundle's readings describe the store its run held, so a comparison against a different store
    answers correctly about something else.
    """
    placed = [entry for entry in entries if "store_identity" in entry]
    comparable = [entry for entry in placed if entry["store_identity"]["comparable"]]
    same = [entry for entry in comparable if not entry["store_identity"]["changed"]]
    recorded = [
        entry for entry in placed if entry["store_identity"].get("reference_source") == "bundle"
    ]
    fallback = [
        entry for entry in placed if entry["store_identity"].get("reference_source") == "explicit"
    ]
    lines = [
        "## The store these bundles read",
        "",
        "Each current run records a DATA_DIR-relative store location beside the digest over the "
        "store manifest it read. Without a flag, a re-read resolves each bundle's own location. "
        "An explicit `--store` wins for every bundle and remains the fallback for an older bundle "
        "without one. A missing recorded location is reported as unavailable and is never turned "
        "into a mismatch against a store that was not read.",
        "",
        f"- resolved from the bundle's recorded location: {len(recorded)} of {len(placed)}",
        f"- resolved from the explicit override/fallback: {len(fallback)} of {len(placed)}",
        f"- resolved store matches the bundle identity: {len(same)} of {len(placed)}",
        f"- resolved store differs from the bundle identity: {len(comparable) - len(same)} of "
        f"{len(placed)}",
        f"- not comparable (location or identity unavailable): {len(placed) - len(comparable)} of "
        f"{len(placed)}",
        "",
        "| run | store placement |",
        "| --- | --- |",
    ]
    lines += [f"| `{entry['label']}` | {entry['store_identity']['detail']} |" for entry in placed]
    return lines


def _budget_section(entries: list[JsonObject]) -> list[str]:
    """The same archive re-read at a smaller candidate budget, which is the one knob a prefix covers."""
    asked = [entry for entry in entries if "at_budget" in entry]
    budget = asked[0]["at_budget"]["budget"]
    moved = sum(1 for entry in asked if entry["at_budget"]["changes"])
    refusals = [entry["at_budget"] for entry in asked if not entry["at_budget"]["recomputable"]]
    lines = [
        f"## The same bundles at candidate budget {budget}",
        "",
        "The budget is a rank cutoff into the store's own cosine ordering, so a SMALLER one returns "
        "a prefix of the same ranking and the recorded prefix answers it exactly. A budget past the "
        "recorded prefix is refused rather than answered from a truncated list.",
        "",
        f"- attribution moves at this budget: {moved} of {len(asked)}",
        f"- refused (no record, or past the recorded prefix): {len(refusals)} of {len(asked)}",
        # The table says only "not recomputable", and the three ways a budget is refused have three
        # different fixes -- re-run the audit, raise `--effort`, raise the record cap. Counted per
        # reason rather than per bundle, because an archive sweep repeats each one many times.
        *(
            f"  - {reason} -- {sum(1 for r in refusals if r['reason'] == reason)} of "
            f"{len(refusals)}"
            for reason in sorted({str(refusal["reason"]) for refusal in refusals})
        ),
        "",
        "| run | pairs returned at this budget | attribution at this budget | moves |",
        "| --- | --- | --- | --- |",
    ]
    for entry in asked:
        at_budget = entry["at_budget"]
        reading = (
            _UNANSWERABLE
            if not at_budget["recomputable"]
            else stage_phrase(at_budget["recomputed"])
        )
        lines.append(
            f"| `{entry['label']}` | {pairs_phrase(at_budget)} | {reading} "
            f"| {_MOVES[at_budget['changes']]} |"
        )
    return lines


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
