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
    """Whether the store `--store` names is the store this bundle's readings are about."""
    identity = entry["store_identity"]
    return f"[stage] {entry['label']}: {identity['detail']}"


def pairs_phrase(at_budget: JsonObject) -> str:
    """`5 of the run's 8 document pairs return` -- what the budget actually took away.

    The count is the reading the NAMED pair cannot give: on a corpus whose corpus-first lost pair is
    lost at every budget, the name is identical at every budget and the count is not.
    """
    returned, run = at_budget["document_pairs"], at_budget["run_document_pairs"]
    if returned is None:
        return "the returned pairs are not recomputable"
    return f"{returned} of the run's {run} document pairs return"


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
    """Which of these bundles were taken over the store `--store` names, and which were not.

    This is the one question the recorded store manifest is asked, and it is an equality test: a
    bundle's readings describe the store its run held, so a bundle taken over a different store
    answers correctly about something the operator is not looking at.
    """
    placed = [entry for entry in entries if "store_identity" in entry]
    comparable = [entry for entry in placed if entry["store_identity"]["comparable"]]
    same = [entry for entry in comparable if not entry["store_identity"]["changed"]]
    lines = [
        "## The store these bundles read",
        "",
        "Each run recorded a digest over the store manifest it read, so a bundle can be PLACED "
        "against a store on disk without keeping a second copy of that manifest. The answer is an "
        "equality test and nothing more: which documents changed is the store's own question, "
        "asked of `store_meta.json` by `llb refresh-index`, not of a finished audit.",
        "",
        f"- taken over this store: {len(same)} of {len(placed)}",
        f"- taken over a different store: {len(comparable) - len(same)} of {len(placed)}",
        f"- not comparable (no store identity recorded): {len(placed) - len(comparable)} of "
        f"{len(placed)}",
        "",
        "| run | this store |",
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
        "| run | document pairs returned | at this budget | moves |",
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
