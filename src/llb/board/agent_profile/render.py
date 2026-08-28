"""Render a composed profile as `agent_profile.json` and the markdown rationale.

The JSON is what a runtime reads; the markdown is what an operator reads before trusting it. Both
carry the same three things per field -- the value, where it came from, and what is missing -- so
neither can quietly say more than the other. Prose lives in `board.agent_profile.*` templates.
"""

from datetime import datetime

from llb.board.agent_profile.artifacts import age_days
from llb.board.agent_profile.model import (
    STATE_MEASURED,
    STATE_UNMEASURED,
    AgentProfile,
    ProfileField,
)
from llb.board.agent_profile.replay import replay_block, replay_commands
from llb.board.recommend.model import _md_table
from llb.core.contracts.common import JsonObject
from llb.prompts.registry import render_text

# The four states, in the order the summary counts them (best news first).
STATE_ORDER = ("measured", "demoted", "refused", "unmeasured")
# A lane reason can run to several hundred characters (a multiplicity-adjusted p-value list).
# The markdown table keeps the head; the JSON keeps every reason in full.
MAX_NOTE_CHARS = 220


def _t(name: str, **values: object) -> str:
    return render_text(f"board.agent_profile.{name}", values)


def _value_text(value: object) -> str:
    """One cell for a field value; a compound value (the loop policy) renders as `k=v` pairs."""
    if value is None:
        return "-"
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items() if v is not None)
    return str(value)


def _clip(text: str) -> str:
    """One table cell's worth of a lane reason, pointing at the JSON for the rest."""
    if len(text) <= MAX_NOTE_CHARS:
        return text
    return text[: MAX_NOTE_CHARS - 3].rstrip() + "... (full text in agent_profile.json)"


def _freshness(item: ProfileField, now: datetime) -> JsonObject:
    return {"measured_at": item.measured_at, "age_days": age_days(item.measured_at, now)}


def _field_payload(item: ProfileField, now: datetime) -> JsonObject:
    return {
        "value": item.value,
        "state": item.state,
        "lane": item.spec.lane,
        "depends_on": sorted(item.spec.depends),
        "evidence_path": item.evidence_path,
        "verdict": item.verdict,
        "uncertainty": item.uncertainty,
        "freshness": _freshness(item, now),
        "measured_against": item.measured_against,
        "notes": list(item.notes),
    }


def profile_payload(profile: AgentProfile) -> JsonObject:
    """The machine-readable profile: one entry per field, plus the drift findings and the replay."""
    now = datetime.fromisoformat(profile.generated_at)
    states = {state: 0 for state in STATE_ORDER}
    for item in profile.fields:
        states[item.state] = states.get(item.state, 0) + 1
    return {
        "generated_at": profile.generated_at,
        "anchor": {
            "resolved": profile.anchor.resolved,
            "model": profile.anchor.model,
            "corpus_root": profile.anchor.corpus_root,
            "retrieval_fingerprint": profile.anchor.retrieval_fingerprint,
        },
        "drift": {"store": profile.store_drift, "adapter": profile.adapter_drift},
        "states": states,
        "fields": {item.name: _field_payload(item, now) for item in profile.fields},
        "replay": replay_block(profile),
    }


def _field_rows(profile: AgentProfile, now: datetime) -> list[list[str]]:
    rows = []
    for item in profile.fields:
        age = age_days(item.measured_at, now)
        rows.append(
            [
                f"`{item.name}`",
                _value_text(item.value),
                item.state,
                item.verdict or "-",
                "-" if age is None else f"{age:.1f}d",
                f"`{item.evidence_path}`" if item.evidence_path else "-",
            ]
        )
    return rows


def _gap_rows(profile: AgentProfile) -> list[list[str]]:
    rows = []
    for item in profile.fields:
        if item.state == STATE_MEASURED:
            continue
        rows.append(
            [
                f"`{item.name}`",
                item.state,
                item.spec.lane,
                _clip("; ".join(item.notes) or item.spec.summary),
            ]
        )
    return rows


def _drift_lines(profile: AgentProfile) -> list[str]:
    lines: list[str] = []
    if profile.store_drift:
        named = ", ".join(
            f"{c['knob']} {c['measured']} -> {c['current']}" for c in profile.store_drift
        )
        lines += ["", _t("store_drift", named=named)]
    if profile.adapter_drift:
        from llb.board.agent_profile.model import FIELD_ADAPTER

        lines += [
            "",
            _t(
                "adapter_drift",
                adapter=profile.by_name(FIELD_ADAPTER).value,
                named="; ".join(profile.adapter_drift),
            ),
        ]
    return lines


def format_profile_md(profile: AgentProfile) -> str:
    """The operator rationale: what is recommended, on what evidence, and what is missing."""
    now = datetime.fromisoformat(profile.generated_at)
    anchor = profile.anchor
    lines = [
        "# Agent operating profile",
        "",
        f"Composed: {profile.generated_at}",
        "",
        (
            _t("anchor", model=anchor.model, corpus=anchor.corpus_root or "unrecorded")
            if anchor.resolved
            else _t("no_anchor")
        ),
        "",
        _t("intro"),
    ]
    lines += _drift_lines(profile)
    if not profile.measured():
        lines += ["", _t("all_unmeasured")]
    lines += [
        "",
        "## Fields",
        "",
        _t("fields_intro"),
        "",
        _md_table(
            ["field", "value", "state", "verdict", "age", "evidence"], _field_rows(profile, now)
        ),
    ]
    gaps = _gap_rows(profile)
    if gaps:
        lines += [
            "",
            "## Gaps",
            "",
            _t("gaps_intro"),
            "",
            _md_table(["field", "state", "lane", "why"], gaps),
        ]
    commands = replay_commands(profile)
    if commands:
        lines += ["", "## Replay", "", _t("replay_intro"), "", "```bash"]
        lines += commands
        lines += ["```"]
    notes = [
        f"- `{item.name}`: {_clip(note)}"
        for item in profile.fields
        if item.state != STATE_UNMEASURED
        for note in item.notes
    ]
    if notes:
        lines += ["", "## Per-field notes", ""] + notes
    return "\n".join(lines)
