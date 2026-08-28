"""Agent-side profile fields: the adapter to load, the context-management policy, and the
controller loop policy -- plus the guard-dependent routing rule the memory boundary surface adds
to the context-policy field.

Same contract as the retrieval side: a lane that never ran leaves its field `unmeasured`.
"""

from pathlib import Path

from llb.board.agent_profile.artifacts import artifact_timestamp, newest_payload
from llb.board.agent_profile.model import (
    FIELD_ADAPTER,
    FIELD_CONTEXT_POLICY,
    FIELD_LOOP_POLICY,
    FIELD_SPECS_BY_NAME,
    KEY_MODEL,
    STATE_MEASURED,
    ProfileField,
)
from llb.bench.loop_policy.report import METHOD as LOOP_POLICY_METHOD
from llb.bench.memory.boundary.surface import METHOD as BOUNDARY_METHOD
from llb.core.contracts.common import JsonObject

# What the profile emits when no adapter is registered for the recommended base model. It is a
# MEASURED answer, not a gap: the registry is the authority on which adapters exist.
ADAPTER_NONE = "none"
# The sweep writes this beside every cell bundle's manifest (`loop_policy/persist.py`).
LOOP_RECOMMENDATION = "recommendation.json"


def _field(name: str) -> ProfileField:
    return ProfileField(FIELD_SPECS_BY_NAME[name])


def adapter_field(data_dir: Path, model: str | None) -> ProfileField:
    """The newest registered adapter for the recommended base model, or `none`.

    The registry's own `staleness()` verdict is carried through unchanged -- the profile never
    re-decides whether an adapter is current, it reports what the registry decided.
    """
    from llb.finetune.registry.io import load_registry, registry_path
    from llb.finetune.registry.staleness import staleness

    item = _field(FIELD_ADAPTER)
    path = registry_path(data_dir)
    if not Path(path).is_file():
        item.note("no adapter registry on this host")
        return item
    entries = load_registry(path)
    item.evidence_path = str(path)
    item.measured_at = artifact_timestamp(Path(path))
    item.state = STATE_MEASURED
    item.measured_against = {KEY_MODEL: model}
    matching = [entry for entry in entries.values() if model and entry.base_model == model]
    if not matching:
        item.value = ADAPTER_NONE
        item.verdict = "no_registered_adapter"
        item.note(
            f"the registry holds {len(entries)} adapter(s), none trained on `{model}`"
            if entries
            else "the registry is empty"
        )
        return item
    entry = max(matching, key=lambda e: e.recency)
    report = staleness(entry)
    item.value = entry.short_id
    item.verdict = report.verdict
    item.uncertainty = {"reasons": list(report.reasons), "n_candidates": len(matching)}
    for reason in report.reasons:
        item.note(reason)
    return item


def context_policy_field(data_dir: Path, model: str | None) -> ProfileField:
    """The context-management policy the agentic-context lane ranks first for the model.

    The value is the best-scoring policy; the verdict is that policy's own paired completion
    reading against the `full` baseline, so a policy that merely edged the baseline out on a point
    estimate does not read like a settled win.
    """
    from llb.board.agentic_context import load_agentic_context_records

    item = _field(FIELD_CONTEXT_POLICY)
    if model is None:
        item.note("no recommended model to read a context policy for")
        return item
    records = [r for r in load_agentic_context_records(data_dir) if r.model == model]
    if not records:
        item.note(f"no agentic-context bundle for `{model}`")
        return item
    best = max(records, key=lambda r: r.result.objective_score)
    item.value = best.policy
    item.state = STATE_MEASURED
    item.evidence_path = str(Path(best.run_dir) / "manifest.json")
    item.measured_at = best.created_at or artifact_timestamp(Path(best.run_dir) / "manifest.json")
    item.measured_against = {KEY_MODEL: best.model}
    stability = _completion_stability(Path(best.run_dir) / "manifest.json")
    item.verdict = str(stability.get("reading", "baseline")) if stability else "baseline"
    item.uncertainty = stability
    if len(records) == 1:
        item.note("only one policy was ever run against this model -- nothing was compared")
    return item


def _completion_stability(manifest_path: Path) -> JsonObject | None:
    """The bundle's own paired completion stability block against the `full` baseline."""
    from llb.board.agent_profile.artifacts import read_json

    payload = read_json(manifest_path)
    config = payload.get("config") if isinstance(payload, dict) else None
    paired = config.get("paired_vs_full") if isinstance(config, dict) else None
    completion = paired.get("completion") if isinstance(paired, dict) else None
    stability = completion.get("stability") if isinstance(completion, dict) else None
    return stability if isinstance(stability, dict) else None


def loop_policy_field(data_dir: Path, model: str | None) -> ProfileField:
    """The loop-policy cell the grid recommends: step budget plus the two handling policies."""
    item = _field(FIELD_LOOP_POLICY)
    found = newest_payload(Path(data_dir) / LOOP_POLICY_METHOD, LOOP_RECOMMENDATION)
    if found is None:
        item.note("no agentic loop-policy sweep has run on this host")
        return item
    path, payload = found
    if payload.get("max_steps") is None:
        item.note(f"{path} carries no recommended cell")
        return item
    item.value = {
        "max_steps": payload.get("max_steps"),
        "malformed_call_policy": payload.get("malformed_call_policy"),
        "repeated_call_policy": payload.get("repeated_call_policy"),
        "repeat_feedback_variant": payload.get("repeat_feedback_variant"),
    }
    item.state = STATE_MEASURED
    item.evidence_path = str(path)
    item.verdict = str(payload.get("verdict", "?"))
    paired = payload.get("paired_completion_vs_baseline")
    item.uncertainty = (
        paired.get("stability") if isinstance(paired, dict) and paired.get("stability") else paired
    )
    item.measured_at = artifact_timestamp(path)
    item.measured_against = {KEY_MODEL: payload.get("model")}
    reason = payload.get("reason")
    if reason:
        item.note(str(reason))
    if not payload.get("changes_shipped_defaults"):
        item.note("the sweep retained the shipped defaults")
    return item


def memory_routing_note(data_dir: Path) -> str | None:
    """The guard-dependent routing line for memory-dependent work, or None when unmapped here.

    Memory-dependent transcripts do not take their context policy from the shape-level comparison
    alone: the cap-fitting boundary surface measured WHERE compact stops repaying its summary call,
    and that crossover is relative to the prompt guard. It is attached to the context-policy field
    as a CONDITION on its value rather than as a second value competing with it.
    """
    found = newest_payload(Path(data_dir) / BOUNDARY_METHOD, "manifest.json")
    if found is None:
        return None
    path, payload = found
    config = payload.get("config")
    analysis = config.get("analysis") if isinstance(config, dict) else None
    if not isinstance(analysis, dict):
        return None
    rule = analysis.get("routing_rule")
    lines = [str(line) for line in rule] if isinstance(rule, list) else []
    if not lines:
        return None
    reading = analysis.get("surface_reading", "?")
    return f"memory-dependent routing ({reading}): {'; '.join(lines)} -- measured by {path}"
