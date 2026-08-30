"""Compose the per-lane readings into ONE agent operating profile.

Three things happen here and nowhere else:

1. **The anchor.** The run-eval pick fixes the model, the corpus, and the store the profile claims
   to be a configuration FOR. Without a run-eval bundle there is no anchor, and nothing downstream
   can be cross-checked -- which is itself reported rather than papered over.
2. **The consistency guard.** A field measured against a different model, corpus, or store
   fingerprint is REFUSED, naming the axis that disagreed. Composing it would silently describe a
   configuration nobody ever ran.
3. **The staleness demotion.** A stale adapter or a store whose retrieval fingerprint moved since
   the runs were taken demotes every field that rests on it, naming the changed knob. The value
   stays visible -- an operator still needs to know what it WAS -- but it stops being a
   recommendation.
"""

from datetime import datetime, timezone
from pathlib import Path

from llb.board.agent_profile.model import (
    DEP_ADAPTER,
    DEP_STORE,
    FIELD_ADAPTER,
    FIELD_CONTEXT_POLICY,
    FIELD_MODEL,
    KEY_CORPUS,
    KEY_MODEL,
    KEY_STORE,
    PROFILE_FIELD_SPECS,
    STATE_DEMOTED,
    STATE_MEASURED,
    AgentProfile,
    Anchor,
    ProfileField,
)
from llb.board.agent_profile.sources_agent import (
    ADAPTER_NONE,
    adapter_field,
    context_policy_field,
    loop_policy_field,
    memory_routing_note,
)
from llb.board.agent_profile.sources_rag import (
    context_order_field,
    prompt_system_field,
    reranker_field,
    run_eval_fields,
)
from llb.board.recommend.model import Recommendation
from llb.core.contracts.common import JsonObject
from llb.finetune.registry.model import RETRIEVAL_FINGERPRINT_KEYS, VERDICT_STALE

# The profile renders in registry order, not in the order the lanes happened to be read.
_FIELD_ORDER = {spec.name: index for index, spec in enumerate(PROFILE_FIELD_SPECS)}


def build_agent_profile(
    data_dir: Path | str,
    rec: Recommendation | None,
    *,
    now: datetime | None = None,
    index_dir: Path | str | None = None,
) -> AgentProfile:
    """Assemble every field, then run the guard and the demotion over the assembled set.

    `index_dir` is the built store the drift check reads back; it defaults to wherever the run
    configuration says this data root's store lives, so an operator needs no argument.
    """
    root = Path(data_dir)
    moment = now or datetime.now(timezone.utc)
    fields = run_eval_fields(rec)
    # The anchor has to exist before the model-scoped lanes are read: three of them select their
    # bundle BY the recommended model, so a profile with no run-eval pick reads no model lane at all
    # rather than guessing which model's bundle to compose.
    anchor = _anchor(fields)
    fields += [
        reranker_field(root),
        context_order_field(root),
        prompt_system_field(root, anchor.model),
        adapter_field(root, anchor.model),
        context_policy_field(root, anchor.model),
        loop_policy_field(root, anchor.model),
    ]
    _attach_memory_routing(root, fields)
    fields.sort(key=lambda item: _FIELD_ORDER[item.name])
    profile = AgentProfile(generated_at=moment.isoformat(), anchor=anchor, fields=fields)
    _apply_consistency_guard(profile)
    _apply_store_drift(profile, root, index_dir)
    _apply_adapter_drift(profile)
    return profile


def _anchor(run_eval_fields_: list[ProfileField]) -> Anchor:
    """The model / corpus / store the profile is a configuration for, from the run-eval pick."""
    model_field = next(item for item in run_eval_fields_ if item.name == FIELD_MODEL)
    if model_field.state != STATE_MEASURED:
        return Anchor()
    against = model_field.measured_against
    fingerprint = against.get(KEY_STORE)
    return Anchor(
        model=str(model_field.value),
        corpus_root=_as_str(against.get(KEY_CORPUS)),
        retrieval_fingerprint=fingerprint if isinstance(fingerprint, dict) else None,
    )


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def _attach_memory_routing(root: Path, fields: list[ProfileField]) -> None:
    note = memory_routing_note(root)
    if note is None:
        return
    context_policy = next(item for item in fields if item.name == FIELD_CONTEXT_POLICY)
    context_policy.note(note)


def _apply_consistency_guard(profile: AgentProfile) -> None:
    """Refuse any field whose lane recorded a different model, corpus, or store fingerprint."""
    anchor = profile.anchor
    if not anchor.resolved:
        for item in profile.fields:
            if item.state == STATE_MEASURED:
                item.note("no run-eval anchor on this host -- cross-lane consistency unchecked")
        return
    for item in profile.fields:
        if item.state != STATE_MEASURED:
            continue
        for axis in _disagreements(item, anchor):
            item.refuse(axis)


def _disagreements(item: ProfileField, anchor: Anchor) -> list[str]:
    """Every consistency axis on which a field's own provenance contradicts the anchor.

    A lane that did not record an axis is not checked on it: silence is not a disagreement, and
    inventing one would refuse fields that are perfectly consistent.
    """
    against = item.measured_against
    out: list[str] = []
    if against.get(KEY_MODEL) not in (None, anchor.model):
        out.append(
            f"measured on model `{against[KEY_MODEL]}`, but the profile anchors on `{anchor.model}`"
        )
    if anchor.corpus_root and against.get(KEY_CORPUS) not in (None, anchor.corpus_root):
        out.append(
            f"measured against corpus `{against[KEY_CORPUS]}`, "
            f"but the profile anchors on `{anchor.corpus_root}`"
        )
    out += _fingerprint_disagreements(against.get(KEY_STORE), anchor.retrieval_fingerprint)
    return out


def _fingerprint_disagreements(recorded: object, anchor: JsonObject | None) -> list[str]:
    if not isinstance(recorded, dict) or not anchor:
        return []
    return [
        f"measured with retrieval {knob}=`{recorded[knob]}`, but the profile anchors on "
        f"`{anchor.get(knob)}`"
        for knob, _meta_key in RETRIEVAL_FINGERPRINT_KEYS
        if knob in recorded
        and recorded[knob] is not None
        and anchor.get(knob) is not None
        and recorded[knob] != anchor.get(knob)
    ]


def _holding(profile: AgentProfile, axis: str) -> list[ProfileField]:
    """Fields that depend on `axis` AND actually hold a value.

    A field with no value cannot be demoted -- it is already a gap, and telling an operator that
    their empty `context_order` rests on a moved store explains nothing about why it is empty.
    Both drift axes can fire on the same field, so a field already demoted still collects the
    second reason.
    """
    return [
        item
        for item in profile.fields
        if axis in item.spec.depends and item.state in (STATE_MEASURED, STATE_DEMOTED)
    ]


def _apply_store_drift(profile: AgentProfile, data_dir: Path, index_dir: Path | str | None) -> None:
    """Demote every store-dependent field when the built store's retrieval knobs have moved.

    The comparison reuses the adapter registry's retrieval-fingerprint axis, so a knob that makes
    an adapter stale makes the profile's retrieval fields stale by the same rule.
    """
    from llb.core.config import RunConfig
    from llb.finetune.registry.staleness import retrieval_fingerprint_for

    anchor = profile.anchor.retrieval_fingerprint
    # The store location is the run configuration's to decide, not this module's to restate.
    store = index_dir if index_dir is not None else RunConfig(data_dir=data_dir).index_dir()
    current = retrieval_fingerprint_for(store)
    if not anchor or current is None:
        return
    changed = [
        {"knob": knob, "measured": anchor.get(knob), "current": current.get(knob)}
        for knob, _meta_key in RETRIEVAL_FINGERPRINT_KEYS
        if anchor.get(knob) != current.get(knob)
    ]
    if not changed:
        return
    profile.store_drift = changed
    named = ", ".join(f"{c['knob']} {c['measured']} -> {c['current']}" for c in changed)
    for item in _holding(profile, DEP_STORE):
        item.demote(f"the built store's retrieval fingerprint changed since this run: {named}")


def _apply_adapter_drift(profile: AgentProfile) -> None:
    """Demote every adapter-dependent field when the recommended adapter is stale."""
    adapter = profile.by_name(FIELD_ADAPTER)
    if adapter.verdict != VERDICT_STALE or adapter.value in (None, ADAPTER_NONE):
        return
    profile.adapter_drift = list(adapter.notes)
    named = "; ".join(profile.adapter_drift) or "training provenance moved"
    for item in _holding(profile, DEP_ADAPTER):
        item.demote(f"adapter `{adapter.value}` is stale: {named}")
