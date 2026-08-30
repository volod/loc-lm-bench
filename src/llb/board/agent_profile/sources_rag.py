"""Retrieval-side profile fields: the model/backend pick, its retrieval knobs, the reranker the
bake-off ranks first, the probed context order, and the prompt-system package.

Each reader returns a fully populated `ProfileField` when its lane ran on this host and an
`unmeasured` one when it did not. Nothing here invents a value: an absent lane is reported as a
gap, never as the default the code would have used anyway.
"""

from pathlib import Path

from llb.board.agent_profile.artifacts import artifact_timestamp, newest_payload
from llb.board.agent_profile.model import (
    FIELD_BACKEND,
    FIELD_CONTEXT_BUDGET,
    FIELD_CONTEXT_ORDER,
    FIELD_MODEL,
    FIELD_PROMPT_SYSTEM,
    FIELD_RERANKER,
    FIELD_TOP_K,
    FIELD_SPECS_BY_NAME,
    KEY_CORPUS,
    KEY_MODEL,
    KEY_STORE,
    STATE_MEASURED,
    ProfileField,
)
from llb.board.recommend.model import Recommendation
from llb.core.contracts.common import JsonObject
from llb.eval.position_probe import PROBE_METHOD
from llb.eval.position_probe_report import PROBE_JSON
from llb.finetune.registry.model import RETRIEVAL_FINGERPRINT_KEYS
from llb.rag.rerank_bakeoff.models import BAKEOFF_METHOD, BAKEOFF_REPORT_JSON


def _field(name: str) -> ProfileField:
    return ProfileField(FIELD_SPECS_BY_NAME[name])


def _run_fingerprint(config: JsonObject) -> JsonObject:
    """The store knobs a run bundle recorded, keyed exactly as the adapter registry keys them."""
    return {knob: config.get(knob) for knob, _meta_key in RETRIEVAL_FINGERPRINT_KEYS}


def run_eval_fields(rec: Recommendation | None) -> list[ProfileField]:
    """model / backend / top_k / context_budget from the host pick of the ranked cohort."""
    fields = [
        _field(FIELD_MODEL),
        _field(FIELD_BACKEND),
        _field(FIELD_TOP_K),
        _field(FIELD_CONTEXT_BUDGET),
    ]
    if rec is None:
        for item in fields:
            item.note("no final-split run-eval bundle on this host")
        return fields
    pick = rec.recommended_for_host
    config = pick.record.config if isinstance(pick.record.config, dict) else {}
    evidence = str(Path(pick.record.run_dir) / "manifest.json")
    row = next((r for r in rec.ranked if r["model"] == pick.model), None)
    verdict = "ranked_first" if row is not None and row.get("rank") == 1 else "host_fit_pick"
    against = {
        KEY_MODEL: pick.model,
        KEY_CORPUS: config.get("corpus_root"),
        KEY_STORE: _run_fingerprint(config),
    }
    values = {
        FIELD_MODEL: pick.model,
        FIELD_BACKEND: pick.result.backend,
        FIELD_TOP_K: config.get("top_k"),
        FIELD_CONTEXT_BUDGET: config.get("context_budget"),
    }
    uncertainty = (
        {"objective_ci": list(row["objective_ci"])} if row and "objective_ci" in row else None
    )
    for item in fields:
        value = values.get(item.name)
        if value is None:
            item.note(f"the winning cell's manifest records no `{item.name}`")
            continue
        item.value = value
        item.state = STATE_MEASURED
        item.evidence_path = evidence
        item.verdict = verdict
        item.uncertainty = uncertainty
        item.measured_at = pick.record.created_at or artifact_timestamp(Path(evidence))
        item.measured_against = dict(against)
    return fields


def reranker_field(data_dir: Path) -> ProfileField:
    """The cross-encoder the bake-off's adoption verdict names, or `none` when it retains no-rerank."""
    item = _field(FIELD_RERANKER)
    found = newest_payload(Path(data_dir) / BAKEOFF_METHOD, BAKEOFF_REPORT_JSON)
    if found is None:
        item.note("no reranker bake-off has run on this host")
        return item
    path, payload = found
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict) or not verdict.get("model"):
        item.note(f"{path} carries no adoption verdict")
        return item
    item.value = str(verdict["model"])
    item.state = STATE_MEASURED
    item.evidence_path = str(path)
    item.verdict = str(verdict.get("decision", "?"))
    item.uncertainty = _reranker_uncertainty(payload, item.value)
    item.measured_at = artifact_timestamp(path)
    item.measured_against = {
        KEY_CORPUS: payload.get("corpus_root"),
        KEY_STORE: {"embedding_model": payload.get("embedding_model")},
    }
    reason = verdict.get("reason")
    if reason:
        item.note(str(reason))
    return item


def _reranker_uncertainty(payload: JsonObject, model: str) -> JsonObject | None:
    """The named candidate's paired stability block -- the same shape every lane persists."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("model") != model:
            continue
        paired = candidate.get("paired_vs_baseline")
        metrics = paired.get("metrics") if isinstance(paired, dict) else None
        if not isinstance(metrics, dict):
            return None
        return {
            metric: block["stability"]
            for metric, block in metrics.items()
            if isinstance(block, dict) and isinstance(block.get("stability"), dict)
        }
    return None


def context_order_field(data_dir: Path) -> ProfileField:
    """The `context_order` the lost-in-the-middle probe recommends for the model it probed."""
    item = _field(FIELD_CONTEXT_ORDER)
    found = newest_payload(Path(data_dir) / PROBE_METHOD, PROBE_JSON)
    if found is None:
        item.note("no context-position probe has run on this host")
        return item
    path, payload = found
    recommendation = payload.get("recommendation")
    if not recommendation:
        item.note(f"{path} carries no context_order recommendation")
        return item
    item.value = str(recommendation)
    item.state = STATE_MEASURED
    item.evidence_path = str(path)
    item.verdict = str(payload.get("verdict", "?"))
    item.uncertainty = {"positions": payload.get("positions")}
    item.measured_at = artifact_timestamp(path)
    item.measured_against = {
        KEY_MODEL: payload.get("model"),
        KEY_STORE: payload.get("retrieval_fingerprint"),
    }
    note = payload.get("recommendation_note")
    if note:
        item.note(str(note))
    return item


def prompt_system_field(data_dir: Path, model: str | None) -> ProfileField:
    """The prompt-system id the RAG comparison ranks first for the recommended model."""
    from llb.board.prompt_systems import load_rag_prompt_system_records

    item = _field(FIELD_PROMPT_SYSTEM)
    if model is None:
        item.note("no recommended model to compare prompt systems for")
        return item
    records = [r for r in load_rag_prompt_system_records(data_dir) if r.model == model]
    if not records:
        item.note(f"no prompt-system-tagged run-eval bundle for `{model}`")
        return item
    best = max(records, key=lambda r: r.result.objective_score)
    item.value = best.prompt_system
    item.state = STATE_MEASURED
    item.evidence_path = str(Path(best.run_dir) / "manifest.json")
    item.verdict = "ranked_first" if len(records) > 1 else "single_candidate"
    item.uncertainty = {"objective": best.result.objective_score, "n_candidates": len(records)}
    item.measured_at = artifact_timestamp(Path(best.run_dir) / "manifest.json")
    item.measured_against = {KEY_MODEL: best.model}
    if len(records) == 1:
        item.note("only one prompt system was ever run against this model -- nothing was compared")
    return item
