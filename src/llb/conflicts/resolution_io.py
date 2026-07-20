"""Resolution artifact IO, stale-finding validation, apply, and rollback."""

import hashlib
import json
from pathlib import Path

from llb.conflicts.constants import (
    CONFLICT_OVERLAY_FILE,
    EFFECT_REPORT_FILE,
    RESOLUTION_PLAN_FILE,
    REVIEW_RECORDS_FILE,
    SUMMARY_FILE,
)
from llb.conflicts.overlay import applied_overlay_path, overlay_from_plan
from llb.conflicts.resolution_policy import (
    ACTION_DROP_DUPLICATE,
    ACTION_KEEP_BOTH,
    STATUS_ACCEPTED,
    build_plan,
)
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text


def load_findings(path: Path | str) -> tuple[list[JsonObject], str]:
    findings_path = Path(path)
    raw = findings_path.read_bytes()
    rows: list[JsonObject] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{findings_path}:{line_no}: finding must be a JSON object")
        rows.append(payload)
    return rows, hashlib.sha256(raw).hexdigest()


def infer_corpus_root(findings_path: Path | str, explicit: Path | str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    summary_path = Path(findings_path).parent / SUMMARY_FILE
    if not summary_path.is_file():
        raise ValueError("--corpus is required when findings.jsonl has no adjacent summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    corpus_root = summary.get("corpus_root") if isinstance(summary, dict) else None
    if not isinstance(corpus_root, str) or not corpus_root:
        raise ValueError(f"{summary_path}: missing corpus_root")
    return Path(corpus_root)


def create_resolution_artifacts(
    findings_path: Path | str,
    out_dir: Path | str,
    *,
    policy: str,
    corpus_root: Path | str,
    reviewed: Path | str | None = None,
) -> tuple[JsonObject, JsonObject, dict[str, Path]]:
    findings, source_sha = load_findings(findings_path)
    plan = build_plan(findings, policy, str(corpus_root))
    plan["source_findings_sha256"] = source_sha
    if reviewed is not None:
        merge_review_decisions(plan, reviewed)
        plan["action_counts"] = _action_counts(plan)
    overlay = overlay_from_plan(plan)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "plan": out / RESOLUTION_PLAN_FILE,
        "overlay": out / CONFLICT_OVERLAY_FILE,
        "review": out / REVIEW_RECORDS_FILE,
        "effect": out / EFFECT_REPORT_FILE,
    }
    atomic_write_text(paths["plan"], _json(plan))
    atomic_write_text(paths["overlay"], _json(overlay))
    atomic_write_text(paths["review"], _review_jsonl(plan))
    return plan, overlay, paths


def install_overlay(corpus_root: Path | str, overlay: JsonObject, plan: JsonObject) -> Path:
    """Validate every referenced source span, then atomically install the overlay."""
    root = Path(corpus_root)
    if not root.is_dir():
        raise ValueError(f"corpus root not found: {root}")
    validate_plan_sources(root, plan)
    path = applied_overlay_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _json(overlay))
    return path


def rollback_overlay(corpus_root: Path | str) -> Path | None:
    path = applied_overlay_path(corpus_root)
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"refusing rollback: overlay path is not a file: {path}")
    path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass
    return path


def validate_plan_sources(corpus_root: Path, plan: JsonObject) -> None:
    """Refuse a stale audit rather than suppressing text at obsolete offsets."""
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        for side in ("a", "b"):
            ref = item.get(side)
            if not isinstance(ref, dict):
                raise ValueError(f"plan item {item.get('finding_id')} has no side {side}")
            _validate_ref(corpus_root, ref, str(item.get("finding_id")), str(item.get("tier", "")))


def _validate_ref(corpus_root: Path, ref: JsonObject, finding_id: str, tier: str) -> None:
    doc_id = ref.get("doc_id")
    if not isinstance(doc_id, str):
        raise ValueError(f"finding {finding_id}: missing doc_id")
    path = corpus_root / doc_id
    if not path.is_file():
        raise ValueError(f"finding {finding_id}: source document is missing: {doc_id}")
    text = path.read_text(encoding="utf-8")
    start, end = ref.get("char_start"), ref.get("char_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError(f"finding {finding_id}: invalid source span for {doc_id}")
    if end > len(text):
        raise ValueError(f"finding {finding_id}: source span exceeds {doc_id}")
    if tier in ("hash", "lexical") and (start != 0 or end != len(text)):
        raise ValueError(
            f"finding {finding_id}: whole-document source changed since audit: {doc_id}"
        )
    quoted = ref.get("text")
    if ref.get("offsets_exact", True) and isinstance(quoted, str) and text[start:end] != quoted:
        raise ValueError(f"finding {finding_id}: source text changed since audit: {doc_id}")


def merge_review_decisions(plan: JsonObject, path: Path | str) -> None:
    decisions = {
        str(row["finding_id"]): str(row.get("resolution_decision", ""))
        for row in _read_jsonl(path)
        if row.get("finding_id")
    }
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        decision = decisions.get(str(item.get("finding_id")))
        if decision == ACTION_KEEP_BOTH:
            item.update(action=ACTION_KEEP_BOTH, status=STATUS_ACCEPTED, target_side=None)
            item["target_doc_id"] = None
            item["rationale"] = "accepted human review decision"
        elif decision in ("drop_a", "drop_b"):
            side = decision[-1]
            ref = item.get(side)
            item.update(action=ACTION_DROP_DUPLICATE, status=STATUS_ACCEPTED, target_side=side)
            item["target_doc_id"] = ref.get("doc_id") if isinstance(ref, dict) else None
            item["rationale"] = "accepted human review decision"


def _read_jsonl(path: Path | str) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _review_jsonl(plan: JsonObject) -> str:
    items = plan.get("items")
    rows = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("status") != "review_required":
            continue
        rows.append(
            {
                "review_type": "corpus_conflict_resolution",
                "finding_id": item.get("finding_id"),
                "relation": item.get("relation"),
                "rationale": item.get("rationale"),
                "a": item.get("a"),
                "b": item.get("b"),
                "staleness": item.get("staleness"),
                "resolution_decision": "",
            }
        )
    return "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)


def _action_counts(plan: JsonObject) -> dict[str, int]:
    counts: dict[str, int] = {}
    items = plan.get("items")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            action = str(item.get("action", ""))
            counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


def _json(payload: JsonObject) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
