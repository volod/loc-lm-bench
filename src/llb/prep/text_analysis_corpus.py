"""Synthetic text-analysis corpus planter (text analysis) -- richer per-kind planted labels.

`prepare-synthetic-corpus` historically planted QA-style `key_fact` labels only (one
question/answer/span triple per fact). This module extends the planter to the FULL text-analysis
sub-task taxonomy -- key_fact / entity / topic / trend / risk / decision / contradiction, plus the
judged narrative / insight -- each emitted as a structured `PlantedLabelRecord` the text analysis scorer
(`llb.scoring.text_analysis`) consumes.

Grounding discipline (so a label can never point at absent text): the planter quotes a verbatim
`evidence` span per label, which is re-grounded against the doc for EXACT offsets (reusing
`frontier.ground_span`). The grounded substring is added to the label's `aliases`, so the
verbatim form is always an accepted surface even when the canonical `value` is a paraphrase.
Quote-bearing kinds (`GROUNDED_REQUIRED_KINDS`) are dropped when their evidence is ungrounded;
analytical kinds (topic / trend / risk / decision / insight / narrative) keep an ungrounded label
(they are legitimately not verbatim) but then carry no offsets.

`litellm` stays lazy (the `[prep]` extra); the completion is injectable, so prompt building,
parsing, grounding, and direction backfill are pure and unit-tested without any network or key.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from llb.contracts import PlantedLabelRecord
from llb.prep.frontier import (
    LLMComplete,
    ProvenanceLog,
    ground_span,
    litellm_complete,
    parse_json_block,
)
from llb.prompts import render_text
from llb.scoring import text_analysis as ta

# A doc prompt builder: (topic, n_per_kind, kinds) -> planter prompt. Swappable so the chat-log
# planter reuses the same generate -> parse -> ground -> bundle flow with a chat-shaped prompt.
DocPromptBuilder = Callable[[str, int, tuple[str, ...]], str]

_LOG = logging.getLogger(__name__)

# Default kinds a synthetic text-analysis doc plants (the objective sub-tasks; judged kinds are
# opt-in because their headline is the gated judge, not the objective matcher).
DEFAULT_KINDS: tuple[str, ...] = (
    ta.KEY_FACT,
    ta.ENTITY,
    ta.TOPIC,
    ta.TREND,
    ta.RISK,
    ta.DECISION,
)
# Quote-bearing kinds: a label whose `evidence` is not grounded in the doc is dropped.
GROUNDED_REQUIRED_KINDS = frozenset({ta.KEY_FACT, ta.ENTITY, ta.CONTRADICTION})


def text_analysis_doc_prompt(topic: str, n_per_kind: int, kinds: tuple[str, ...]) -> str:
    """Ask the planter for a short UA factual doc plus `n_per_kind` planted labels per kind."""
    asks = "\n".join(f'  - "{kind}"' for kind in kinds)
    return render_text(
        "prep.text_analysis_corpus.doc",
        {"topic": topic, "n_per_kind": n_per_kind, "asks": asks},
    )


def _alias_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _get_kind(raw: dict[str, Any], doc_id: str) -> str | None:
    kind = str(raw.get("kind", "")).strip()
    if kind not in ta.ALL_KINDS:
        _LOG.warning("[plant] %s: drop label with unknown kind %r", doc_id, kind)
        return None
    return kind


def _get_grounded(
    ground_span: Callable[[str, str], tuple[int, str] | None],
    document: str,
    value: str,
    evidence: str,
    kind: str,
    doc_id: str,
) -> tuple[bool, tuple[int, str] | None, str | None]:
    # Ground the VALUE first (offsets then point at the value mention); fall back to the
    # verbatim EVIDENCE quote, whose grounded substring becomes an accepted alias.
    is_grounded = True
    grounded = ground_span(document, value)
    evidence_alias: str | None = None
    if grounded is None and evidence:
        grounded = ground_span(document, evidence)
        if grounded is not None:
            evidence_alias = grounded[1]
    if grounded is None and kind in GROUNDED_REQUIRED_KINDS:
        _LOG.warning(
            "[plant] %s: drop %s label with ungrounded value/evidence %r",
            doc_id,
            kind,
            (value or evidence)[:40],
        )
        is_grounded = False
    return is_grounded, grounded, evidence_alias


def _create_record(
    doc_id: str,
    kind: str,
    value: str,
    grounded: tuple[int, str] | None,
    evidence: str,
    evidence_alias: str | None,
    aliases: list[str],
    index: int,
    attrs: dict[str, Any],
) -> PlantedLabelRecord:
    record: PlantedLabelRecord = {
        "label_id": f"{doc_id}-{kind}-{index}",
        "kind": kind,
        "value": value,
        "scoring": "objective" if kind in ta.OBJECTIVE_KINDS else "judged",
    }
    if grounded is not None:
        start, exact_text = grounded
        record["doc_id"] = doc_id
        record["char_start"] = start
        record["char_end"] = start + len(exact_text)
        if evidence_alias and evidence_alias not in aliases and evidence_alias != value:
            aliases.append(evidence_alias)  # verbatim evidence is an accepted surface
    if aliases:
        record["aliases"] = aliases
    if kind == ta.TREND and "direction" not in attrs:
        inferred = ta.direction_of(f"{value} {evidence}")
        if inferred is not None:
            attrs["direction"] = inferred
    if attrs:
        record["attrs"] = attrs
    return record


def _append_record(
    records: list[PlantedLabelRecord],
    kind: str,
    raw: dict[str, Any],
    doc_id: str,
    document: str,
    index: int,
) -> bool:
    value = str(raw.get("value", "")).strip()
    if not value:
        return False
    aliases = _alias_list(raw.get("aliases"))
    attrs = dict(raw.get("attrs", {}) or {})
    evidence = str(raw.get("evidence", "")).strip()

    is_grounded, grounded, evidence_alias = _get_grounded(
        ground_span, document, value, evidence, kind, doc_id
    )
    if not is_grounded:
        return False

    record = _create_record(
        doc_id, kind, value, grounded, evidence, evidence_alias, aliases, index, attrs
    )
    records.append(record)
    return True


def plant_labels(
    doc_id: str, document: str, raw_labels: list[dict[str, Any]]
) -> list[PlantedLabelRecord]:
    """Turn the planter's raw label objects into grounded `PlantedLabelRecord`s.

    Unknown kinds and (for quote-bearing kinds) ungrounded evidence are dropped. A trend's
    `attrs.direction` is backfilled from its evidence/value when the planter omitted it.
    """
    records: list[PlantedLabelRecord] = []
    counters: dict[str, int] = {}
    for raw in raw_labels:
        kind = _get_kind(raw, doc_id)
        if kind is None:
            continue

        index = counters.get(kind, 0)
        if _append_record(records, kind, raw, doc_id, document, index):
            counters[kind] = index + 1

    return records


def prepare_text_analysis_corpus(
    topics: list[str],
    *,
    planter_model: str,
    judge_model: str,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    n_per_kind: int = 2,
    complete: LLMComplete | None = None,
    out_dir: Path | str | None = None,
    log: ProvenanceLog | None = None,
    prompt_builder: DocPromptBuilder = text_analysis_doc_prompt,
    provenance_kind: str = "synthetic-text-analysis",
) -> tuple[dict[str, str], list[PlantedLabelRecord]]:
    """Generate synthetic docs with structured per-kind planted labels. Planter MUST differ from
    the judge (a model grading answers it authored is circular).

    `prompt_builder` swaps the doc-generation prompt (the chat-log planter passes a chat-shaped one),
    `provenance_kind` tags the bundle. Writes (when `out_dir` is given) `corpus/<doc>.md`,
    `text_analysis_labels.jsonl` (`PlantedLabelRecord`s), and a `provenance.json` tagging
    `synthetic: true`.
    """
    if planter_model == judge_model:
        raise ValueError(
            "planter_model must differ from judge_model: a model must not grade answers it "
            "authored (planter != judge)."
        )
    unknown = [kind for kind in kinds if kind not in ta.ALL_KINDS]
    if unknown:
        raise ValueError(f"unknown text-analysis kinds: {unknown}")
    log = log if log is not None else ProvenanceLog()
    complete = complete or litellm_complete(planter_model, log=log)
    out_dir = Path(out_dir) if out_dir is not None else None
    corpus_dir = out_dir / "corpus" if out_dir is not None else None

    docs: dict[str, str] = {}
    records: list[PlantedLabelRecord] = []
    for i, topic in enumerate(topics):
        doc_id = f"synth-{i:03d}"
        raw = complete(prompt_builder(topic, n_per_kind, kinds))
        try:
            payload = parse_json_block(raw)
        except json.JSONDecodeError:
            _LOG.warning("[prepare-ta] unparseable completion for topic %r; skipping", topic)
            continue
        if not isinstance(payload, dict):
            _LOG.warning("[prepare-ta] expected a JSON object for topic %r; skipping", topic)
            continue
        document = str(payload.get("document", "")).strip()
        if not document:
            continue
        docs[doc_id] = document
        raw_labels = [entry for entry in payload.get("labels", []) if isinstance(entry, dict)]
        records += plant_labels(doc_id, document, raw_labels)
        if corpus_dir is not None:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")

    if out_dir is not None:
        labels_path = out_dir / "text_analysis_labels.jsonl"
        labels_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        (out_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "kind": provenance_kind,
                    "synthetic": True,
                    "planter_model": planter_model,
                    "judge_model": judge_model,
                    "kinds": list(kinds),
                    "n_docs": len(docs),
                    "n_labels": len(records),
                    "labels_by_kind": _count_by_kind(records),
                    "corpus_root": "corpus",
                    "cost": log.summary(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _LOG.info("[prepare-ta] %d docs, %d planted labels -> %s", len(docs), len(records), out_dir)
    return docs, records


def _count_by_kind(records: list[PlantedLabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    return cast(dict[str, int], counts)
