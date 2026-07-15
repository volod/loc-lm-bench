"""Focused text analysis labels implementation."""

import logging
from collections.abc import Callable
from typing import Any
from llb.core.contracts.benchmarks import PlantedLabelRecord
from llb.prep.frontier import ground_span
from llb.prompts.registry import render_text
from llb.scoring import text_analysis_labels as ta

_LOG = logging.getLogger(__name__)

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
