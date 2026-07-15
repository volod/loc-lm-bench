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
from collections.abc import Callable
from pathlib import Path
from typing import cast

from llb.core.contracts.benchmarks import PlantedLabelRecord
from llb.prep.frontier import litellm_complete, parse_json_block
from llb.prep.frontier_telemetry import LLMComplete, ProvenanceLog
from llb.scoring import text_analysis_labels as ta
from llb.prep.text_analysis_labels import _LOG, plant_labels, text_analysis_doc_prompt

# A doc prompt builder: (topic, n_per_kind, kinds) -> planter prompt. Swappable so the chat-log
# planter reuses the same generate -> parse -> ground -> bundle flow with a chat-shaped prompt.
DocPromptBuilder = Callable[[str, int, tuple[str, ...]], str]


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


def _make_corpus_dir(
    corpus_dir: Path | None,
    docs: dict[str, str],
    records: list[PlantedLabelRecord],
    complete: LLMComplete,
    prompt_builder: DocPromptBuilder,
    topics: list[str],
    n_per_kind: int,
    kinds: tuple[str, ...],
) -> None:
    for i, topic in enumerate(topics):
        doc_id = f"synth-{i:03d}"
        raw = complete(prompt_builder(topic, n_per_kind, kinds))
        try:
            payload = parse_json_block(raw)
        except json.JSONDecodeError:
            _LOG.warning("[prepare-ta] unparseable completion for topic %r; skipping", topic)
            return
        if not isinstance(payload, dict):
            _LOG.warning("[prepare-ta] expected a JSON object for topic %r; skipping", topic)
            return
        document = str(payload.get("document", "")).strip()
        if not document:
            return
        docs[doc_id] = document
        raw_labels = [entry for entry in payload.get("labels", []) if isinstance(entry, dict)]
        records += plant_labels(doc_id, document, raw_labels)
        if corpus_dir is not None:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")


def _write_provenance(
    out_dir: Path | None,
    docs: dict[str, str],
    records: list[PlantedLabelRecord],
    planter_model: str,
    judge_model: str,
    kinds: tuple[str, ...],
    log: ProvenanceLog,
    provenance_kind: str,
) -> None:
    if out_dir is None:
        return
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
    _make_corpus_dir(corpus_dir, docs, records, complete, prompt_builder, topics, n_per_kind, kinds)

    _write_provenance(
        out_dir, docs, records, planter_model, judge_model, kinds, log, provenance_kind
    )

    return docs, records


def _count_by_kind(records: list[PlantedLabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    return cast(dict[str, int], counts)
