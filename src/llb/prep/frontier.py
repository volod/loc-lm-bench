"""Frontier-LLM data-prep utilities (frontier drafting) -- GPU-free, litellm-backed.

Two utilities, both producing UNVERIFIED material a human then reviews (only `verified=True`
items ever score a model):

  prepare_goldset           Draft (question, reference_answer, exact source span) triples from
                            real corpus docs -- a head start for the human gold-set author.
                            Every drafted span is re-grounded against the doc; a draft whose
                            quoted answer is not a verbatim substring is dropped, so a label
                            can never point at text that is not there.

  prepare_synthetic_corpus  Generate synthetic docs with STRUCTURED planted labels. The
                            planter model must NOT be the eval judge (a model grading answers it
                            authored is circular), so the planter id is recorded and a guard
                            rejects planter == judge.

`litellm` is imported lazily (the `[prep]` extra); the completion call is injectable, so the
prompt building, JSON parsing, span grounding, and split assignment are pure and unit-tested
without any network or key.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

from llb.goldset.schema import GoldItem, Provenance, SourceSpan, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.prep.frontier_telemetry import LLMComplete, ProvenanceLog
from llb.prompts.registry import render_text
from llb.rag.chunking.corpus import iter_docs
from llb.prep.frontier_parsing import ground_span, parse_json_block

_LOG = logging.getLogger(__name__)

PROVENANCE_DRAFTED: Provenance = "frontier-drafted"

# Markdown / PDF-extraction decoration that carries no semantic content -- dropped before matching
# so a clean drafted span still grounds against markdown-decorated corpus text (and vice versa).


# --- fuzzy-but-exact span grounding (frontier drafting) ------------------------------------------------


# Curly quote folding applied during normalization (grounding treats them as ASCII quotes).


def litellm_complete(
    model: str, temperature: float = 0.2, log: ProvenanceLog | None = None
) -> LLMComplete:
    """Default completion via litellm (needs the `[prep]` extra + a provider key in the env).
    When `log` is given, each call's model/tokens/cost is recorded for the draft provenance."""

    def complete(prompt: str) -> str:
        from litellm import completion, completion_cost

        started = time.monotonic()
        try:
            resp = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except Exception as exc:
            if log is not None:
                log.record(model, 0, 0, 0.0, latency_s=time.monotonic() - started, error=str(exc))
            raise
        if log is not None:
            usage = resp.get("usage", {}) or {}
            try:
                cost = float(completion_cost(resp))
            except Exception:  # cost unavailable for some providers -- record 0, keep going
                cost = 0.0
            log.record(
                model,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                cost,
                latency_s=time.monotonic() - started,
            )
        return str(resp["choices"][0]["message"]["content"])

    return complete


# --- prepare_goldset ----------------------------------------------------------------------


def goldset_draft_prompt(doc_id: str, text: str, n: int) -> str:
    """Ask for `n` UA QA pairs grounded in `text`, each quoting an EXACT answer substring."""
    return render_text(
        "prep.frontier.goldset_draft",
        {"doc_id": doc_id, "text": text, "n": n},
    )


def build_drafted_items(
    doc_id: str,
    doc_text: str,
    drafts: list[dict[str, Any]],
    split: Split,
    *,
    provenance: Provenance = PROVENANCE_DRAFTED,
    id_prefix: str = "draft",
) -> list[GoldItem]:
    """Turn raw drafts into GoldItems, dropping any whose answer span is not in the doc.

    `provenance` / `id_prefix` let other drafters (e.g. the ontology-assisted pipeline) reuse the
    exact-grounding + GoldItem construction while tagging their own provenance and id namespace.
    """
    items: list[GoldItem] = []
    for i, draft in enumerate(drafts):
        span_text = str(draft.get("answer_span", "")).strip()
        question = str(draft.get("question", "")).strip()
        reference = str(draft.get("reference_answer", span_text)).strip()
        if not span_text or not question:
            continue
        grounded = ground_span(doc_text, span_text)  # exact, then normalized-but-exact fallback
        if grounded is None:  # ungrounded -> never emit a label that points at absent text
            _LOG.warning("[prepare-goldset] drop ungrounded span in %s: %r", doc_id, span_text[:40])
            continue
        start, exact_text = grounded
        items.append(
            GoldItem(
                id=f"{doc_id}-{id_prefix}-{i}",
                question=question,
                reference_answer=reference,
                source_doc_id=doc_id,
                source_spans=[
                    SourceSpan(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=start + len(exact_text),
                        text=exact_text,
                    )
                ],
                provenance=provenance,
                verified=False,
                split=split,
            )
        )
    return items


def _object_list(value: Any, *, source: str) -> list[dict[str, Any]]:
    """Keep object entries from an LLM JSON array; malformed shapes are skipped safely."""
    if not isinstance(value, list):
        _LOG.warning("[prepare] expected a JSON array for %s; skipping", source)
        return []
    objects = [entry for entry in value if isinstance(entry, dict)]
    if len(objects) != len(value):
        _LOG.warning(
            "[prepare] ignored %d non-object entries for %s", len(value) - len(objects), source
        )
    return objects


def prepare_goldset(
    corpus_root: Path | str,
    *,
    model: str,
    n_per_doc: int = 3,
    complete: LLMComplete | None = None,
    out_path: Path | str | None = None,
    seed: int = 13,
    log: ProvenanceLog | None = None,
) -> list[GoldItem]:
    """Draft a review-ready gold set from a corpus. Items are `verified=False` pending review."""
    log = log if log is not None else ProvenanceLog()
    complete = complete or litellm_complete(model, log=log)
    docs = list(iter_docs(Path(corpus_root)))
    if not docs:
        raise ValueError(f"no .md/.txt documents under {corpus_root}")

    drafted: list[GoldItem] = []
    for doc_id, text in docs:
        raw = complete(goldset_draft_prompt(doc_id, text, n_per_doc))
        try:
            parsed = parse_json_block(raw)
        except json.JSONDecodeError:
            _LOG.warning("[prepare-goldset] unparseable completion for %s; skipping", doc_id)
            continue
        drafted += build_drafted_items(doc_id, text, _object_list(parsed, source=doc_id), "final")

    splits = assign_splits([it.id for it in drafted], seed=seed)
    for it in drafted:
        it.split = cast(Split, splits[it.id])
    if out_path is not None:
        out_path = Path(out_path)
        dump_goldset(drafted, out_path)
        provenance = {
            "kind": "frontier-drafted",
            "synthetic": False,  # drafted FROM a real corpus (vs planted synthetic docs)
            "model": model,
            "n_items": len(drafted),
            "cost": log.summary(),
        }
        out_path.with_suffix(".provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _LOG.info("[prepare-goldset] drafted %d items -> %s", len(drafted), out_path)
    return drafted


# --- prepare_synthetic_corpus -------------------------------------------------------------
