"""Frontier-LLM data-prep utilities (M3.5) -- GPU-free, litellm-backed.

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
import re
from pathlib import Path
from typing import Any, Callable

from llb.goldset.schema import GoldItem, Provenance, SourceSpan, dump_goldset
from llb.goldset.splits import assign_splits

_LOG = logging.getLogger(__name__)

LLMComplete = Callable[[str], str]  # prompt -> raw completion text
PROVENANCE_DRAFTED: Provenance = "frontier-drafted"
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_block(text: str) -> Any:
    """Parse JSON from a completion, tolerating a ```json ... ``` fence or surrounding prose."""
    fenced = _JSON_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = min((i for i in (candidate.find("["), candidate.find("{")) if i >= 0), default=-1)
        end = max(candidate.rfind("]"), candidate.rfind("}"))
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def litellm_complete(model: str, temperature: float = 0.2) -> LLMComplete:
    """Default completion via litellm (needs the `[prep]` extra + a provider key in the env)."""

    def complete(prompt: str) -> str:
        from litellm import completion

        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return str(resp["choices"][0]["message"]["content"])

    return complete


# --- prepare_goldset ----------------------------------------------------------------------


def goldset_draft_prompt(doc_id: str, text: str, n: int) -> str:
    """Ask for `n` UA QA pairs grounded in `text`, each quoting an EXACT answer substring."""
    return (
        "Ти укладач набору запитань для оцінювання україномовних RAG-моделей.\n"
        f"З наведеного документа склади рівно {n} пар «запитання-відповідь» українською.\n"
        "Відповідь МАЄ бути дослівною підрядковою цитатою з документа (скопіюй точно).\n"
        'Поверни лише JSON-масив об\'єктів {"question": ..., "reference_answer": ..., '
        '"answer_span": ...}, де answer_span -- точна цитата з тексту.\n\n'
        f"Документ [{doc_id}]:\n{text}\n"
    )


def build_drafted_items(
    doc_id: str, doc_text: str, drafts: list[dict[str, Any]], split: str
) -> list[GoldItem]:
    """Turn raw drafts into GoldItems, dropping any whose answer span is not in the doc."""
    items: list[GoldItem] = []
    for i, draft in enumerate(drafts):
        span_text = str(draft.get("answer_span", "")).strip()
        question = str(draft.get("question", "")).strip()
        reference = str(draft.get("reference_answer", span_text)).strip()
        if not span_text or not question:
            continue
        start = doc_text.find(span_text)
        if start < 0:  # ungrounded -> never emit a label that points at absent text
            _LOG.warning("[prepare-goldset] drop ungrounded span in %s: %r", doc_id, span_text[:40])
            continue
        items.append(
            GoldItem(
                id=f"{doc_id}-draft-{i}",
                question=question,
                reference_answer=reference,
                source_doc_id=doc_id,
                source_spans=[
                    SourceSpan(
                        doc_id=doc_id,
                        char_start=start,
                        char_end=start + len(span_text),
                        text=span_text,
                    )
                ],
                provenance=PROVENANCE_DRAFTED,
                verified=False,
                split=split,  # type: ignore[arg-type]
            )
        )
    return items


def prepare_goldset(
    corpus_root: Path | str,
    *,
    model: str,
    n_per_doc: int = 3,
    complete: LLMComplete | None = None,
    out_path: Path | str | None = None,
    seed: int = 13,
) -> list[GoldItem]:
    """Draft a review-ready gold set from a corpus. Items are `verified=False` pending review."""
    complete = complete or litellm_complete(model)
    docs = sorted(Path(corpus_root).rglob("*.md")) + sorted(Path(corpus_root).rglob("*.txt"))
    if not docs:
        raise ValueError(f"no .md/.txt documents under {corpus_root}")

    drafted: list[GoldItem] = []
    for path in docs:
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        raw = complete(goldset_draft_prompt(doc_id, text, n_per_doc))
        try:
            parsed = parse_json_block(raw)
        except json.JSONDecodeError:
            _LOG.warning("[prepare-goldset] unparseable completion for %s; skipping", doc_id)
            continue
        drafted += build_drafted_items(doc_id, text, list(parsed), "final")

    splits = assign_splits([it.id for it in drafted], seed=seed)
    for it in drafted:
        it.split = splits[it.id]  # type: ignore[assignment]
    if out_path is not None:
        dump_goldset(drafted, out_path)
        _LOG.info("[prepare-goldset] drafted %d items -> %s", len(drafted), out_path)
    return drafted


# --- prepare_synthetic_corpus -------------------------------------------------------------


def synthetic_doc_prompt(topic: str, n_labels: int) -> str:
    """Ask for a short UA factual doc on `topic` plus `n_labels` planted, span-grounded QA."""
    return (
        "Ти генеруєш синтетичний україномовний документ із контрольованими фактами для "
        "оцінювання RAG.\n"
        f"Напиши короткий фактичний документ (3-6 абзаців) на тему: {topic}.\n"
        f"Поряд сплануй рівно {n_labels} «закладені» факти у вигляді пар запитання-відповідь, "
        "де відповідь -- дослівна цитата з документа.\n"
        'Поверни лише JSON {"document": ..., "labels": [{"question": ..., '
        '"reference_answer": ..., "answer_span": ...}]}.\n'
    )


def prepare_synthetic_corpus(
    topics: list[str],
    *,
    planter_model: str,
    judge_model: str,
    n_labels: int = 3,
    complete: LLMComplete | None = None,
    out_dir: Path | str | None = None,
    seed: int = 13,
) -> tuple[dict[str, str], list[GoldItem]]:
    """Generate synthetic docs + planted-label gold items. Planter MUST differ from the judge."""
    if planter_model == judge_model:
        raise ValueError(
            "planter_model must differ from judge_model: a model must not grade answers it "
            "authored (planter != judge)."
        )
    complete = complete or litellm_complete(planter_model)
    out_dir = Path(out_dir) if out_dir is not None else None

    docs: dict[str, str] = {}
    items: list[GoldItem] = []
    for i, topic in enumerate(topics):
        doc_id = f"synth-{i:03d}"
        raw = complete(synthetic_doc_prompt(topic, n_labels))
        try:
            payload = parse_json_block(raw)
        except json.JSONDecodeError:
            _LOG.warning("[prepare-corpus] unparseable completion for topic %r; skipping", topic)
            continue
        document = str(payload.get("document", "")).strip()
        if not document:
            continue
        docs[doc_id] = document
        items += build_drafted_items(doc_id, document, list(payload.get("labels", [])), "final")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")

    splits = assign_splits([it.id for it in items], seed=seed)
    for it in items:
        it.split = splits[it.id]  # type: ignore[assignment]
    if out_dir is not None:
        dump_goldset(items, out_dir / "planted_labels.jsonl")
        (out_dir / "provenance.json").write_text(
            json.dumps({"planter_model": planter_model, "judge_model": judge_model}, indent=2),
            encoding="utf-8",
        )
        _LOG.info(
            "[prepare-corpus] %d docs, %d planted items -> %s", len(docs), len(items), out_dir
        )
    return docs, items
