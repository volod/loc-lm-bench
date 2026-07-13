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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from llb.goldset.schema import GoldItem, Provenance, SourceSpan, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.prompts import render_text
from llb.rag.chunking.corpus import iter_docs

_LOG = logging.getLogger(__name__)

LLMComplete = Callable[[str], str]  # prompt -> raw completion text
PROVENANCE_DRAFTED: Provenance = "frontier-drafted"
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Markdown / PDF-extraction decoration that carries no semantic content -- dropped before matching
# so a clean drafted span still grounds against markdown-decorated corpus text (and vice versa).
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)  # e.g. per-page `<!-- source_pdf: ... -->`
_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)  # inline table `<br>` -> whitespace
_DROP_CHARS = set("*_#|•")  # bold/italic/heading/table-pipe/bullet markers
_DASHES = "–—―"  # en/em/horizontal-bar dashes -> ascii hyphen
_SPACE_BEFORE_PUNCT = ".,;:)"  # PDF extraction often inserts a space before these


# --- per-call cost / model provenance (frontier drafting) ----------------------------------------------


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class ProvenanceLog:
    """Accumulates one record per frontier call so a draft carries its provider/model/cost."""

    calls: list[CallRecord] = field(default_factory=list)

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float
    ) -> None:
        self.calls.append(CallRecord(model, prompt_tokens, completion_tokens, cost_usd))

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, int] = {}
        for call in self.calls:
            by_model[call.model] = by_model.get(call.model, 0) + 1
        return {
            "calls": len(self.calls),
            "models": sorted(by_model),
            "total_prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "total_completion_tokens": sum(c.completion_tokens for c in self.calls),
            "total_cost_usd": round(sum(c.cost_usd for c in self.calls), 6),
        }


# --- fuzzy-but-exact span grounding (frontier drafting) ------------------------------------------------


# Curly quote folding applied during normalization (grounding treats them as ASCII quotes).
_QUOTE_FOLD = {**{c: '"' for c in "“”„"}, **{c: "'" for c in "‘’‚‛"}}


class _Normalizer:
    """Char-by-char state machine behind `_normalize` (see its docstring for the rules)."""

    def __init__(self, masked: str) -> None:
        self.masked = masked
        self.out: list[str] = []
        self.index: list[int] = []
        self.prev_space = False
        self.i = 0

    @staticmethod
    def _is_dash(ch: str) -> bool:
        return ch in _DASHES or ch == "-"

    def _append_boundary_space(self, original_index: int) -> None:
        if not self.prev_space and self.out:
            self.out.append(" ")
            self.index.append(original_index)
            self.prev_space = True

    def _consume_dashes(self) -> None:
        """A single dash is kept; a `--` run (table rule / separator) becomes a word boundary."""
        run = 1
        while self.i + run < len(self.masked) and self._is_dash(self.masked[self.i + run]):
            run += 1
        if run >= 2:
            self._append_boundary_space(self.i)
            self.i += run
            return
        self.out.append("-")
        self.index.append(self.i)
        self.prev_space = False
        self.i += 1

    def _consume_char(self, ch: str) -> None:
        if ch in _SPACE_BEFORE_PUNCT and self.prev_space:
            self.out.pop()  # drop the space PDF extraction wedged before this punctuation
            self.index.pop()
        self.out.append(_QUOTE_FOLD.get(ch, ch).casefold())
        self.index.append(self.i)
        self.prev_space = False
        self.i += 1

    def run(self) -> tuple[str, list[int]]:
        while self.i < len(self.masked):
            ch = self.masked[self.i]
            if self._is_dash(ch):
                self._consume_dashes()
            elif ch in _DROP_CHARS:
                self.i += 1
            elif ch.isspace():
                self.i += 1
                self._append_boundary_space(self.i - 1)
            else:
                self._consume_char(ch)
        while self.out and self.out[-1] == " ":  # no trailing boundary space (offsets stay exact)
            self.out.pop()
            self.index.pop()
        return "".join(self.out), self.index


def _normalize(text: str) -> tuple[str, list[int]]:
    """Casefold, collapse whitespace, and drop markdown / PDF-extraction decoration, returning the
    normalized string and, for each normalized char, its index in the ORIGINAL text (so a match
    maps back to EXACT offsets). Dropping decoration lets a clean drafted span ground against
    markdown-decorated corpus text: `**bold**`, `#### headings`, `| table |` pipes, `<br>`, list
    bullets, HTML comments (per-page `<!-- source_pdf ... -->` markers), curly quotes/dashes, table
    `----` rules, and the stray space PDF extraction inserts before `.,;:)` are all normalized away."""
    # Blank out multi-char decoration first (keeping length so offsets stay aligned to the original).
    masked = _HTML_COMMENT.sub(lambda m: " " * len(m.group()), text)
    masked = _LINE_BREAK.sub(lambda m: " " * len(m.group()), masked)
    return _Normalizer(masked).run()


def ground_span(doc_text: str, span_text: str) -> tuple[int, str] | None:
    """Locate `span_text` in `doc_text`. Exact substring first; then a casefold/whitespace-
    normalized match mapped back to EXACT original offsets. Returns (start, exact_doc_substring)
    or None when ungrounded (so a label can never point at text that is not there)."""
    span_text = span_text.strip()
    if not span_text:
        return None
    exact = doc_text.find(span_text)
    if exact >= 0:
        return exact, span_text
    norm_doc, index = _normalize(doc_text)
    norm_span, _ = _normalize(span_text)
    if not norm_span:
        return None
    pos = norm_doc.find(norm_span)
    if pos < 0:
        return None
    start = index[pos]
    end = index[pos + len(norm_span) - 1] + 1  # inclusive last char -> exclusive end
    return start, doc_text[start:end]  # exact original substring (offsets stay exact)


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


def litellm_complete(
    model: str, temperature: float = 0.2, log: ProvenanceLog | None = None
) -> LLMComplete:
    """Default completion via litellm (needs the `[prep]` extra + a provider key in the env).
    When `log` is given, each call's model/tokens/cost is recorded for the draft provenance."""

    def complete(prompt: str) -> str:
        from litellm import completion, completion_cost

        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
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


def synthetic_doc_prompt(topic: str, n_labels: int) -> str:
    """Ask for a short UA factual doc on `topic` plus `n_labels` planted, span-grounded QA."""
    return render_text(
        "prep.frontier.synthetic_doc",
        {"topic": topic, "n_labels": n_labels},
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
    log: ProvenanceLog | None = None,
) -> tuple[dict[str, str], list[GoldItem]]:
    """Generate synthetic docs + planted-label gold items. Planter MUST differ from the judge.

    The corpus is written under `out_dir/corpus/` so `build-index --corpus-root <that>` can
    index it directly, and the planted labels under `out_dir/planted_labels.jsonl` -- a
    self-contained, explicitly-synthetic bundle for a separately reported scored run.
    """
    if planter_model == judge_model:
        raise ValueError(
            "planter_model must differ from judge_model: a model must not grade answers it "
            "authored (planter != judge)."
        )
    log = log if log is not None else ProvenanceLog()
    complete = complete or litellm_complete(planter_model, log=log)
    out_dir = Path(out_dir) if out_dir is not None else None
    corpus_dir = out_dir / "corpus" if out_dir is not None else None

    docs: dict[str, str] = {}
    items: list[GoldItem] = []
    for i, topic in enumerate(topics):
        doc_id = f"synth-{i:03d}"
        payload = _synthetic_doc_payload(complete, topic, n_labels)
        if payload is None:
            continue
        document = str(payload.get("document", "")).strip()
        if not document:
            continue
        docs[doc_id] = document
        labels = _object_list(payload.get("labels", []), source=doc_id)
        items += build_drafted_items(doc_id, document, labels, "final")
        if corpus_dir is not None:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")

    splits = assign_splits([it.id for it in items], seed=seed)
    for it in items:
        it.split = cast(Split, splits[it.id])
    if out_dir is not None:
        _write_synthetic_bundle(out_dir, docs, items, planter_model, judge_model, log)
    return docs, items


def _synthetic_doc_payload(
    complete: LLMComplete, topic: str, n_labels: int
) -> dict[str, Any] | None:
    """The planter's parsed JSON object for one topic, or None (logged) when unusable."""
    raw = complete(synthetic_doc_prompt(topic, n_labels))
    try:
        payload = parse_json_block(raw)
    except json.JSONDecodeError:
        _LOG.warning("[prepare-corpus] unparseable completion for topic %r; skipping", topic)
        return None
    if not isinstance(payload, dict):
        _LOG.warning("[prepare-corpus] expected a JSON object for topic %r; skipping", topic)
        return None
    return payload


def _write_synthetic_bundle(
    out_dir: Path,
    docs: dict[str, str],
    items: list[GoldItem],
    planter_model: str,
    judge_model: str,
    log: ProvenanceLog,
) -> None:
    """Persist the planted-labels goldset + `synthetic: true` provenance beside the corpus."""
    dump_goldset(items, out_dir / "planted_labels.jsonl")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "synthetic-planted",
                "synthetic": True,  # planted docs, NOT real corpus -- tag every scored run
                "planter_model": planter_model,
                "judge_model": judge_model,
                "n_docs": len(docs),
                "n_items": len(items),
                "corpus_root": "corpus",
                "cost": log.summary(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _LOG.info("[prepare-corpus] %d docs, %d planted items -> %s", len(docs), len(items), out_dir)
