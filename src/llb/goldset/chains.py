"""Canonical chain-of-questions schema and validation.

Chain items are separate from flat ``GoldItem`` rows: each chain has 2-4 ordered steps, and
each step carries its own question, answer, dependency note, and exact source spans. The
validator keeps the same corpus-offset discipline as ``validate-goldset`` while adding the
chain-specific checks the context-policy benchmark needs.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from llb.core.contracts.common import ValidationReport
from llb.goldset.schema import Provenance, SourceSpan, Split

CHAINS_FILENAME = "chains.jsonl"
MIN_CHAIN_STEPS = 2
MAX_CHAIN_STEPS = 4


class ChainStep(BaseModel):
    """One step in an ordered chain-of-questions item."""

    order: int = Field(ge=1)
    question: str
    reference_answer: str
    source_doc_id: str
    source_spans: list[SourceSpan]
    dependency_note: str = ""

    @model_validator(mode="after")
    def _check_step(self) -> "ChainStep":
        if not self.question.strip():
            raise ValueError("chain step question is required")
        if not self.reference_answer.strip():
            raise ValueError("chain step reference_answer is required")
        if not self.source_doc_id.strip():
            raise ValueError("chain step source_doc_id is required")
        if not self.source_spans:
            raise ValueError("chain step requires at least one source span")
        if self.source_spans[0].doc_id != self.source_doc_id:
            raise ValueError("chain step source_doc_id must match the primary span doc_id")
        return self


class ChainItem(BaseModel):
    """One human-reviewable chain-of-questions item."""

    chain_id: str
    lang: str = "uk"
    steps: list[ChainStep]
    provenance: Provenance = "ontology-drafted"
    verified: bool = False
    split: Split = "final"

    @model_validator(mode="after")
    def _check_chain(self) -> "ChainItem":
        if not self.chain_id.strip():
            raise ValueError("chain_id is required")
        if not (MIN_CHAIN_STEPS <= len(self.steps) <= MAX_CHAIN_STEPS):
            raise ValueError(
                f"chain must have {MIN_CHAIN_STEPS}-{MAX_CHAIN_STEPS} steps, has {len(self.steps)}"
            )
        orders = [step.order for step in self.steps]
        if orders != list(range(1, len(self.steps) + 1)):
            raise ValueError(f"chain step orders must be 1..n, got {orders}")
        return self


ChainKind = Literal["goldset", "chains"]


def load_chains(path: Path | str) -> list[ChainItem]:
    """Load + schema-validate a JSONL chain set. Raises ValueError with line context."""
    path = Path(path)
    chains: list[ChainItem] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chains.append(ChainItem.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid chain item: {exc}") from exc
    return chains


def dump_chains(chains: list[ChainItem], path: Path | str) -> None:
    """Write chain items as UTF-8 JSONL (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for chain in chains:
            out.write(json.dumps(chain.model_dump(), ensure_ascii=False) + "\n")


def chain_stratum_key(chain: ChainItem) -> str:
    """Sampling stratum for chain review: provenance x split x first source doc."""
    first_doc = chain.steps[0].source_doc_id if chain.steps else "(none)"
    return f"{chain.provenance}|{chain.split}|{first_doc}"


def _span_key(span: SourceSpan) -> tuple[str, int, int]:
    return (span.doc_id, span.char_start, span.char_end)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _get_corpus_text(
    corpus_root: Path,
    doc_id: str,
    cache: dict[str, str | None],
    errors: list[str],
    item_id: str,
) -> str | None:
    if doc_id in cache:
        return cache[doc_id]
    path = corpus_root / doc_id
    if not path.exists():
        errors.append(f"{item_id}: missing corpus doc {doc_id}")
        cache[doc_id] = None
        return None
    cache[doc_id] = path.read_text(encoding="utf-8")
    return cache[doc_id]


def _validate_span(item_id: str, span: SourceSpan, text: str, errors: list[str]) -> None:
    if span.char_end > len(text):
        errors.append(f"{item_id}: span out of range in {span.doc_id}")
        return
    got = text[span.char_start : span.char_end]
    if got != span.text:
        errors.append(f"{item_id}: span mismatch ({got!r} != {span.text!r})")


def _validate_chain_steps(
    chain: ChainItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    errors: list[str],
) -> None:
    """Per-step checks: dependency notes, span reuse, and span grounding against the corpus."""
    span_keys: set[tuple[str, int, int]] = set()
    for step in chain.steps:
        if step.order > 1 and not step.dependency_note.strip():
            errors.append(f"{chain.chain_id}: step {step.order} missing dependency_note")
        for span in step.source_spans:
            key = _span_key(span)
            if key in span_keys:
                errors.append(f"{chain.chain_id}: step {step.order} reuses span {key}")
            span_keys.add(key)
            text = _get_corpus_text(corpus_root, span.doc_id, cache, errors, chain.chain_id)
            if text is not None:
                _validate_span(chain.chain_id, span, text, errors)


def _validate_final_answer_progression(chain: ChainItem, errors: list[str]) -> None:
    """The final answer must not already be findable from the step-1 passage."""
    first_context = _normalize(" ".join(span.text for span in chain.steps[0].source_spans))
    final_answer = _normalize(chain.steps[-1].reference_answer)
    if final_answer and final_answer in first_context:
        errors.append(f"{chain.chain_id}: final answer is findable from step-1 passage")


def validate_chains(chains: list[ChainItem], corpus_root: Path) -> ValidationReport:
    """Return a report dict: {n, splits, errors}. Empty errors means PASS."""
    corpus_root = Path(corpus_root)
    errors: list[str] = []
    cache: dict[str, str | None] = {}
    seen: set[str] = set()
    splits: dict[str, int] = {}

    for chain in chains:
        if chain.chain_id in seen:
            errors.append(f"duplicate chain_id: {chain.chain_id}")
        seen.add(chain.chain_id)
        splits[chain.split] = splits.get(chain.split, 0) + 1
        _validate_chain_steps(chain, corpus_root, cache, errors)
        _validate_final_answer_progression(chain, errors)

    return {"n": len(chains), "splits": splits, "errors": errors}
