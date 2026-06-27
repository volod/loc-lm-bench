"""Second-frontier cross-check (verified-data hardening / ontology-assisted drafting carry-over) -- the automated verified-data gate.

Every gold/eval item is AI-DRAFTED (frontier drafting / ontology-assisted drafting planter). Before a human stratified sample-verify
(human verification gate) flips it to `verified=true`, this module runs the in-pipeline cross-check the plan calls
"the verified-data gate": a SECOND, independent endpoint (different from the drafter) re-confirms
each item, layered on top of cheap deterministic pre-checks:

  1. GROUNDED        -- at least one labeled span still resolves in the source doc (`ground_span`);
  2. NON-CIRCULAR    -- the answer is not leaked in the question (normalized substring);
  3. SUPPORTED       -- (second frontier) the cited span actually supports the reference answer;
  4. ANSWERABLE      -- (second frontier) the question is sensible + answerable from the doc.

The second-frontier verifier is INJECTABLE (`SecondFrontierVerify`), so the gate is pure and
unit-tested without a key; `second_frontier_verify` builds the litellm-backed default. The
deterministic pre-checks run first so a clearly-broken item never spends a frontier call. Passing
the cross-check does NOT set `verified=true` (only human verification gate does) -- it produces the report a human
sample-verifies, and gates which items are even eligible.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.goldset.schema import GoldItem
from llb.prep.frontier import (
    LLMComplete,
    ProvenanceLog,
    ground_span,
    litellm_complete,
    parse_json_block,
)
from llb.rag.chunking import iter_docs
from llb.scoring.text_analysis import normalize_surface

_LOG = logging.getLogger(__name__)

# (supported, answerable, note) -- the second frontier's semantic verdict for one item.
SecondFrontierVerify = Callable[[GoldItem, str], tuple[bool, bool, str]]


@dataclass(frozen=True)
class CrossCheckVerdict:
    item_id: str
    grounded: bool
    non_circular: bool
    supported: bool
    answerable: bool
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.grounded and self.non_circular and self.supported and self.answerable


def is_grounded(item: GoldItem, doc_text: str) -> bool:
    """At least one labeled span still resolves (verbatim or normalized-but-exact) in the doc."""
    return any(ground_span(doc_text, span.text) is not None for span in item.source_spans)


def is_non_circular(item: GoldItem) -> bool:
    """The answer is not trivially leaked in the question (and neither side is empty)."""
    question = normalize_surface(item.question)
    answer = normalize_surface(item.reference_answer)
    if not question or not answer:
        return False
    return answer not in question


def verify_prompt(item: GoldItem, doc_text: str) -> str:
    """Ask the SECOND frontier model to confirm support + answerability (JSON verdict)."""
    span = item.source_spans[0].text if item.source_spans else ""
    return (
        "Ти незалежний перевіряючий якості оцінювальних даних. Підтверди два факти про пару "
        "питання-відповідь, спираючись на наведений документ.\n\n"
        f"Документ:\n{doc_text}\n\n"
        f"Питання: {item.question}\n"
        f"Відповідь: {item.reference_answer}\n"
        f"Цитата-підстава: {span}\n\n"
        'Поверни ЛИШЕ JSON {"supported": <true|false>, "answerable": <true|false>, '
        '"note": <короткий коментар>}, де supported = чи підтверджує цитата відповідь, '
        "answerable = чи є питання осмисленим і відповідним документу.\n"
    )


def second_frontier_verify(
    model: str, *, complete: LLMComplete | None = None, log: ProvenanceLog | None = None
) -> SecondFrontierVerify:
    """The litellm-backed default verifier (a DIFFERENT model from the drafter -- the egress is the
    human decision decision). The completion is injectable, so the gate is tested without a key."""
    completer = complete or litellm_complete(model, log=log)

    def verify(item: GoldItem, doc_text: str) -> tuple[bool, bool, str]:
        raw = completer(verify_prompt(item, doc_text))
        try:
            payload = parse_json_block(raw)
        except json.JSONDecodeError:
            return False, False, "unparseable verifier output"
        if not isinstance(payload, dict):
            return False, False, "non-object verifier output"
        return (
            bool(payload.get("supported", False)),
            bool(payload.get("answerable", False)),
            str(payload.get("note", "")),
        )

    return verify


def cross_check_item(
    item: GoldItem, doc_text: str, verify: SecondFrontierVerify
) -> CrossCheckVerdict:
    """Run the deterministic pre-checks, then (only if they pass) the second-frontier verifier."""
    grounded = is_grounded(item, doc_text)
    non_circular = is_non_circular(item)
    if not (grounded and non_circular):
        return CrossCheckVerdict(
            item.id, grounded, non_circular, False, False, "failed deterministic pre-check"
        )
    supported, answerable, note = verify(item, doc_text)
    return CrossCheckVerdict(item.id, grounded, non_circular, supported, answerable, note)


@dataclass
class CrossCheckReport:
    verdicts: list[CrossCheckVerdict]

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def passed_ids(self) -> set[str]:
        return {v.item_id for v in self.verdicts if v.passed}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": len(self.verdicts),
            "n_passed": self.n_passed,
            "verdicts": [
                {
                    "item_id": v.item_id,
                    "grounded": v.grounded,
                    "non_circular": v.non_circular,
                    "supported": v.supported,
                    "answerable": v.answerable,
                    "passed": v.passed,
                    "note": v.note,
                }
                for v in self.verdicts
            ],
        }


def load_doc_texts(corpus_root: Path | str) -> dict[str, str]:
    """Corpus docs keyed by BOTH relative path and bare stem (drafts may use either id form)."""
    texts: dict[str, str] = {}
    for rel, text in iter_docs(Path(corpus_root)):
        texts[rel] = text
        texts.setdefault(Path(rel).stem, text)
    return texts


def cross_check_goldset(
    items: list[GoldItem], doc_texts: dict[str, str], verify: SecondFrontierVerify
) -> CrossCheckReport:
    """Cross-check every drafted item against its source doc + the second-frontier verifier."""
    verdicts: list[CrossCheckVerdict] = []
    for item in items:
        doc_text = doc_texts.get(item.source_doc_id) or doc_texts.get(
            Path(item.source_doc_id).stem, ""
        )
        verdict = cross_check_item(item, doc_text, verify)
        if not verdict.passed:
            _LOG.info("[cross-check] %s did not pass: %s", item.id, verdict.note)
        verdicts.append(verdict)
    report = CrossCheckReport(verdicts=verdicts)
    _LOG.info(
        "[cross-check] %d/%d items passed the verified-data gate", report.n_passed, len(items)
    )
    return report
