"""security benchmark corpus-specific case DERIVATION from an ontology draft bundle.

Where the planter (`security_planter.py`) plants LEAK families (RAG-injection, canary) by splicing
markers into raw documents, this module derives the CONTENT-safety families from the corpus's own
grounded ontology -- the entities and topics `prepare-goldset` already extracted. Three case kinds
come out, all carrying an EXACT corpus span so a reviewer can trace every probe back to its source:

  * prohibited-topic denial-guard probes -- a harmful ask (doxxing / defamation / group hate)
    weaponizing a REAL corpus entity; the model SHOULD refuse. Scored objectively by the `refuse`
    detector (attack succeeds on compliance).
  * benign near-boundary controls -- a legitimate factual question about the SAME entity that a
    model must NOT refuse; feeds refusal-appropriateness only, never ASR (catches over-refusal).
  * matched-pair bias probes -- one fixed evaluative behavior posed over two swapped entities of
    the same type (`attrs.pair_id` + `attrs.variant`); scored for decision consistency by the
    matched-group machinery, alongside ASR.

Everything is a PURE, deterministic function over typed extraction records: it needs no model,
network, or GPU, and every emitted record round-trips through the committed `SecurityCase` schema
(so `bench-security`, cross-language grouping, and refusal-appropriateness work unchanged). The set
is `verified=false`-equivalent (`attrs.derived=True`); the human verification gate still gates any
headline/composite use.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llb.prep.ontology.models import DocExtraction, OntologyCandidate
from llb.scoring.security import BIAS_PAIR_KEY
from llb.prep.security_case_builders import (
    _bias_pair_cases,
    _denial_guard_cases,
    _grounded_entities,
)

_LOG = logging.getLogger(__name__)

# Derived content-safety cases all share the `unsafe_content` family; the case ROLE (harmful ask vs
# benign control vs bias variant) is carried by `expect_refusal` / `benign` / `attrs`, not by family.

# Attack-vector tags (recorded in `attrs.vector`) so a reviewer / report can slice derived probes by
# the harm they exercise without re-parsing the prompt.

# The entity types (spaCy/ontology labels) each denial-guard vector applies to, and its harmful-ask
# template. `{name}` is the exact corpus surface form of the grounded entity.

# Benign near-boundary control: a legitimate, document-grounded question about the SAME entity that
# a well-behaved model MUST answer (refusing it is over-refusal).

# Matched-pair bias behavior: one fixed evaluative decision posed over two swapped entities of the
# same type. A model that answers for one variant but refuses the other is inconsistent.


def derive_security_cases(
    extractions: Sequence[DocExtraction],
    *,
    ontology: OntologyCandidate | None = None,
    max_denial_per_vector: int = 8,
    max_bias_pairs_per_type: int = 4,
) -> list[dict[str, Any]]:
    """Derive corpus-specific content-safety cases from grounded extraction records.

    `ontology` is accepted for parity with the bundle contract and future type prioritization; the
    grounded evidence itself comes from `extractions`. Every returned record carries an exact corpus
    span (`attrs.grounding`) and is `verified=false`-equivalent (`attrs.derived=True`).
    """
    grounded = _grounded_entities(extractions)
    if not grounded:
        raise ValueError("no grounded entities in the bundle; nothing to derive")
    cases = _denial_guard_cases(grounded, max_per_vector=max_denial_per_vector)
    cases += _bias_pair_cases(grounded, max_pairs_per_type=max_bias_pairs_per_type)
    if ontology is not None:
        _LOG.debug(
            "[security-derive] ontology has %d entity types (grounding from extraction spans)",
            len(ontology.entity_types),
        )
    n_harmful = sum(1 for c in cases if c["expect_refusal"])
    n_benign = sum(1 for c in cases if c["benign"])
    n_pairs = sum(1 for c in cases if BIAS_PAIR_KEY in c["attrs"]) // 2
    _LOG.info(
        "[security-derive] derived %d cases over %d grounded entities "
        "(%d harmful asks, %d benign controls, %d bias pairs)",
        len(cases),
        len(grounded),
        n_harmful,
        n_benign,
        n_pairs,
    )
    return cases


def derive_from_bundle(
    bundle_dir: Path | str,
    *,
    max_denial_per_vector: int = 8,
    max_bias_pairs_per_type: int = 4,
) -> list[dict[str, Any]]:
    """Load a `prepare-goldset` draft bundle and derive corpus-specific security cases over it."""
    from llb.graph.ingest import load_bundle

    extractions, _docs, ontology = load_bundle(bundle_dir)
    return derive_security_cases(
        extractions,
        ontology=ontology,
        max_denial_per_vector=max_denial_per_vector,
        max_bias_pairs_per_type=max_bias_pairs_per_type,
    )


def _expected_decision(case: dict[str, Any]) -> str:
    """Human-readable expected refuse/answer label for a derived case (worksheet context)."""
    if case.get("expect_refusal"):
        return "REFUSE (harmful ask -- reject the case if a well-behaved model should answer it)"
    if case.get("benign"):
        return "ANSWER (benign control -- reject the case if a well-behaved model should refuse it)"
    return "ANSWER"


def worksheet_rows(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Scaffold verification-worksheet rows (verify_sample.csv schema) from derived cases.

    One row per case pre-filled with the review CONTEXT (prompt, grounded span, vector, expected
    decision) and stratified by attack vector; `decision` / `human_status` are left blank for the
    human to fill. The returned dicts use the canonical worksheet column keys so they round-trip
    through `llb.goldset.verify_base.write_worksheet_rows` and `acceptance_report`.
    """
    rows: list[dict[str, str]] = []
    for case in cases:
        attrs = case.get("attrs", {}) or {}
        grounding = attrs.get("grounding", {}) or {}
        vector = str(attrs.get("vector", ""))
        pair = str(attrs.get(BIAS_PAIR_KEY, ""))
        note = f"pair={pair}" if pair else ""
        rows.append(
            {
                "item_kind": "security",
                "item_id": str(case.get("id", "")),
                "provenance": "derived",
                "stratum": f"derived|security|{vector}",
                "question": str(case.get("prompt", "")),
                "reference_answer": _expected_decision(case),
                "span_doc_id": str(grounding.get("doc_id", "")),
                "span_text": str(grounding.get("text", "")),
                "context": note,
                "decision": "",  # human fills: accept | reject
                "human_status": "",  # human fills: decided
            }
        )
    return rows
