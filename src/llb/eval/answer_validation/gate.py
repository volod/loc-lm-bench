"""Step two of the answer gate: does the declared answer contradict the accepted axioms or the
corpus ledger the retrieved context came from?

The check is one pass of the SHIPPED axiom checker over a merged ledger -- the corpus facts the
retrieved chunks carried, plus the answer's own declared triples. That merge is what makes one
mechanism answer both halves of the question the plan asks:

  - a violation whose facts are all the answer's is a self-contradiction (a claim outside a
    relation's declared range, one name asserted as two disjoint types, a relation asserted of
    itself);
  - a violation mixing an answer fact with a corpus fact is a CONTRADICTION OF THE LEDGER whose
    evidence is in the retrieved chunks -- exactly the case groundedness cannot see, because the
    answer's tokens really do appear in a chunk.

A violation the answer is not responsible for is dropped, and responsibility is decided by running
the SAME axiom over the scoped corpus ALONE first. Citing an answer fact is not enough: a bound of
two objects that the retrieved chunks already break with three of their own is broken whatever the
answer says, and a subject the corpus already gives two conflicting values has no single fact left
for an answer to contradict. In both cases the corpus is contradicting itself -- a data problem
`validate-ontology-axioms` reports -- and refusing the answer for it would blame the model for the
ledger. So a violation counts only when it cites an answer fact AND the same axiom holds at that
same anchor without the answer.

Only SIGNED axioms are ever enabled. An axiom is a domain claim a named reviewer dated, so an
unsigned file is refused with a named error rather than defaulted into an enabled constraint set.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from llb.core.contracts.rag import ChunkRecord
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.eval.answer_validation.answer_ledger import answer_extraction, declared_triples
from llb.eval.answer_validation.constants import (
    ANSWER_DOC_ID,
    EXCLUDED_GATE_KINDS,
    GATE_KINDS,
)
from llb.eval.answer_validation.models import GateVerdict
from llb.eval.answer_validation.scope import CorpusLedger, scoped_fact_count
from llb.prep.ontology.axioms.checker import check_axiom
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.loader import load_axioms
from llb.prep.ontology.axioms.models import Axiom, AxiomSet, Violation
from llb.prep.ontology.models import DocExtraction
from llb.prep.ontology.naming import normalize_name

_LOG = logging.getLogger(__name__)


def gate_axioms(axiom_set: AxiomSet) -> list[Axiom]:
    """The signed axioms whose class may refuse an answer, in file order."""
    return [axiom for axiom in axiom_set.signed if axiom.kind in GATE_KINDS]


class OntologyGate:
    """A signed axiom set plus the corpus ledger, applied to one declared answer at a time."""

    def __init__(self, axioms: Sequence[Axiom], ledger: CorpusLedger) -> None:
        self._axioms = list(axioms)
        self._ledger = ledger

    @property
    def axioms(self) -> list[Axiom]:
        return list(self._axioms)

    @property
    def kinds(self) -> list[str]:
        """The axiom classes this gate can refuse an answer with, sorted."""
        return sorted({axiom.kind for axiom in self._axioms})

    def check(self, envelope: AnswerEnvelope, chunks: Sequence[ChunkRecord]) -> GateVerdict:
        """Check one envelope's declared triples against the axioms and the retrieved ledger."""
        triples = declared_triples(envelope)
        scoped = self._ledger.scoped(chunks)
        if not triples:
            return GateVerdict(
                checked_claims=len(envelope.claims), scoped_facts=scoped_fact_count(scoped)
            )
        answer = answer_extraction(envelope, self._ledger.canonical)
        return GateVerdict(
            violations=_answer_violations(self._axioms, Ledger(scoped), Ledger([*scoped, answer])),
            checked_claims=len(envelope.claims),
            checked_triples=len(triples),
            scoped_facts=scoped_fact_count(scoped),
        )


def _answer_violations(axioms: Sequence[Axiom], corpus: Ledger, merged: Ledger) -> list[Violation]:
    """Every signed axiom the ANSWER broke, in a deterministic order.

    Each axiom is evaluated twice -- once over the retrieved corpus alone, once over the corpus
    plus the answer -- and an anchor the corpus ALREADY broke is subtracted. That subtraction is
    what keeps the gate measuring the answer: without it, a subject the retrieved chunks already
    over-fill or already disagree about refuses whatever the model says about it.
    """
    found: list[Violation] = []
    for axiom in axioms:
        already = {_anchor(v) for v in check_axiom(axiom, corpus).violations}
        found += [
            v
            for v in check_axiom(axiom, merged).violations
            if _implicates_answer(v) and _anchor(v) not in already
        ]
    found.sort(key=lambda violation: violation.key())
    return found


def _anchor(violation: Violation) -> tuple[str, str]:
    """The (axiom, subject) a violation is about -- what "the corpus already broke this" means.

    Deliberately coarser than `Violation.key()`, which includes the endpoints: adding one answer
    fact to an over-filled group CHANGES those endpoints, so keying on them would report every such
    violation as new.
    """
    return violation.axiom_id, normalize_name(violation.subject or "")


def _implicates_answer(violation: Violation) -> bool:
    """Whether the answer is one of the parties to this violation."""
    return any(fact.evidence.doc_id == ANSWER_DOC_ID for fact in violation.facts)


def load_gate(axioms_path: Path | str, extractions: Sequence[DocExtraction]) -> OntologyGate:
    """Build a gate from a signed axiom file and a loaded extraction ledger.

    Both refusals are NAMED: an axiom set nobody signed cannot gate an answer at all, and a signed
    set whose every axiom is of an excluded class would gate nothing while looking enabled.
    """
    axiom_set = load_axioms(axioms_path)
    signed = axiom_set.signed
    if not signed:
        raise SystemExit(
            f"the answer gate refuses {axioms_path}: none of its {len(axiom_set.axioms)} axioms "
            "is signed. An axiom is a domain claim a named reviewer accepted and dated "
            "(dcterms:creator + dcterms:date); sign the set before enabling it at answer time."
        )
    enabled = gate_axioms(axiom_set)
    if not enabled:
        raise SystemExit(
            f"the answer gate refuses {axioms_path}: all {len(signed)} signed axioms are of "
            f"classes the gate does not decide ({', '.join(EXCLUDED_GATE_KINDS)})."
        )
    skipped = len(signed) - len(enabled)
    _LOG.info(
        "[answer-gate] %d signed axioms enabled (%d signed of an excluded class, %d unsigned "
        "candidates not enabled) over %d ledger facts",
        len(enabled),
        skipped,
        len(axiom_set.axioms) - len(signed),
        sum(len(extraction.facts) for extraction in extractions),
    )
    return OntologyGate(enabled, CorpusLedger(extractions))


def load_gate_from_paths(axioms_path: Path | str, ledger_path: Path | str) -> OntologyGate:
    """The run-time entry point: a signed axiom file plus an `extraction.jsonl` (or bundle dir)."""
    from llb.graph.ingest import load_extractions
    from llb.prep.ontology.axioms.run import resolve_ledger_path

    return load_gate(axioms_path, load_extractions(resolve_ledger_path(Path(ledger_path))))
