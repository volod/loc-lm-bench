"""The declared-answer columns of one score row: the envelope contract, then the ontology gate.

Split out of `cases.py` along the subject seam -- everything here reads state a DECLARED answer
produced, and every column is present only when that lane ran. That is what keeps a free-text
bundle and an ungated envelope bundle each exactly the shape they had, so two validation lanes stay
comparable column by column.
"""

from llb.core.contracts.results import CaseScoreRow
from llb.eval.answer_envelope.models import AnswerEnvelope
from llb.eval.graph_contracts import RagState


def attach_envelope_columns(
    row: CaseScoreRow, state: RagState, envelope: AnswerEnvelope | None
) -> None:
    """Attach the declared-answer columns (typed-rag-answer-envelope) to `row`.

    Present only on an envelope-format run, so every bundle recorded with the envelope off keeps
    exactly the shape it had. `envelope_status` is the parse verdict, `repaired` says the bounded
    reprompt was spent (which makes first-attempt conformance readable as `1 - repair_rate`), and
    `n_claims` / `envelope_abstained` are read straight off the declaration.
    """
    if "envelope_status" not in state:
        return
    row["envelope_status"] = str(state["envelope_status"])
    row["repaired"] = bool(state.get("envelope_repaired", False))
    row["n_claims"] = len(envelope.claims) if envelope is not None else 0
    row["envelope_abstained"] = bool(envelope.abstained) if envelope is not None else False
    _attach_validation_columns(row, state)


def _attach_validation_columns(row: CaseScoreRow, state: RagState) -> None:
    """Attach the step-two gate columns (ontology-validated-answer-gate) to `row`.

    Present only when the gate RAN, so a `pydantic` lane bundle keeps exactly the shape it had and
    the two lanes stay comparable column by column. `validation_checked_triples` is the population
    the verdict rests on: an envelope that declared no triple was unchecked, not cleared.
    """
    if "validation_checked_triples" not in state:
        return
    row["validation_checked_triples"] = int(state["validation_checked_triples"])
    row["validation_violations"] = int(state.get("validation_violations", 0))
    row["validation_classes"] = [str(name) for name in state.get("validation_classes", [])]
    row["validation_axioms"] = [str(name) for name in state.get("validation_axioms", [])]
    row["validation_repaired"] = bool(state.get("validation_repaired", False))


def declared_envelope(state: RagState) -> AnswerEnvelope | None:
    """The validated envelope this case declared, if it produced one.

    The state carries it as a plain dict (the durability journal serializes state to JSON), so it
    is revalidated here through the same contract that admitted it at the generation boundary.
    """
    payload = state.get("envelope")
    return AnswerEnvelope.model_validate(payload) if payload is not None else None
