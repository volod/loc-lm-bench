"""Vocabulary, lane labels, and artifact names for the ontology-validated answer gate.

Step one of the gate is the typed contract (`llb.eval.answer_envelope`): the completion either
parses into `AnswerEnvelope` or ends in a typed status. Step two is HERE: the envelope's declared
triples are checked against the accepted axiom set AND against the corpus ledger the retrieved
context came from. Everything the two steps are named by lives in this module so the lane runner,
the gate, the study, and the report read from one list.
"""

from llb.prep.ontology.axioms.constants import (
    ASYMMETRIC,
    DISJOINT_TYPES,
    DOMAIN,
    FUNCTIONAL,
    INVERSE_FUNCTIONAL,
    IRREFLEXIVE,
    MAX_CARDINALITY,
    RANGE,
    SYMMETRIC,
)

# --- the three compared lanes ------------------------------------------------------------------
# `off` is the shipped free-text path and must reproduce a recorded run bundle byte for byte;
# `pydantic` is step one alone; `pydantic+ontology` is the whole two-step gate.
LANE_OFF = "off"
LANE_PYDANTIC = "pydantic"
LANE_PYDANTIC_ONTOLOGY = "pydantic+ontology"
VALIDATION_LANES: tuple[str, ...] = (LANE_OFF, LANE_PYDANTIC, LANE_PYDANTIC_ONTOLOGY)

# --- the run-time gate setting ------------------------------------------------------------------
GATE_OFF = "off"
GATE_ONTOLOGY = "ontology"
GATE_MODES: tuple[str, ...] = (GATE_OFF, GATE_ONTOLOGY)

# --- which axiom classes may refuse an ANSWER ---------------------------------------------------
# Every class the ledger checker decides is a candidate EXCEPT `symmetric`. At the ledger,
# `owl:SymmetricProperty` is reported as the GAP it is -- the counterpart the corpus never
# asserted. An answer is never asked to state both directions of a symmetric relation, so enabling
# that class here would refuse correct one-way answers by construction; it is excluded for that
# stated reason, not for convenience, and the fixture carries a one-way symmetric answer as an
# adversarial ACCEPT case so the exclusion stays measured rather than assumed.
GATE_KINDS: tuple[str, ...] = (
    FUNCTIONAL,
    INVERSE_FUNCTIONAL,
    MAX_CARDINALITY,
    DOMAIN,
    RANGE,
    DISJOINT_TYPES,
    ASYMMETRIC,
    IRREFLEXIVE,
)
EXCLUDED_GATE_KINDS: tuple[str, ...] = (SYMMETRIC,)

# The synthetic document id the answer's own declared triples are grounded in. A merged ledger
# holds the corpus facts the retrieved chunks carried PLUS the answer's; a violation counts as the
# ANSWER's only when at least one of its facts cites this id, so a contradiction the corpus already
# had with itself never refuses an answer that did not make it.
ANSWER_DOC_ID = "answer://declared"

# --- artifact layout ----------------------------------------------------------------------------
METHOD_DIR = "answer-validation"  # $DATA_DIR/answer-validation/<run>/
REPORT_FILENAME = "report.md"
COMPARISON_FILENAME = "comparison.json"
RUN_NAME_PREFIX = "answer-validation"

# The committed adversarial fixture (project-relative), and the axiom set it is checked against.
FIXTURE_PATH = "samples/benchmarks/ontology_violations_uk.json"

# --- reading the gate ---------------------------------------------------------------------------
# The correctness signal a rejection is read against when deciding whether it was a CATCH or a
# FALSE REJECTION. `contains` is deliberate: the token-F1 objective mixes needle-finding with
# terseness, so a verbose but correct answer would inflate the false-rejection rate under it.
REFERENCE_CORRECT_COLUMN = "contains"
# How many violated axiom ids the repair reprompt names. All of them would let one badly-typed
# answer fill the reprompt; the first few name the constraints that actually broke.
MAX_REPORTED_VIOLATIONS = 3
