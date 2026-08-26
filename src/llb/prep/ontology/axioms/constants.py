"""Vocabulary, artifact names, and defaults for the ontology axiom layer.

The induced ontology (`extraction/induce.py`) is a type INVENTORY: it says which entity and
relation types the extractor emitted and how often, and nothing in it can be violated. An AXIOM is
the other half -- a claim about the domain that a ledger can BREAK ("a patent has one duration",
"PERSON and ORG are different things"). Everything an axiom can say is named here so the checker,
the Turtle serializer, and the report read from one list.
"""

# --- axiom classes -----------------------------------------------------------------------------
# Each class maps to one standard OWL/RDFS construct (see `rdf.py`), so the committed constraint
# set is readable by anyone who reads OWL rather than only by this codebase.
FUNCTIONAL = "functional"  # owl:FunctionalProperty -- at most one object per subject
INVERSE_FUNCTIONAL = "inverse_functional"  # owl:InverseFunctionalProperty -- one subject per object
DOMAIN = "domain"  # rdfs:domain -- the subject's entity type must be in the allowed set
RANGE = "range"  # rdfs:range -- the object's entity type must be in the allowed set
DISJOINT_TYPES = "disjoint_types"  # owl:disjointWith -- one name cannot carry both types
SYMMETRIC = "symmetric"  # owl:SymmetricProperty -- an asserted edge needs its counterpart
ASYMMETRIC = "asymmetric"  # owl:AsymmetricProperty -- both directions cannot hold
IRREFLEXIVE = "irreflexive"  # owl:IrreflexiveProperty -- nothing relates to itself
MAX_CARDINALITY = "max_cardinality"  # owl:maxCardinality -- at most N objects per subject

AXIOM_KINDS: tuple[str, ...] = (
    FUNCTIONAL,
    INVERSE_FUNCTIONAL,
    DOMAIN,
    RANGE,
    DISJOINT_TYPES,
    SYMMETRIC,
    ASYMMETRIC,
    IRREFLEXIVE,
    MAX_CARDINALITY,
)

# Classes carried by a RELATION (the rest are carried by a pair of entity types).
RELATION_KINDS: tuple[str, ...] = tuple(k for k in AXIOM_KINDS if k != DISJOINT_TYPES)
# The classes whose OWL reading is a genuine INCONSISTENCY CONDITION, so an OWL 2 RL reasoner can
# be held to the same verdict (`crosscheck.py` evaluates each rule's antecedent over the closure).
# The rest are excluded for stated reasons, not for convenience:
#   - `domain` / `range`: OWL is open-world, so `rdfs:domain` ENTAILS the subject's type rather
#     than refusing a different one. Our reading is closed-world ("the asserted type must be in the
#     set"), which is the useful one for a ledger but is not what a reasoner computes.
#   - `symmetric`: `owl:SymmetricProperty` entails the missing counterpart; we report it as the
#     ledger gap it is, which again a reasoner fills rather than flags.
#   - `max_cardinality`: the OWL RL rule set covers cardinality 0 and 1 only (`cls-maxc1` /
#     `cls-maxc2`), so an N > 1 bound has no reasoner reading to agree with.
REASONER_KINDS: tuple[str, ...] = (
    FUNCTIONAL,
    INVERSE_FUNCTIONAL,
    ASYMMETRIC,
    IRREFLEXIVE,
    DISJOINT_TYPES,
)

# --- artifact layout ---------------------------------------------------------------------------
METHOD_DIR = "ontology-validation"  # $DATA_DIR/ontology-validation/<run>/
VIOLATIONS_FILENAME = "violations.jsonl"
REPORT_FILENAME = "report.md"
SUMMARY_FILENAME = "summary.json"
AXIOM_EVIDENCE_FILENAME = "axiom_evidence.jsonl"
AXIOMS_COPY_FILENAME = "axioms.ttl"

# The committed candidate set and its typed mirror (both under `samples/ontology/`).
SAMPLES_SUBDIR = "samples/ontology"
CANDIDATE_AXIOMS_TURTLE = "axioms_uk_v1.ttl"
CANDIDATE_AXIOMS_JSON = "axioms_uk_v1.json"

# --- report shaping ----------------------------------------------------------------------------
# How many supporting / contradicting examples per axiom the evidence rows carry. The sign-off
# reviewer reads examples, not the whole ledger; the counts beside them are the full population.
N_AXIOM_EXAMPLES = 3
# Violation rows printed in full (with both spans) in the Markdown report; the JSONL keeps all.
N_REPORT_VIOLATIONS = 20
# Evidence text longer than this is elided in the Markdown only -- the JSONL keeps exact spans.
EVIDENCE_PREVIEW_CHARS = 160

# --- optional cross-check extra ----------------------------------------------------------------
EXTRA_HINT = 'uv pip install -e ".[ontology]"'
REASONER_MISSING = (
    "the OWL reasoner cross-check needs the [ontology] extra (rdflib + owlrl). Run: " + EXTRA_HINT
)
