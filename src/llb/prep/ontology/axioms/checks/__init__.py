"""The axiom-class dispatch table: one pure checker per class, grouped by what it reads.

Three families, each with its own shared body: `cardinality` (how many values an endpoint may
carry), `types` (what type an endpoint may carry -- the only classes that can be unchecked), and
`direction` (which way a relation may run). Adding a class means adding a checker to its family and
one row here; nothing else dispatches on the kind.
"""

from collections.abc import Callable

from llb.prep.ontology.axioms.checks.base import Outcome
from llb.prep.ontology.axioms.checks.cardinality import (
    check_functional,
    check_inverse_functional,
    check_max_cardinality,
)
from llb.prep.ontology.axioms.checks.direction import (
    check_asymmetric,
    check_irreflexive,
    check_symmetric,
)
from llb.prep.ontology.axioms.checks.types import (
    check_disjoint_types,
    check_domain,
    check_range,
)
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
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.models import Axiom

Check = Callable[[Axiom, Ledger], Outcome]

CHECKS: dict[str, Check] = {
    FUNCTIONAL: check_functional,
    INVERSE_FUNCTIONAL: check_inverse_functional,
    MAX_CARDINALITY: check_max_cardinality,
    DOMAIN: check_domain,
    RANGE: check_range,
    DISJOINT_TYPES: check_disjoint_types,
    SYMMETRIC: check_symmetric,
    ASYMMETRIC: check_asymmetric,
    IRREFLEXIVE: check_irreflexive,
}
