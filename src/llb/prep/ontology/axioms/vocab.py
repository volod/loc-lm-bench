"""The OWL/RDFS vocabulary the axiom set is written in, and the IRI naming that makes it stable.

One axiom class maps to exactly one standard construct, so the committed `.ttl` is ordinary OWL:
a reviewer who knows `owl:FunctionalProperty` needs nothing from this repository to read it. The
identity, the Ukrainian gloss, and the sign-off ride on a standard `owl:Axiom` annotation block
rather than on a private predicate, for the same reason.

IRI local names are transliterated ASCII (`llb:rel_diie`), never the Ukrainian surface: a relation
surface may carry spaces and apostrophes, which a Turtle prefixed name cannot. The surface itself
is preserved verbatim as the property's `rdfs:label`, which is what the checker matches on.
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

# A non-dereferenceable namespace under the RFC 2606 reserved `.example` TLD: the axiom set is a
# local artifact and must never look like it resolves to a published ontology.
NS = "https://loc-lm-bench.example/ontology/uk#"
OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
DCTERMS = "http://purl.org/dc/terms/"

PREFIXES: tuple[tuple[str, str], ...] = (
    ("llb", NS),
    ("owl", OWL),
    ("rdf", RDF),
    ("rdfs", RDFS),
    ("xsd", XSD),
    ("dcterms", DCTERMS),
)

RDF_TYPE = RDF + "type"
RDFS_LABEL = RDFS + "label"
RDFS_COMMENT = RDFS + "comment"
RDFS_DOMAIN = RDFS + "domain"
RDFS_RANGE = RDFS + "range"
RDFS_SUBCLASS_OF = RDFS + "subClassOf"
OWL_AXIOM = OWL + "Axiom"
OWL_THING = OWL + "Thing"
OWL_CLASS = OWL + "Class"
OWL_RESTRICTION = OWL + "Restriction"
OWL_ON_PROPERTY = OWL + "onProperty"
OWL_MAX_CARDINALITY = OWL + "maxCardinality"
OWL_UNION_OF = OWL + "unionOf"
OWL_DISJOINT_WITH = OWL + "disjointWith"
OWL_OBJECT_PROPERTY = OWL + "ObjectProperty"
OWL_ONTOLOGY = OWL + "Ontology"
OWL_VERSION_INFO = OWL + "versionInfo"
OWL_ANNOTATED_SOURCE = OWL + "annotatedSource"
OWL_ANNOTATED_PROPERTY = OWL + "annotatedProperty"
OWL_ANNOTATED_TARGET = OWL + "annotatedTarget"
DCTERMS_CREATOR = DCTERMS + "creator"
DCTERMS_DATE = DCTERMS + "date"
XSD_NON_NEGATIVE_INTEGER = XSD + "nonNegativeInteger"

ONTOLOGY_IRI = NS.rstrip("#")
RELATION_PREFIX = "rel_"

# axiom class <-> the property characteristic that expresses it (`rdf:type` axioms).
CHARACTERISTIC_BY_KIND: dict[str, str] = {
    FUNCTIONAL: OWL + "FunctionalProperty",
    INVERSE_FUNCTIONAL: OWL + "InverseFunctionalProperty",
    SYMMETRIC: OWL + "SymmetricProperty",
    ASYMMETRIC: OWL + "AsymmetricProperty",
    IRREFLEXIVE: OWL + "IrreflexiveProperty",
}
KIND_BY_CHARACTERISTIC: dict[str, str] = {v: k for k, v in CHARACTERISTIC_BY_KIND.items()}

# The remaining classes are expressed by the predicate of their annotated triple.
KIND_BY_PREDICATE: dict[str, str] = {
    RDFS_DOMAIN: DOMAIN,
    RDFS_RANGE: RANGE,
    OWL_DISJOINT_WITH: DISJOINT_TYPES,
    RDFS_SUBCLASS_OF: MAX_CARDINALITY,
}

# Ukrainian -> ASCII, the KMU 2010 romanization simplified to a context-free table (the
# word-initial `є/ї/й/ю/я` variants are dropped: this is an IRI slug, not a passport spelling).
# Cyrillic letters outside Ukrainian are mapped too, so a stray Russian surface still slugs.
_TRANSLIT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "h",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ie",
    "ж": "zh",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ь": "",
    "ю": "iu",
    "я": "ia",
    "ы": "y",
    "э": "e",
    "ъ": "",
    "ё": "e",
    "'": "",
    "’": "",
}


def slug(text: str) -> str:
    """A stable ASCII IRI local name for a Ukrainian relation surface."""
    out: list[str] = []
    for char in text.strip().casefold():
        if char in _TRANSLIT:
            out.append(_TRANSLIT[char])
        elif char.isascii() and (char.isalnum() or char == "_"):
            out.append(char)
        else:
            out.append("_")
    collapsed = "_".join(part for part in "".join(out).split("_") if part)
    return collapsed or "relation"


def type_iri(entity_type: str) -> str:
    """The class IRI for one member of the closed entity-type vocabulary."""
    return NS + entity_type


def type_name(iri: str) -> str:
    """The entity type behind a class IRI (inverse of `type_iri`)."""
    return iri[len(NS) :] if iri.startswith(NS) else iri


def compact(iri: str) -> str:
    """Render an IRI as a prefixed name when a declared prefix covers it."""
    for prefix, namespace in PREFIXES:
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace) :]}"
    return f"<{iri}>"
