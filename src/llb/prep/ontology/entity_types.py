"""The CLOSED entity-type vocabulary for the constrained ontology (MH.2-signed schema).

The graph ontology is meant to be a closed, reviewable type set, not whatever the extractor
invents. Two things make that true here:

  - the extractor is INSTRUCTED to use this vocabulary (`entity_types_prompt_block` is injected
    into `extraction_prompt`), and
  - every emitted/NER type is NORMALIZED back into it (`normalize_entity_type`): a synonym maps to
    its canonical type and anything out-of-vocabulary collapses to `MISC`, so the schema can never
    silently expand.

The set is an OntoNotes-derived granularity adapted to Ukrainian benchmark corpora (legal,
encyclopedic, news): beyond the generic PERSON/ORG/LOC it adds LAW (codes/treaties/conventions),
WORK (intellectual-property + creative objects), PRODUCT, NORP, DURATION (distinct from a DATE
point), MONEY, and QUANTITY -- the types that make a fact more granular. To extend it, edit this
one module (the prompt + the normalizer + the docs/tests all read from here).
"""

MISC = "MISC"

# (canonical type, short Ukrainian gloss). The order is the order shown to the extractor.
ENTITY_TYPE_GLOSSES: tuple[tuple[str, str], ...] = (
    ("PERSON", "особа: людина, автор, виконавець, власник"),
    ("NORP", "національність, етнічна, релігійна чи політична група"),
    ("ORG", "організація, установа, орган влади, компанія, суд"),
    ("LOC", "місце, географічний об'єкт, країна, юрисдикція"),
    ("LAW", "закон, кодекс, договір, конвенція, угода, нормативний акт"),
    (
        "WORK",
        "твір чи об'єкт інтелектуальної власності: книга, винахід, торговельна марка, зразок, ПЗ",
    ),
    ("PRODUCT", "продукт, товар, технологія, послуга"),
    ("EVENT", "подія, процес або процедура"),
    ("DATE", "дата або момент часу"),
    ("DURATION", "тривалість чи період часу, напр. двадцять років"),
    ("MONEY", "грошова сума"),
    ("QUANTITY", "вимірювана величина з одиницею або відсоток"),
    (MISC, "інше: абстрактне поняття, право, термін"),
)

# The closed vocabulary, in display order.
ENTITY_TYPES: tuple[str, ...] = tuple(name for name, _gloss in ENTITY_TYPE_GLOSSES)
DEFAULT_ENTITY_TYPE = MISC
_CANONICAL = frozenset(ENTITY_TYPES)

# Synonyms / common alternate labels (uppercased, underscores) -> canonical type. Covers what an
# LLM or a spaCy/Stanza pipeline (uk_core_news emits PER/ORG/LOC/MISC; OntoNotes adds GPE/FAC/...)
# tends to produce, so those collapse into the closed set instead of expanding it.
_SYNONYMS: dict[str, str] = {
    "PER": "PERSON",
    "PERS": "PERSON",
    "PEOPLE": "PERSON",
    "ROLE": "PERSON",
    "TITLE": "PERSON",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "COMPANY": "ORG",
    "INSTITUTION": "ORG",
    "AGENCY": "ORG",
    "GPE": "LOC",
    "FAC": "LOC",
    "FACILITY": "LOC",
    "GEO": "LOC",
    "PLACE": "LOC",
    "COUNTRY": "LOC",
    "CITY": "LOC",
    "LOCATION": "LOC",
    "NATIONALITY": "NORP",
    "RELIGION": "NORP",
    "STATUTE": "LAW",
    "CODE": "LAW",
    "TREATY": "LAW",
    "CONVENTION": "LAW",
    "AGREEMENT": "LAW",
    "ACT": "LAW",
    "REGULATION": "LAW",
    "WORK_OF_ART": "WORK",
    "WORKOFART": "WORK",
    "ARTWORK": "WORK",
    "INVENTION": "WORK",
    "TRADEMARK": "WORK",
    "PATENT": "WORK",
    "DESIGN": "WORK",
    "GOODS": "PRODUCT",
    "TECHNOLOGY": "PRODUCT",
    "SERVICE": "PRODUCT",
    "TIME": "DATE",
    "DATETIME": "DATE",
    "PERIOD": "DURATION",
    "AGE": "DURATION",
    "TERM": "DURATION",
    "CURRENCY": "MONEY",
    "PRICE": "MONEY",
    "COST": "MONEY",
    "PERCENT": "QUANTITY",
    "PERCENTAGE": "QUANTITY",
    "MEASURE": "QUANTITY",
    "MEASUREMENT": "QUANTITY",
    "CARDINAL": "QUANTITY",
    "ORDINAL": "QUANTITY",
    "NUMBER": "QUANTITY",
    "LANGUAGE": "MISC",
}


def normalize_entity_type(raw: str) -> str:
    """Map a raw extractor / NER type into the closed vocabulary (else `MISC`).

    Case-insensitive; spaces and hyphens fold to underscores so `"work of art"` -> `WORK`.
    An empty or unknown type becomes `MISC`, so the schema stays closed.
    """
    key = "_".join(raw.strip().upper().replace("-", "_").split())
    if key in _CANONICAL:
        return key
    return _SYNONYMS.get(key, DEFAULT_ENTITY_TYPE)


def entity_types_prompt_block() -> str:
    """The vocabulary (canonical name + Ukrainian gloss) injected into the extraction prompt."""
    return "; ".join(f"{name} -- {gloss}" for name, gloss in ENTITY_TYPE_GLOSSES)
