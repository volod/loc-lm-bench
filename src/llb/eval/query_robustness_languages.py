"""Committed drafted language variants for the paired query-robustness lane."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.goldset.schema import GoldItem, load_goldset
from llb.prep.ontology.constants import UKRAINIAN_MIN_CYRILLIC_FRACTION
from llb.rag.query_prep.base import QueryEdit, QueryPrepResult

LANGUAGE_RU = "language_ru"
LANGUAGE_MIXED = "language_mixed"
LANGUAGE_VARIANT_CLASSES = (LANGUAGE_RU, LANGUAGE_MIXED)
LANGUAGE_FIXTURE_SUFFIX = "_ru"
VARIANT_ID_SEPARATOR = "--"
_EXPECTED_LANG = {LANGUAGE_RU: "ru", LANGUAGE_MIXED: "uk-ru"}
FIXTURE_TRANSLATE_STEP = "fixture_translate"
FIXTURE_DRAFTED = "drafted"
FIXTURE_VERIFIED = "verified"


def infer_language_fixture(goldset_path: Path) -> Path:
    """Resolve the conventional sibling drafted overlay for a baseline gold set."""
    root = goldset_path.parent
    return root.with_name(f"{root.name}{LANGUAGE_FIXTURE_SUFFIX}") / "goldset.jsonl"


def language_variant_id(item_id: str, variant_class: str) -> str:
    return f"{item_id}{VARIANT_ID_SEPARATOR}{variant_class}"


def compose_mixed_question(ukrainian: str, russian: str) -> str:
    """Deterministically alternate differing aligned tokens from a paired UA/RU question."""
    uk_tokens = ukrainian.strip().split()
    ru_tokens = russian.strip().split()
    mixed = list(ru_tokens)
    differing = [
        index
        for index in range(min(len(uk_tokens), len(ru_tokens)))
        if uk_tokens[index] != ru_tokens[index]
    ]
    for offset, index in enumerate(differing):
        if offset % 2 == 0:
            mixed[index] = uk_tokens[index]
    result = " ".join(mixed)
    if result in {ukrainian.strip(), russian.strip()} and differing:
        mixed[differing[0]] = uk_tokens[differing[0]]
        result = " ".join(mixed)
    if result in {ukrainian.strip(), russian.strip()}:
        raise ValueError("paired questions cannot produce a distinct mixed-language variant")
    return result


def select_ukrainian_baseline(items: Sequence[GoldItem]) -> tuple[list[GoldItem], list[str]]:
    """Exclude mislabeled Latin questions without rejecting valid UA lacking specific letters."""
    selected = [item for item in items if _is_cyrillic_dominant(item.question)]
    excluded = [item.id for item in items if item not in selected]
    return selected, excluded


def _is_cyrillic_dominant(text: str) -> bool:
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in text.casefold())
    latin = sum("a" <= char <= "z" for char in text.casefold())
    return (
        bool(cyrillic + latin) and cyrillic / (cyrillic + latin) >= UKRAINIAN_MIN_CYRILLIC_FRACTION
    )


def _unchanged_payload(item: GoldItem) -> dict[str, object]:
    payload = item.model_dump()
    for field in ("id", "lang", "question", "provenance", "verified"):
        payload.pop(field)
    return payload


def language_fixture_status(path: Path) -> str:
    """Require one uniform drafted or human-verified review state across the overlay."""
    states = {(item.provenance, item.verified) for item in load_goldset(path)}
    if states == {("frontier-drafted", False)}:
        return FIXTURE_DRAFTED
    if states == {("human-verified", True)}:
        return FIXTURE_VERIFIED
    raise ValueError("query-language fixture must be uniformly drafted or human-verified")


def _validate_variant_item(
    item: GoldItem,
    base: GoldItem,
    variant_class: str,
    seen_ids: set[str],
) -> str:
    if item.id in seen_ids:
        raise ValueError(f"{item.id}: duplicate language variant")
    seen_ids.add(item.id)
    if item.lang != _EXPECTED_LANG[variant_class]:
        raise ValueError(f"{item.id}: expected lang={_EXPECTED_LANG[variant_class]!r}")
    if _unchanged_payload(item) != _unchanged_payload(base):
        raise ValueError(f"{item.id}: language variant changed gold content beyond the question")
    if item.question == base.question:
        raise ValueError(f"{item.id}: language variant question is identical to Ukrainian")
    return item.question


def load_language_variants(
    path: Path,
    baseline: Sequence[GoldItem],
    classes: Sequence[str],
) -> dict[tuple[str, str], str]:
    """Load a drafted overlay and prove that only id, language, and question changed."""
    selected = tuple(name for name in classes if name in LANGUAGE_VARIANT_CLASSES)
    if not selected:
        return {}
    if not path.exists():
        raise ValueError(f"query-language fixture not found: {path}")
    language_fixture_status(path)
    if len({item.id for item in baseline}) != len(baseline):
        raise ValueError("baseline gold set has duplicate item ids")
    expected_ids = {
        language_variant_id(item.id, variant_class): (item, variant_class)
        for item in baseline
        for variant_class in selected
    }
    variants: dict[tuple[str, str], str] = {}
    seen_ids: set[str] = set()
    for item in load_goldset(path):
        expected = expected_ids.get(item.id)
        if expected is None:
            continue
        base, variant_class = expected
        variants[(base.id, variant_class)] = _validate_variant_item(
            item, base, variant_class, seen_ids
        )
    missing = [
        variant_id
        for variant_id, (item, variant_class) in expected_ids.items()
        if (item.id, variant_class) not in variants
    ]
    if missing:
        raise ValueError(f"query-language fixture is missing paired variants: {missing[:3]}")
    return variants


def fixture_translation_queries(
    variants: Mapping[tuple[str, str], str], baseline: Sequence[GoldItem]
) -> dict[str, str]:
    """Map each exact drafted query to Ukrainian as a benchmark-only retrieval upper bound."""
    clean = {item.id: item.question for item in baseline}
    translations: dict[str, str] = {}
    for (item_id, _variant_class), question in variants.items():
        translated = clean[item_id]
        previous = translations.setdefault(question, translated)
        if previous != translated:
            raise ValueError("one language variant maps ambiguously to two Ukrainian questions")
    return translations


@dataclass(frozen=True)
class FixtureTranslationPrep:
    """Query-prep adapter used only to measure exact paired retrieval recovery."""

    translations: Mapping[str, str]

    def process(self, query: str) -> QueryPrepResult:
        translated = self.translations.get(query)
        if translated is None:
            raise ValueError("fixture translation lane received an unpaired query")
        edit = QueryEdit(FIXTURE_TRANSLATE_STEP, "translate", query, translated)
        return QueryPrepResult(
            raw=query,
            processed=translated,
            steps=(FIXTURE_TRANSLATE_STEP,),
            edits=(edit,),
        )
