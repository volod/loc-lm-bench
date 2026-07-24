"""Shared query-prep vocabulary: step ids, the edit/result records, and the injected seams.

Every step submodule imports the recorded-edit dataclass and its own step id from here, so the
step modules never import each other and the pipeline can compose them without a cycle.
"""

from collections.abc import Callable
from dataclasses import dataclass

# Canonical step ids and their canonical order (a configured list may use any subset/order).
STEP_NORMALIZE = "normalize"
STEP_TYPOS = "typos"
STEP_GLOSSARY = "glossary"
STEP_REWRITE = "rewrite"
STEP_HYDE = "hyde"
STEP_DECOMPOSE = "decompose"
QUERY_PREP_STEPS: tuple[str, ...] = (
    STEP_NORMALIZE,
    STEP_TYPOS,
    STEP_GLOSSARY,
    STEP_REWRITE,
    STEP_HYDE,
    STEP_DECOMPOSE,
)

QUERY_GLOSSARY_VERSION = "query-glossary-v1"

# Edit `kind` values. The two normalize kinds are also the provenance kinds the typos step
# reverses when it checks a correction candidate against the token as originally typed.
KIND_TRANSLITERATE = "transliterate"
KIND_HOMOGLYPH = "homoglyph"
KIND_TYPO = "typo"
KIND_ALIAS = "alias"
KIND_REWRITE = "rewrite"

# Injected local-LLM rewrite seam: original query -> rewritten query (identity when absent).
Rewriter = Callable[[str], str]
QueryGenerator = Callable[[str], str]

# Injected morphology probe for the typos step's opt-in guard: True when the token is a known
# valid Ukrainian word form (pymorphy3 `word_is_known`; `llb.rag.lexical.load_uk_word_probe`).
KnownWordProbe = Callable[[str], bool]

# Injected plausibility probe for the normalize step's language gate: True when a decoded
# (Cyrillic) token looks like real Ukrainian -- it is in the corpus vocabulary or a valid word
# form. Foreign-language text decodes to nonsense the probe rejects.
PlausibilityProbe = Callable[[str], bool]


@dataclass(frozen=True)
class LanguageGate:
    """The whole-query transliteration decision for the normalize step, with its evidence.

    Transliteration is decided for the query as a WHOLE, not per token: a query written in a
    foreign language decodes to Cyrillic nonsense that no later step can undo, so it is left
    untouched. `transliterate` is the verdict; the counts are the share of the query's Latin word
    tokens whose decoded form the plausibility probe accepted.
    """

    transliterate: bool
    latin_tokens: int
    plausible_tokens: int
    threshold: float

    @property
    def plausible_share(self) -> float:
        """Fraction of Latin word tokens that decode to plausible Ukrainian (1.0 when there are none)."""
        return self.plausible_tokens / self.latin_tokens if self.latin_tokens else 1.0

    def as_provenance(self) -> dict[str, object]:
        """Compact per-query record of the gate decision for the A/B report and audit logs."""
        return {
            "transliterated": self.transliterate,
            "latin_tokens": self.latin_tokens,
            "plausible_tokens": self.plausible_tokens,
            "plausible_share": round(self.plausible_share, 3),
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class QueryEdit:
    """One recorded transformation, so the A/B report and logs can attribute every change.

    `kind` names the mechanism (`transliterate` / `typo` / `alias` / `rewrite`); `original` and
    `replacement` are the token (or whole query, for `rewrite`) before and after.
    """

    step: str
    kind: str
    original: str
    replacement: str


@dataclass(frozen=True)
class QueryPrepResult:
    """The processed query plus a full transformation log; `raw` is always the untouched input."""

    raw: str
    processed: str
    steps: tuple[str, ...]
    edits: tuple[QueryEdit, ...] = ()
    rewrite: str | None = None
    hypothetical_answer: str | None = None
    decomposition: str | None = None
    subqueries: tuple[str, ...] = ()
    normalize_gate: LanguageGate | None = None

    @property
    def changed(self) -> bool:
        return bool(self.processed != self.raw or self.hypothetical_answer or self.subqueries)

    def provenance(self) -> dict[str, object]:
        """Per-case query text needed to reproduce and audit the retrieval call."""
        out: dict[str, object] = {
            "query_processed": self.processed,
            "query_corrections": sum(
                edit.step not in {STEP_HYDE, STEP_DECOMPOSE} for edit in self.edits
            ),
        }
        if self.normalize_gate is not None and not self.normalize_gate.transliterate:
            # Only surfaced when the gate actually suppressed transliteration, so a normal run's
            # provenance is unchanged and an operator sees exactly why a foreign query passed
            # through untouched.
            out["query_normalize_gate"] = self.normalize_gate.as_provenance()
        if self.hypothetical_answer is not None:
            out["query_hypothetical_answer"] = self.hypothetical_answer
        if self.decomposition is not None:
            out["query_decomposition"] = self.decomposition
            out["query_subqueries"] = list(self.subqueries)
        return out
