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
QUERY_PREP_STEPS: tuple[str, ...] = (STEP_NORMALIZE, STEP_TYPOS, STEP_GLOSSARY, STEP_REWRITE)

QUERY_GLOSSARY_VERSION = "query-glossary-v1"

# Injected local-LLM rewrite seam: original query -> rewritten query (identity when absent).
Rewriter = Callable[[str], str]

# Injected morphology probe for the typos step's opt-in guard: True when the token is a known
# valid Ukrainian word form (pymorphy3 `word_is_known`; `llb.rag.lexical.load_uk_word_probe`).
KnownWordProbe = Callable[[str], bool]


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

    @property
    def changed(self) -> bool:
        return self.processed != self.raw
