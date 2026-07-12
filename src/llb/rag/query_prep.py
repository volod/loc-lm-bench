"""Opt-in query-side processing lane between the user question and retrieval (uk-query-processing).

A pure, unit-testable pipeline of NAMED steps that transforms a query BEFORE it reaches the
retrieval store. It never touches the stored corpus text -- original word forms stay untouched;
this is the query-side twin of the index-side lexical normalization already shipped in
`llb.rag.lexical`. Every step is honest: it reports what it changed so an A/B report can attribute
a per-step retrieval delta before anyone turns the lane on by default.

Steps (applied in the configured order):
  - normalize: matching-side casefold, apostrophe-variant unification, and a small transliteration
    table that maps Latin-typed Ukrainian tokens back to Cyrillic (`zakon` -> `закон`).
  - typos:     deterministic corpus-vocabulary typo tolerance. A query token ABSENT from the
    indexed corpus vocabulary is corrected to its nearest in-vocabulary token within
    Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars); a token the corpus already
    contains is NEVER altered.
  - glossary:  alias/glossary expansion. When the query mentions a known term (or one of its
    surzhyk / transliterated aliases) the entry's other surface forms are appended, so retrieval
    catches the variant the corpus actually uses. Sourced from a `query_glossary.json` built from a
    draft bundle's `prompt_dictionary_candidates.jsonl` (`build_glossary_from_candidates`).
  - rewrite:   an optional local-LLM query rewrite through an injected endpoint callable; OFF by
    default and only present when explicitly requested. Records the rewritten query.

The module has no heavy dependencies (it reuses the pure tokenizer in `llb.rag.lexical`), so the
whole lane is testable without a model, a store, or the [rag] extra.
"""

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.rag.lexical import normalize_token, tokenize

_LOG = logging.getLogger(__name__)

# Canonical step ids and their canonical order (a configured list may use any subset/order).
STEP_NORMALIZE = "normalize"
STEP_TYPOS = "typos"
STEP_GLOSSARY = "glossary"
STEP_REWRITE = "rewrite"
QUERY_PREP_STEPS: tuple[str, ...] = (STEP_NORMALIZE, STEP_TYPOS, STEP_GLOSSARY, STEP_REWRITE)

# Typo-tolerance thresholds (Damerau-Levenshtein / OSA edit distance). A longer token tolerates a
# second edit because a single transposition + substitution is common in long Ukrainian inflections.
TYPO_MAX_DISTANCE_SHORT = 1
TYPO_LONG_TOKEN_CHARS = 8
TYPO_MAX_DISTANCE_LONG = 2

QUERY_GLOSSARY_VERSION = "query-glossary-v1"

# Injected local-LLM rewrite seam: original query -> rewritten query (identity when absent).
Rewriter = Callable[[str], str]

# Injected morphology probe for the typos step's opt-in guard: True when the token is a known
# valid Ukrainian word form (pymorphy3 `word_is_known`; `llb.rag.lexical.load_uk_word_probe`).
KnownWordProbe = Callable[[str], bool]

# Reversible-ish Ukrainian romanization used to invert Latin-typed terms back to Cyrillic. The map
# is injective, so the Latin->Cyrillic inverse is well-defined by longest-key-first matching; the
# soft sign and a few rare digraph collisions (shch) are intentionally out of the reversible core.
CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "h",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ye",
    "ж": "zh",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "yi",
    "й": "j",
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
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ю": "yu",
    "я": "ya",
}
# Latin -> Cyrillic, matched longest-first so digraphs (`shch`, `kh`, `ye`) win over single letters.
LATIN_TO_CYRILLIC: dict[str, str] = {lat: cyr for cyr, lat in CYRILLIC_TO_LATIN.items()}
_LATIN_KEYS_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(LATIN_TO_CYRILLIC, key=len, reverse=True))


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


# --------------------------------------------------------------------------------------------
# Transliteration (normalize step)
# --------------------------------------------------------------------------------------------


def _is_latin_word(token: str) -> bool:
    """True for a non-empty token whose every character is an ASCII letter."""
    return bool(token) and all("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in token)


def transliterate_latin_to_cyrillic(token: str) -> str:
    """Map a Latin-typed token to Cyrillic via the romanization table (longest-match, greedy).

    Characters with no table entry pass through unchanged, so a token that is not Latin-typed
    Ukrainian degrades to (mostly) itself. Case is folded first (the matching side never keeps
    case). Non-Latin tokens are returned unchanged by the caller.
    """
    lowered = token.casefold()
    out: list[str] = []
    i = 0
    while i < len(lowered):
        for key in _LATIN_KEYS_LONGEST_FIRST:
            if lowered.startswith(key, i):
                out.append(LATIN_TO_CYRILLIC[key])
                i += len(key)
                break
        else:
            out.append(lowered[i])
            i += 1
    return "".join(out)


# Characters dropped when romanizing (the soft sign and apostrophe variants have no Latin form).
_ROMANIZE_DROP = frozenset("ь'’ʼ")


def cyrillic_to_latin(text: str) -> str:
    """Romanize Ukrainian text with the reversible table (used to seed transliterated aliases).

    The soft sign and apostrophes are dropped so a seeded alias is clean Latin; any character with
    no table entry (a space, a digit) passes through unchanged.
    """
    return "".join(
        "" if ch in _ROMANIZE_DROP else CYRILLIC_TO_LATIN.get(ch, ch) for ch in text.casefold()
    )


def apply_normalize(query: str) -> tuple[str, list[QueryEdit]]:
    """Casefold + apostrophe-unify the whole query, then transliterate Latin-typed tokens.

    Casefolding and apostrophe unification are silent matching-side normalization (they never
    change which corpus terms match). Each Latin->Cyrillic transliteration is recorded as an edit
    because it is a real, auditable substitution.
    """
    from llb.rag.lexical import _APOSTROPHE_VARIANTS, _TOKEN_RE

    folded = query.translate(_APOSTROPHE_VARIANTS).casefold()
    edits: list[QueryEdit] = []

    def _replace(match: "re.Match[str]") -> str:
        token = match.group(0)
        if not _is_latin_word(token):
            return token
        cyrillic = transliterate_latin_to_cyrillic(token)
        if cyrillic == token:
            return token
        edits.append(
            QueryEdit(STEP_NORMALIZE, "transliterate", original=token, replacement=cyrillic)
        )
        _LOG.debug("[query-prep] transliterate %r -> %r", token, cyrillic)
        return cyrillic

    processed = _TOKEN_RE.sub(_replace, folded)
    return processed, edits


# --------------------------------------------------------------------------------------------
# Typo tolerance (typos step)
# --------------------------------------------------------------------------------------------


def damerau_levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Optimal string alignment (Damerau-Levenshtein with adjacent transpositions), bounded.

    Returns `max_distance + 1` early once the running minimum exceeds `max_distance`, so the
    per-token vocabulary scan stays cheap. Exact distance when `max_distance` is None.
    """
    la, lb = len(a), len(b)
    if max_distance is not None and abs(la - lb) > max_distance:
        return max_distance + 1
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, prev2[j - 2] + 1)
            cur[j] = best
            row_best = min(row_best, best)
        if max_distance is not None and row_best > max_distance:
            return max_distance + 1
        prev2, prev = prev, cur
    return prev[lb]


def _typo_budget(token: str) -> int:
    return TYPO_MAX_DISTANCE_LONG if len(token) > TYPO_LONG_TOKEN_CHARS else TYPO_MAX_DISTANCE_SHORT


def nearest_vocab_token(token: str, vocabulary: "frozenset[str]", max_distance: int) -> str | None:
    """The closest in-vocabulary token within `max_distance` edits, else None.

    Deterministic: ties break on smaller edit distance, then lexicographically smaller token, so
    the result is independent of the (unordered) vocabulary iteration order.
    """
    best: tuple[int, str] | None = None
    for candidate in vocabulary:
        if abs(len(candidate) - len(token)) > max_distance:
            continue
        distance = damerau_levenshtein(token, candidate, max_distance)
        if distance <= max_distance:
            key = (distance, candidate)
            if best is None or key < best:
                best = key
    return best[1] if best is not None else None


def apply_typos(
    query: str,
    vocabulary: "frozenset[str]",
    known_word: KnownWordProbe | None = None,
) -> tuple[str, list[QueryEdit]]:
    """Correct out-of-vocabulary word tokens to their nearest in-vocabulary neighbor.

    An in-vocabulary token is never altered. Purely numeric tokens are left untouched so an
    article/law number or code is never "corrected" into a different one. Every correction is
    recorded and logged.

    `known_word` is the opt-in morphology guard (morphology-aware-typo-guard): an OOV token the
    probe recognizes as a valid Ukrainian word form is a grammatical inflection, not a typo, so
    it is left for index+query lemmatization to match instead of being edit-distance "corrected"
    to a different corpus surface form. Genuine misspellings stay unknown to the probe and are
    still corrected.
    """
    from llb.rag.lexical import _TOKEN_RE

    edits: list[QueryEdit] = []

    def _replace(match: "re.Match[str]") -> str:
        raw = match.group(0)
        token = normalize_token(raw)
        if not token or token.isdigit() or token in vocabulary:
            return raw
        if known_word is not None and known_word(token):
            _LOG.debug("[query-prep] typo guard: %r is a known word form; left unchanged", token)
            return raw
        correction = nearest_vocab_token(token, vocabulary, _typo_budget(token))
        if correction is None or correction == token:
            return raw
        edits.append(QueryEdit(STEP_TYPOS, "typo", original=token, replacement=correction))
        _LOG.info("[query-prep] typo %r -> %r (corpus vocabulary)", token, correction)
        return correction

    processed = _TOKEN_RE.sub(_replace, query)
    return processed, edits


def build_vocabulary(texts: Iterable[str]) -> "frozenset[str]":
    """The set of normalized corpus tokens (the in-vocabulary set the typo step protects)."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(tokenize(text))
    return frozenset(vocab)


# --------------------------------------------------------------------------------------------
# Glossary / alias expansion (glossary step)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GlossaryEntry:
    """A canonical term and its alias surface forms (aliases include surzhyk / transliteration)."""

    canonical: str
    aliases: tuple[str, ...] = ()

    def surface_forms(self) -> tuple[str, ...]:
        forms = [self.canonical, *self.aliases]
        seen: set[str] = set()
        unique: list[str] = []
        for form in forms:
            if form and form not in seen:
                seen.add(form)
                unique.append(form)
        return tuple(unique)


def _normalized_form(text: str) -> str:
    """Space-joined normalized token string, so alias matching respects word boundaries."""
    return " ".join(tokenize(text))


@dataclass(frozen=True)
class Glossary:
    """Alias-expansion lookup built from a draft bundle's dictionary candidates (or hand-authored).

    `expand` appends the missing surface forms of any entry whose surface form appears in the query,
    so the retriever sees every spelling the corpus might use without the raw query being lost.
    """

    entries: tuple[GlossaryEntry, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Glossary":
        entries = [
            GlossaryEntry(
                canonical=str(row["canonical"]),
                aliases=tuple(str(a) for a in row.get("aliases", []) if str(a).strip()),
            )
            for row in data.get("entries", [])
            if str(row.get("canonical", "")).strip()
        ]
        return cls(tuple(entries))

    @classmethod
    def load(cls, path: Path | str) -> "Glossary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self, source_bundle: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": QUERY_GLOSSARY_VERSION,
            "entries": [
                {"canonical": entry.canonical, "aliases": list(entry.aliases)}
                for entry in self.entries
            ],
        }
        if source_bundle is not None:
            payload["source_bundle"] = source_bundle
        return payload


def apply_glossary(query: str, glossary: "Glossary") -> tuple[str, list[QueryEdit]]:
    """Append the alias/canonical surface forms of every glossary entry the query triggers.

    Matching is word-boundary substring matching on the normalized token string, so a multi-word
    canonical term matches as a phrase. The raw query text is preserved; expansions are appended.
    """
    normalized = _normalized_form(query)
    if not normalized:
        return query, []
    haystack = f" {normalized} "
    present: set[str] = set(normalized.split())
    additions: list[str] = []
    edits: list[QueryEdit] = []
    for entry in glossary.entries:
        forms = [(_normalized_form(form), form) for form in entry.surface_forms()]
        if not any(norm and f" {norm} " in haystack for norm, _ in forms):
            continue
        for norm, original in forms:
            if not norm or all(token in present for token in norm.split()):
                continue
            additions.append(norm)
            present.update(norm.split())
            edits.append(
                QueryEdit(STEP_GLOSSARY, "alias", original=entry.canonical, replacement=original)
            )
            _LOG.info("[query-prep] alias expand %r += %r", entry.canonical, original)
    if not additions:
        return query, []
    return f"{query} {' '.join(additions)}", edits


def build_glossary_from_candidates(
    rows: Iterable[dict[str, Any]], *, add_transliterations: bool = True
) -> Glossary:
    """Turn `prompt_dictionary_candidates.jsonl` rows into glossary entries.

    Each candidate `term` becomes a canonical entry; its recorded `aliases` carry over, and (when
    `add_transliterations`) a romanized Latin variant of the term is added so a Latin-typed query
    still expands. Deterministic: entries are sorted by canonical term. Hand-added surzhyk /
    transliteration aliases can be appended to the emitted JSON afterwards.
    """
    entries: list[GlossaryEntry] = []
    for row in rows:
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        aliases = _candidate_aliases(term, row, add_transliterations=add_transliterations)
        entries.append(GlossaryEntry(canonical=term, aliases=tuple(aliases)))
    entries.sort(key=lambda entry: entry.canonical.casefold())
    return Glossary(tuple(entries))


def _candidate_aliases(term: str, row: dict[str, Any], *, add_transliterations: bool) -> list[str]:
    """Distinct recorded aliases for a term, optionally plus its romanized Latin variant."""
    aliases: list[str] = []
    seen: set[str] = {term.casefold()}
    for alias in row.get("aliases", []) or []:
        text = str(alias).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            aliases.append(text)
    if add_transliterations:
        romanized = cyrillic_to_latin(term)
        if romanized and romanized != term.casefold() and romanized not in seen:
            aliases.append(romanized)
    return aliases


# --------------------------------------------------------------------------------------------
# LLM rewrite (rewrite step)
# --------------------------------------------------------------------------------------------


def apply_rewrite(query: str, rewriter: "Rewriter") -> tuple[str, list[QueryEdit], str | None]:
    """Rewrite the query through the injected local-LLM endpoint; record both forms.

    A blank or unchanged rewrite is a no-op (the original query is kept), so a degenerate model
    response never silently drops the question.
    """
    rewritten = (rewriter(query) or "").strip()
    if not rewritten or rewritten == query:
        return query, [], rewritten or None
    _LOG.info("[query-prep] llm rewrite %r -> %r", query, rewritten)
    return (
        rewritten,
        [QueryEdit(STEP_REWRITE, "rewrite", original=query, replacement=rewritten)],
        (rewritten),
    )


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


@dataclass
class QueryPrep:
    """An ordered pipeline of query-prep steps with their resolved dependencies.

    `process` runs the steps in order, threading the query through each and accumulating the edit
    log. An empty step list is an exact no-op (the processed query is byte-identical to the raw
    query), which is the off-by-default behavior the acceptance gate requires.
    """

    steps: tuple[str, ...] = ()
    vocabulary: "frozenset[str]" = field(default_factory=frozenset)
    glossary: Glossary | None = None
    rewriter: Rewriter | None = None
    known_word: KnownWordProbe | None = None

    @classmethod
    def build(
        cls,
        steps: Iterable[str],
        *,
        vocabulary: "frozenset[str] | None" = None,
        glossary: Glossary | None = None,
        rewriter: Rewriter | None = None,
        known_word: KnownWordProbe | None = None,
    ) -> "QueryPrep":
        """Validate step names and their required dependencies, then build the pipeline."""
        ordered = tuple(steps)
        unknown = [step for step in ordered if step not in QUERY_PREP_STEPS]
        if unknown:
            raise ValueError(
                f"unknown query-prep step(s): {unknown}; choose from {list(QUERY_PREP_STEPS)}"
            )
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"duplicate query-prep step(s): {ordered}")
        if STEP_TYPOS in ordered and vocabulary is None:
            raise ValueError("the 'typos' step needs a corpus vocabulary")
        if known_word is not None and STEP_TYPOS not in ordered:
            raise ValueError("the typo morphology guard needs the 'typos' step")
        if STEP_GLOSSARY in ordered and glossary is None:
            raise ValueError("the 'glossary' step needs a query glossary")
        if STEP_REWRITE in ordered and rewriter is None:
            raise ValueError("the 'rewrite' step needs a rewrite endpoint callable")
        return cls(
            steps=ordered,
            vocabulary=vocabulary if vocabulary is not None else frozenset(),
            glossary=glossary,
            rewriter=rewriter,
            known_word=known_word,
        )

    def process(self, query: str) -> QueryPrepResult:
        current = query
        edits: list[QueryEdit] = []
        rewrite_text: str | None = None
        for step in self.steps:
            if step == STEP_NORMALIZE:
                current, step_edits = apply_normalize(current)
            elif step == STEP_TYPOS:
                current, step_edits = apply_typos(
                    current, self.vocabulary, known_word=self.known_word
                )
            elif step == STEP_GLOSSARY:
                assert self.glossary is not None  # guaranteed by build()
                current, step_edits = apply_glossary(current, self.glossary)
            else:  # STEP_REWRITE
                assert self.rewriter is not None  # guaranteed by build()
                current, step_edits, rewrite_text = apply_rewrite(current, self.rewriter)
            edits.extend(step_edits)
        return QueryPrepResult(
            raw=query,
            processed=current,
            steps=self.steps,
            edits=tuple(edits),
            rewrite=rewrite_text,
        )


# --------------------------------------------------------------------------------------------
# A/B report (validate-retrieval / compare-retrieval --query-prep-ab)
# --------------------------------------------------------------------------------------------

# (question, gold source spans) -- the per-item A/B input (matches `llb.rag.compare.CompareItem`).
AbItem = tuple[str, list[Any]]
# A retriever seam: processed query + k -> ranked chunk records (any RAG-store `.retrieve`).
RetrieveFn = Callable[[str, int], list[Any]]

AB_BASELINE_LABEL = "baseline"


def cumulative_pipelines(
    steps: Iterable[str],
    *,
    vocabulary: "frozenset[str] | None" = None,
    glossary: Glossary | None = None,
    rewriter: Rewriter | None = None,
    known_word: KnownWordProbe | None = None,
) -> list[tuple[str, "QueryPrep"]]:
    """`baseline` (no steps) then one pipeline per cumulative prefix (`+normalize`, `+typos`, ...).

    The A/B report scores each stage so a per-step marginal retrieval delta is attributable. Every
    prefix reuses the same resolved dependencies (`known_word` only binds to prefixes that
    include the typos step).
    """
    ordered = tuple(steps)
    stages: list[tuple[str, QueryPrep]] = [(AB_BASELINE_LABEL, QueryPrep.build(()))]
    for index in range(1, len(ordered) + 1):
        prefix = ordered[:index]
        pipeline = QueryPrep.build(
            prefix,
            vocabulary=vocabulary,
            glossary=glossary,
            rewriter=rewriter,
            known_word=known_word if STEP_TYPOS in prefix else None,
        )
        stages.append((f"+{ordered[index - 1]}", pipeline))
    return stages


def query_prep_ab_report(
    items: list[AbItem],
    retrieve: RetrieveFn,
    k: int,
    stages: list[tuple[str, "QueryPrep"]],
) -> dict[str, Any]:
    """Score retrieval at every cumulative stage and attribute per-step recall@k / MRR deltas.

    Pure over the injected `retrieve` seam (fake store in tests). Each stage's delta is measured
    against the PREVIOUS stage, so the marginal contribution of each added step is explicit.
    """
    from llb.rag.retrieval import evaluate_retrieval

    rows: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    for label, pipeline in stages:
        pairs = [
            (retrieve(pipeline.process(question).processed, k), spans) for question, spans in items
        ]
        metrics = evaluate_retrieval(pairs, k)
        row: dict[str, Any] = {
            "stage": label,
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "delta_recall": 0.0 if prev is None else metrics["recall_at_k"] - prev["recall_at_k"],
            "delta_mrr": 0.0 if prev is None else metrics["mrr"] - prev["mrr"],
        }
        rows.append(row)
        prev = {"recall_at_k": metrics["recall_at_k"], "mrr": metrics["mrr"]}
    return {"k": k, "n": len(items), "stages": rows}


def format_query_prep_ab(report: dict[str, Any]) -> str:
    """Render the A/B stages as an ASCII table (AGENTS.md: ASCII-only, no box-drawing)."""
    lines = [f"[query-prep A/B] n={report['n']} k={report['k']}"]
    width = max((len(row["stage"]) for row in report["stages"]), default=len("stage"))
    lines.append(f"  {'stage'.ljust(width)}   recall@k   d(recall)      mrr    d(mrr)")
    for row in report["stages"]:
        lines.append(
            f"  {row['stage'].ljust(width)}   {row['recall_at_k']:8.3f}  {row['delta_recall']:+8.3f} "
            f"{row['mrr']:8.3f} {row['delta_mrr']:+8.3f}"
        )
    return "\n".join(lines)
