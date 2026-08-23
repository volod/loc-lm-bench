"""Response-integrity guard: did the model leak its reasoning, and did it answer in the
prompt's language?

Two failure modes score as ordinary content without it, both measured on this host's roster.

**Leaked reasoning.** A hybrid-thinking tag can emit its deliberation into the ANSWER BODY even
though the launcher sends the backend's native suppression flag on every call (Ollama `think:
false`, vLLM `chat_template_kwargs.enable_thinking=false`). The deliberation is generated text, so
it inflates `completion_tokens` -- the denominator throughput and cost are derived from -- and on a
bounded `max_tokens` budget it can consume the whole budget so that no answer is emitted at all.
The observed shape is not a well-formed `<think>...</think>` block: the OPENING tag is consumed by
the chat template and only the closing tag survives into the body, so a matched-pair scan misses
it. Both tags are therefore searched INDEPENDENTLY, and a leak with no terminator at all is caught
by its first-person deliberation opener instead.

**Off-language answer.** The benchmark prompt and its system prompt are Ukrainian, so an answer in
English (or Russian) is a delivery failure, not a content one -- but token F1 against a Ukrainian
reference simply reads low, which is the same reading a wrong Ukrainian answer earns. The guard
separates them.

Both verdicts are ADDITIVE diagnostics. They do not change `status`, they do not rewrite the
answer, and nothing here feeds the objective: an answer that leaked reasoning is scored exactly as
it was before this module existed, and the flags are what say so in the record.
"""

import re
import unicodedata

from typing import NamedTuple

__all__ = [
    "LeakVerdict",
    "LanguageVerdict",
    "answer_language",
    "guard_verdicts",
    "language_verdict",
    "reasoning_leak",
]

# --- reasoning leak ---------------------------------------------------------------------

# Reasoning delimiters, matched INDEPENDENTLY of their partner. The measured qwen3:30b leak
# carries a bare closing `</think>` because the chat template already emitted the opening tag into
# the prompt, so requiring a pair would miss exactly the case this guard exists for.
_LEAK_DELIMITERS = (
    "</think>",
    "<think>",
    "</thinking>",
    "<thinking>",
    "<|channel|>analysis",  # harmony-style channel markers
    "<|start|>assistant<|channel|>",
    "[think]",
    "[/think]",
)

# First-person deliberation openers, for a leak that carries NO delimiter (the budget ran out
# mid-thought, so the terminator was never emitted). Every marker is a planning frame a delivered
# answer does not open with -- the model narrating its own next step, or restating the asker in the
# third person. They are matched only against the answer's HEAD (`LEAK_HEAD_CHARS`): a leak is a
# PREFIX by construction, and an answer that mentions "the user" three paragraphs in is discussing
# the subject, not deliberating.
_LEAK_OPENERS = (
    # English (the observed language of every leak measured on this host)
    "okay, ",
    "okay.",
    "ok, so",
    "alright, ",
    "let me think",
    "let's think",
    "let's tackle",
    "let me tackle",
    "let's see",
    "first, i need",
    "first, let",
    "i need to figure",
    "i need to check",
    "i should check",
    "the user is asking",
    "the user asks",
    "the user wants",
    "the question is asking",
    "so the user",
    "we need to answer",
    # Ukrainian / Russian. NOT hypothetical: on the RAG benchmark the same tag that deliberates in
    # English on a short prompt deliberates in UKRAINIAN once the prompt carries retrieved Ukrainian
    # context, and every one of those leaks opens on a "let us work through the context" frame. The
    # deliberation verb is required -- a bare "давайте" opens plenty of ordinary prose.
    "давайте проаналізуємо",
    "давайте розглянемо",
    "давайте подивимо",  # subsumes подивимось / подивимося
    "давайте розберемо",
    "проаналізуємо контекст",
    "розглянемо контекст",
    "переглянемо контекст",
    "гаразд, ",
    "добре, мені",
    "мені потрібно з'ясувати",
    "спочатку мені потрібно",
    "користувач запитує",
    "давайте проанализируем",
    "давайте рассмотрим",
    "проанализируем контекст",
    "хорошо, ",
    "пользователь спрашивает",
)

# How much of the answer's head an opener may match in. Wide enough for a leading blank line or a
# short preamble, far short of a real answer's body.
LEAK_HEAD_CHARS = 120

_DELIMITER_RE = re.compile("|".join(re.escape(marker) for marker in _LEAK_DELIMITERS))
_OPENER_RE = re.compile("|".join(re.escape(marker) for marker in _LEAK_OPENERS))


class LeakVerdict(NamedTuple):
    """One answer's leaked-reasoning reading.

    `marker` names the signal that fired, so a flagged case is auditable without re-running it.
    `leak_chars` is the completion text attributable to the leak: everything up to and including
    the LAST reasoning delimiter, or -- when the leak has no terminator -- the whole completion,
    because there is then no boundary at which the answer could be said to begin. It is a
    diagnostic on how much of the generation budget the leak consumed, deliberately biased to
    over-report rather than to invent an answer boundary that is not in the text.
    """

    leaked: bool
    marker: str
    leak_chars: int


def reasoning_leak(text: str) -> LeakVerdict:
    """Read whether this completion leaked deliberation into the answer body."""
    if not text or not text.strip():
        return LeakVerdict(False, "", 0)
    lowered = text.casefold()
    delimiters = list(_DELIMITER_RE.finditer(lowered))
    if delimiters:
        last = delimiters[-1]
        return LeakVerdict(True, last.group(0), last.end())
    opener = _OPENER_RE.search(lowered[:LEAK_HEAD_CHARS])
    if opener is not None:
        return LeakVerdict(True, opener.group(0).strip(), len(text))
    return LeakVerdict(False, "", 0)


# --- answer language --------------------------------------------------------------------

UK = "uk"
RU = "ru"
EN = "en"
# Cyrillic-dominant, but carrying none of the letters that separate Ukrainian from Russian
# ("Що таке авторське право?" -- this benchmark's own wording). A real reading, not a failure: it
# settles the SCRIPT,
# which is what catches the measured failure mode, and declines to guess the language.
CYRILLIC = "cyrillic"
# No letter evidence at all: an empty answer, or a bare numeral.
UNDETERMINED = "undetermined"

_CYRILLIC_LANGUAGES = frozenset({UK, RU, CYRILLIC})

# Letters present in exactly one of the two Cyrillic languages this benchmark sees. Shared letters
# (including "и", which both use) say nothing and are not listed.
_UK_ONLY = frozenset("їієґ")
_RU_ONLY = frozenset("ыэъё")

# Below this many letters a dominance count is noise rather than evidence. Four keeps short but
# real answers readable ("70 років" -> uk) while an answer of pure digits stays undetermined.
MIN_LETTERS = 4


# An all-caps run of two or more letters is an acronym or a symbol, not evidence of a language, and
# it is dropped from BOTH scripts so the rule stays symmetric ("BPP, ZPP та RP." is a Ukrainian
# answer about complexity classes; "США" does not make one more Ukrainian). Measured: without this,
# that one benchmark item -- answered perfectly, objective 1.0 -- was the only off-language flag on
# both clean models, purely because three Latin acronyms outnumbered the "та" between them.
_ACRONYM_RE = re.compile(r"\b\w*[^\W\d_]\w*\b")
_MIN_ACRONYM_LETTERS = 2


def _is_acronym(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    return len(letters) >= _MIN_ACRONYM_LETTERS and all(char.isupper() for char in letters)


def _script_counts(text: str) -> tuple[int, int]:
    """(cyrillic letters, latin letters) in `text`, ignoring acronyms and every other script."""
    cyrillic = latin = 0
    for token in _ACRONYM_RE.findall(text):
        if _is_acronym(token):
            continue
        for char in token:
            if not char.isalpha():
                continue
            name = unicodedata.name(char, "")
            if name.startswith("CYRILLIC"):
                cyrillic += 1
            elif name.startswith("LATIN"):
                latin += 1
    return cyrillic, latin


def answer_language(text: str) -> str:
    """The dominant language of `text`: `uk` / `ru` / `cyrillic` / `en` / `undetermined`.

    Latin dominance is read as English: it is the only Latin-script language the roster's models
    fall back into on a Ukrainian prompt, and the alternative -- a full language identifier -- would
    add a dependency and a model pin to a guard whose whole job is to be cheap and always on. A
    Ukrainian answer quoting Latin-script terms stays Cyrillic-dominant, so the rule needs Latin to
    STRICTLY outnumber Cyrillic before it fires.

    Inside Cyrillic the call is made on letters only one of the two languages has. Plenty of
    genuine Ukrainian carries none of them, so `cyrillic` is a common and correct answer here --
    the script is settled, the language is not, and `language_verdict` uses exactly that much.
    """
    cyrillic, latin = _script_counts(text)
    if cyrillic + latin < MIN_LETTERS:
        return UNDETERMINED
    if latin > cyrillic:
        return EN
    folded = text.casefold()
    uk_hits = sum(1 for char in folded if char in _UK_ONLY)
    ru_hits = sum(1 for char in folded if char in _RU_ONLY)
    if uk_hits > ru_hits:
        return UK
    if ru_hits > uk_hits:
        return RU
    return CYRILLIC


class LanguageVerdict(NamedTuple):
    """The prompt's language, the answer's language, and whether they disagree."""

    prompt_language: str
    answer_language: str
    mismatch: bool


def language_verdict(question: str, answer: str) -> LanguageVerdict:
    """Compare the answer's dominant language against the question's.

    A mismatch is called at the coarsest level both sides actually settle. Different SCRIPTS is a
    mismatch outright -- that is the measured failure, a Ukrainian prompt answered in English, and
    it does not depend on either side carrying a language-distinguishing letter. Inside Cyrillic a
    mismatch needs both sides decided as `uk` or `ru`; a Cyrillic answer whose language the letters
    do not settle is recorded as `cyrillic` and never counted against the model.

    The reading is taken over the WHOLE completion, leaked reasoning included, because that is the
    text the objective scored -- an English deliberation with a Ukrainian sentence at the end is an
    English response by delivered volume, and `reasoning_leak` in the same row is what says why.
    """
    prompt = answer_language(question)
    delivered = answer_language(answer)
    if prompt == UNDETERMINED or delivered == UNDETERMINED:
        return LanguageVerdict(prompt, delivered, False)
    if (prompt in _CYRILLIC_LANGUAGES) != (delivered in _CYRILLIC_LANGUAGES):
        return LanguageVerdict(prompt, delivered, True)
    decided = {UK, RU}
    mismatch = prompt in decided and delivered in decided and prompt != delivered
    return LanguageVerdict(prompt, delivered, mismatch)


def guard_verdicts(question: str, answer: str) -> tuple[LeakVerdict, LanguageVerdict]:
    """Both per-response guard readings for one case."""
    return reasoning_leak(answer), language_verdict(question, answer)
