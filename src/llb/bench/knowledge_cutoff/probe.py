"""Date-blind, position-balanced multiple-choice probes and answer parsing."""

import hashlib
import re
from dataclasses import dataclass

from llb.bench.knowledge_cutoff.data import CutoffEvent, LETTERS

SHUFFLE_VERSION = "llb-cutoff-v1"
ANSWER_PATTERN = re.compile(r"^\s*(?:answer\s*(?:is|:)\s*)?\(?([A-D])\)?(?:[.):\s]|$)", re.I)
CHOICE_PREFIX = re.compile(r"^\s*[A-D]\s*[.):]\s*", re.I)


@dataclass(frozen=True, slots=True)
class PreparedProbe:
    prompt: str
    expected: str
    choice_order: tuple[str, ...]


def _choice_order(event: CutoffEvent) -> tuple[int, ...]:
    """Stable per-event permutation so source answer positions cannot become a shortcut."""
    ranked = []
    for index in range(len(LETTERS)):
        digest = hashlib.sha256(f"{SHUFFLE_VERSION}:{event.id}:{index}".encode()).digest()
        ranked.append((digest, index))
    return tuple(index for _digest, index in sorted(ranked))


def prepare_probe(event: CutoffEvent) -> PreparedProbe:
    order = _choice_order(event)
    source_answer = LETTERS.index(event.mcq_answer)
    expected = LETTERS[order.index(source_answer)]
    choices = []
    source_labels = []
    for display_index, source_index in enumerate(order):
        clean = CHOICE_PREFIX.sub("", event.mcq_choices[source_index]).strip()
        choices.append(f"{LETTERS[display_index]}) {clean}")
        source_labels.append(LETTERS[source_index])
    prompt = (
        "Answer the multiple-choice question using only one letter: A, B, C, or D. "
        "Do not explain your answer.\n\n"
        f"{event.mcq_question}\n\n" + "\n".join(choices) + "\n\nAnswer:"
    )
    return PreparedProbe(prompt, expected, tuple(source_labels))


def parse_answer(text: str) -> str | None:
    """Accept a leading answer letter while rejecting letters buried in prose."""
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).upper() if match else None
