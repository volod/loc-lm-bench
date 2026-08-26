"""Per-SPAN question affinity: the tie-break signal the graph lane's node relevance cannot carry.

Both graph strategies score a NODE and then emit every one of that node's mention spans at the
node's score, so a node with twenty mentions contributes twenty exactly-tied candidates and the
rank-k cut lands inside that block. What decides which of them is retrieved is then
`(doc_id, char_start, char_end)` -- a document id, which is not a relevance signal.

This module supplies the missing per-span term from what a mention already carries: its own text
and the title of the section containing it. It reuses the entity linker's tokenizer, stem key, and
exact-over-stem weighting (`llb.graph.linking`) rather than introducing a second notion of "this
text matches the question", and returns a bounded share so the caller can place it strictly INSIDE
one node-relevance level (`llb.graph.retrieval`) -- the signal orders spans the lane had not
scored apart, and never reorders spans it had.
"""

from llb.graph.constants import SECTION_TITLE_MATCH_WEIGHT, STEM_MATCH_WEIGHT
from llb.graph.linking import morph_key, tokenize
from llb.graph.model import GraphMention


class QuestionKeys:
    """The question's content tokens and stem keys, tokenized once per query, not per span.

    It also memoizes per-text coverage, because a subgraph serializes hundreds of spans drawn from
    a handful of distinct section titles and repeated entity surface forms.
    """

    __slots__ = ("_covered", "keys", "size", "tokens")

    def __init__(self, question: str) -> None:
        self.tokens = tokenize(question)
        self.keys = {morph_key(token) for token in self.tokens}
        self.size = len(self.tokens)
        self._covered: dict[str, float] = {}

    def coverage(self, text: str) -> float:
        """Question tokens `text` covers: exact hits at full weight, stem-only hits discounted."""
        hit = self._covered.get(text)
        if hit is None:
            hit = self._covered[text] = self._coverage(text)
        return hit

    def _coverage(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        exact = len(self.tokens & tokens)
        stem_only = max(len(self.keys & {morph_key(token) for token in tokens}) - exact, 0)
        return exact + STEM_MATCH_WEIGHT * stem_only


def span_affinity(question: QuestionKeys, mention: GraphMention) -> float:
    """Share of the question this span covers, in [0, 1] (0.0 when the question has no content).

    The span's own text counts full; its section title counts at `SECTION_TITLE_MATCH_WEIGHT`,
    because a title says what the surrounding passage is about rather than what this span says.
    Within each, an exact token hit counts full and a shared-stem hit counts `STEM_MATCH_WEIGHT`,
    exactly as entity linking scores them.
    """
    if not question.size:
        return 0.0
    covered = question.coverage(mention["text"]) + SECTION_TITLE_MATCH_WEIGHT * question.coverage(
        mention["section_title"]
    )
    return min(covered / question.size, 1.0)
