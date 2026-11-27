"""One worked example per dangerous-substitution class.

The windows come from the same masked reading the confusion counts use. A class
the pilot never produced is reported with ``found`` false, so the list cannot drop
a class by omitting it.
"""

from llb.prep.book.metrics.align import alignment
from llb.prep.book.metrics.confusion import (
    equation_numbers,
    masked_pair,
    normalize_equation,
    step_class,
    unicode_script_tokens,
)
from llb.prep.book.metrics.constants import CLASS_EQUATION, CLASS_SCRIPT, CONFUSION_CLASSES


def worked_examples(pages: list[tuple[str, int, str, str, str]]) -> list[dict[str, object]]:
    """``(lane, pdf_page, page_class, reference, hypothesis)`` in report order.

    The first hit for a class wins. Later pages do not replace it.
    """
    found: dict[str, dict[str, object]] = {}
    for lane, pdf_page, page_class, reference, hypothesis in pages:
        for kind, (left, right) in _hits(reference, hypothesis).items():
            found.setdefault(
                kind,
                {
                    "confusion_class": kind,
                    "found": True,
                    "lane": lane,
                    "pdf_page": pdf_page,
                    "page_class": page_class,
                    "reference": left,
                    "hypothesis": right,
                },
            )
    rows = []
    for kind in CONFUSION_CLASSES:
        rows.append(
            found.get(
                kind,
                {
                    "confusion_class": kind,
                    "found": False,
                    "lane": None,
                    "pdf_page": None,
                    "page_class": None,
                    "reference": None,
                    "hypothesis": None,
                },
            )
        )
    return rows


def _hits(reference: str, hypothesis: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    _equations(found, reference, hypothesis)
    _scripts(found, reference, hypothesis)
    ref_text, hyp_text = masked_pair(reference, hypothesis)
    for op, left, right in alignment(ref_text, hyp_text):
        kind = step_class(op, left, right, ref_text, hyp_text)
        if kind is None or kind in found:
            continue
        found[kind] = (_window(ref_text, left), _window(hyp_text, right))
    return found


def _equations(found: dict[str, tuple[str, str]], reference: str, hypothesis: str) -> None:
    ref_nums = equation_numbers(reference)
    hyp_nums = equation_numbers(hypothesis)
    for index in range(max(len(ref_nums), len(hyp_nums))):
        left = ref_nums[index] if index < len(ref_nums) else ""
        right = hyp_nums[index] if index < len(hyp_nums) else ""
        if normalize_equation(left) != normalize_equation(right):
            found[CLASS_EQUATION] = (left, right)
            return


def _scripts(found: dict[str, tuple[str, str]], reference: str, hypothesis: str) -> None:
    ref = unicode_script_tokens(reference)
    hyp = unicode_script_tokens(hypothesis)
    if ref == hyp:
        return
    found[CLASS_SCRIPT] = (_tokens(ref), _tokens(hyp))


def _tokens(tokens: list[tuple[str, str, str]]) -> str:
    if not tokens:
        return ""
    return " ".join(f"{base}_{sub}^{sup}" for base, sub, sup in tokens)


def _window(text: str, index: int) -> str:
    if index < 0 or not text:
        return ""
    start = max(0, index - 3)
    end = min(len(text), index + 4)
    return text[start:end]
