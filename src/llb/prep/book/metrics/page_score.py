"""Score one reference page against one submission.

Character error is the edit distance of the reading strings. Dangerous substitutions are counted
beside it. A text-only lane scores formula terms as plain text and reports figure boxes, table
grids, and caption links absent. Where the blocks sit is scored in ``layout``.
"""

from llb.core.contracts.data_prep.book_blocks import UNCERTAINTY_CLASSES
from llb.prep.book.metrics.align import text_score
from llb.prep.book.metrics.constants import FIELD_FORMULA, FIELD_TABLE
from llb.prep.book.metrics.confusion import confusion_counts, script_events, unicode_script_tokens
from llb.prep.book.metrics.latex import (
    FormulaRead,
    LatexError,
    formula_fidelity,
    latex_parses,
    script_tokens,
    terms_of,
)
from llb.prep.book.metrics.layout import captions, figures, order, pairs, printed
from llb.prep.book.metrics.reading import (
    block_reading,
    cell_labels,
    collapsed,
    page_reading,
    reference_reading,
)
from llb.prep.book.metrics.reference import RefBlock, RefPage
from llb.prep.book.metrics.report import (
    MetricValue,
    PageScore,
    absent,
    maybe,
    not_applicable,
    scored,
)
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.metrics.tables import Cell, table_distance, table_size


def score_page(page: RefPage, submission: PageSubmission) -> PageScore:
    """Every metric for one page. The substitution counts are not added to the error rates."""
    reference_text = page_reading(
        (block.reading_order, reference_reading(block)) for block in page.blocks
    )
    hypothesis_text = _hypothesis_text(submission)
    text = text_score(reference_text, hypothesis_text)
    counts = confusion_counts(
        reference_text, hypothesis_text, _script_count(page, submission, hypothesis_text)
    )
    fidelity, parse_valid = _formulas(page, submission, hypothesis_text)
    return PageScore(
        lane=submission.lane,
        pdf_page=page.pdf_page,
        page_class=page.page_class,
        character_error_rate=maybe(text.character_error_rate),
        word_error_rate=maybe(text.word_error_rate),
        missing_words=text.missing_words,
        duplicated_words=text.duplicated_words,
        reference_chars=text.reference_chars,
        reference_words=text.reference_words,
        reading_order_agreement=order(page, submission, hypothesis_text),
        dangerous_substitutions=counts,
        formula_fidelity=fidelity,
        formula_parse_valid=parse_valid,
        table_similarity=_tables(page, submission),
        figure_overlap=figures(page, submission),
        caption_association=captions(page, submission),
        printed_page_accuracy=printed(page, submission, hypothesis_text),
        submitted=submission.present,
        abstentions=_abstentions(submission),
    )


def _hypothesis_text(submission: PageSubmission) -> str:
    if submission.text_only:
        return collapsed(" ".join(block.text for block in submission.blocks))
    parts = [(block.reading_order, _hyp_reading(block)) for block in submission.blocks]
    return page_reading(parts)


def _hyp_reading(block: HypBlock) -> str:
    cells = cell_labels(block.table) if block.table is not None else None
    return block_reading(block.kind, block.text, block.equation_number, cells)


def _script_count(page: RefPage, submission: PageSubmission, hypothesis_text: str) -> int:
    reference: list[tuple[str, str, str]] = []
    for block in page.blocks:
        if block.kind == "formula" and block.latex:
            reference.extend(_scripts(block.latex))
        else:
            reference.extend(unicode_script_tokens(block.text))
    hypothesis: list[tuple[str, str, str]] = []
    if submission.text_only:
        hypothesis.extend(_scripts(hypothesis_text))
        hypothesis.extend(unicode_script_tokens(hypothesis_text))
    else:
        for item in submission.blocks:
            if item.latex:
                hypothesis.extend(_scripts(item.latex))
            else:
                hypothesis.extend(unicode_script_tokens(item.text))
    return script_events(reference, hypothesis)


def _scripts(source: str) -> list[tuple[str, str, str]]:
    try:
        return script_tokens(source)
    except LatexError:
        return []


def _formulas(
    page: RefPage, submission: PageSubmission, hypothesis_text: str
) -> tuple[MetricValue, bool | None]:
    formulas = [block for block in page.blocks if block.kind == "formula" and block.latex]
    parse_valid = _emitted_formulas_parse(submission)
    if not formulas:
        return not_applicable(), parse_valid
    if FIELD_FORMULA in submission.unsupported:
        return absent(), None
    if submission.text_only:
        plain = [
            formula_fidelity(block.latex or "", hypothesis_text, hypothesis_is_latex=False)
            for block in formulas
        ]
        return _fidelity(plain), None
    structured: list[FormulaRead] = []
    for reference, hypothesis in pairs(page.blocks, submission.blocks, "formula"):
        latex = reference.latex or ""
        if hypothesis is None:
            structured.append(_missed(latex))
        elif hypothesis.latex:
            structured.append(formula_fidelity(latex, hypothesis.latex, hypothesis_is_latex=True))
        else:
            structured.append(formula_fidelity(latex, hypothesis.text, hypothesis_is_latex=False))
    return _fidelity(structured), parse_valid


def _emitted_formulas_parse(submission: PageSubmission) -> bool | None:
    formulas = [block for block in submission.blocks if block.kind == "formula"]
    if submission.text_only or not formulas:
        return None
    return all(latex_parses(block.latex or block.text) for block in formulas)


def _missed(latex: str) -> FormulaRead:
    count = len(terms_of(latex))
    return FormulaRead(True, 0.0 if count else None, count, 0)


def _fidelity(reads: list[FormulaRead]) -> MetricValue:
    matched = sum(read.matched_terms for read in reads if read.fidelity is not None)
    total = sum(read.reference_terms for read in reads if read.fidelity is not None)
    if total == 0:
        return not_applicable()
    return scored(matched / total)


def _tables(page: RefPage, submission: PageSubmission) -> MetricValue:
    tables = [block for block in page.blocks if block.kind == "table" and block.table is not None]
    if not tables:
        return not_applicable()
    if submission.text_only or FIELD_TABLE in submission.unsupported:
        return absent()
    distance = 0
    total = 0
    for reference, hypothesis in pairs(page.blocks, submission.blocks, "table"):
        ref_rows = _ref_rows(reference)
        hyp_rows = (
            hypothesis.table if hypothesis is not None and hypothesis.table is not None else ()
        )
        distance += table_distance(ref_rows, hyp_rows)
        total += table_size(ref_rows) + table_size(hyp_rows)
    if total == 0:
        return not_applicable()
    return scored(1.0 - distance / total)


def _abstentions(submission: PageSubmission) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(UNCERTAINTY_CLASSES, 0)
    for name, count in submission.abstentions:
        if name in counts:
            counts[name] += count
    return counts


def _ref_rows(block: RefBlock) -> tuple[tuple[Cell, ...], ...]:
    if block.table is None:
        return ()
    return tuple(
        tuple(Cell(collapsed(cell.text), cell.colspan, cell.rowspan) for cell in row)
        for row in block.table.rows
    )
