"""The gold-item record table and the comparison specification it is linked under.

A gold item carries more than the one question cosine the shipped drop constant reads: the
question and reference-answer embeddings, the document it was drafted from, the character span it
cites, and its question type. Each becomes an agreement ladder, so a drop decision arrives as a
match probability with the agreements behind it instead of a single number over a single field.

Split is retained but NOT compared: the drafting pipeline assigns splits AFTER deduplication, so
every candidate carries the same placeholder at drop time. Agreement on it would price when a
field gets filled in, not whether two items are the same question.
"""

from llb.core.contracts.common import JsonObject
from llb.goldset.schema import GoldItem
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec
from llb.linkage.constants import (
    KIND_ARRAY_INTERSECT,
    KIND_COSINE,
    KIND_EXACT,
)
from llb.prep.ontology.constants import QUESTION_TYPE_MULTI_HOP
from llb.prep.ontology.drafting.question_types import classify_question_type
from llb.prep.ontology.extraction.dedup import QuestionEmbedder, Vector
from llb.prep.ontology.linkage.constants import (
    ANSWER_COSINE_THRESHOLDS,
    ANSWER_VECTOR_COLUMN,
    BLOCK_KEY_COLUMN,
    BLOCK_KEY_VALUE,
    ITEM_ID_COLUMN,
    QUESTION_COSINE_THRESHOLDS,
    QUESTION_TYPE_COLUMN,
    QUESTION_VECTOR_COLUMN,
    ROLE_CANDIDATE,
    ROLE_COLUMN,
    ROLE_PRIOR,
    SOURCE_DOC_COLUMN,
    SPAN_BLOCKS_COLUMN,
    SPAN_BLOCK_CHARS,
    SPAN_BLOCK_SIZES,
    SPLIT_COLUMN,
)
from llb.prep.ontology.models import ItemLabels


def question_type_of(item: GoldItem, label: ItemLabels | None = None) -> str:
    """The item's question type: the recorded label, else the same derivation the drafter used.

    Prior bundles ship gold items without their review labels, so a prior row's type is derived --
    multi-span items are the multi-hop drafter's output, and everything else goes through the
    classifier that labelled the flat drafts. Both sides of a pair therefore get the same rule.
    """
    if label is not None:
        return label.question_type
    if len(item.source_spans) > 1:
        return QUESTION_TYPE_MULTI_HOP
    return classify_question_type(item.question, item.reference_answer)


def span_blocks(item: GoldItem) -> list[str]:
    """The character grid cells the item's source spans cover, as `<doc-id>:<cell>` keys."""
    cells: set[str] = set()
    for span in item.source_spans:
        first = span.char_start // SPAN_BLOCK_CHARS
        last = max(span.char_end - 1, span.char_start) // SPAN_BLOCK_CHARS
        cells.update(f"{span.doc_id}:{cell}" for cell in range(first, last + 1))
    return sorted(cells)


def record_id(role: str, index: int) -> str:
    """A table identifier that stays unique when a prior bundle and a draft batch share item ids."""
    return f"{role}:{index:05d}"


def embed_columns(
    embedder: QuestionEmbedder, items: list[GoldItem]
) -> tuple[list[Vector], list[Vector]]:
    """Embed every question, then every reference answer -- one call per column.

    One call per column is what keeps a column's vectors the same width: an embedder is only
    required to be consistent within a call, and a fixed-width DuckDB array column is what the
    cosine comparison is defined on.
    """
    if not items:
        return [], []
    return (
        embedder.embed([item.question for item in items]),
        embedder.embed([item.reference_answer for item in items]),
    )


def build_records(
    prior_items: list[GoldItem],
    candidates: list[GoldItem],
    candidate_labels: dict[str, ItemLabels],
    question_vectors: list[Vector],
    answer_vectors: list[Vector],
) -> tuple[list[JsonObject], dict[str, GoldItem]]:
    """One record per gold item, priors first, plus the record id -> item map the report reads."""
    items = [*prior_items, *candidates]
    if len(question_vectors) != len(items) or len(answer_vectors) != len(items):
        raise ValueError("question and answer vectors must align with the record table")
    records: list[JsonObject] = []
    by_record: dict[str, GoldItem] = {}
    for index, item in enumerate(items):
        prior = index < len(prior_items)
        role = ROLE_PRIOR if prior else ROLE_CANDIDATE
        identifier = record_id(role, index)
        by_record[identifier] = item
        records.append(
            {
                "unique_id": identifier,
                ITEM_ID_COLUMN: item.id,
                ROLE_COLUMN: role,
                SPLIT_COLUMN: item.split,
                BLOCK_KEY_COLUMN: BLOCK_KEY_VALUE,
                QUESTION_VECTOR_COLUMN: question_vectors[index],
                ANSWER_VECTOR_COLUMN: answer_vectors[index],
                SOURCE_DOC_COLUMN: item.source_doc_id,
                SPAN_BLOCKS_COLUMN: span_blocks(item),
                QUESTION_TYPE_COLUMN: question_type_of(
                    item, None if prior else candidate_labels.get(item.id)
                ),
            }
        )
    return records, by_record


def build_gold_item_spec(question_dimension: int, answer_dimension: int) -> LinkageSpec:
    """The gold-item comparison specification at the embedding widths the run produced."""
    spec = LinkageSpec(
        comparisons=(
            ComparisonSpec(
                column=QUESTION_VECTOR_COLUMN,
                kind=KIND_COSINE,
                thresholds=QUESTION_COSINE_THRESHOLDS,
                dimension=question_dimension,
            ),
            ComparisonSpec(
                column=ANSWER_VECTOR_COLUMN,
                kind=KIND_COSINE,
                thresholds=ANSWER_COSINE_THRESHOLDS,
                dimension=answer_dimension,
            ),
            ComparisonSpec(column=SOURCE_DOC_COLUMN, kind=KIND_EXACT),
            ComparisonSpec(
                column=SPAN_BLOCKS_COLUMN,
                kind=KIND_ARRAY_INTERSECT,
                thresholds=SPAN_BLOCK_SIZES,
            ),
            ComparisonSpec(column=QUESTION_TYPE_COLUMN, kind=KIND_EXACT),
        ),
        blocking_rules=(BlockingRule((BLOCK_KEY_COLUMN,)),),
        # Each expectation-maximisation pass holds one compared column fixed and learns the rest;
        # the two rules cover each other, so no comparison goes untrained.
        training_rules=(
            BlockingRule((SOURCE_DOC_COLUMN,)),
            BlockingRule((QUESTION_TYPE_COLUMN,)),
        ),
        retain_columns=(ITEM_ID_COLUMN, ROLE_COLUMN, SPLIT_COLUMN, BLOCK_KEY_COLUMN),
    )
    spec.validate()
    return spec
