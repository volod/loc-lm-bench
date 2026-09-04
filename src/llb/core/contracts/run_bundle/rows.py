"""Row contracts for the per-case files a run bundle publishes.

Every one of these rows already had a `TypedDict` the producers build directly, and those
declarations are where the columns and their meanings are documented -- `CaseScoreRow` alone
carries fifty of them, each with the reading it stands for. Restating that column list a second
time as a Pydantic model would put the two copies one release apart the first time a metric is
added, and the copy the registry validated would be the stale one.

So the contract model is DERIVED from the TypedDict the producer already writes: a required key
becomes a required field, a `NotRequired` key becomes an optional one (a column a run did not
measure is absent, never zero), and the identity fields are added on top. The derivation is
deterministic, so the generated JSON Schema is stable, and a column added to the TypedDict reaches
the schema and the registry in the same change that adds it.
"""

from typing import Literal, Optional, cast, get_args, get_type_hints

from pydantic import create_model
from typing_extensions import NotRequired, get_origin

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.benchmarks import (
    AgenticCaseRow,
    SecurityCaseRow,
    StructuredCaseRow,
    SummarizationCaseRow,
    TextAnalysisCaseRow,
    ToolingCaseRow,
)
from llb.core.contracts.rag import CaseRetrievalRecord
from llb.core.contracts.results import CaseScoreRow

CASE_SCORE_SCHEMA_ID = "llb.case-score"
RETRIEVAL_CASE_SCHEMA_ID = "llb.retrieval-case"
AGENTIC_CASE_SCHEMA_ID = "llb.agentic-case"
SECURITY_CASE_SCHEMA_ID = "llb.security-case"
STRUCTURED_CASE_SCHEMA_ID = "llb.structured-case"
SUMMARIZATION_CASE_SCHEMA_ID = "llb.summarization-case"
TEXT_ANALYSIS_CASE_SCHEMA_ID = "llb.text-analysis-case"
TOOLING_CASE_SCHEMA_ID = "llb.tooling-case"


def _column(annotation: object) -> object:
    """The field type one TypedDict key contributes, with its `NotRequired` wrapper removed."""
    if get_origin(annotation) is NotRequired:
        return get_args(annotation)[0]
    return annotation


def row_contract(
    columns: type, *, model_name: str, schema_id: str, version: str, description: str
) -> type[ArtifactContract]:
    """The strict row contract for a `TypedDict` the producers already build.

    A row is written in its compact form -- the columns and nothing else -- because a bundle holds
    one per scored case and stamping an identity on every line would multiply the file for no
    reader. The identity lives on the model so the registry can dispatch a row a manifest binds,
    and `ContractRegistry.read_as` stamps it at read time.
    """
    hints = get_type_hints(columns, include_extras=True)
    required = set(getattr(columns, "__required_keys__", frozenset()))
    fields: dict[str, object] = {
        "schema_id": (Literal[schema_id], ...),
        "schema_version": (Literal[version], ...),
    }
    for name, annotation in hints.items():
        column = _column(annotation)
        fields[name] = (column, ...) if name in required else (Optional[column], None)
    model = create_model(model_name, __base__=ArtifactContract, **fields)  # type: ignore[call-overload]
    model.__doc__ = description
    return cast(type[ArtifactContract], model)


CaseScoreRecord = row_contract(
    CaseScoreRow,
    model_name="CaseScoreRecord",
    schema_id=CASE_SCORE_SCHEMA_ID,
    version="1.0.0",
    description="One scored RAG evaluation case: its correctness, retrieval, and delivery columns.",
)

RetrievalCaseRecord = row_contract(
    CaseRetrievalRecord,
    model_name="RetrievalCaseRecord",
    schema_id=RETRIEVAL_CASE_SCHEMA_ID,
    version="1.0.0",
    description="What one case's scored context actually contained, against its gold spans.",
)

AgenticCaseRecord = row_contract(
    AgenticCaseRow,
    model_name="AgenticCaseRecord",
    schema_id=AGENTIC_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One agentic episode: its outcome, its loop counters, and its context accounting.",
)

SecurityCaseRecord = row_contract(
    SecurityCaseRow,
    model_name="SecurityCaseRecord",
    schema_id=SECURITY_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One security attack or benign-control case and the refusal it earned.",
)

StructuredCaseRecord = row_contract(
    StructuredCaseRow,
    model_name="StructuredCaseRecord",
    schema_id=STRUCTURED_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One structured-output case: schema conformance and per-field accuracy.",
)

SummarizationCaseRecord = row_contract(
    SummarizationCaseRow,
    model_name="SummarizationCaseRecord",
    schema_id=SUMMARIZATION_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One summarization case: the coverage it reached and its faithfulness.",
)

TextAnalysisCaseRecord = row_contract(
    TextAnalysisCaseRow,
    model_name="TextAnalysisCaseRecord",
    schema_id=TEXT_ANALYSIS_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One text-analysis document: the objective label recovery over its subtasks.",
)

ToolingCaseRecord = row_contract(
    ToolingCaseRow,
    model_name="ToolingCaseRecord",
    schema_id=TOOLING_CASE_SCHEMA_ID,
    version="1.0.0",
    description="One tooling case: which tool was selected and whether its arguments held.",
)
