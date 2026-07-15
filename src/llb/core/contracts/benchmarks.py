"""Category benchmark inputs, per-case rows, and aggregate reports."""

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.common import JsonObject


class PlantedLabelRecord(TypedDict):
    """One planted ground-truth label for the text-analysis benchmark."""

    label_id: str
    kind: str
    value: str
    aliases: NotRequired[list[str]]
    doc_id: NotRequired[str]
    char_start: NotRequired[int]
    char_end: NotRequired[int]
    attrs: NotRequired[JsonObject]
    scoring: NotRequired[str]


class SubtaskScore(TypedDict):
    """Objective recovery score for one text-analysis subtask over one document."""

    kind: str
    objective: bool
    n_labels: int
    n_pred: int
    matched: list[tuple[str, float]]
    precision: float
    recall: float
    f1: float


class TextAnalysisCaseRow(TypedDict):
    """Per-document objective score for one text-analysis case."""

    item_id: str
    status: str
    objective_score: float
    n_objective_subtasks: int
    n_labels: int
    subtask_f1_json: str
    judged_quality: NotRequired[float]
    long_doc_answer: NotRequired[str]


class ReliabilityReport(TypedDict):
    """Counts and reliability for the shared benchmark failure taxonomy."""

    n: int
    n_ok: int
    reliability: float
    failures: dict[str, int]


class SummarizationCaseRow(TypedDict):
    """Per-case outcome for a summarization benchmark case."""

    item_id: str
    status: str
    coverage: float
    objective_score: float
    faithfulness: NotRequired[float]
    answer_preview: str


class StructuredCaseRow(TypedDict):
    """Per-case outcome for a structured-output benchmark case."""

    item_id: str
    conformant: float
    field_accuracy: float
    score: float
    objective_score: float


class AgenticCaseRow(TypedDict):
    """Per-task outcome for one agentic episode."""

    item_id: str
    status: str
    success: float
    objective_score: float
    n_steps: int
    n_tool_calls: int
    trajectory_quality: NotRequired[float]
    answer_preview: str


class ToolDef(TypedDict):
    """An OpenAI-style function definition used by tooling benchmarks."""

    name: str
    description: str
    parameters: JsonObject


class ToolingCaseRow(TypedDict):
    """Per-case outcome for a tooling or function-calling benchmark case."""

    item_id: str
    expected_tool: str | None
    called_tool: str | None
    attempted: bool
    tool_selected: float
    schema_valid: float
    arguments_exact: float
    no_hallucinated_tool: float
    well_formed: float
    correct: float
    objective_score: float


class SecurityCaseRow(TypedDict):
    """Per-case outcome for a security attack or benign-control case."""

    item_id: str
    family: str
    benign: bool
    expect_refusal: bool
    status: str
    attack_success: float
    defended: float
    objective_score: NotRequired[float]
    refused: float
    appropriate_refusal: float
    refusal_quality: NotRequired[float]
    lang: NotRequired[str]
    xlang_group: NotRequired[str]
    pair_id: NotRequired[str]
    answer_preview: str
