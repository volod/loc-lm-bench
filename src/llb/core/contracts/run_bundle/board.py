"""Board-side analysis records: the misses a run made and what they point at."""

from typing import Final, Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.run_bundle.common import RunBundleRow

MISS_ANALYSIS_SCHEMA_ID: Final = "llb.miss-analysis"
MISS_CASE_SCHEMA_ID: Final = "llb.miss-case"


class MissCaseRow(ArtifactContract):
    """One classified miss (`misses.jsonl`): a pointer a human follows back into the run."""

    schema_id: Literal["llb.miss-case"] = MISS_CASE_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    item_id: str = Field(min_length=1)
    miss_class: str = Field(min_length=1)
    status: str
    objective_score: float
    judge_score: float | None = None
    retrieval_hit: bool
    first_hit_rank: int | None = None
    question: str
    source_doc_id: str
    topic: str
    question_type: str
    answer_preview: str
    retrieved_docs: list[str] = Field(default_factory=list)


class MissClusterRow(RunBundleRow):
    """Miss density for one cluster key (document, topic, or question type)."""

    key: str
    n_misses: int = Field(ge=0)
    n_cases: int = Field(ge=0)
    miss_rate: float


class MissAnalysisDocument(ArtifactContract):
    """`analysis.json`: the machine-readable miss summary `llb recommend` reads."""

    schema_id: Literal["llb.miss-analysis"] = MISS_ANALYSIS_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    created_at: str = Field(min_length=1)
    run_dir: str
    model: str
    backend: str
    split: str
    n_cases: int = Field(ge=0)
    n_misses: int = Field(ge=0)
    threshold: float
    rag_config: JsonObject
    class_counts: dict[str, int]
    clusters: dict[str, list[MissClusterRow]]
    # A deeper or shallower retrieval probe's readings, and the tuning moves they support. Both
    # are shaped by the probe and the recommendation vocabulary that produced them.
    probes: list[JsonObject] = Field(default_factory=list)
    recommendations: list[JsonObject] = Field(default_factory=list)
