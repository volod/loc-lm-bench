"""Record-linkage bundle contracts.

`settings.json` is the one member a replay must read before it re-scores anything: it carries the
whole decision -- what was compared, where it was blocked, and where it was cut. The two versions
are the two forms this project has written. The older one omits the tuning knobs that were added
after it, and `LinkageSpec.from_payload` filled them from its own defaults; the current one states
every knob, so a bundle records the run's settings instead of a reader's build's defaults.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.corpus import DataPrepRow


class LinkageSpecRecordV1(DataPrepRow):
    """The pre-contract specification payload: every tuning knob may be absent."""

    comparisons: list[JsonObject] = Field(default_factory=list)
    blocking_rules: list[JsonObject] = Field(default_factory=list)
    training_rules: list[JsonObject] = Field(default_factory=list)
    unique_id_column: str | None = None
    retain_columns: list[str] | None = None
    match_threshold: float | None = None
    max_pairs: int | None = None
    seed: int | None = None
    em_max_iterations: int | None = None
    em_convergence: float | None = None
    random_match_probability: float | None = None
    duckdb_threads: int | None = None
    min_level_probability: float | None = None
    retain_matching_columns: bool | None = None


class LinkageSpecRecord(DataPrepRow):
    """The current specification payload: every knob the run used, stated."""

    comparisons: list[JsonObject] = Field(default_factory=list)
    blocking_rules: list[JsonObject] = Field(default_factory=list)
    training_rules: list[JsonObject] = Field(default_factory=list)
    unique_id_column: str = Field(min_length=1)
    retain_columns: list[str] = Field(default_factory=list)
    match_threshold: float = Field(gt=0.0, le=1.0)
    max_pairs: int = Field(gt=0)
    seed: int
    em_max_iterations: int = Field(ge=0)
    em_convergence: float = Field(ge=0.0)
    random_match_probability: float = Field(ge=0.0, le=1.0)
    duckdb_threads: int = Field(ge=1)
    min_level_probability: float = Field(ge=0.0, lt=0.5)
    retain_matching_columns: bool


class LinkageSummaryRecord(DataPrepRow):
    """The numbers an operator reads first, and the run bundle's metadata block."""

    n_records: int = Field(ge=0)
    n_scored_pairs: int = Field(ge=0)
    n_matched_pairs: int = Field(ge=0)
    n_clusters: int = Field(ge=0)
    n_multi_record_clusters: int = Field(ge=0)
    largest_cluster: int = Field(ge=0)
    match_threshold: float = Field(gt=0.0, le=1.0)
    seed: int
    trained_from_labels: bool
    n_accuracy_points: int = Field(ge=0)
    n_untrained_levels: int = Field(ge=0)


class LinkageSettingsV1(ArtifactContract):
    """The pre-contract bundle settings: a specification that leans on a reader's defaults."""

    schema_id: Literal["llb.linkage-settings"]
    schema_version: Literal["1.0.0"]
    specification: LinkageSpecRecordV1
    summary: LinkageSummaryRecord
    metadata: JsonObject | None = None


class LinkageSettings(ArtifactContract):
    """The current bundle settings: the whole decision, replayable without this build."""

    schema_id: Literal["llb.linkage-settings"]
    schema_version: Literal["2.0.0"]
    specification: LinkageSpecRecord
    summary: LinkageSummaryRecord
    metadata: JsonObject | None = None
