"""The run manifest: the immutable head of every run bundle this project publishes.

A run bundle is a SET of files -- the manifest, the per-case score rows, an optional retrieval
sidecar, and whatever additional records the method that produced it published beside them. The
manifest is what makes the set addressable: it names the run, pins the configuration and
environment that produced it, carries the headline metrics a board ranks on, and (from version 2)
SAYS what every other member of the bundle is.
"""

from typing import Final, Literal

from pydantic import Field, model_validator

from llb.core.contracts.artifacts import DIGEST_PATTERN, ArtifactContract, ContractReference
from llb.core.contracts.common import JsonObject
from llb.core.contracts.hardware import ContentionReport, TelemetryReport
from llb.core.contracts.judging import JudgeStatus
from llb.core.contracts.rag import RetrievalMetrics
from llb.core.contracts.run_bundle.common import RunBundleRow
from llb.core.contracts.run_bundle.studies import (
    STUDY_ANALYSIS_SCHEMA_ID,
    STUDY_DESIGN_SCHEMA_ID,
)
from llb.core.contracts.runs import DurabilityStatus, RunEnvironment, RunMetrics

RUN_MANIFEST_SCHEMA_ID: Final = "llb.run-manifest"
STUDY_RECORD_SCHEMA_IDS = (STUDY_DESIGN_SCHEMA_ID, STUDY_ANALYSIS_SCHEMA_ID)


class ScoreRowsDeclaration(RunBundleRow):
    """What the rows of this bundle's `scores.jsonl` answer to.

    A run evaluation and the six category benchmarks each write a row shape this project models,
    so those bundles name the registered contract. A benchmark STUDY writes its own cell, seed, or
    crossover table instead -- one row per grid cell, with the columns that study measured -- and
    those columns belong to the study rather than to a family a cross-cutting reader could know
    ahead of time. Such a bundle names the study that owns the rows and the exact column set it
    published, which is a weaker claim than a contract and a checkable one: a reader can still ask
    whether the rows it just opened are the rows the run said it wrote.
    """

    record_contract: ContractReference | None = None
    owner: str | None = Field(default=None, min_length=1)
    columns: list[str] | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> "ScoreRowsDeclaration":
        study_form = self.owner is not None and self.columns is not None
        if (self.record_contract is not None) == study_form:
            raise ValueError(
                "score rows declare either a record contract or an owning study with its columns"
            )
        return self


class RunArtifactDeclaration(RunBundleRow):
    """One additional file published inside the bundle, and what makes it readable.

    `persist_run` accepts nothing here it cannot describe: an artifact either resolves to a
    registered contract that validated its content before publication, or it is a human report --
    a rendered table or narrative nobody parses -- and says so in `human_report`, which carries
    the reason the exemption applies rather than merely asserting one.
    """

    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    digest: str = Field(pattern=DIGEST_PATTERN)
    n_bytes: int = Field(ge=0)
    record_contract: ContractReference | None = None
    human_report: str | None = Field(default=None, min_length=1)
    # The study a design or analysis belongs to. A design names itself, but an analysis is written
    # beside the design that produced it and states nothing, so without this the bundle would hold
    # a reading nobody could attribute -- and attribution is the whole value of an archived one.
    study_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_declaration(self) -> "RunArtifactDeclaration":
        if (self.record_contract is None) == (self.human_report is None):
            raise ValueError(
                "an additional artifact declares either a record contract or a human-report reason"
            )
        contract = self.record_contract
        study = contract is not None and contract.schema_id in STUDY_RECORD_SCHEMA_IDS
        if study and self.study_id is None:
            raise ValueError("a study record artifact must name the study it belongs to")
        if not study and self.study_id is not None:
            raise ValueError("only a study record artifact names a study")
        return self


class RunManifestV1(ArtifactContract):
    """`manifest.json` as this project wrote it before the registry existed.

    Every field here is exactly what a pre-contract bundle carries, which is why version 1 is the
    version a bundle with no identity is read at. What it could NOT say is what the rest of its
    own directory is: the score rows had no declared contract and an additional artifact was any
    name with any content.
    """

    schema_id: Literal["llb.run-manifest"] = RUN_MANIFEST_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    split: str | None = None
    created_at: str = Field(min_length=1)
    config: JsonObject
    env: RunEnvironment
    metrics: RunMetrics | None = None
    retrieval: RetrievalMetrics | None = None
    judge: JudgeStatus | None = None
    telemetry: TelemetryReport | None = None
    contention: ContentionReport | None = None
    durability: DurabilityStatus | None = None
    prompt_system_provenance: JsonObject | None = None
    # Set only by a context lane that laid whole documents into the prompt: the declared window,
    # the window the backend was probed as serving, and which of the two bound the skip threshold.
    context_window: JsonObject | None = None
    n_cases: int = Field(default=0, ge=0)


class RunManifestDocument(RunManifestV1):
    """The current manifest: the bundle head, plus what every other member of the bundle is.

    Version 2 extends version 1 rather than restating it -- the run identity, config, environment,
    and metrics did not move -- and adds the two declarations a downstream reader could previously
    only guess at from filenames.
    """

    schema_version: Literal["2.0.0"] = "2.0.0"  # type: ignore[assignment]
    # Absent in a bundle migrated from version 1: an older run never recorded what its rows
    # answered to. A reader treats that as "this bundle does not state its score-row contract",
    # never as "its rows have none".
    score_rows: ScoreRowsDeclaration | None = None
    # Empty in a bundle migrated from version 1, for the same reason.
    artifacts: list[RunArtifactDeclaration] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact_names(self) -> "RunManifestDocument":
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("additional artifact names must be unique")
        return self
