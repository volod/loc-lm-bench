"""A published run bundle described as a dataset, plus the auto-RAG run it may have come from.

A run bundle is the project's primary downstream API and, like a vector store, it is a SET of
files that only mean something together: the manifest says what ran, the score rows say how each
case did, the retrieval rows say what each case's context held, and the probe rows say what the
same model did without one. Describing it is what lets `persist_run` validate every member before
it renames staging into place, and what lets a board or an outside consumer answer "can I read
this bundle" once instead of per file.

Which contract the score rows are bound to is the producer's to state: an evaluation run writes
`llb.case-score` rows and a benchmark category run writes `llb.benchmark-cell` rows, and no
inspection of a legacy bundle's columns can tell the two apart reliably. The published
`dataset_manifest.json` records the answer, which is what makes a published bundle
self-describing; a bundle written before this existed is described with the kind its reader
supplies.
"""

from pathlib import Path

from llb.artifacts.datasets import MemberSpec, describe_dataset
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.runs.members import RunMember, structured_members
from llb.core.contracts.artifacts import DatasetManifest
from llb.core.contracts.orchestration import (
    AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID,
    AUTO_RAG_MANIFEST_SCHEMA_ID,
    AUTO_RAG_RECOMMENDATION_SCHEMA_ID,
    AUTO_RAG_STAGE_LINKS_SCHEMA_ID,
)
from llb.core.contracts.run_bundle import (
    BENCHMARK_CELL_SCHEMA_ID,
    CASE_RETRIEVAL_SCHEMA_ID,
    CASE_SCORE_SCHEMA_ID,
    CONTEXT_PROBE_SCHEMA_ID,
    RUN_ABORT_SCHEMA_ID,
)
from llb.core.contracts.runs import RUN_MANIFEST_SCHEMA_ID

RUN_BUNDLE_DATASET_ID = "llb-run-bundle"
AUTO_RAG_DATASET_ID = "llb-auto-rag-run"
MISS_ANALYSIS_DATASET_ID = "llb-miss-analysis"
AGENT_PROFILE_DATASET_ID = "llb-agent-profile"

MANIFEST_FILE = "manifest.json"
SCORES_FILE = "scores.jsonl"
RETRIEVAL_FILE = "retrieval.jsonl"
PROBES_FILE = "probes.jsonl"
ABORT_FILE = "scorer/abort.json"

# Which score-row contract a bundle's `scores.jsonl` is bound to. `run` is one evaluation over a
# gold set; `benchmark` is one category cell whose columns that lane names.
KIND_RUN = "run"
KIND_BENCHMARK = "benchmark"
CASE_CONTRACTS = {KIND_RUN: CASE_SCORE_SCHEMA_ID, KIND_BENCHMARK: BENCHMARK_CELL_SCHEMA_ID}

AUTO_RAG_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("auto-rag-manifest", "manifest.json", AUTO_RAG_MANIFEST_SCHEMA_ID),
    MemberSpec("journal", "journal.jsonl", AUTO_RAG_JOURNAL_EVENT_SCHEMA_ID, "jsonl"),
    MemberSpec("stage-links", "artifacts.json", AUTO_RAG_STAGE_LINKS_SCHEMA_ID),
    MemberSpec(
        "recommendation", "rag_recommendation.yaml", AUTO_RAG_RECOMMENDATION_SCHEMA_ID, "yaml"
    ),
)


def run_bundle_members(
    case_schema_id: str = CASE_SCORE_SCHEMA_ID, extra: tuple[RunMember, ...] = ()
) -> tuple[MemberSpec, ...]:
    """Every project-owned member a run bundle can hold, canonical members first."""
    additional = tuple(
        MemberSpec(_member_id(member.name), member.name, member.schema_id)
        for member in structured_members(extra)
    )
    return (
        MemberSpec("run-manifest", MANIFEST_FILE, RUN_MANIFEST_SCHEMA_ID),
        MemberSpec("scores", SCORES_FILE, case_schema_id, "jsonl"),
        MemberSpec("retrieval", RETRIEVAL_FILE, CASE_RETRIEVAL_SCHEMA_ID, "jsonl"),
        MemberSpec("probes", PROBES_FILE, CONTEXT_PROBE_SCHEMA_ID, "jsonl"),
        MemberSpec("budget-abort", ABORT_FILE, RUN_ABORT_SCHEMA_ID),
        *additional,
    )


def run_bundle_manifest(
    run_dir: Path | str,
    *,
    kind: str = KIND_RUN,
    extra: tuple[RunMember, ...] = (),
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> DatasetManifest:
    """Describe one run bundle: its manifest, its rows, and the sidecars its lane added."""
    if kind not in CASE_CONTRACTS:
        raise ValueError(f"unknown run bundle kind {kind!r}; choose {' | '.join(CASE_CONTRACTS)}")
    return describe_dataset(
        Path(run_dir),
        RUN_BUNDLE_DATASET_ID,
        "One published run bundle: what ran, how every case scored, and what it was served.",
        run_bundle_members(CASE_CONTRACTS[kind], extra),
        registry,
    )


def auto_rag_manifest(
    run_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one auto-RAG run directory: the settings, the journal, and what it recommended."""
    return describe_dataset(
        Path(run_dir),
        AUTO_RAG_DATASET_ID,
        "One auto-RAG run: the settings it is resumable against and the configuration it picked.",
        AUTO_RAG_MEMBERS,
        registry,
    )


def _member_id(name: str) -> str:
    """A stable member id for an added sidecar: its file name without the extension."""
    return Path(name).stem
