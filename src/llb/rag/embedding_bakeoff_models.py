"""Contracts of the embedder bake-off: the per-item input, the store seam, and the report rows.

Kept apart from the scoring lane (`llb.rag.embedding_bakeoff`) for the same reason
`fusion_evidence/models.py` is: the row shape is read by the scorer, the report renderer, and the
CLI, while the build/score orchestration is read by none of them. A candidate row carries THREE
kinds of fact -- retrieval quality, build cost (throughput / size / device / API spend), and the
paired uncertainty of its lead over the baseline embedder -- and every consumer sees the same one.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.rag import SourceSpanRecord
from llb.rag.candidate_screen import SkippedCandidate
from llb.rag.encoder_throughput import HostThroughputSummary, ThroughputProfile
from llb.rag.embedding_bakeoff_uncertainty import (
    BakeoffVerdict,
    PairedRow,
    UncertaintySettings,
)

if TYPE_CHECKING:  # imported lazily: the floor is opt-in and costs an extra retrieval pass
    from llb.rag.noise_floor_models import NoiseFloorReport

# (question, gold source spans) -- the per-item input shared across every candidate.
BakeoffItem = tuple[str, list[SourceSpanRecord]]

KIND_LOCAL = "local"
KIND_API = "api"

# Default LOCAL candidates for Ukrainian RAG, in three bands: the incumbents (current default
# first, then its cheap CUDA sibling and the larger retrieval-tuned alternatives), the current
# multilingual retrieval generation an operator would actually shortlist today, and the
# paraphrase/STS model whose objective differs (which is why the ranking is measured, not assumed).
# Every id here MUST carry a declared convention in `llb.rag.embedding_families`; two of the
# current-generation rows need `trust_remote_code` and are skipped unless opted into
# (`llb.rag.embedding_bakeoff_roster`).
DEFAULT_LOCAL_CANDIDATES = [
    "intfloat/multilingual-e5-base",  # current RunConfig default
    "intfloat/multilingual-e5-small",  # sub-base retrieval-tuned; cheap CUDA alternative
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large-instruct",  # instruct-format sibling of e5-large
    "Alibaba-NLP/gte-multilingual-base",  # trust_remote_code
    "jinaai/jina-embeddings-v3",  # trust_remote_code; task LoRA adapters
    "Qwen/Qwen3-Embedding-0.6B",  # decoder-based, instruct query format
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
]

BYTES_PER_MB = 1024 * 1024


def slugify_model(model: str) -> str:
    """Filesystem-safe slug for a model id, for the per-candidate store directory."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "model"


@dataclass
class BuiltStore:
    """One built candidate store plus the build measurements the report ranks on.

    `store` exposes `.retrieve(question, k) -> list[ChunkRecord]` and `.meta` (dim / n_indexed /
    embedding_model). `cost_usd` is set only for the API row. `throughput_profile` is set when
    `compare-embeddings` runs the cold/warm encoder decomposition.
    """

    store: Any
    embed_seconds: float
    index_bytes: int
    kind: str = KIND_LOCAL
    device: str | None = None
    cost_usd: float | None = None
    throughput_profile: ThroughputProfile | None = None


# embedding_model -> BuiltStore. The CLI binds the heavy real builder; tests inject a fake.
StoreBuilder = Callable[[str], BuiltStore]


class CandidateResult(TypedDict):
    """One embedder's row: retrieval quality plus throughput / size / device fit.

    `family` is the declared query/passage convention the row was scored under
    (`llb.rag.embedding_families`); without it on the row, a reader cannot tell whether two
    candidates were even asked the same question.
    """

    model: str
    kind: str
    family: str
    recall_at_k: float
    mrr: float
    n: int
    k: int
    dim: int
    n_indexed: int
    embed_seconds: float
    index_bytes: int
    device: NotRequired[str]
    cost_usd: NotRequired[float]
    # True when this row was scored with repository-supplied modelling code executing.
    trust_remote_code: NotRequired[bool]
    # Cold/warm encoder decomposition (optional; present when --encoder-throughput ran).
    throughput_profile: NotRequired[ThroughputProfile]
    # Paired percentile-bootstrap delta against the baseline embedder over shared resample index
    # sets, plus the item-level ledger -- absent only when the baseline was not scored in this run.
    # This is the reading that says whether a point-estimate lead is an item set or a ranking.
    paired_vs_baseline: NotRequired[PairedRow]


class BakeoffReport(TypedDict):
    k: int
    n: int
    corpus_root: str
    candidates: list[CandidateResult]
    best_recall: str | None
    # Roster entries that produced no row (see `llb.rag.embedding_bakeoff_roster`). Present only
    # when something was skipped, so a report that ranks fewer models than the roster names says so.
    skipped: NotRequired[list[SkippedCandidate]]
    # How the paired intervals on the rows were drawn (baseline / resamples / confidence / seed),
    # so the report is reproducible from its own header.
    uncertainty: NotRequired[UncertaintySettings]
    # Adopt-or-retain: point-estimate rank order is NOT the recommendation; this is.
    verdict: NotRequired[BakeoffVerdict]
    # Per-item metric vectors retained so a later run can price a declared paired delta before
    # rebuilding any store. Item order is the stable identity this pure lane receives.
    paired_items: NotRequired[list[dict[str, Any]]]
    power_analysis: NotRequired[dict[str, Any]]
    # Measurement floor across the candidate stores, present only when it was asked for
    # (`compare-embeddings --noise-floor`). This lane is exactly where a sub-item recall delta
    # gets read as a recommendation, so the floor states whether the winner is separated from
    # the runner-up at all. See `llb.rag.noise_floor`.
    noise_floor: NotRequired["NoiseFloorReport"]
    # Cold/warm encoder decomposition host summary (optional; see `llb.rag.encoder_throughput`).
    encoder_throughput: NotRequired[HostThroughputSummary]
