"""Miss-analysis vocabulary: the disjoint miss classes, the tuning thresholds, the artifact
filenames, the question-type / topic heuristics, and the three record dataclasses.

Everything here is a leaf (constants + dataclasses + the report-template helper `_t`); the loader,
classifier, recommendation, and report modules build on it.
"""

from dataclasses import dataclass, field

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.core.contracts import JsonObject
from llb.eval import common as eval_common
from llb.prompts.registry import render_text

# Miss classes (each miss lands in exactly ONE, decided in precedence order; see classify_case).
MISS_RETRIEVAL = "retrieval_miss"
MISS_GENERATION = "generation_miss"
MISS_REFUSAL = "refusal"
MISS_ARTIFACT = "format_artifact"
MISS_JUDGE = "judge_disagreement"
MISS_CLASSES = (MISS_RETRIEVAL, MISS_GENERATION, MISS_REFUSAL, MISS_ARTIFACT, MISS_JUDGE)

# A scoreable (status=ok) case below this objective score counts as a miss.
DEFAULT_MISS_THRESHOLD = 0.5
# A low-objective case the trusted judge rated at/above this is a judge DISAGREEMENT, not a
# generation miss -- the two signals conflict, so a human (or the calibration gate) should look.
JUDGE_AGREEMENT_MIN = 0.7
# Terminal statuses that are output/transport artifacts rather than model knowledge failures.
ARTIFACT_STATUSES = frozenset({eval_common.EMPTY, eval_common.MALFORMED, ERR_TIMEOUT, ERR_BACKEND})

# Probe interpretation thresholds: a deeper probe CONFIRMS the retrieval hypothesis when it
# recovers at least this fraction of the retrieval misses; a shallower probe recommends
# lowering top_k only when its mean objective beats the subset baseline by at least this much.
PROBE_CONFIRM_MIN = 0.5
PROBE_MIN_OBJECTIVE_GAIN = 0.05

# A generation-miss cluster this large (and holding at least this share of generation misses)
# earns a "prompt-system dictionary terms" recommendation for the clustered document/topic.
DICTIONARY_CLUSTER_MIN = 2
DICTIONARY_CLUSTER_SHARE = 0.5

MISS_ANALYSIS_METHOD = "miss-analysis"
RETRIEVAL_FILENAME = "retrieval.jsonl"
MISSES_FILENAME = "misses.jsonl"
REPORT_FILENAME = "report.md"
ANALYSIS_FILENAME = "analysis.json"
ITEM_PROVENANCE_FILENAME = "item_provenance.jsonl"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# The RAG knobs cited as recommendation evidence (mirrors board/recommend RAG_CONFIG_KEYS).
RAG_CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")

# Question-type heuristic: leading interrogative -> coarse type (UA + EN). Used only when the
# goldset has no `item_provenance.jsonl` sidecar carrying drafted `question_type` labels.
_QUESTION_TYPE_MARKERS = (
    ("хто", "who"),
    ("кого", "who"),
    ("коли", "when"),
    ("де", "where"),
    ("куди", "where"),
    ("скільки", "how_many"),
    ("чому", "why"),
    ("навіщо", "why"),
    ("як", "how"),
    ("який", "which"),
    ("яка", "which"),
    ("яке", "which"),
    ("які", "which"),
    ("якого", "which"),
    ("якої", "which"),
    ("що", "what"),
    ("чим", "what"),
    ("чого", "what"),
    ("who", "who"),
    ("when", "when"),
    ("where", "where"),
    ("why", "why"),
    ("how", "how"),
    ("which", "which"),
    ("what", "what"),
)
DEFAULT_QUESTION_TYPE = "other"

# Topic heuristic: the longest content token of the question (casefolded), skipping
# interrogatives/particles. Coarse but deterministic; a provenance sidecar `topic` wins.
_TOPIC_MIN_TOKEN_CHARS = 4
_TOPIC_STOPWORDS = frozenset(marker for marker, _ in _QUESTION_TYPE_MARKERS) | frozenset(
    {
        "було",
        "буде",
        "цього",
        "цієї",
        "року",
        "році",
        "щодо",
        "через",
        "після",
        "перед",
        "тому",
        "того",
        "може",
        "does",
        "with",
        "from",
    }
)

CLUSTER_DIMENSIONS = ("document", "topic", "question_type")


def _t(name: str, **values: object) -> str:
    """Render a `board.miss.<name>` text template (report prose lives in template files)."""
    return render_text(f"board.miss.{name}", values)


@dataclass(slots=True)
class MissRecord:
    """One classified miss (a `misses.jsonl` line)."""

    item_id: str
    miss_class: str
    status: str
    objective_score: float
    judge_score: float | None
    retrieval_hit: bool
    first_hit_rank: int | None
    question: str
    source_doc_id: str
    topic: str
    question_type: str
    answer_preview: str

    def as_dict(self) -> JsonObject:
        return {
            "item_id": self.item_id,
            "miss_class": self.miss_class,
            "status": self.status,
            "objective_score": self.objective_score,
            "judge_score": self.judge_score,
            "retrieval_hit": self.retrieval_hit,
            "first_hit_rank": self.first_hit_rank,
            "question": self.question,
            "source_doc_id": self.source_doc_id,
            "topic": self.topic,
            "question_type": self.question_type,
            "answer_preview": self.answer_preview,
        }


@dataclass(slots=True)
class ClusterRow:
    """Miss density for one cluster key (document / topic / question type)."""

    key: str
    n_misses: int
    n_cases: int

    @property
    def miss_rate(self) -> float:
        return self.n_misses / self.n_cases if self.n_cases else 0.0

    def as_dict(self) -> JsonObject:
        return {
            "key": self.key,
            "n_misses": self.n_misses,
            "n_cases": self.n_cases,
            "miss_rate": round(self.miss_rate, 4),
        }


@dataclass
class MissAnalysis:
    """The full analysis of one run bundle: classified misses, clusters, recommendations."""

    run_dir: str
    model: str
    backend: str
    split: str
    n_cases: int
    threshold: float
    rag_config: JsonObject
    misses: list[MissRecord]
    class_counts: dict[str, int]
    clusters: dict[str, list[ClusterRow]]
    recommendations: list[JsonObject] = field(default_factory=list)
    probes: list[JsonObject] = field(default_factory=list)
