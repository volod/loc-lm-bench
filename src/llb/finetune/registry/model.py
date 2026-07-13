"""Registry data contracts and stable event constants."""

from dataclasses import dataclass, field
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.core.paths import resolve_project_path
from llb.finetune.trainer import ADAPTER_DIGEST_SHORT_CHARS

ADAPTERS_METHOD = "adapters"
REGISTRY_FILENAME = "registry.jsonl"
MERGED_DIRNAME = "merged"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

EVENT_REGISTER = "register"
EVENT_MERGE = "merge"
EVENT_DELETE = "delete"

VERDICT_CURRENT = "current"
VERDICT_STALE = "stale"
VERDICT_UNKNOWN = "unknown"

RETRIEVAL_FINGERPRINT_KEYS = (
    ("embedding_model", "embedding_model"),
    ("strategy", "strategy"),
    ("chunk_size", "size"),
    ("chunk_overlap", "overlap"),
    ("retrieval_mode", "mode"),
)

MIN_ADAPTER_ID_PREFIX = 6


@dataclass(frozen=True)
class AdapterEntry:
    """One registered adapter: what trained it, what it scored, and where it lives."""

    adapter_id: str
    base_model: str
    adapter_label: str
    adapter_dir: Path
    dataset_digest: str
    dataset_item_ids: tuple[str, ...] = ()
    dataset_split_counts: dict[str, int] = field(default_factory=dict)
    goldset_digest: str | None = None
    corpus_digest: str | None = None
    goldset_path: str | None = None
    corpus_root: str | None = None
    retrieval_fingerprint: JsonObject | None = None
    index_dir: str | None = None
    source_run: str | None = None
    eval_summary: JsonObject = field(default_factory=dict)
    created_at: str = ""
    merges: tuple[JsonObject, ...] = ()
    sequence: int = 0

    @property
    def short_id(self) -> str:
        return self.adapter_id[:ADAPTER_DIGEST_SHORT_CHARS]

    @property
    def recency(self) -> tuple[str, int]:
        """Sort key for newest-adapter selection, exact under same-second registrations."""
        return (self.created_at, self.sequence)

    @property
    def resolved_dir(self) -> Path:
        """Absolute adapter directory (committed fixtures record project-relative paths)."""
        return resolve_project_path(self.adapter_dir)

    def as_dict(self) -> JsonObject:
        """The register-event payload; merge history is carried by separate events."""
        return {
            "adapter_id": self.adapter_id,
            "base_model": self.base_model,
            "adapter_label": self.adapter_label,
            "adapter_dir": str(self.adapter_dir),
            "dataset_digest": self.dataset_digest,
            "dataset_item_ids": list(self.dataset_item_ids),
            "dataset_split_counts": dict(self.dataset_split_counts),
            "goldset_digest": self.goldset_digest,
            "corpus_digest": self.corpus_digest,
            "goldset_path": self.goldset_path,
            "corpus_root": self.corpus_root,
            "retrieval_fingerprint": (
                dict(self.retrieval_fingerprint) if self.retrieval_fingerprint is not None else None
            ),
            "index_dir": self.index_dir,
            "source_run": self.source_run,
            "eval": dict(self.eval_summary),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class StalenessReport:
    """Whether an entry's recorded evidence still describes the present benchmark."""

    verdict: str
    reasons: tuple[str, ...] = ()

    @property
    def is_stale(self) -> bool:
        return self.verdict == VERDICT_STALE

    def describe(self) -> str:
        return f"{self.verdict}: {'; '.join(self.reasons)}" if self.reasons else self.verdict
