"""Adapter registry, staleness, serving, and garbage-collection lifecycle tests."""

import json
from pathlib import Path


from llb.backends.base import BackendLauncher, ChatResult
from llb.finetune.registry.io import append_event
from llb.finetune.registry.model import (
    AdapterEntry,
)
from llb.finetune.serving.merge import (
    ollama_tag,
)
from llb.finetune.serving.model import (
    ADAPTER_LORA_NAME,
    MergeArtifacts,
    MergeRequest,
    ServePlan,
)
from llb.finetune.trainer import fake_train_adapter
from llb.goldset.schema import GoldItem, dump_goldset

FIXTURE_REGISTRY = Path("samples/finetune/registry/registry.jsonl")
LAUNDERED_ADAPTER = Path("samples/finetune/laundered-adapter")
POISONED_ADAPTER = Path("samples/finetune/poisoned-adapter")
STALE_FIXTURE_ID = "5741e0c2f4a1b3d6e8907c5a2b4d6f8091a3c5e7d9f1b3a5c7e9d1f3b5a7c9e1"


def _item(item_id: str, split: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer=f"Answer {item_id}",
        source_doc_id=f"{item_id}.txt",
        source_spans=[
            {"doc_id": f"{item_id}.txt", "char_start": 0, "char_end": 5, "text": "alpha"}
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def _goldset(tmp_path: Path, *items: GoldItem) -> Path:
    path = tmp_path / "goldset.jsonl"
    dump_goldset(list(items) or [_item("tune-1", "tuning")], path)
    return path


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "doc.md").write_text("alpha beta\n", encoding="utf-8")
    return corpus


def _dataset(tmp_path: Path, name: str = "dataset") -> Path:
    dataset_dir = tmp_path / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_digest": "abc",
                "item_ids": ["tune-1"],
                "split_counts": {"tuning": 1},
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _trained_adapter(data_dir: Path, *, model: str = "base-model", seed: int = 13) -> Path:
    """Fake-train an adapter INSIDE data_dir so GC is allowed to delete it."""
    dataset_dir = _dataset(data_dir, f"dataset-{seed}")
    adapter_dir = data_dir / "self-improve" / f"round-{seed}" / "adapter"
    fake_train_adapter(dataset_dir=dataset_dir, model=model, out_dir=adapter_dir, seed=seed)
    return adapter_dir


def _register_event(registry: Path, entry: AdapterEntry) -> None:
    """Append a register event with an explicit `created_at`, so supersession is deterministic."""
    append_event(registry, {"event": "register", **entry.as_dict()})


def _entry(adapter_dir: Path, *, created_at: str, base_model: str = "base-model") -> AdapterEntry:
    manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    return AdapterEntry(
        adapter_id=str(manifest["adapter_digest"]),
        base_model=base_model,
        adapter_label=str(manifest["adapter_label"]),
        adapter_dir=adapter_dir,
        dataset_digest=str(manifest["dataset_digest"]),
        dataset_item_ids=tuple(manifest["dataset_item_ids"]),
        dataset_split_counts=dict(manifest["dataset_split_counts"]),
        created_at=created_at,
    )


def _run_bundle(run_root: Path, name: str, *, model: str, adapter_digest: str | None) -> Path:
    run = run_root / name
    run.mkdir(parents=True)
    config: dict[str, object] = {"model": model, "backend": "vllm"}
    if adapter_digest is not None:
        config["adapter"] = {"adapter_digest": adapter_digest}
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "split": "final",
                "config": config,
                "metrics": {"objective_score": 0.5, "reliability": 1.0, "tokens_per_s": 1.0},
                "n_cases": 1,
            }
        ),
        encoding="utf-8",
    )
    (run / "scores.jsonl").write_text(
        json.dumps({"item_id": "final-1", "split": "final", "objective_score": 0.5}) + "\n",
        encoding="utf-8",
    )
    return run


class _FakeLauncher(BackendLauncher):
    """Records lifecycle calls; answers the serving probe without any real backend."""

    def __init__(self, plan: ServePlan):
        super().__init__(model=plan.served_model, meta={"backend": plan.backend})
        self.request_model = ADAPTER_LORA_NAME if plan.adapter_path else plan.served_model
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def chat(self, messages, max_tokens, temperature, timeout) -> ChatResult:  # type: ignore[no-untyped-def]
        return ChatResult(text="OK", completion_tokens=1, latency_s=0.01)

    def stop(self) -> None:
        self.stopped = True


def _fake_merge(request: MergeRequest) -> MergeArtifacts:
    merged_dir = request.out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf = request.out_dir / "model.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    tag = ollama_tag(request.entry) if request.backend == "ollama" else None
    return MergeArtifacts(merged_dir, gguf, tag, tool="fake-merge")
