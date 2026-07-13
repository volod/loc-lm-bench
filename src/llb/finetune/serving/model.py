"""Contracts and constants for serving registered adapters."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts import JsonObject
from llb.finetune.registry.model import AdapterEntry, StalenessReport

BACKEND_VLLM = "vllm"
BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"
SERVING_BACKENDS = (BACKEND_VLLM, BACKEND_OLLAMA, BACKEND_LLAMACPP)

MERGE_MANIFEST = "merge.json"
MERGED_WEIGHTS_DIRNAME = "merged"
CONVERT_SCRIPT = "convert_hf_to_gguf.py"
GGUF_OUTTYPE = "f16"
OLLAMA_TAG_PREFIX = "llb-adapter-"
MODELFILE_NAME = "Modelfile"
MERGE_TOOL_PEFT_GGUF = "peft-merge+llama.cpp-convert"
ADAPTER_LORA_NAME = "adapter"
PROBE_PROMPT = "Reply with OK."
PROBE_MAX_TOKENS = 16
HOLD_POLL_S = 1.0


@dataclass(frozen=True)
class MergeRequest:
    entry: AdapterEntry
    backend: str
    out_dir: Path
    data_dir: Path


@dataclass(frozen=True)
class MergeArtifacts:
    merged_dir: Path
    gguf_path: Path | None = None
    model_tag: str | None = None
    tool: str = MERGE_TOOL_PEFT_GGUF

    def as_dict(self) -> JsonObject:
        return {
            "merged_dir": str(self.merged_dir),
            "gguf_path": str(self.gguf_path) if self.gguf_path else None,
            "model_tag": self.model_tag,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, payload: JsonObject) -> "MergeArtifacts":
        gguf = payload.get("gguf_path")
        return cls(
            merged_dir=Path(str(payload["merged_dir"])),
            gguf_path=Path(str(gguf)) if gguf else None,
            model_tag=payload.get("model_tag"),
            tool=str(payload.get("tool") or MERGE_TOOL_PEFT_GGUF),
        )


@dataclass(frozen=True)
class ServePlan:
    entry: AdapterEntry
    backend: str
    served_model: str
    adapter_path: Path | None = None
    merged: MergeArtifacts | None = None


@dataclass(frozen=True)
class ServeResult:
    adapter_id: str
    base_model: str
    backend: str
    served_model: str
    request_model: str
    endpoint: str
    staleness: StalenessReport
    merged: MergeArtifacts | None = None
    probe_text: str | None = None
    probe_error: str | None = None


MergeFn = Callable[[MergeRequest], MergeArtifacts]
LauncherFn = Callable[[ServePlan, RunConfig], BackendLauncher]
ReadyFn = Callable[[ServeResult], None]
