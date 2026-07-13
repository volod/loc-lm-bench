"""Cached LoRA merge, GGUF conversion, and Ollama Modelfile construction."""

import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.finetune.registry.io import merged_root
from llb.finetune.registry.model import AdapterEntry
from llb.finetune.registry.register import record_merge
from llb.finetune.serving.model import (
    BACKEND_OLLAMA,
    CONVERT_SCRIPT,
    GGUF_OUTTYPE,
    MERGE_MANIFEST,
    MERGE_TOOL_PEFT_GGUF,
    MERGED_WEIGHTS_DIRNAME,
    MODELFILE_NAME,
    OLLAMA_TAG_PREFIX,
    MergeArtifacts,
    MergeFn,
    MergeRequest,
)

_LOG = logging.getLogger(__name__)
TOKENIZER_ASSET_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

_OLLAMA_TEMPLATE_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "<|im_start|>",
        "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
        "{{ .Content }}<|im_end|>\n"
        "{{ end }}<|im_start|>assistant\n",
        ("<|im_start|>", "<|im_end|>"),
    ),
    (
        "<start_of_turn>",
        '{{- range .Messages }}<start_of_turn>{{ if eq .Role "assistant" }}model{{ else }}user{{ end }}\n'
        "{{ .Content }}<end_of_turn>\n"
        "{{ end }}<start_of_turn>model\n",
        ("<start_of_turn>", "<end_of_turn>"),
    ),
    (
        "<|start_header_id|>",
        "{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n"
        "{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n",
        ("<|eot_id|>",),
    ),
)


def ensure_merged(
    entry: AdapterEntry,
    *,
    backend: str,
    data_dir: Path | str,
    registry: Path | str,
    merge_fn: MergeFn | None = None,
) -> MergeArtifacts:
    """Reuse a cached merge or build and record one immutable merge artifact."""
    out_dir = merged_root(data_dir) / entry.short_id / backend
    manifest = out_dir / MERGE_MANIFEST
    if manifest.is_file():
        return MergeArtifacts.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (merge_fn or merge_adapter)(
        MergeRequest(entry=entry, backend=backend, out_dir=out_dir, data_dir=Path(data_dir))
    )
    atomic_write_text(
        manifest, json.dumps(artifacts.as_dict(), ensure_ascii=False, indent=2) + "\n"
    )
    record_merge(
        registry=registry,
        adapter_id=entry.adapter_id,
        backend=backend,
        artifacts=artifacts.as_dict(),
    )
    return artifacts


def merge_adapter(request: MergeRequest) -> MergeArtifacts:
    merged_dir = request.out_dir / MERGED_WEIGHTS_DIRNAME
    _merge_lora_weights(request.entry, merged_dir)
    gguf_path = _convert_to_gguf(merged_dir, request.out_dir, request.entry, request.data_dir)
    model_tag = (
        _ollama_create(gguf_path, request.out_dir, request.entry)
        if request.backend == BACKEND_OLLAMA
        else None
    )
    return MergeArtifacts(merged_dir, gguf_path, model_tag, MERGE_TOOL_PEFT_GGUF)


def _merge_lora_weights(entry: AdapterEntry, merged_dir: Path) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "[serve-adapter] merging needs the finetune extra on the CUDA host: "
            'uv pip install -e ".[finetune]"'
        ) from exc
    base = AutoModelForCausalLM.from_pretrained(entry.base_model, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(entry.resolved_dir)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(entry.base_model, trust_remote_code=True).save_pretrained(
        merged_dir
    )
    copy_base_tokenizer_assets(entry.base_model, merged_dir)


def copy_base_tokenizer_assets(
    base_model: str,
    merged_dir: Path,
    downloader: Callable[[str, str], str] | None = None,
) -> None:
    """Restore base tokenizer assets after saving merged weights."""
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    for filename in TOKENIZER_ASSET_FILES:
        try:
            source = downloader(base_model, filename)
        except Exception as exc:
            _LOG.debug("[serve-adapter] no %s carried from %s: %s", filename, base_model, exc)
            continue
        shutil.copy2(source, merged_dir / filename)


def _convert_to_gguf(merged_dir: Path, out_dir: Path, entry: AdapterEntry, data_dir: Path) -> Path:
    script = data_dir / "llb" / "llamacpp" / "src" / CONVERT_SCRIPT
    if not script.is_file():
        raise SystemExit(
            f"[serve-adapter] {CONVERT_SCRIPT} not found at {script}; run `make build-llamacpp`"
        )
    gguf_path = out_dir / f"{OLLAMA_TAG_PREFIX}{entry.short_id}.gguf"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            GGUF_OUTTYPE,
        ],
        check=True,
    )
    return gguf_path


def ollama_tag(entry: AdapterEntry) -> str:
    return f"{OLLAMA_TAG_PREFIX}{entry.short_id}"


def _ollama_create(gguf_path: Path, out_dir: Path, entry: AdapterEntry) -> str:
    tag = ollama_tag(entry)
    modelfile = out_dir / MODELFILE_NAME
    chat_template = read_chat_template(out_dir / MERGED_WEIGHTS_DIRNAME)
    atomic_write_text(modelfile, modelfile_text(gguf_path, chat_template))
    subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True)
    return tag


def read_chat_template(merged_dir: Path) -> str:
    jinja = merged_dir / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config = merged_dir / "tokenizer_config.json"
    if config.is_file():
        return str(json.loads(config.read_text(encoding="utf-8")).get("chat_template") or "")
    return ""


def modelfile_text(gguf_path: Path, chat_template: str) -> str:
    """Build an Ollama Modelfile with the recognized model-family chat template."""
    lines = [f"FROM {gguf_path}"]
    for marker, template, stops in _OLLAMA_TEMPLATE_FAMILIES:
        if marker in chat_template:
            lines.append(f'TEMPLATE """{template}"""')
            lines.extend(f'PARAMETER stop "{stop}"' for stop in stops)
            break
    else:
        _LOG.warning(
            "[serve-adapter] unrecognized chat template; %s has no TEMPLATE",
            MODELFILE_NAME,
        )
    return "\n".join(lines) + "\n"
