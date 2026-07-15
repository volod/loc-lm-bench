"""Tests for adapter serving."""

import json
from pathlib import Path
import pytest
from llb.backends.base import BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.register import register_adapter
from llb.finetune.serving.merge import (
    copy_base_tokenizer_assets,
    modelfile_text,
    ollama_tag,
    read_chat_template,
)
from llb.finetune.serving.model import (
    ADAPTER_LORA_NAME,
    MergeArtifacts,
    MergeRequest,
    ServePlan,
)
from llb.finetune.serving.run import serve_adapter
from adapter_registry_helpers import _FakeLauncher, _fake_merge, _trained_adapter


@pytest.mark.parametrize("backend", ["vllm", "ollama", "llamacpp"])
def test_serving_smoke_passes_for_every_backend(tmp_path: Path, backend: str):
    registry = registry_path(tmp_path)
    entry = register_adapter(registry=registry, adapter_dir=_trained_adapter(tmp_path))
    cfg = RunConfig(data_dir=tmp_path, model="base-model", backend=backend)
    plans: list[ServePlan] = []

    def launcher_factory(plan: ServePlan, _config: RunConfig) -> BackendLauncher:
        plans.append(plan)
        return _FakeLauncher(plan)

    result = serve_adapter(
        cfg,
        adapter=entry.short_id,
        registry=registry,
        merge_fn=_fake_merge,
        launcher_factory=launcher_factory,
    )

    assert result.probe_error is None and result.probe_text == "OK"
    assert result.backend == backend
    assert plans[0].backend == backend
    if backend == "vllm":
        assert plans[0].adapter_path == entry.resolved_dir
        assert result.request_model == ADAPTER_LORA_NAME
        assert result.merged is None
    else:
        assert plans[0].adapter_path is None
        assert result.merged is not None


def test_serve_reports_readiness_while_the_backend_is_still_up(tmp_path: Path):
    """`on_ready` must fire before `--hold` blocks, else the operator sees nothing while serving."""
    registry = registry_path(tmp_path)
    entry = register_adapter(registry=registry, adapter_dir=_trained_adapter(tmp_path))
    cfg = RunConfig(data_dir=tmp_path, model="base-model", backend="vllm")
    launchers: list[_FakeLauncher] = []
    seen: list[tuple[str, bool]] = []

    def launcher_factory(plan: ServePlan, _config: RunConfig) -> BackendLauncher:
        launchers.append(_FakeLauncher(plan))
        return launchers[-1]

    serve_adapter(
        cfg,
        adapter=entry.short_id,
        registry=registry,
        launcher_factory=launcher_factory,
        on_ready=lambda ready: seen.append((ready.endpoint, launchers[0].stopped)),
    )

    assert seen == [(cfg.vllm_host, False)], "readiness must be reported before the backend stops"
    assert launchers[0].stopped, "the launcher is always released afterwards"


def test_serve_never_holds_on_a_failed_probe(tmp_path: Path):
    registry = registry_path(tmp_path)
    entry = register_adapter(registry=registry, adapter_dir=_trained_adapter(tmp_path))
    cfg = RunConfig(data_dir=tmp_path, model="base-model", backend="vllm")

    class _FailingLauncher(_FakeLauncher):
        def chat(self, messages, max_tokens, temperature, timeout) -> ChatResult:  # type: ignore[no-untyped-def]
            return ChatResult(text="", error="backend_error")

    # `hold=True` would block forever if the failed probe did not short-circuit it.
    result = serve_adapter(
        cfg,
        adapter=entry.short_id,
        registry=registry,
        launcher_factory=lambda plan, _cfg: _FailingLauncher(plan),
        hold=True,
    )

    assert result.probe_error == "backend_error"


def test_serve_fails_on_an_empty_probe_completion(tmp_path: Path):
    """A served-but-mute endpoint (e.g. a template-less merge emitting an immediate EOS) must
    fail the smoke -- an empty answer is not an answer."""
    registry = registry_path(tmp_path)
    entry = register_adapter(registry=registry, adapter_dir=_trained_adapter(tmp_path))
    cfg = RunConfig(data_dir=tmp_path, model="base-model", backend="vllm")

    class _MuteLauncher(_FakeLauncher):
        def chat(self, messages, max_tokens, temperature, timeout) -> ChatResult:  # type: ignore[no-untyped-def]
            return ChatResult(text="  ", error=None)

    # `hold=True` would block forever if the empty probe did not short-circuit it.
    result = serve_adapter(
        cfg,
        adapter=entry.short_id,
        registry=registry,
        launcher_factory=lambda plan, _cfg: _MuteLauncher(plan),
        hold=True,
    )

    assert result.probe_error == "probe returned an empty completion"


@pytest.mark.parametrize(
    ("marker", "stop"),
    [
        ("<|im_start|>", "<|im_end|>"),
        ("<start_of_turn>", "<end_of_turn>"),
        ("<|start_header_id|>", "<|eot_id|>"),
    ],
)
def test_modelfile_carries_the_detected_chat_template_family(marker: str, stop: str):
    """Ollama ignores the GGUF chat template on `create`, so the Modelfile must restate it --
    a bare FROM line serves raw completions and a merged instruct model degrades to gibberish."""
    text = modelfile_text(Path("/x/model.gguf"), f"jinja using {marker} somewhere")

    assert text.startswith("FROM /x/model.gguf\n")
    assert 'TEMPLATE """' in text and marker in text
    assert f'PARAMETER stop "{stop}"' in text


def test_modelfile_for_an_unrecognized_template_stays_bare():
    text = modelfile_text(Path("/x/model.gguf"), "{% some exotic jinja %}")

    assert text == "FROM /x/model.gguf\n"


def test_merge_restores_the_base_repos_pristine_tokenizer_files(tmp_path: Path):
    """The transformers >= 5 tokenizer resave drops `tokenizer.model` and the control-token
    markings, which breaks GGUF conversion (Gemma-3 vocab assert) and Ollama chat (turn markers
    exported as NORMAL tokens -> empty answers). The merge must carry the base repo's originals,
    overwriting the resaved copies."""
    hub = tmp_path / "hub"
    hub.mkdir()
    available = {"tokenizer.model": b"spm", "tokenizer_config.json": b'{"pristine": true}'}
    for name, body in available.items():
        (hub / name).write_bytes(body)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "tokenizer_config.json").write_bytes(b'{"resaved": true}')

    def downloader(repo: str, filename: str) -> str:
        assert repo == "google/base"
        if filename not in available:
            raise FileNotFoundError(filename)
        return str(hub / filename)

    copy_base_tokenizer_assets("google/base", merged, downloader=downloader)

    assert (merged / "tokenizer.model").read_bytes() == b"spm"
    # The pristine original WINS over the transformers resave.
    assert (merged / "tokenizer_config.json").read_bytes() == b'{"pristine": true}'
    # Files the base repo does not ship (e.g. Qwen has no sentencepiece model) are a silent
    # per-file no-op, never a failure.
    assert not (merged / "tokenizer.json").exists()


def test_read_chat_template_uses_the_transformers_jinja_file(tmp_path: Path):
    (tmp_path / "chat_template.jinja").write_text("template", encoding="utf-8")
    assert read_chat_template(tmp_path) == "template"

    assert read_chat_template(tmp_path / "missing") == ""


def test_merge_is_recorded_once_per_backend_and_cached(tmp_path: Path):
    registry = registry_path(tmp_path)
    entry = register_adapter(registry=registry, adapter_dir=_trained_adapter(tmp_path))
    cfg = RunConfig(data_dir=tmp_path, model="base-model", backend="ollama")
    merges: list[str] = []

    def counting_merge(request: MergeRequest) -> MergeArtifacts:
        merges.append(request.backend)
        return _fake_merge(request)

    for _ in range(2):
        serve_adapter(
            cfg,
            adapter=entry.adapter_id,
            registry=registry,
            merge_fn=counting_merge,
            launcher_factory=lambda plan, _cfg: _FakeLauncher(plan),
        )

    assert merges == ["ollama"], "the second serve must reuse the cached merge"
    events = [
        json.loads(line)
        for line in registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    merge_events = [event for event in events if event["event"] == "merge"]
    assert len(merge_events) == 1
    assert merge_events[0]["model_tag"] == ollama_tag(entry)
    assert load_registry(registry)[entry.adapter_id].merges[0]["backend"] == "ollama"
