"""Adapter registry, staleness, serving, and garbage-collection lifecycle tests."""

import json
from pathlib import Path

import pytest

from llb.backends.base import BackendLauncher, ChatResult
from llb.board.runs import STALE_STAMP, load_run_records
from llb.core.config import RunConfig
from llb.finetune.guard import validate_adapter_for_eval
from llb.finetune.lifecycle import GC_DELETE, GC_KEEP, GC_REFUSE, cited_adapters, gc_adapters
from llb.finetune.registry import (
    VERDICT_CURRENT,
    VERDICT_STALE,
    VERDICT_UNKNOWN,
    AdapterEntry,
    append_event,
    load_registry,
    register_adapter,
    registry_path,
    resolve_adapter,
    staleness,
)
from llb.finetune.serving import (
    ADAPTER_LORA_NAME,
    MergeArtifacts,
    MergeRequest,
    ServePlan,
    copy_base_tokenizer_assets,
    modelfile_text,
    ollama_tag,
    read_chat_template,
    serve_adapter,
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


def test_registry_round_trip_is_idempotent(tmp_path: Path):
    goldset = _goldset(tmp_path)
    registry = registry_path(tmp_path)
    adapter_dir = _trained_adapter(tmp_path)

    entry = register_adapter(
        registry=registry,
        adapter_dir=adapter_dir,
        goldset_path=goldset,
        source_run=tmp_path / "run-tuning",
        eval_summary={"objective_score": 0.75, "delta": 0.1},
    )
    again = register_adapter(
        registry=registry,
        adapter_dir=adapter_dir,
        goldset_path=goldset,
        source_run=tmp_path / "run-tuning",
        eval_summary={"objective_score": 0.75, "delta": 0.1},
    )

    assert again.adapter_id == entry.adapter_id
    assert registry.read_text(encoding="utf-8").count('"event"') == 1
    entries = load_registry(registry)
    loaded = entries[entry.adapter_id]
    assert loaded.base_model == "base-model"
    assert loaded.dataset_split_counts == {"tuning": 1}
    assert loaded.eval_summary["objective_score"] == 0.75
    assert loaded.goldset_digest is not None
    assert resolve_adapter(entries, entry.adapter_id[:8]).adapter_id == entry.adapter_id
    assert resolve_adapter(entries, entry.adapter_label).adapter_id == entry.adapter_id


def test_staleness_flips_when_the_goldset_digest_changes(tmp_path: Path):
    goldset = _goldset(tmp_path, _item("tune-1", "tuning"))
    corpus = _corpus(tmp_path)
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=goldset,
        corpus_root=corpus,
    )
    assert staleness(entry).verdict == VERDICT_CURRENT

    dump_goldset([_item("tune-1", "tuning"), _item("tune-2", "tuning")], goldset)

    report = staleness(entry)
    assert report.verdict == VERDICT_STALE
    assert report.is_stale
    assert report.reasons == ("goldset changed since training",)


def test_staleness_is_unknown_when_a_digest_was_never_recorded(tmp_path: Path):
    """A missing corpus digest can never read as `current` -- absence of evidence is not evidence."""
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=_goldset(tmp_path),
    )

    report = staleness(entry)

    assert report.verdict == VERDICT_UNKNOWN
    assert report.reasons == ("corpus digest unavailable",)


def test_committed_fixture_stamps_the_stale_entry():
    entries = load_registry(FIXTURE_REGISTRY)
    stale = entries[STALE_FIXTURE_ID]

    report = staleness(stale)

    assert report.verdict == VERDICT_STALE
    assert [merge["backend"] for merge in stale.merges] == ["ollama"]


def test_guard_reads_recorded_digests_not_the_adapter_manifest():
    """The laundered manifest claims a clean tuning set; the registry records the final-split ids."""
    protected = [_item("sample-final-item", "final")]

    validate_adapter_for_eval(adapter_path=LAUNDERED_ADAPTER, items=protected, model="sample/base")

    with pytest.raises(SystemExit, match="sample-final-item"):
        validate_adapter_for_eval(
            adapter_path=LAUNDERED_ADAPTER,
            items=protected,
            model="sample/base",
            registry=FIXTURE_REGISTRY,
        )


def test_guard_still_refuses_an_unregistered_poisoned_manifest():
    with pytest.raises(SystemExit, match="sample-final-item"):
        validate_adapter_for_eval(
            adapter_path=POISONED_ADAPTER,
            items=[_item("sample-final-item", "final")],
            model="sample/base",
            registry=FIXTURE_REGISTRY,
        )


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


def test_read_chat_template_prefers_the_transformers5_jinja_file(tmp_path: Path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "legacy"}), encoding="utf-8"
    )
    assert read_chat_template(tmp_path) == "legacy"

    (tmp_path / "chat_template.jinja").write_text("modern", encoding="utf-8")
    assert read_chat_template(tmp_path) == "modern"

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


def test_gc_refuses_a_cited_adapter_until_forced(tmp_path: Path):
    registry = registry_path(tmp_path)
    old = _trained_adapter(tmp_path, seed=1)
    new = _trained_adapter(tmp_path, seed=2)
    old_entry = _entry(old, created_at="2026-01-01T00:00:00Z")
    new_entry = _entry(new, created_at="2026-02-01T00:00:00Z")
    _register_event(registry, old_entry)
    _register_event(registry, new_entry)
    run_root = tmp_path / "run-eval"
    _run_bundle(run_root, "cited-run", model="base-model", adapter_digest=old_entry.adapter_id)

    assert cited_adapters(run_root, load_registry(registry)) == {
        old_entry.adapter_id: (str(run_root / "cited-run"),)
    }

    refused = gc_adapters(data_dir=tmp_path)
    assert [d.action for d in refused.decisions if d.entry.adapter_id == old_entry.adapter_id] == [
        GC_REFUSE
    ]
    assert [d.action for d in refused.decisions if d.entry.adapter_id == new_entry.adapter_id] == [
        GC_KEEP
    ]
    assert old.is_dir(), "a cited adapter must survive an unforced GC"
    assert old_entry.adapter_id in load_registry(registry)

    forced = gc_adapters(data_dir=tmp_path, force=True)
    assert [d.entry.adapter_id for d in forced.deleted] == [old_entry.adapter_id]
    assert not old.exists()
    assert new.is_dir()
    assert old_entry.adapter_id not in load_registry(registry)


def test_supersession_uses_log_order_when_created_at_ties(tmp_path: Path):
    """`created_at` has second resolution, so two fast rounds tie; the append log still orders them."""
    registry = registry_path(tmp_path)
    same_second = "2026-03-01T12:00:00Z"
    first = _entry(_trained_adapter(tmp_path, seed=1), created_at=same_second)
    second = _entry(_trained_adapter(tmp_path, seed=2), created_at=same_second)
    _register_event(registry, first)
    _register_event(registry, second)
    assert first.adapter_id > second.adapter_id, "id order must disagree with log order here"

    plan = gc_adapters(data_dir=tmp_path, dry_run=True)

    assert [d.entry.adapter_id for d in plan.deleted] == [first.adapter_id]
    assert [d.entry.adapter_id for d in plan.kept] == [second.adapter_id]


def test_gc_deletes_a_superseded_uncited_adapter(tmp_path: Path):
    registry = registry_path(tmp_path)
    old = _trained_adapter(tmp_path, seed=1)
    new = _trained_adapter(tmp_path, seed=2)
    _register_event(registry, _entry(old, created_at="2026-01-01T00:00:00Z"))
    _register_event(registry, _entry(new, created_at="2026-02-01T00:00:00Z"))

    plan = gc_adapters(data_dir=tmp_path, dry_run=True)
    assert [d.action for d in plan.decisions if d.entry.resolved_dir == old.resolve()] == [
        GC_DELETE
    ]
    assert old.is_dir(), "a dry run never deletes"

    gc_adapters(data_dir=tmp_path)
    assert not old.exists()
    assert new.is_dir()


def test_gc_never_deletes_an_adapter_outside_the_data_dir(tmp_path: Path):
    """The committed sample adapters are superseded by each other, and must survive even --force."""
    plan = gc_adapters(data_dir=tmp_path, registry=FIXTURE_REGISTRY, force=True)

    assert plan.deleted == []
    assert [decision.action for decision in plan.refused] == [GC_REFUSE]
    assert Path("samples/finetune/stale-adapter/adapter_manifest.json").is_file()


def test_register_adapter_cli_rescues_a_hand_trained_adapter(tmp_path: Path):
    """`finetune-adapter` alone does not register, so its row would be dropped by the board."""
    from typer.testing import CliRunner

    from llb.main import app

    adapter_dir = _trained_adapter(tmp_path)
    run_root = tmp_path / "run-eval"
    manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    _run_bundle(
        run_root,
        "hand-run",
        model=manifest["adapter_label"],
        adapter_digest=manifest["adapter_digest"],
    )
    assert load_run_records(run_root, data_dir=tmp_path) == []

    result = CliRunner().invoke(
        app,
        ["register-adapter", "--adapter-dir", str(adapter_dir)],
        env={"DATA_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert manifest["adapter_digest"][:12] in result.output
    rendered = [record.result.model for record in load_run_records(run_root, data_dir=tmp_path)]
    assert rendered == [manifest["adapter_label"]]


def test_board_drops_unregistered_and_stamps_stale_adapter_rows(tmp_path: Path):
    goldset = _goldset(tmp_path, _item("tune-1", "tuning"))
    registry = registry_path(tmp_path)
    entry = register_adapter(
        registry=registry, adapter_dir=_trained_adapter(tmp_path), goldset_path=goldset
    )
    run_root = tmp_path / "run-eval"
    _run_bundle(run_root, "base-run", model="base-model", adapter_digest=None)
    _run_bundle(run_root, "ghost-run", model="base-model+adapter-ghost", adapter_digest="ghost")
    _run_bundle(run_root, "tuned-run", model=entry.adapter_label, adapter_digest=entry.adapter_id)

    fresh = {record.result.model for record in load_run_records(run_root, data_dir=tmp_path)}
    assert fresh == {"base-model", entry.adapter_label}, "the unregistered adapter row is dropped"

    dump_goldset([_item("tune-1", "tuning"), _item("tune-2", "tuning")], goldset)

    stamped = {record.result.model for record in load_run_records(run_root, data_dir=tmp_path)}
    assert stamped == {"base-model", f"{entry.adapter_label} [{STALE_STAMP}]"}
