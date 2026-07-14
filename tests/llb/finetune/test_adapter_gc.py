"""Tests for adapter gc."""

import json
from pathlib import Path
from llb.board.runs import STALE_STAMP, load_run_records
from llb.core.paths import PROJECT_ROOT
from llb.finetune.lifecycle import (
    GC_DELETE,
    GC_KEEP,
    GC_REFUSE,
    cited_adapters,
    gc_adapters,
    gc_rows,
    plan_gc,
)
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.register import register_adapter
from llb.goldset.schema import dump_goldset
from adapter_registry_gc_helpers import _superseded_pair
from adapter_registry_helpers import (
    FIXTURE_REGISTRY,
    STALE_FIXTURE_ID,
    _entry,
    _goldset,
    _item,
    _register_event,
    _run_bundle,
    _trained_adapter,
)


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
        old_entry.adapter_id: (f"run-bundle:{run_root / 'cited-run'}",)
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


def test_gc_refuses_adapter_cited_only_by_campaign_journal(tmp_path: Path):
    """A superseded adapter no run bundle cites, but a campaign journal still links."""
    old, _new, old_entry, _ = _superseded_pair(tmp_path)
    journal = tmp_path / "finetune-campaign" / "2026-03-01" / "campaign.progress.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps({"model": "base-model", "status": "completed", "adapter_dir": str(old)}) + "\n",
        encoding="utf-8",
    )

    plan = gc_adapters(data_dir=tmp_path, dry_run=True)
    decision = next(d for d in plan.decisions if d.entry.adapter_id == old_entry.adapter_id)
    assert decision.action == GC_REFUSE
    assert decision.cited_by == (f"campaign-journal:{journal}",)
    assert str(journal) in decision.reason  # the refusal names the citing journal
    row = next(r for r in gc_rows(plan) if r["action"] == GC_REFUSE)
    assert row["cited_kinds"] == "campaign-journal"

    forced = gc_adapters(data_dir=tmp_path, force=True)
    assert [d.entry.adapter_id for d in forced.deleted] == [old_entry.adapter_id]
    assert not old.exists()


def test_gc_refuses_adapter_cited_by_self_improve_state(tmp_path: Path):
    old, _new, old_entry, _ = _superseded_pair(tmp_path)
    state = tmp_path / "self-improve" / "2026-03-02" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps({"rounds": [{"round": 1, "adapter_dir": str(old)}]}), encoding="utf-8"
    )

    plan = gc_adapters(data_dir=tmp_path, dry_run=True)
    decision = next(d for d in plan.decisions if d.entry.adapter_id == old_entry.adapter_id)
    assert decision.action == GC_REFUSE
    assert decision.cited_by == (f"self-improve-state:{state}",)
    assert old.is_dir()


def test_committed_campaign_journal_fixture_blocks_unforced_gc(tmp_path: Path):
    """The committed journal fixture cites the committed stale adapter; GC must refuse it."""
    entries = load_registry(FIXTURE_REGISTRY)
    cited = cited_adapters(
        tmp_path / "no-runs", entries, data_dir=Path("samples/finetune/gc-journals")
    )
    journal = (
        "samples/finetune/gc-journals/finetune-campaign/campaign-fixture/campaign.progress.jsonl"
    )
    assert [c.split(":", 1)[0] for c in cited[STALE_FIXTURE_ID]] == ["campaign-journal"]
    assert cited[STALE_FIXTURE_ID][0].endswith(journal)

    # plan_gc is pure; PROJECT_ROOT as data root keeps the fixture inside the deletable zone
    # so the CITATION (not the outside-$DATA_DIR rule) is what blocks the unforced plan.
    plan = plan_gc(entries, cited=cited, data_dir=PROJECT_ROOT, force=False)
    decision = next(d for d in plan.decisions if d.entry.adapter_id == STALE_FIXTURE_ID)
    assert decision.action == GC_REFUSE
    assert journal in decision.reason

    forced = plan_gc(entries, cited=cited, data_dir=PROJECT_ROOT, force=True)
    assert [d.entry.adapter_id for d in forced.deleted] == [STALE_FIXTURE_ID]
    assert Path("samples/finetune/stale-adapter/adapter_manifest.json").is_file()


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
