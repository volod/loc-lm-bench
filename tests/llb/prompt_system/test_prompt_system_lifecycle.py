"""Tests for prompt system lifecycle."""

import json
from llb.bench.agentic.run import run_agentic
from llb.board.prompt_systems import load_prompt_system_records, prompt_system_comparison
from llb.prompt_system import budget as bud
from llb.prompt_system import review as rv
from llb.prompt_system import tuning as tn
from llb.prompt_system.pipeline import (
    CANDIDATES_FILE,
    MANIFEST_FILE,
    prepare_prompt_system,
)
from llb.prompt_system.selection import resolve_prompt_package, prompt_system_id_from_package_path
from llb.prompt_system.template import (
    GRAPH_INLINE,
    GRAPH_NONE,
    TemplateFields,
)
from test_prompt_system import SAMPLE_CORPUS, _corpus, _two_tasks


def test_review_transitions_and_roundtrip(tmp_path):
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    c = rv.make_candidate(_corpus(), TemplateFields(anthology_size=2), b, tok)
    rv.approve(c, "looks good")
    assert c.status == rv.STATUS_APPROVED and c.note == "looks good"
    path = tmp_path / "candidates.json"
    rv.save_candidates([c], path)
    loaded = rv.load_candidates(path)
    assert loaded[0].prompt_system_id == c.prompt_system_id
    assert loaded[0].status == rv.STATUS_APPROVED
    assert loaded[0].fields.anthology_size == 2


def test_review_revise_creates_new_id():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    corpus = _corpus()
    c = rv.make_candidate(corpus, TemplateFields(anthology_size=2), b, tok)
    revised = rv.revise(c, TemplateFields(anthology_size=4), corpus, b, tok)
    assert revised.status == rv.STATUS_REVISED and revised.prompt_system_id != c.prompt_system_id


def test_variant_grid_dedupes_and_validates():
    grid = tn.variant_grid(
        TemplateFields(), anthology_sizes=[2, 2, 4], graph_styles=[GRAPH_NONE, GRAPH_INLINE]
    )
    keys = {(f.anthology_size, f.metadata_density, f.graph_reference_style) for f in grid}
    assert len(keys) == len(grid)  # no duplicates
    assert all(f.anthology_size in (2, 4) for f in grid)


def test_generate_candidates_dedupes_by_id():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    grid = tn.variant_grid(
        TemplateFields(), anthology_sizes=[2, 4], graph_styles=[GRAPH_NONE, GRAPH_INLINE]
    )
    candidates = tn.generate_candidates(_corpus(), grid, b, tok)
    ids = [c.prompt_system_id for c in candidates]
    assert len(ids) == len(set(ids)) and len(candidates) >= 4


def test_prepare_prompt_system_writes_artifacts(tmp_path):
    run = prepare_prompt_system(
        SAMPLE_CORPUS, data_dir=tmp_path, context_window=4096, max_passages=6
    )
    assert run.candidates and (run.run_dir / MANIFEST_FILE).exists()
    manifest = json.loads((run.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["method"] == "prompt-system" and manifest["n_candidates"] == len(run.candidates)
    assert manifest["corpus_digest"] and manifest["context_window"] == 4096
    candidates = json.loads((run.run_dir / CANDIDATES_FILE).read_text(encoding="utf-8"))
    assert len(candidates) == len(run.candidates)


def test_prepare_prompt_system_supports_stable_out_dir(tmp_path):
    out_dir = tmp_path / "sample_prompt_system"
    run = prepare_prompt_system(
        SAMPLE_CORPUS,
        data_dir=tmp_path,
        out_dir=out_dir,
        context_window=4096,
        max_passages=4,
    )

    assert run.run_dir == out_dir
    assert (out_dir / MANIFEST_FILE).exists()
    assert (out_dir / CANDIDATES_FILE).exists()


def test_resolve_prompt_package_from_compact_path(tmp_path):
    out_dir = tmp_path / "prompt-system" / "stable"
    run = prepare_prompt_system(
        SAMPLE_CORPUS,
        data_dir=tmp_path,
        out_dir=out_dir,
        context_window=4096,
        max_passages=4,
    )
    target = run.candidates[0].prompt_system_id
    selector = out_dir / target

    assert prompt_system_id_from_package_path(selector) == target
    selected = resolve_prompt_package(tmp_path, target, selector)

    assert selected.run_dir == out_dir
    assert selected.package.system_prompt == run.candidates[0].system_prompt
    assert selected.provenance["prompt_system_id"] == target
    assert selected.provenance["context_window"] == 4096


def test_run_agentic_records_prompt_system_and_board_compares(tmp_path):
    # ps1: both succeed (answer contains x); ps2: fails (empty answer)
    run_agentic(
        _two_tasks(),
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":"x"}}',
        prompt_system="ps1",
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    run_agentic(
        _two_tasks(),
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',
        prompt_system="ps2",
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    records = load_prompt_system_records(tmp_path)
    assert {(r.model, r.prompt_system) for r in records} == {("m", "ps1"), ("m", "ps2")}
    rows, table, ids = prompt_system_comparison(tmp_path, "m")
    assert {row["model"] for row in rows} == {"ps1", "ps2"}
    top = next(row for row in rows if row["rank"] == 1)
    assert top["model"] == "ps1" and "policy:" in table
