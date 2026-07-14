"""Tests for goldset sampling."""

import json
from llb.goldset.chains import CHAINS_FILENAME, dump_chains, load_chains, validate_chains
from llb.goldset.verify_acceptance import (
    emit_accepted_chain_ledger,
    run_accept,
)
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.goldset.verify_sampling.context import (
    corpus_window,
    load_cross_check,
)
from llb.goldset.verify_sampling.strata import draw_stratified_sample, stratify
from llb.goldset.verify_sampling.worksheet import (
    build_sample_worksheet,
)
from tests.llb.goldset._verify_helpers import (
    DOC,
    TEXT,
    _bundle,
    _chain,
    _chain_bundle,
    _item,
)


def test_stratify_splits_by_provenance_split_doc():
    items = [
        _item("a", split="calibration"),
        _item("b", split="calibration"),
        _item("c", split="final", doc="squad/doc2.txt"),
    ]
    strata = stratify(items)
    assert len(strata) == 2  # a,b share a stratum; c (different split + doc) is its own
    assert sorted(len(v) for v in strata.values()) == [1, 2]


def test_sample_is_deterministic_and_covers_every_stratum():
    items = [_item(f"d{i}", split="calibration") for i in range(8)]
    items += [_item(f"s{i}", split="final", doc="squad/doc2.txt") for i in range(4)]
    one = [it.id for it in draw_stratified_sample(items, 6, seed=7)]
    two = [it.id for it in draw_stratified_sample(items, 6, seed=7)]
    assert one == two  # deterministic given the seed
    assert any(i.startswith("s") for i in one)  # the small second stratum is represented


def test_sample_returns_all_when_n_exceeds_population():
    items = [_item(f"d{i}") for i in range(3)]
    assert len(draw_stratified_sample(items, 99)) == 3


def test_sample_draws_exactly_n_when_proportional_rounding_undershoots():
    """verify-sample-exact-allocation: 3 strata of 7/7/6 at n=4 used to yield 3 rows (all
    quotas rounded down to the floor of one); the largest-remainder top-up restores the 4th."""
    items = [_item(f"a{i}", doc="squad/doc-a.txt") for i in range(7)]
    items += [_item(f"b{i}", doc="squad/doc-b.txt") for i in range(7)]
    items += [_item(f"c{i}", doc="squad/doc-c.txt") for i in range(6)]
    for seed in range(20):
        sample = draw_stratified_sample(items, 4, seed=seed)
        assert len(sample) == 4, f"seed {seed} drew {len(sample)} rows"
        docs = {it.source_doc_id for it in sample}
        assert len(docs) == 3  # the floor of one still covers every stratum
    one = [it.id for it in draw_stratified_sample(items, 4, seed=13)]
    two = [it.id for it in draw_stratified_sample(items, 4, seed=13)]
    assert one == two  # seeded draws stay reproducible


def test_stratum_quotas_sum_exactly_and_respect_sizes():
    from llb.goldset.verify_sampling.strata import stratum_quotas

    quotas = stratum_quotas({"a": 7, "b": 7, "c": 6}, 4)
    assert sum(quotas.values()) == 4
    assert all(q >= 1 for q in quotas.values())
    capped = stratum_quotas({"a": 2, "b": 1}, 40)  # budget capped at the population
    assert capped == {"a": 2, "b": 1}
    tight = stratum_quotas({"a": 5, "b": 4, "c": 3}, 2)  # n below the stratum count:
    assert sum(tight.values()) == 2  # largest strata get the floor first, deterministically
    assert tight["a"] == 1 and tight["b"] == 1 and tight["c"] == 0


def test_corpus_window_delimits_span():
    win = corpus_window(TEXT, TEXT.find("1871"), TEXT.find("1871") + 4, ctx=10)
    assert ">>>1871<<<" in win


def test_build_sample_worksheet_writes_rows_and_manifest(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b"), _item("c")])
    out = tmp_path / "verify_sample.csv"
    n, strata = build_sample_worksheet(bundle, out, n=2, seed=1)
    assert n == 2
    rows, _ = load_worksheet(out)
    assert len(rows) == 2
    assert all(">>>" in r["context"] for r in rows)  # the cited span is captured in context
    manifest = json.loads((out.with_name("sample_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["sample_size"] == 2 and manifest["population"] == 3


def test_build_sample_worksheet_marks_synthetic_from_bundle_meta(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")], synthetic=True)
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    assert rows[0]["synthetic"] == "true"  # bundle-level provenance.json flag, not per-item
    manifest = json.loads((out.with_name("sample_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True


def test_build_sample_worksheet_reads_planted_labels_filename(tmp_path):
    from llb.goldset.schema import dump_goldset

    # A synthetic bundle names its gold file planted_labels.jsonl, not goldset.jsonl.
    dump_goldset([_item("a")], tmp_path / "planted_labels.jsonl")
    doc = tmp_path / "corpus" / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(TEXT + "\n", encoding="utf-8")
    out = tmp_path / "ws.csv"
    n, _ = build_sample_worksheet(tmp_path, out, n=1)
    assert n == 1


def test_validate_chains_checks_spans_and_step_rules(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("ok")])
    report = validate_chains(load_chains(bundle / CHAINS_FILENAME), bundle / "corpus")
    assert report["errors"] == []

    bad = _chain("bad")
    bad.steps[1].source_spans = bad.steps[0].source_spans
    dump_chains([bad], bundle / CHAINS_FILENAME)
    report = validate_chains(load_chains(bundle / CHAINS_FILENAME), bundle / "corpus")
    assert any("reuses span" in err for err in report["errors"])


def test_build_sample_worksheet_auto_samples_chains_when_present(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("c1"), _chain("c2")])
    out = tmp_path / "verify_sample.csv"
    n, strata = build_sample_worksheet(bundle, out, n=1)
    assert n == 1 and strata
    rows, _ = load_worksheet(out)
    assert rows[0]["item_kind"] == "chains"
    assert rows[0]["chain_steps"]
    manifest = json.loads((out.with_name("sample_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["kind"] == "chains"


def test_emit_accepted_chain_ledger_and_accept_command(tmp_path):
    bundle = _chain_bundle(tmp_path, [_chain("c1"), _chain("c2")])
    ws = bundle / "verify_sample.csv"
    build_sample_worksheet(bundle, ws, n=2)
    rows, fields = load_worksheet(ws)
    for row in rows:
        row["decision"] = "accept"
    write_worksheet_rows(ws, rows, fields)

    assert emit_accepted_chain_ledger(bundle, ["c1"], tmp_path / "manual") == 1
    assert load_chains(tmp_path / "manual" / CHAINS_FILENAME)[0].verified is True
    assert run_accept(ws, bundle, None, tolerance=0.05) == 0
    accepted = load_chains(bundle / "accepted" / CHAINS_FILENAME)
    assert [chain.chain_id for chain in accepted] == ["c1", "c2"]
    assert all(chain.verified for chain in accepted)


def test_cross_check_sidecar_is_loaded(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    (bundle / "goldset.cross_check.json").write_text(
        json.dumps(
            {"verdicts": [{"item_id": "a", "grounded": True, "supported": False, "note": "weak"}]}
        ),
        encoding="utf-8",
    )
    verdicts = load_cross_check(bundle)
    assert verdicts["a"]["supported"] is False
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    assert rows[0]["cc_supported"] == "false" and rows[0]["cc_note"] == "weak"
