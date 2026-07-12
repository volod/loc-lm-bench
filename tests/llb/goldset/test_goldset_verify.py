"""Tests for the pure human verification gate pieces (`llb.goldset.verify`).

Stratification, deterministic sampling, the acceptance arithmetic, verification-reference
validation, and the accepted-ledger round-trip -- all checked directly with no terminal / model /
endpoint / GPU. The interactive session loop is covered in `test_goldset_verify_session.py`; both
modules share the factories in `_verify_helpers.py`.
"""

import json

import pytest

from llb.bench.common import verified_data_config
from llb.goldset.chains import CHAINS_FILENAME, dump_chains, load_chains, validate_chains
from llb.goldset.schema import load_goldset
from llb.goldset.verify import (
    acceptance_report,
    accepted_ids,
    build_sample_worksheet,
    check_verification_ref,
    confidence_order,
    corpus_window,
    draw_stratified_sample,
    emit_accepted_chain_ledger,
    emit_accepted_ledger,
    format_verification_status,
    ground_answer,
    infer_reject_code,
    load_cross_check,
    load_retrieval_ranks,
    load_worksheet,
    merge_sample_worksheet,
    rejection_reasons_summary,
    row_confidence,
    run_accept,
    stratify,
    write_worksheet_rows,
)
from llb.goldset.verify_session import format_card
from llb.prep.verified_ledger import apply_verified_ledger, load_verified_ledger

from tests.llb.goldset._verify_helpers import (
    DOC,
    TEXT,
    _bundle,
    _chain,
    _chain_bundle,
    _item,
    _ws_row,
)


# --- pure: strata + sampling --------------------------------------------------------------


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
    from llb.goldset.verify import stratum_quotas

    quotas = stratum_quotas({"a": 7, "b": 7, "c": 6}, 4)
    assert sum(quotas.values()) == 4
    assert all(q >= 1 for q in quotas.values())
    capped = stratum_quotas({"a": 2, "b": 1}, 40)  # budget capped at the population
    assert capped == {"a": 2, "b": 1}
    tight = stratum_quotas({"a": 5, "b": 4, "c": 3}, 2)  # n below the stratum count:
    assert sum(tight.values()) == 2  # largest strata get the floor first, deterministically
    assert tight["a"] == 1 and tight["b"] == 1 and tight["c"] == 0


# --- pure: corpus window ------------------------------------------------------------------


def test_corpus_window_delimits_span():
    win = corpus_window(TEXT, TEXT.find("1871"), TEXT.find("1871") + 4, ctx=10)
    assert ">>>1871<<<" in win


# --- sample worksheet ---------------------------------------------------------------------


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


# --- pure: acceptance arithmetic ----------------------------------------------------------


def test_acceptance_pass_within_tolerance():
    rows = [_ws_row(f"a{i}", "accept") for i in range(19)] + [_ws_row("r0", "reject")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["decided"] == 20 and report["rejected"] == 1
    assert report["reject_rate"] == 0.05 and report["passed"] is True


def test_acceptance_fail_over_tolerance():
    rows = [_ws_row("a0", "accept"), _ws_row("r0", "reject"), _ws_row("r1", "reject")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["passed"] is False


def test_acceptance_flags_undecided_failures():
    rows = [_ws_row("a0", "accept"), _ws_row("u0", "", chk_grounded="fail")]
    report = acceptance_report(rows)
    assert report["undecided"] == 1 and report["undecided_with_failures"] == 1


def test_check_verification_ref_accepts_decided_worksheet(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,accept\n", encoding="utf-8")
    status = check_verification_ref(path)
    assert status.valid is True and status.kind == "worksheet"


def test_check_verification_ref_rejects_undecided_worksheet(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,\n", encoding="utf-8")
    status = check_verification_ref(path)
    assert status.valid is False and "undecided" in status.reason
    assert status.stats["undecided"] == 1
    message = format_verification_status(status)
    assert "stats:" in message
    assert "undecided: 1" in message
    assert "make verify-review VERIFY_WS=" in message
    assert "--data-verified --verification-ref" in message


def test_verified_data_config_rejects_invalid_ref_with_operator_diagnostics(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        verified_data_config(data_verified=True, verification_ref=str(path))

    message = str(excinfo.value)
    assert "verification reference cannot be used with --data-verified" in message
    assert "undecided: 1" in message
    assert "make verify-review VERIFY_WS=" in message


def test_check_verification_ref_accepts_sample_manifest(tmp_path):
    path = tmp_path / "verify_sample.csv"
    path.write_text("item_id,stratum,decision\nok,s,accept\n", encoding="utf-8")
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(json.dumps({"worksheet": str(path)}), encoding="utf-8")
    status = check_verification_ref(manifest)
    assert status.valid is True and status.kind == "sample_manifest"


def test_per_stratum_failure_is_isolated():
    rows = [_ws_row(f"x{i}", "accept", stratum="clean") for i in range(10)]
    rows += [_ws_row("y0", "reject", stratum="dirty"), _ws_row("y1", "accept", stratum="dirty")]
    report = acceptance_report(rows, tolerance=0.05)
    assert report["per_stratum"]["clean"]["passed"] == 1.0
    assert report["per_stratum"]["dirty"]["passed"] == 0.0  # 50% reject hides at the overall level


# --- accepted-ledger round-trip (the flip is an ADOPTION) ---------------------------------


def test_emit_accepted_ledger_round_trips_through_the_ledger(tmp_path):
    bundle = _bundle(tmp_path, [_item("keep"), _item("drop")])
    out_dir = tmp_path / "accepted"
    n = emit_accepted_ledger(bundle, ["keep"], out_dir)
    assert n == 1
    accepted = load_goldset(out_dir / "goldset.jsonl")
    assert accepted[0].id == "keep" and accepted[0].verified is True
    assert (out_dir / "corpus" / DOC).is_file()  # grounding doc copied -> self-contained
    # The ingester adopts the accepted id by REPLACEMENT (not a boolean flip on the draft).
    ledger = load_verified_ledger([out_dir / "goldset.jsonl"])
    drafts = [_item("keep"), _item("drop")]
    merged, docs, n_verified = apply_verified_ledger(drafts, ledger)
    assert n_verified == 1
    assert next(i for i in merged if i.id == "keep").verified is True
    assert next(i for i in merged if i.id == "drop").verified is False


def test_check_verification_ref_accepts_accepted_ledger(tmp_path):
    bundle = _bundle(tmp_path / "draft", [_item("keep")])
    out_dir = tmp_path / "accepted"
    emit_accepted_ledger(bundle, ["keep"], out_dir)
    status = check_verification_ref(out_dir)
    assert status.valid is True and status.kind == "accepted_ledger"


def test_accepted_ids_only_accept_decisions():
    rows = [_ws_row("a", "accept"), _ws_row("r", "reject"), _ws_row("u", "")]
    assert accepted_ids(rows) == ["a"]


# --- reviewer signals: retrieval rank + page citation ---------------------------------------


def test_worksheet_carries_retrieval_rank_and_page_citation(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    (bundle / "needle_items.jsonl").write_text(
        json.dumps({"id": "a", "retrieval_rank": 2}) + "\n", encoding="utf-8"
    )
    sidecar = bundle / "corpus" / "squad" / "doc1.citations.json"
    sidecar.write_text(
        json.dumps(
            {
                "source": "orig/doc1.pdf",
                "pages": [{"page": 3, "char_start": 0, "char_end": len(TEXT)}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    assert rows[0]["retrieval_rank"] == "2"
    assert rows[0]["page_citation"] == "doc1.pdf p.3"
    # The card renders both so the reviewer sees them without leaving the terminal.
    card = format_card(rows[0], 1, 1, 0)
    assert "doc1.pdf p.3" in card and "retrieval_rank=2" in card


def test_load_retrieval_ranks_reads_both_sidecars(tmp_path):
    (tmp_path / "needle_items.jsonl").write_text(
        json.dumps({"id": "a", "retrieval_rank": 1}) + "\n", encoding="utf-8"
    )
    (tmp_path / "item_provenance.jsonl").write_text(
        json.dumps({"id": "b", "retrieval_rank": 4})
        + "\n"
        + json.dumps({"id": "c", "retrieval_rank": None})
        + "\n",
        encoding="utf-8",
    )
    ranks = load_retrieval_ranks(tmp_path)
    assert ranks == {"a": 1, "b": 4}  # a null rank (retrieval miss) is simply absent


def test_confidence_order_puts_least_confident_first():
    good = _ws_row("good", cc_grounded="true", cc_supported="true", retrieval_rank="1")
    bad = _ws_row("bad", cc_grounded="false")
    mid = _ws_row("mid")
    assert row_confidence(good) > row_confidence(mid) > row_confidence(bad)
    assert confidence_order([good, bad, mid]) == [1, 2, 0]


# --- additive sample enlargement (merge mode) -----------------------------------------------


def test_merge_adds_only_new_rows_and_preserves_decided_bytes(tmp_path):
    bundle = _bundle(tmp_path, [_item(f"i{k}") for k in range(6)])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    rows[0]["decision"] = "accept"
    rows[0]["human_note"] = "ok, з комою"  # the comma forces CSV quoting on this row
    write_worksheet_rows(out, rows, fields)
    before_lines = out.read_bytes().splitlines(keepends=True)
    decided_id = rows[0]["item_id"]

    added, total = merge_sample_worksheet(bundle, out, n=5, seed=1)
    assert added == 3 and total == 5  # same-seed draw is a superset; only new ids appended

    after_rows, _ = load_worksheet(out)
    ids = [r["item_id"] for r in after_rows]
    assert len(set(ids)) == len(ids)  # a decided row is never re-drawn
    assert ids[:2] == [r["item_id"] for r in rows]  # existing rows keep their order
    after_lines = out.read_bytes().splitlines(keepends=True)
    assert after_lines[: len(before_lines)] == before_lines  # decided rows byte-for-byte
    assert next(r for r in after_rows if r["item_id"] == decided_id)["decision"] == "accept"
    manifest = json.loads((out.with_name("sample_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["merged_added"] == 3 and manifest["sample_size"] == 5


def test_merge_is_idempotent(tmp_path):
    bundle = _bundle(tmp_path, [_item(f"i{k}") for k in range(6)])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    merge_sample_worksheet(bundle, out, n=5, seed=1)
    snapshot = out.read_bytes()
    added, total = merge_sample_worksheet(bundle, out, n=5, seed=1)
    assert added == 0 and total == 5
    assert out.read_bytes() == snapshot


def test_merge_falls_back_to_fresh_build(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b")])
    out = tmp_path / "ws.csv"
    added, total = merge_sample_worksheet(bundle, out, n=2, seed=1)
    assert added == 2 and total == 2
    assert out.is_file()


# --- coded rejection reasons ----------------------------------------------------------------


def test_infer_reject_code_prefers_first_failed_check():
    assert infer_reject_code(_ws_row("a", chk_reference="fail")) == "wrong_reference"
    assert (
        infer_reject_code(_ws_row("a", chk_grounded="fail", chk_reference="fail")) == "ungrounded"
    )
    assert infer_reject_code(_ws_row("a")) == "other"


def test_run_accept_exports_rejection_reasons(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b")])
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    rows[0]["decision"] = "accept"
    rows[1]["decision"] = "reject"
    rows[1]["reject_code"] = "bad_question"
    rows[1]["human_note"] = "тривiальне"
    write_worksheet_rows(out, rows, fields)
    assert run_accept(out, bundle, None, tolerance=0.6) == 0
    reasons = json.loads(
        (bundle / "accepted" / "rejection_reasons.json").read_text(encoding="utf-8")
    )
    assert reasons["rejected"] == 1
    cell = reasons["by_code"]["bad_question"]
    assert cell["count"] == 1 and cell["items"][0]["note"] == "тривiальне"


def test_rejection_reasons_summary_infers_missing_codes():
    rows = [_ws_row("a", "reject", chk_grounded="fail"), _ws_row("b", "accept")]
    summary = rejection_reasons_summary(rows)
    assert summary["rejected"] == 1 and "ungrounded" in summary["by_code"]


# --- accept-with-edit re-grounding ----------------------------------------------------------


def test_ground_answer_prefers_occurrence_nearest_hint():
    text = "рік 1871 та ще раз 1871 у кінці"
    late = text.rfind("1871")
    assert ground_answer(text, "1871", hint_start=late) == (late, late + 4)
    assert ground_answer(text, "відсутнє") is None


def test_emit_accepted_ledger_applies_regrounded_edit(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    out_dir = tmp_path / "accepted"
    n = emit_accepted_ledger(bundle, ["a"], out_dir, edits={"a": "Новограді-Волинському"})
    assert n == 1
    item = load_goldset(out_dir / "goldset.jsonl")[0]
    assert item.verified is True
    assert item.reference_answer == "Новограді-Волинському"
    assert item.source_spans[0].text == "Новограді-Волинському"
    start = item.source_spans[0].char_start
    assert TEXT[start : start + len("Новограді-Волинському")] == "Новограді-Волинському"


def test_emit_accepted_ledger_blocks_ungrounded_edit(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    with pytest.raises(ValueError, match="verbatim"):
        emit_accepted_ledger(
            bundle, ["a"], tmp_path / "accepted", edits={"a": "вигадана вiдповiдь"}
        )
