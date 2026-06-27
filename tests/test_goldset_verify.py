"""Tests for human verification gate data verification (`llb.goldset.verify` + `verify_session`).

The pure pieces (stratification, deterministic sampling, acceptance arithmetic, the
accepted-ledger round-trip, parse_command) are checked directly; the session loop is driven by
an INJECTED input iterator + output sink, so no terminal / model / endpoint / GPU is needed -- it
operates only on the CSV.
"""

import json

import pytest

from llb.bench.common import verified_data_config
from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.goldset.verify import (
    WORKSHEET_COLS,
    acceptance_report,
    accepted_ids,
    build_sample_worksheet,
    check_verification_ref,
    corpus_window,
    draw_stratified_sample,
    emit_accepted_ledger,
    format_verification_status,
    load_cross_check,
    load_worksheet,
    stratify,
    write_worksheet_rows,
)
from llb.goldset.verify_session import (
    ACCEPT_CMD,
    CHECK,
    HELP,
    JUMP,
    NEXT,
    PREV,
    QUIT,
    REJECT_CMD,
    decided_count,
    first_undecided_index,
    format_card,
    parse_command,
    run_session,
)
from llb.prep.verified_ledger import apply_verified_ledger, load_verified_ledger

DOC = "squad/doc1.txt"
TEXT = "Леся Українка народилася 1871 року в Новограді-Волинському. Вона була поетесою."


def _item(item_id, *, answer="1871", provenance="frontier-drafted", split="calibration", doc=DOC):
    start = TEXT.find(answer)
    return GoldItem(
        id=item_id,
        question=f"Коли подія {item_id}?",
        reference_answer=answer,
        source_doc_id=doc,
        source_spans=[
            SourceSpan(doc_id=doc, char_start=start, char_end=start + len(answer), text=answer)
        ],
        provenance=provenance,
        split=split,
    )


def _bundle(tmp_path, items, *, synthetic=False):
    """Write a minimal draft bundle (goldset.jsonl + corpus/) under tmp_path."""
    from llb.goldset.schema import dump_goldset

    dump_goldset(items, tmp_path / "goldset.jsonl")
    doc = tmp_path / "corpus" / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(TEXT + "\n", encoding="utf-8")
    if synthetic:
        (tmp_path / "provenance.json").write_text(
            json.dumps({"synthetic": True, "kind": "synthetic-planted"}), encoding="utf-8"
        )
    return tmp_path


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


def _ws_row(item_id, decision="", stratum="s", **over):
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": item_id, "stratum": stratum, "decision": decision})
    row.update(over)
    return row


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


# --- session: parse_command ---------------------------------------------------------------


def test_parse_check_pass_and_fail():
    assert parse_command("g") == __import__(
        "llb.goldset.verify_session", fromlist=["Command"]
    ).Command(CHECK, field="chk_grounded", value=True)
    assert parse_command("R").kind == CHECK and parse_command("R").value is False


def test_parse_decisions_and_nav():
    assert parse_command("y").kind == ACCEPT_CMD
    assert parse_command("x").kind == REJECT_CMD
    assert parse_command("").kind == NEXT
    assert parse_command("b").kind == PREV
    assert parse_command("j5") == __import__(
        "llb.goldset.verify_session", fromlist=["Command"]
    ).Command(JUMP, value=5)
    assert parse_command("q").kind == QUIT
    assert parse_command("?").kind == HELP


def test_format_card_hides_crosscheck_by_default():
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": "a", "question": "q", "context": "ctx>>>x<<<", "cc_supported": "false"})
    assert "crosscheck" not in format_card(row, 1, 1, 0)
    assert "crosscheck" in format_card(row, 1, 1, 0, show_crosscheck=True)


def test_format_card_omits_planted_for_real_items():
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": "a", "synthetic": "false"})
    assert "chk_planted" not in format_card(row, 1, 1, 0)
    row["synthetic"] = "true"
    assert "chk_planted" in format_card(row, 1, 1, 0)


# --- session: the interactive loop (injected I/O) -----------------------------------------


def _ws(tmp_path, rows):
    path = tmp_path / "verify.csv"
    write_worksheet_rows(path, rows, WORKSHEET_COLS)
    return path


def test_session_marks_checks_and_decides(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", stratum="s", synthetic="false"), _ws_row("b", stratum="s")])
    out: list[str] = []
    # item a: grounded pass, reference fail, accept -> advances; item b: reject; then finish.
    decided = run_session(
        path,
        inputs=iter(["g", "R", "y", "x", "q"]),
        output=out.append,
        show_crosscheck=False,
    )
    assert decided == 2
    rows, _ = load_worksheet(path)
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["a"]["chk_grounded"] == "pass" and by_id["a"]["chk_reference"] == "fail"
    assert by_id["a"]["decision"] == "accept" and by_id["b"]["decision"] == "reject"


def test_session_resumes_at_first_undecided(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", "accept"), _ws_row("b", ""), _ws_row("c", "")])
    assert first_undecided_index(load_worksheet(path)[0]) == 1
    out: list[str] = []
    run_session(path, inputs=iter(["y", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    # The session opened on item b (first undecided) and accepted it; a stayed accepted, c untouched.
    assert {r["item_id"]: r["decision"] for r in rows} == {"a": "accept", "b": "accept", "c": ""}


def test_session_planted_check_rejected_for_real_item(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", synthetic="false")])
    out: list[str] = []
    run_session(path, inputs=iter(["p", "q"]), output=out.append)
    rows, _ = load_worksheet(path)
    assert rows[0]["chk_planted"] == ""  # the N/A planted mark was refused
    assert any("N/A" in line for line in out)


def test_session_save_preserves_context_column(tmp_path):
    path = _ws(tmp_path, [_ws_row("a", context="some>>>span<<<text")])
    run_session(path, inputs=iter(["y", "q"]), output=[].append)
    rows, _ = load_worksheet(path)
    assert rows[0]["context"] == "some>>>span<<<text"  # sampler-owned column not clobbered


def test_decided_count(tmp_path):
    rows = [_ws_row("a", "accept"), _ws_row("b", "reject"), _ws_row("c", "")]
    assert decided_count(rows) == 2
