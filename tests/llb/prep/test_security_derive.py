"""security benchmark corpus-specific case derivation from an ontology draft bundle."""

import json
from pathlib import Path

import pytest

from llb.prep import security_derive as sd
from llb.prep import security_case_builders as scb
from llb.prep.ontology.models import DocExtraction, Entity
from llb.goldset.schema import SourceSpan
from llb.scoring import security
from llb.scoring import security_cases

# A tiny grounded corpus: doc text with entity mentions at exact offsets. char_end - char_start must
# equal len(text) for every span (SourceSpan enforces it), so the derived grounding is exact.
DOC_ID = "corpus/doc-001.md"
DOC_TEXT = (
    "Олена Ткач керує компанією ТОВ Приклад. Ігор Пасічник очолює групу Русини у регіоні. "
    "Група Латиняни також згадана."
)


def _span(text: str) -> SourceSpan:
    start = DOC_TEXT.index(text)
    return SourceSpan(doc_id=DOC_ID, char_start=start, char_end=start + len(text), text=text)


def _entity(name: str, etype: str) -> Entity:
    return Entity(name=name, type=etype, mentions=[_span(name)])


def _extractions() -> list[DocExtraction]:
    return [
        DocExtraction(
            doc_id=DOC_ID,
            entities=[
                _entity("Олена Ткач", "PERSON"),
                _entity("Ігор Пасічник", "PERSON"),
                _entity("ТОВ Приклад", "ORG"),
                _entity("Русини", "NORP"),
                _entity("Латиняни", "NORP"),
            ],
        )
    ]


def test_derive_requires_grounded_entities():
    with pytest.raises(ValueError, match="no grounded entities"):
        sd.derive_security_cases([DocExtraction(doc_id=DOC_ID)])


def test_every_probe_cites_an_exact_corpus_span():
    # Acceptance gate: every generated probe cites a corpus topic/entity with an EXACT span; the
    # span text must equal the substring of the source document at its offsets.
    cases = sd.derive_security_cases(_extractions())
    assert cases
    for case in cases:
        grounding = case["attrs"]["grounding"]
        assert grounding["doc_id"] == DOC_ID
        exact = DOC_TEXT[grounding["char_start"] : grounding["char_end"]]
        assert exact == grounding["text"]
        assert grounding["text"] in case["prompt"]  # the grounded entity appears in the probe
        assert case["attrs"]["grounding_hint"] == grounding["text"]
        assert case["attrs"]["derived"] is True  # verified=false-equivalent


def test_denial_guard_probes_expect_refusal_and_benign_controls_do_not():
    cases = sd.derive_security_cases(_extractions())
    harmful = [c for c in cases if c["expect_refusal"]]
    benign = [c for c in cases if c["benign"]]
    assert harmful and benign
    # harmful asks are refuse-detector attack cases; benign controls never expect refusal.
    assert all(c["detector_kind"] == security_cases.DETECT_REFUSE for c in harmful)
    assert all(not c["benign"] for c in harmful)
    assert all(not c["expect_refusal"] for c in benign)
    # every harmful vector has a matched benign control referencing the same vector.
    controls = {c["attrs"].get("control_for") for c in benign if "control_for" in c["attrs"]}
    assert scb.VECTOR_PII in controls and scb.VECTOR_DEFAMATION in controls


def test_benign_controls_feed_refusal_appropriateness_only_never_asr():
    # A model that refuses EVERYTHING defends every harmful ask (ASR=0) but over-refuses the benign
    # controls, so refusal-appropriateness drops -- benign cases must not enter the ASR denominator.
    cases = security_cases.load_security_cases(sd.derive_security_cases(_extractions()))
    refuse_all = ["Вибачте, я не можу допомогти." for _ in cases]
    score = security.score_security(cases, refuse_all)
    n_harmful = sum(1 for c in cases if not c.benign)
    assert score.n_attacks == n_harmful  # benign controls excluded from ASR
    assert score.asr == 0.0  # every harmful ask defended
    assert score.refusal_appropriateness < 1.0  # over-refusal on benign controls is penalized


def test_bias_pairs_are_matched_and_score_for_consistency():
    cases = security_cases.load_security_cases(sd.derive_security_cases(_extractions()))
    pairs: dict[str, set[str]] = {}
    for c in cases:
        pid = c.attrs.get(security.BIAS_PAIR_KEY)
        if pid:
            pairs.setdefault(pid, set()).add(c.attrs[security.BIAS_VARIANT_KEY])
    assert pairs  # at least one matched bias pair emitted (PERSON and/or NORP)
    assert all(len(variants) >= 2 for variants in pairs.values())  # each pair is comparable
    # A model that answers one variant of a pair but refuses the other is INCONSISTENT.
    outputs = []
    flip = True
    for c in cases:
        if c.attrs.get(security.BIAS_PAIR_KEY):
            outputs.append("Так, обґрунтування..." if flip else "Вибачте, не можу.")
            flip = not flip
        else:
            outputs.append("Так, обґрунтування...")
    bias = security.bias_pair_consistency(cases, outputs)
    assert bias.n_pairs >= 1
    assert bias.consistency < 1.0  # the alternating decision made every pair inconsistent


def test_bias_pair_consistency_all_answered_is_consistent():
    cases = security_cases.load_security_cases(sd.derive_security_cases(_extractions()))
    answered = ["Так, обґрунтування..." for _ in cases]
    bias = security.bias_pair_consistency(cases, answered)
    assert bias.consistency == 1.0  # same decision across every swap -> consistent


def test_caps_bound_the_derived_set():
    cases = sd.derive_security_cases(
        _extractions(), max_denial_per_vector=1, max_bias_pairs_per_type=0
    )
    assert all(security.BIAS_PAIR_KEY not in c["attrs"] for c in cases)  # bias pairs suppressed
    # one PERSON denial-guard per PII/defamation vector + its benign control, etc.
    assert cases  # still yields denial-guard probes


def test_derived_cases_round_trip_and_are_json_serializable():
    cases = sd.derive_security_cases(_extractions())
    text = json.dumps(cases, ensure_ascii=False)
    reloaded = security_cases.load_security_cases(json.loads(text))
    assert len(reloaded) == len(cases)
    assert all(c.family == security_cases.UNSAFE_CONTENT for c in reloaded)


def test_worksheet_rows_scaffold_one_row_per_case_undecided():
    cases = sd.derive_security_cases(_extractions())
    rows = sd.worksheet_rows(cases)
    assert len(rows) == len(cases)
    for row, case in zip(rows, cases):
        assert row["item_id"] == case["id"]
        assert row["stratum"] == f"derived|security|{case['attrs']['vector']}"
        assert row["question"] == case["prompt"]
        assert row["span_text"] == case["attrs"]["grounding"]["text"]
        assert row["decision"] == "" and row["human_status"] == ""  # human fills these


def test_scaffolded_worksheet_round_trips_through_acceptance_gate(tmp_path):
    # The scaffold must be a VALID verification worksheet: once every row is decided (accept), the
    # real acceptance checker reports it passed with a zero reject rate -- so bench-security can
    # consume it as a verification ref.
    from llb.goldset.verify_acceptance import acceptance_report
    from llb.goldset.verify_base import (
        load_worksheet,
        worksheet_fieldnames,
        write_worksheet_rows,
    )

    cases = sd.derive_security_cases(_extractions())
    rows = sd.worksheet_rows(cases)
    path = tmp_path / "verify_sample.csv"
    write_worksheet_rows(path, rows, worksheet_fieldnames())
    loaded, _ = load_worksheet(path)
    # an undecided worksheet is NOT accepted...
    assert acceptance_report(loaded)["undecided"] == len(loaded)
    # ...decide every row accept -> passes with zero reject rate.
    for row in loaded:
        row["decision"] = "accept"
        row["human_status"] = "decided"
    report = acceptance_report(loaded)
    assert report["undecided"] == 0 and report["rejected"] == 0
    assert report["passed"] is True


def test_scaffolded_worksheet_drives_shared_review_session(tmp_path):
    # The derived worksheet must drive the SAME interactive review session as goldset/chain
    # verification: feeding `y` (accept) per row + `q` decides every row through the shared UI.
    from llb.goldset.verify_base import load_worksheet, worksheet_fieldnames, write_worksheet_rows
    from llb.goldset.verify_session.loop import run_session

    cases = sd.derive_security_cases(_extractions())
    path = tmp_path / "verify_sample.csv"
    write_worksheet_rows(path, sd.worksheet_rows(cases), worksheet_fieldnames())
    run_session(path, inputs=["y"] * len(cases) + ["q"], output=lambda _line: None)
    rows, _ = load_worksheet(path)
    assert all((r.get("decision") or "") == "accept" for r in rows)  # session persisted decisions


def test_committed_verified_sample_passes_the_gate_and_matches_cases():
    # The human-reviewed derived worksheet is committed as a durable verification reference; it must
    # keep passing the verification-ref checker and stay aligned with the committed derived cases.
    from llb.goldset.verify_refcheck import check_verification_ref
    from llb.goldset.verify_base import load_worksheet

    manifest = Path("samples/verification/security_derived/sample_manifest.json")
    status = check_verification_ref(str(manifest))
    assert status.valid and status.kind == "sample_manifest"  # accepted as verified data
    ws_rows, _ = load_worksheet(manifest.parent / "verify_sample.csv")
    assert ws_rows and all((r.get("decision") or "") in ("accept", "reject") for r in ws_rows)
    cases = json.loads(
        Path("samples/benchmarks/security_cases_derived_uk.json").read_text(encoding="utf-8")
    )
    assert {r["item_id"] for r in ws_rows} == {c["id"] for c in cases}  # worksheet covers the set


def test_committed_derived_sample_is_grounded_and_covers_vectors():
    # A small derived-and-committed sample (a real corpus bundle) guards the derivation shape for
    # regression: every case round-trips, is grounded, and the three case kinds are all present.
    path = Path("samples/benchmarks/security_cases_derived_uk.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = security_cases.load_security_cases(raw)
    assert cases
    vectors = {c["attrs"]["vector"] for c in raw}
    assert {scb.VECTOR_BENIGN_CONTROL, scb.VECTOR_BIAS_PAIR} <= vectors
    assert vectors & {scb.VECTOR_PII, scb.VECTOR_DEFAMATION, scb.VECTOR_GROUP_HATE}
    for case in raw:
        grounding = case["attrs"]["grounding"]
        assert grounding["text"] and grounding["text"] in case["prompt"]
        assert grounding["char_end"] > grounding["char_start"]
        assert case["attrs"]["derived"] is True
    # bias pairs are complete (>=2 variants each) so decision-consistency always has a pair.
    pairs: dict[str, set[str]] = {}
    for c in raw:
        pid = c["attrs"].get(security.BIAS_PAIR_KEY)
        if pid:
            pairs.setdefault(pid, set()).add(c["attrs"][security.BIAS_VARIANT_KEY])
    assert pairs and all(len(v) >= 2 for v in pairs.values())
