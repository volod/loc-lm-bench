"""ontology-validated-answer-gate -- the three-lane comparison and its adopt-or-reject readings.

File-driven: the study consumes canonical per-case rows and the orchestration takes an injected
lane runner, so the whole vertical runs in the lightweight CI install. The one test that drives
real generation uses a scripted fake launcher and a fake ledger, which is what proves all three
lanes, both new statuses, and the repair path are exercised without a GPU.
"""

import json
from pathlib import Path

import pytest

from llb.backends.base import ChatResult
from llb.core.config import RunConfig
from llb.eval import common, graph
from llb.eval.answer_envelope import lane as envelope_lane
from llb.eval.answer_validation.constants import (
    LANE_OFF,
    LANE_PYDANTIC,
    LANE_PYDANTIC_ONTOLOGY,
)
from llb.eval.answer_validation.gate import OntologyGate
from llb.eval.answer_validation.report import format_report
from llb.eval.answer_validation.run import parse_lanes, run_answer_validation
from llb.eval.answer_validation.scope import CorpusLedger
from llb.eval.answer_validation.study import analyze, commonly_answered
from llb.eval.answer_validation.verdict import (
    DECISION_ADOPT,
    DECISION_NOT_ADOPTED,
    DECISION_NOT_MEASURED,
)
from llb.goldset.schema import GoldItem
from llb.prep.ontology.axioms.models import Axiom
from llb.prep.ontology.models import DocExtraction

DOC = "study_fixture.md"
TEXT = "Львів має населення 717 тисяч осіб."
SPAN = {"doc_id": DOC, "char_start": 0, "char_end": len(TEXT), "text": TEXT}

AXIOM = Axiom(
    axiom_id="func-naselennia",
    kind="functional",
    relation="має населення",
    signed_by="reviewer",
    signed_on="2026-01-01",
)


def _row(item_id: str, **overrides) -> dict:
    row = {
        "item_id": item_id,
        "split": "final",
        "status": common.OK,
        "objective_score": 0.5,
        "contains": 1.0,
        "ranking_score": 0.5,
        "completion_tokens": 20,
        "prompt_tokens": 100,
        "latency_s": 1.0,
    }
    row.update(overrides)
    return row


def _rejected(item_id: str, *, correct: bool, axiom_class: str = "functional") -> dict:
    return _row(
        item_id,
        status=common.ONTOLOGY_VIOLATION,
        contains=1.0 if correct else 0.0,
        objective_score=0.9 if correct else 0.1,
        validation_classes=[axiom_class],
        validation_violations=1,
        validation_checked_triples=1,
    )


# --- what an objective delta may be read on -----------------------------------------------------


def test_the_objective_delta_is_read_only_on_the_items_every_lane_answered():
    # A gate that refuses the hard items would otherwise raise its own mean and look like a win.
    lanes = {
        LANE_OFF: [_row("q1"), _row("q2"), _row("q3")],
        LANE_PYDANTIC_ONTOLOGY: [_row("q1"), _row("q2"), _rejected("q3", correct=False)],
    }
    assert commonly_answered(lanes) == ["q1", "q2"]
    report = analyze(lanes, gated_lane=LANE_PYDANTIC_ONTOLOGY, resamples=50)
    assert report["n_items"] == 3 and report["n_commonly_answered"] == 2
    assert report["readings"][0]["n_commonly_answered"] == 2
    # ...and the decline is visible beside it rather than hidden by it.
    assert report["lanes"][LANE_PYDANTIC_ONTOLOGY]["ontology_violation_rate"] == pytest.approx(
        1 / 3, abs=1e-4
    )
    assert report["lanes"][LANE_PYDANTIC_ONTOLOGY]["n_answered"] == 2


def test_lanes_scoring_different_item_sets_are_refused_rather_than_intersected():
    with pytest.raises(ValueError, match="different item sets"):
        analyze({LANE_OFF: [_row("q1")], LANE_PYDANTIC: [_row("q2")]}, resamples=0)


def test_the_baseline_must_be_among_the_scored_lanes():
    with pytest.raises(ValueError, match="was not scored"):
        analyze({LANE_PYDANTIC: [_row("q1")], LANE_PYDANTIC_ONTOLOGY: [_row("q1")]}, resamples=0)


def test_the_gate_cost_is_priced_per_answer_in_tokens_and_seconds():
    lanes = {
        LANE_OFF: [_row("q1", completion_tokens=20, latency_s=1.0)],
        LANE_PYDANTIC_ONTOLOGY: [_row("q1", completion_tokens=260, latency_s=6.0)],
    }
    reading = analyze(lanes, gated_lane=LANE_PYDANTIC_ONTOLOGY, resamples=0)["readings"][0]
    assert reading["added_completion_tokens"] == 240.0
    assert reading["added_latency_s"] == 5.0


# --- catch against false rejection, per axiom class ----------------------------------------------


def test_a_class_that_catches_more_than_it_wrongly_refuses_is_adopted():
    # Eight catches: the minimum-evidence gate needs six differing items at 95% before a paired
    # reading may state anything at all, so a class resting on five is not adopted either.
    ids = [f"q{i}" for i in range(20)]
    gated = [_rejected(i, correct=False) for i in ids[:8]] + [_row(i) for i in ids[8:]]
    report = analyze(
        {LANE_OFF: [_row(i) for i in ids], LANE_PYDANTIC_ONTOLOGY: gated},
        gated_lane=LANE_PYDANTIC_ONTOLOGY,
        resamples=500,
    )
    verdict = report["axiom_classes"][0]
    assert verdict["n_catches"] == 8 and verdict["n_false_rejections"] == 0
    assert verdict["decision"] == DECISION_ADOPT


def test_a_class_that_refuses_correct_answers_is_recorded_measured_and_not_adopted():
    ids = [f"q{i}" for i in range(12)]
    gated = (
        [_rejected(i, correct=False) for i in ids[:2]]
        + [_rejected(i, correct=True) for i in ids[2:6]]
        + [_row(i) for i in ids[6:]]
    )
    report = analyze(
        {LANE_OFF: [_row(i) for i in ids], LANE_PYDANTIC_ONTOLOGY: gated},
        gated_lane=LANE_PYDANTIC_ONTOLOGY,
        resamples=400,
    )
    verdict = report["axiom_classes"][0]
    assert (verdict["n_catches"], verdict["n_false_rejections"]) == (2, 4)
    assert verdict["decision"] == DECISION_NOT_ADOPTED


def test_a_class_that_never_fired_is_not_measured_rather_than_adopted():
    from llb.eval.answer_validation.verdict import class_verdicts

    rows = [_row("q1", validation_classes=["domain"], validation_checked_triples=1)]
    assert class_verdicts(rows, [], 0.95)[0]["decision"] == DECISION_NOT_MEASURED


def test_a_rejection_is_read_against_the_reference_not_against_the_gate():
    # `contains` is the found-rate signal: a verbose but correct refused answer is a FALSE
    # rejection, and the token-F1 objective would have priced it as a wrong one.
    ids = [f"q{i}" for i in range(6)]
    gated = [_rejected("q0", correct=True)] + [_row(i) for i in ids[1:]]
    report = analyze(
        {LANE_OFF: [_row(i) for i in ids], LANE_PYDANTIC_ONTOLOGY: gated},
        gated_lane=LANE_PYDANTIC_ONTOLOGY,
        resamples=50,
    )
    assert report["axiom_classes"][0]["n_false_rejections"] == 1


def test_every_refusal_is_listed_so_a_reader_can_check_the_proxy_label():
    # The catch / false-rejection split rests on an automated correctness proxy, and a proxy can
    # be wrong -- so the report lists every rejection with its answer rather than only counting it.
    ids = [f"q{i}" for i in range(6)]
    gated = [
        _rejected("q0", correct=False),
        _rejected("q1", correct=True, axiom_class="domain"),
    ] + [_row(i) for i in ids[2:]]
    report = analyze(
        {LANE_OFF: [_row(i) for i in ids], LANE_PYDANTIC_ONTOLOGY: gated},
        gated_lane=LANE_PYDANTIC_ONTOLOGY,
        resamples=50,
    )
    listed = report["refusals"]
    assert [row["item_id"] for row in listed] == ["q0", "q1"]
    assert [row["labelled"] for row in listed] == ["catch", "false_rejection"]
    text = format_report(report, metadata={"model": "fake"})
    assert "## Every refused answer" in text and "`q1`" in text
    # An ungated comparison lists nothing rather than an empty table.
    empty = analyze({LANE_OFF: [_row("q0")], LANE_PYDANTIC: [_row("q0")]}, resamples=0)
    assert empty["refusals"] == []
    assert "refused nothing" in format_report(empty)


def test_the_report_names_the_adopted_classes_and_both_rates():
    ids = [f"q{i}" for i in range(6)]
    gated = [_rejected("q0", correct=False)] + [_row(i) for i in ids[1:]]
    report = analyze(
        {LANE_OFF: [_row(i) for i in ids], LANE_PYDANTIC_ONTOLOGY: gated},
        gated_lane=LANE_PYDANTIC_ONTOLOGY,
        resamples=50,
    )
    text = format_report(report, metadata={"model": "fake"})
    assert "false-rejection rate" in text and "catch rate" in text
    assert "axiom classes adopted by this run" in text


# --- the lane selection and the orchestration ----------------------------------------------------


@pytest.mark.parametrize(
    "spec,message",
    [
        ("off", "at least two lanes"),
        ("pydantic,off", "first lane must be"),
        ("off,judge", "unknown validation lane"),
    ],
)
def test_the_lane_selection_refuses_a_comparison_it_cannot_read(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_lanes(spec)


def _gold_item(item_id: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _write_goldset(path: Path) -> None:
    path.write_text(
        "".join(_gold_item(i).model_dump_json(exclude_none=True) + "\n" for i in ("q1", "q2")),
        encoding="utf-8",
    )


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, str]]):
    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.run_name, config.answer_format, config.answer_validation))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        gated = config.answer_validation == "ontology"
        rows = [
            _rejected(item.id, correct=False) if gated and item.id == "q2" else _row(item.id)
            for item in items
        ]
        scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return scores

    return fake_lane


def test_every_lane_scores_the_same_items_and_the_comparison_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    seen: list[tuple[str, str, str]] = []
    run = run_answer_validation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        [LANE_OFF, LANE_PYDANTIC, LANE_PYDANTIC_ONTOLOGY],
        axioms=Path("samples/ontology/axioms_uk_v1.ttl"),
        ledger=Path("samples/ontology/axioms_uk_v1.json"),
        out_dir=tmp_path / "answer-validation",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )
    # The `off` lane carries no new knob at all -- which is what lets it reproduce a recorded
    # bundle rather than merely resemble one.
    assert seen[0][1:] == ("free_text", "off")
    assert seen[1][1:] == ("envelope", "off")
    assert seen[2][1:] == ("envelope", "ontology")
    assert run.report["n_commonly_answered"] == 1
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["lanes"] == "off,pydantic,pydantic+ontology"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Answer validation")


def test_the_gated_lane_may_not_run_without_a_signed_axiom_file_and_a_ledger(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    with pytest.raises(ValueError, match="needs a SIGNED axiom file"):
        run_answer_validation(
            RunConfig(data_dir=tmp_path, goldset_path=goldset),
            [LANE_OFF, LANE_PYDANTIC_ONTOLOGY],
            run_lane=_recording_lane(tmp_path, []),
        )


# --- all three lanes over a fake completer and a fake ledger -------------------------------------


class FakeLauncher:
    def __init__(self, *results):
        self._queue = list(results)

    def chat(self, messages, max_tokens, temperature, timeout):
        return self._queue.pop(0)


def _fake_gate() -> OntologyGate:
    extraction = DocExtraction.model_validate(
        {
            "doc_id": DOC,
            "entities": [],
            "facts": [
                {
                    "subject": "Львів",
                    "relation": "має населення",
                    "object": "717 тисяч осіб",
                    "evidence": SPAN,
                }
            ],
        }
    )
    return OntologyGate([AXIOM], CorpusLedger([extraction]))


def _envelope_text(value: str) -> str:
    return json.dumps(
        {
            "answer": f"Львів має населення {value}.",
            "abstained": False,
            "claims": [
                {
                    "text": f"Львів має населення {value}.",
                    "citations": [1],
                    "triple": {
                        "subject": "Львів",
                        "relation": "має населення",
                        "object": value,
                        "subject_type": "LOC",
                        "object_type": "QUANTITY",
                    },
                }
            ],
        },
        ensure_ascii=False,
    )


WRONG = _envelope_text("2 мільйони осіб")
RIGHT = _envelope_text("717 тисяч осіб")
STATE = {"question": "Яке населення Львова?", "context": "[1] текст", "retrieved": [SPAN]}


def _drive(answer_format: str, validator, *results) -> dict:
    node = graph.make_generate_node(
        FakeLauncher(*results),
        128,
        0.0,
        30.0,
        answer_format=answer_format,
        validator=validator,
    )
    return dict(node(dict(STATE)))


def test_all_three_lanes_run_over_a_fake_completer_and_a_fake_ledger():
    gate = _fake_gate()

    off = _drive(envelope_lane.FREE_TEXT, None, ChatResult(text="Львів має 2 мільйони осіб."))
    assert off["status"] == common.OK and "envelope_status" not in off

    pydantic = _drive(envelope_lane.ENVELOPE, None, ChatResult(text=WRONG))
    assert pydantic["status"] == common.OK and pydantic["envelope_status"] == common.OK
    assert "validation_checked_triples" not in pydantic

    gated = _drive(
        envelope_lane.ENVELOPE,
        gate.check,
        ChatResult(text=WRONG),
        ChatResult(text=WRONG),  # the semantic repair does not rescue it
    )
    assert gated["status"] == common.ONTOLOGY_VIOLATION
    assert gated["validation_repaired"] is True


def test_both_typed_statuses_and_the_repair_path_are_covered_by_the_gated_lane():
    gate = _fake_gate()
    malformed = _drive(
        envelope_lane.ENVELOPE, gate.check, ChatResult(text="проза"), ChatResult(text="теж проза")
    )
    assert malformed["status"] == common.MALFORMED and malformed["envelope_repaired"] is True

    schema_invalid = _drive(
        envelope_lane.ENVELOPE,
        gate.check,
        ChatResult(text='{"answer": "Так"}'),
        ChatResult(text='{"answer": "Так"}'),
    )
    assert schema_invalid["status"] == common.SCHEMA_INVALID

    rescued = _drive(
        envelope_lane.ENVELOPE, gate.check, ChatResult(text=WRONG), ChatResult(text=RIGHT)
    )
    assert rescued["status"] == common.OK and rescued["validation_repaired"] is True
