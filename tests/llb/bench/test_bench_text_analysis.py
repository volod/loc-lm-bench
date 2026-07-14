"""text analysis scored text-analysis runner + richer planted-label emit."""

import json
from pathlib import Path

from llb.bench.text_analysis.constants import TEXT_ANALYSIS_LABELS
from llb.bench.text_analysis.run import run_text_analysis
from llb.prep.text_analysis_corpus import plant_labels, prepare_text_analysis_corpus
from llb.prep.text_analysis_labels import GROUNDED_REQUIRED_KINDS
from llb.scoring import text_analysis_labels as ta
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS


def make_similarity(table):
    def similarity(a, b):
        return table.get((a, b), table.get((b, a), 0.0))

    return similarity


ZERO_SIM = make_similarity({})


# --- planter (richer per-kind PlantedLabelRecords) ----------------------------------------


def test_plant_labels_grounds_evidence_and_adds_alias():
    doc = "У 2023 році частка відновлюваної енергії зросла. Київ став центром реформ."
    raw = [
        {"kind": "entity", "value": "Київ", "evidence": "Київ став центром"},
        {
            "kind": "trend",
            "value": "Зростання частки ВДЕ",
            "evidence": "частка відновлюваної енергії зросла",
            "attrs": {"subject": "ВДЕ"},
        },
    ]
    records = plant_labels("synth-000", doc, raw)
    by_kind = {r["kind"]: r for r in records}
    # entity grounded -> exact offsets + verbatim alias
    ent = by_kind["entity"]
    assert doc[ent["char_start"] : ent["char_end"]] == "Київ"
    assert "Київ" in (ent.get("aliases") or []) or ent["value"] == "Київ"
    # trend direction backfilled from evidence ("зросла" -> up)
    assert by_kind["trend"]["attrs"]["direction"] == ta.DIRECTION_UP


def test_plant_labels_drops_ungrounded_quote_kind():
    doc = "Короткий документ без потрібної цитати."
    raw = [{"kind": "entity", "value": "Марс", "evidence": "Марс далеко"}]
    assert plant_labels("d", doc, raw) == []
    assert "entity" in GROUNDED_REQUIRED_KINDS


def test_plant_labels_keeps_ungrounded_analytical_kind():
    doc = "Документ про економіку."
    raw = [{"kind": "topic", "value": "макроекономічна стабільність", "evidence": ""}]
    records = plant_labels("d", doc, raw)
    assert len(records) == 1
    assert "char_start" not in records[0]  # analytical kinds may be ungrounded (no offsets)


def test_plant_labels_rejects_unknown_kind():
    assert plant_labels("d", "x", [{"kind": "bogus", "value": "v"}]) == []


def test_prepare_text_analysis_corpus_writes_bundle(tmp_path):
    doc = "Інфляція зросла. Ризик дефіциту бюджету. Уряд ухвалив рішення скоротити витрати."

    def fake_complete(prompt):
        return json.dumps(
            {
                "document": doc,
                "labels": [
                    {"kind": "trend", "value": "інфляція зросла", "evidence": "Інфляція зросла"},
                    {
                        "kind": "risk",
                        "value": "дефіцит бюджету",
                        "evidence": "Ризик дефіциту бюджету",
                    },
                    {
                        "kind": "decision",
                        "value": "скоротити витрати",
                        "evidence": "рішення скоротити витрати",
                    },
                ],
            }
        )

    docs, records = prepare_text_analysis_corpus(
        ["бюджет"],
        planter_model="planter",
        judge_model="judge",
        kinds=(ta.TREND, ta.RISK, ta.DECISION),
        complete=fake_complete,
        out_dir=tmp_path,
    )
    assert len(docs) == 1 and len(records) == 3
    assert (tmp_path / "corpus" / "synth-000.md").exists()
    labels_path = tmp_path / "text_analysis_labels.jsonl"
    assert labels_path.exists()
    loaded = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    assert {r["kind"] for r in loaded} == {"trend", "risk", "decision"}
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["synthetic"] is True and prov["n_labels"] == 3


def test_prepare_text_analysis_rejects_planter_equals_judge(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="planter != judge"):
        prepare_text_analysis_corpus(["t"], planter_model="m", judge_model="m", out_dir=tmp_path)


# --- scored runner ------------------------------------------------------------------------


def _write_bundle(tmp_path, doc_id="synth-000"):
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    (corpus / f"{doc_id}.md").write_text("Київ. Львів. Тема: економіка.", encoding="utf-8")
    records = [
        {"label_id": f"{doc_id}-entity-0", "kind": "entity", "value": "Київ", "doc_id": doc_id},
        {"label_id": f"{doc_id}-entity-1", "kind": "entity", "value": "Львів", "doc_id": doc_id},
        {"label_id": f"{doc_id}-topic-0", "kind": "topic", "value": "економіка", "doc_id": doc_id},
        # a judged kind that must stay OUT of the objective headline
        {"label_id": f"{doc_id}-insight-0", "kind": "insight", "value": "ринок", "doc_id": doc_id},
    ]
    (tmp_path / TEXT_ANALYSIS_LABELS).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return tmp_path


def test_run_text_analysis_scores_and_persists(tmp_path):
    bundle = _write_bundle(tmp_path / "b")

    def complete(prompt):
        # perfect entity recovery, perfect topic, plus an insight answer
        return json.dumps(
            {"entity": ["Київ", "Львів"], "topic": ["економіка"], "insight": ["ринок"]}
        )

    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=complete,
        similarity=ZERO_SIM,
        data_dir=tmp_path / "data",
        mirror=lambda *_: None,
    )
    assert run.result.tier == TIER_TEXT_ANALYSIS
    assert run.result.objective_score == 1.0  # entity + topic both perfect; insight excluded
    assert run.result.reliability == 1.0
    assert run.rows[0]["status"] == "ok"
    assert run.rows[0]["n_objective_subtasks"] == 2  # entity + topic, not insight
    # the board ranks under the text-analysis tier
    assert run.board[0]["tier"] == TIER_TEXT_ANALYSIS
    assert run.paths is not None and "text-analysis" in run.paths["manifest"]


def test_run_text_analysis_reports_meter_throughput(tmp_path):
    from llb.bench.common_backend import ThroughputMeter

    bundle = _write_bundle(tmp_path / "b")
    meter = ThroughputMeter()
    meter.completion_tokens, meter.generation_s, meter.calls = 100, 4.0, 4  # 25 tok/s
    run = run_text_analysis(
        bundle,
        model="m",
        backend="ollama",
        complete=lambda _: json.dumps({"entity": ["Київ"], "topic": ["економіка"]}),
        similarity=ZERO_SIM,
        data_dir=tmp_path / "data",
        mirror=lambda *_: None,
        meter=meter,
    )
    assert run.result.tokens_per_s == 25.0  # real throughput flows onto the board row
    manifest = json.loads(Path(run.paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["tokens_per_s"] == 25.0


# --- gated judge for narrative/insight + long_doc map-reduce (category expansion residual) ---------------


def fake_judge(faith, relevancy):
    def scorer(records, _model):
        return [{"faithfulness": faith, "answer_relevancy": relevancy} for _ in records]

    return scorer


def _write_long_doc_bundle(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "synth-000.md").write_text(
        "Перший розділ про бюджет міста та його видатки. " * 40
        + "Висновок: бюджет зріс на 15 відсотків.",
        encoding="utf-8",
    )
    records = [
        {
            "label_id": "synth-000-long_doc-0",
            "kind": "long_doc",
            "value": "бюджет зріс на 15 відсотків",
            "doc_id": "synth-000",
            "attrs": {"question": "На скільки зріс бюджет?"},
        }
    ]
    (tmp_path / TEXT_ANALYSIS_LABELS).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return tmp_path
