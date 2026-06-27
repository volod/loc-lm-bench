"""security benchmark public adversarial-set adapters (AdvBench / HarmBench / JailbreakBench), UA-adapted."""

import json

import pytest

from llb.bench import security as bench_sec
from llb.prep import security_sources as ss
from llb.scoring import security


def test_from_advbench_normalizes_goal():
    behaviors = ss.from_advbench([{"goal": "do harm", "target": "Sure"}, {"goal": ""}])
    assert len(behaviors) == 1  # empty goal dropped
    assert behaviors[0].text == "do harm" and behaviors[0].source == ss.ADVBENCH


def test_from_harmbench_keeps_category_and_id():
    behaviors = ss.from_harmbench(
        [{"Behavior": "b", "SemanticCategory": "cyber", "BehaviorID": "x1"}]
    )
    assert behaviors[0].id == "harmbench-x1" and behaviors[0].category == "cyber"


def test_from_jailbreakbench_uses_goal():
    behaviors = ss.from_jailbreakbench([{"Goal": "g", "Category": "Harassment", "Index": "3"}])
    assert behaviors[0].id == "jbb-3" and behaviors[0].text == "g"


def test_behavior_to_case_default_unsafe_content_refuse():
    b = ss.Behavior(id="advbench-0000", text="harmful ask", source=ss.ADVBENCH)
    case = ss.behavior_to_case(b)
    assert case["family"] == security.UNSAFE_CONTENT
    assert case["detector_kind"] == security.DETECT_REFUSE and case["expect_refusal"] is True
    assert "українською" in case["prompt"] and "harmful ask" in case["prompt"]
    assert case["attrs"]["public_source"] == ss.ADVBENCH


def test_behavior_to_case_translate_injected():
    b = ss.Behavior(id="x", text="harmful ask", source=ss.ADVBENCH)
    case = ss.behavior_to_case(b, translate=lambda _t: "шкідливий запит")
    assert "шкідливий запит" in case["prompt"] and "harmful ask" not in case["prompt"]


def test_jailbreak_wrap_tags_jailbreak_family():
    b = ss.Behavior(id="x", text="harmful ask", source=ss.JAILBREAKBENCH)
    case = ss.behavior_to_case(b, jailbreak_wrap=ss.JAILBREAK_TEMPLATES[0])
    assert case["family"] == security.JAILBREAK and "DAN" in case["prompt"]


def test_adapt_public_set_limit_and_default_family():
    rows = [{"goal": f"ask {i}"} for i in range(5)]
    cases = ss.adapt_public_set(ss.ADVBENCH, rows, limit=2)
    assert len(cases) == 2 and all(c["family"] == security.UNSAFE_CONTENT for c in cases)


def test_adapt_public_set_jailbreak_default_family():
    rows = [{"Goal": "g", "Index": "1"}]
    cases = ss.adapt_public_set(ss.JAILBREAKBENCH, rows, jailbreak_wrap=ss.JAILBREAK_TEMPLATES[1])
    assert cases[0]["family"] == security.JAILBREAK


def test_adapt_public_set_unknown_source():
    with pytest.raises(ValueError, match="unknown public security source"):
        ss.adapt_public_set("bogus", [])


def test_adapted_cases_load_as_security_cases(tmp_path):
    rows = [{"goal": "make a weapon"}, {"goal": "phish someone"}]
    cases = ss.adapt_public_set(ss.ADVBENCH, rows)
    path = tmp_path / "adapted.json"
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    loaded = bench_sec.load_cases_file(path)  # round-trips through the SecurityCase schema
    assert len(loaded) == 2 and all(c.detector_kind == security.DETECT_REFUSE for c in loaded)


def test_load_rows_csv(tmp_path):
    path = tmp_path / "advbench.csv"
    path.write_text("goal,target\nharm one,Sure\nharm two,Okay\n", encoding="utf-8")
    rows = ss.load_rows(path)
    assert [r["goal"] for r in rows] == ["harm one", "harm two"]
