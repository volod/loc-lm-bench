"""Committed sample fixtures for the guarded composite pipeline."""

import json
from pathlib import Path

from llb.bench.text_analysis import run_text_analysis
from llb.goldset.verify import check_verification_ref


SAMPLE_VERIFICATION_ROOT = Path("samples/verification/composite_samples")
TEXT_ANALYSIS_BUNDLE = Path("samples/text_analysis_bundle_uk")


def test_committed_sample_verification_refs_are_valid():
    for category in (
        "text_analysis",
        "summarization",
        "structured",
        "security",
        "agentic",
        "tooling",
    ):
        manifest = SAMPLE_VERIFICATION_ROOT / category / "sample_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        artifact = (manifest.parent / str(payload["artifact"])).resolve()
        assert artifact.exists()

        status = check_verification_ref(manifest)
        assert status.valid is True
        assert status.kind == "sample_manifest"
        assert status.stats["undecided"] == 0


def test_committed_text_analysis_bundle_scores_with_exact_predictions():
    def complete(prompt: str) -> str:
        if "sample-001" in prompt:
            return json.dumps(
                {
                    "key_fact": ["пасажиропотік зріс на 18 відсотків"],
                    "entity": ["Київська міська рада"],
                    "topic": ["міський транспорт"],
                    "trend": ["пасажиропотік зріс"],
                    "risk": ["нестача водіїв"],
                    "decision": ["модернізувати трамвайні маршрути"],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "key_fact": ["320 консультацій"],
                "entity": ["Львівська лікарня"],
                "topic": ["телемедицина"],
                "risk": ["нестабільний інтернет"],
                "decision": ["додати мобільну бригаду"],
            },
            ensure_ascii=False,
        )

    run = run_text_analysis(
        TEXT_ANALYSIS_BUNDLE,
        model="fixture-model",
        backend="fake",
        complete=complete,
        similarity=lambda _a, _b: 0.0,
        persist=False,
    )

    assert run.result.objective_score == 1.0
    assert run.result.n_cases == 2
    assert {row["item_id"] for row in run.rows} == {"sample-001", "sample-002"}
