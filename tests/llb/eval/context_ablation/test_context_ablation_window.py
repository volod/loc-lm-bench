"""What a document lane's skip count was measured against, and how the report says so.

The min(declared, served) arithmetic itself is pinned in `tests/llb/backends/test_prompt_window.py`;
what belongs here is that both document lanes read that window, and that the binding survives into
the run manifest, the comparison, and the report.
"""

import json
from pathlib import Path

from llb.backends.prompt_window import PromptWindow
from llb.backends.served_window import BUDGET_SOURCE_DECLARED, BUDGET_SOURCE_SERVED
from llb.core.config import RunConfig
from llb.eval import common as eval_common
from llb.eval.context_ablation.models import LANE_LONG_CONTEXT, LANE_RETRIEVED_DOCUMENT
from llb.eval.context_ablation.report import _window_note
from llb.eval.context_ablation.run import lane_context_windows
from llb.eval.context_ablation.sources import build_context_lane

# Big enough to overflow a 4096-token window (~10k usable chars) and small enough to fit a
# 32768-token one, so ONE document separates the two binding directions.
DOCUMENT = "документ " * 4000
GOLD_STATE = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 10}]}
UNLISTED_MODEL = "not-in-any-roster:test"


class _Launcher:
    def __init__(self, served: int | None):
        self._served = served

    def served_context(self) -> int | None:
        return self._served


def _lane(tmp_path: Path, strategy: str, *, declared: int, served: int | None):
    (tmp_path / "d1.txt").write_text(DOCUMENT, encoding="utf-8")
    config = RunConfig(
        context_strategy=strategy,
        corpus_root=tmp_path,
        model=UNLISTED_MODEL,
        context_budget=declared,
        max_tokens=256,
    )
    window = PromptWindow(config, launcher=_Launcher(served=served))
    return build_context_lane(config, window.fits), window


def test_the_long_context_lane_skips_on_the_served_window(tmp_path: Path):
    """The whole point: a document the DECLARED window admits and the served one cannot."""
    lane, window = _lane(tmp_path, LANE_LONG_CONTEXT, declared=32768, served=4096)

    assert lane is not None
    assert lane.source(GOLD_STATE)["status"] == eval_common.CONTEXT_OVERFLOW
    assert window.provenance()["budget_source"] == BUDGET_SOURCE_SERVED


def test_the_long_context_lane_lays_in_a_document_both_windows_admit(tmp_path: Path):
    """The fit path is untouched: taking the minimum must not start skipping what fits."""
    lane, window = _lane(tmp_path, LANE_LONG_CONTEXT, declared=32768, served=32768)

    assert lane is not None
    state = lane.source(GOLD_STATE)
    assert state.get("status") is None
    assert state["retrieved"][0]["text"] == DOCUMENT
    assert window.provenance()["budget_source"] == BUDGET_SOURCE_DECLARED


def test_the_retrieved_document_lane_reads_the_same_window(tmp_path: Path):
    """Both document lanes share one rule, so they must share one window too."""
    lane, window = _lane(tmp_path, LANE_RETRIEVED_DOCUMENT, declared=32768, served=4096)

    assert lane is not None
    retrieved = {"retrieved": [{"doc_id": "d1.txt", "chunk_id": "c", "text": "chunk"}]}
    assert lane.refiner({}, retrieved)["status"] == eval_common.CONTEXT_OVERFLOW
    assert window.provenance()["budget_source"] == BUDGET_SOURCE_SERVED


def test_lane_context_windows_reads_the_binding_back_off_the_run_manifests(tmp_path: Path):
    """The comparison reports the window the RUN recorded, never one it recomputes."""
    bundle = tmp_path / "long_context"
    bundle.mkdir()
    binding = {
        "declared_max_model_len": 32768,
        "served_max_model_len": 4096,
        "budget_source": BUDGET_SOURCE_SERVED,
    }
    (bundle / "manifest.json").write_text(json.dumps({"context_window": binding}), encoding="utf-8")
    missing = tmp_path / "closed_book"
    missing.mkdir()

    windows = lane_context_windows({"long_context": [str(bundle)], "closed_book": [str(missing)]})
    assert windows == {"long_context": binding, "closed_book": None}


def test_the_report_names_which_window_bound_the_lane():
    served = {
        "context_window": {
            "declared_max_model_len": 32768,
            "served_max_model_len": 4096,
            "budget_source": BUDGET_SOURCE_SERVED,
        }
    }
    declared = {
        "context_window": {
            "declared_max_model_len": 4096,
            "served_max_model_len": None,
            "budget_source": BUDGET_SOURCE_DECLARED,
        }
    }
    assert _window_note(served) == " -- window 4096 tokens (served, declared 32768)"
    assert _window_note(declared) == " -- window 4096 tokens (declared)"
    assert _window_note({"context_window": None}) == ""
