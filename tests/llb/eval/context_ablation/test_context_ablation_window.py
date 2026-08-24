"""The window a document lane's skips are measured against (min of declared and served).

Both document lanes promise to SKIP an item whose documents exceed the model's usable window
rather than truncate it. That promise is only kept if the window checked is the one the backend is
actually SERVING: Ollama serves `num_ctx` 4096 however large a window the model card advertises,
so a declared-only check hands the backend a document it then truncates silently -- and the lane
reports the truncated answer as a delivered long-context result.

These fixtures pin both binding directions and the provenance each one records.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.backends.served_window import BUDGET_SOURCE_DECLARED, BUDGET_SOURCE_SERVED
from llb.eval import common as eval_common
from llb.eval.context_ablation.models import LANE_LONG_CONTEXT, LANE_RETRIEVED_DOCUMENT
from llb.eval.context_ablation.report import _window_note
from llb.eval.context_ablation.run import lane_context_windows
from llb.eval.context_ablation.sources import build_context_lane
from llb.eval.context_ablation.window import DocumentWindow

# Big enough to overflow a 4096-token window (~12k chars at 3 chars/token) and small enough to fit
# a 32768-token one, so ONE document separates the two binding directions.
DOCUMENT = "документ " * 4000
GOLD_STATE = {"gold_spans": [{"doc_id": "d1.txt", "char_start": 0, "char_end": 10}]}

# No roster entry, so nothing but the explicit budget can declare a window -- the fixture states
# the declared side itself instead of depending on what the model manifest happens to price.
UNLISTED_MODEL = "not-in-any-roster:test"


class _Launcher:
    """A started launcher reporting the window it serves (None == the probe could not see one)."""

    def __init__(self, served: int | None):
        self._served = served
        self.warmed = 0

    def served_context(self) -> int | None:
        return self._served


def _config(tmp_path: Path, strategy: str, declared: int) -> RunConfig:
    (tmp_path / "d1.txt").write_text(DOCUMENT, encoding="utf-8")
    return RunConfig(
        context_strategy=strategy,
        corpus_root=tmp_path,
        model=UNLISTED_MODEL,
        context_budget=declared,
        max_tokens=256,
    )


def test_the_served_window_binds_the_skip_when_it_is_smaller_than_the_declared_one(tmp_path: Path):
    """The whole point: a document the DECLARED window admits and the served one cannot."""
    config = _config(tmp_path, LANE_LONG_CONTEXT, declared=32768)
    lane = build_context_lane(config, launcher=_Launcher(served=4096))

    assert lane is not None and lane.window is not None
    assert lane.source(GOLD_STATE)["status"] == eval_common.CONTEXT_OVERFLOW
    assert lane.window.provenance() == {
        "declared_max_model_len": 32768,
        "served_max_model_len": 4096,
        "budget_source": BUDGET_SOURCE_SERVED,
    }


def test_the_declared_window_still_binds_when_the_backend_serves_a_larger_one(tmp_path: Path):
    """The mirror direction: a generous backend must not widen what the config declared."""
    config = _config(tmp_path, LANE_LONG_CONTEXT, declared=4096)
    lane = build_context_lane(config, launcher=_Launcher(served=32768))

    assert lane is not None and lane.window is not None
    assert lane.source(GOLD_STATE)["status"] == eval_common.CONTEXT_OVERFLOW
    assert lane.window.provenance() == {
        "declared_max_model_len": 4096,
        "served_max_model_len": 32768,
        "budget_source": BUDGET_SOURCE_DECLARED,
    }


def test_a_document_inside_both_windows_is_laid_in_whole(tmp_path: Path):
    """The fit path is untouched: taking the minimum must not start skipping what fits."""
    config = _config(tmp_path, LANE_LONG_CONTEXT, declared=32768)
    lane = build_context_lane(config, launcher=_Launcher(served=32768))

    assert lane is not None and lane.window is not None
    state = lane.source(GOLD_STATE)
    assert state.get("status") is None
    assert state["retrieved"][0]["text"] == DOCUMENT
    assert lane.window.provenance()["budget_source"] == BUDGET_SOURCE_DECLARED


def test_the_retrieved_document_lane_reads_the_same_served_window(tmp_path: Path):
    """Both document lanes share one rule, so they must share one window too."""
    config = _config(tmp_path, LANE_RETRIEVED_DOCUMENT, declared=32768)
    lane = build_context_lane(config, launcher=_Launcher(served=4096))

    assert lane is not None and lane.window is not None
    retrieved = {"retrieved": [{"doc_id": "d1.txt", "chunk_id": "c", "text": "chunk"}]}
    assert lane.refiner({}, retrieved)["status"] == eval_common.CONTEXT_OVERFLOW
    assert lane.window.provenance()["budget_source"] == BUDGET_SOURCE_SERVED


def test_the_window_is_resolved_once_and_only_when_an_item_asks(tmp_path: Path):
    """A probe before the first item would read a backend that is not serving yet."""
    config = _config(tmp_path, LANE_LONG_CONTEXT, declared=32768)
    window = DocumentWindow(config, launcher=_Launcher(served=4096))

    assert window.provenance() is None, "nothing was checked, so nothing bound anything"
    assert window.fits(10) is True
    first = window.resolve()
    assert window.resolve() is first, "the probe must not repeat per item"


def test_an_explicit_fits_predicate_keeps_the_lane_free_of_a_window(tmp_path: Path):
    """The injected-predicate seam CI runs on resolves nothing and claims no provenance."""
    config = _config(tmp_path, LANE_LONG_CONTEXT, declared=32768)
    lane = build_context_lane(config, lambda chars: False)

    assert lane is not None and lane.window is None
    assert lane.source(GOLD_STATE)["status"] == eval_common.CONTEXT_OVERFLOW


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
