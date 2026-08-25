"""The `rag` lane's prompt is checked against the window the backend actually serves.

`top_k * chunk_size` is the same on every item, so a `rag` overflow is a CONFIGURATION error, not a
per-item outcome: either every prompt fits or none do. The run is not refused -- the estimate is an
upper bound on a context that short chunks and filters routinely make smaller -- but it must never
be silent, because the backend truncates without saying so and the score would then read as that
configuration's quality.
"""

from llb.backends.prompt_window import PromptWindow
from llb.core.config import RunConfig
from llb.eval.context_ablation.models import LANE_CLOSED_BOOK, LANE_LONG_CONTEXT, LANE_RAG
from llb.executor.runner_setup import check_rag_prompt_window

UNLISTED_MODEL = "not-in-any-roster:test"


class _Launcher:
    def __init__(self, served: int | None):
        self._served = served

    def served_context(self) -> int | None:
        return self._served


def _window(config: RunConfig, served: int | None) -> PromptWindow:
    return PromptWindow(config, launcher=_Launcher(served=served))


def _config(strategy: str = LANE_RAG, *, top_k: int = 12, chunk_size: int = 1280) -> RunConfig:
    return RunConfig(
        context_strategy=strategy,
        model=UNLISTED_MODEL,
        context_budget=131072,
        max_tokens=512,
        top_k=top_k,
        chunk_size=chunk_size,
    )


def test_a_prompt_the_declared_window_admits_and_the_served_one_cannot_is_named():
    """top_k 12 x chunk_size 1280 = 15,360 chars ~ 6,144 tokens: inside 131072, past 4096."""
    config = _config()
    warning = check_rag_prompt_window(config, _window(config, served=4096))

    assert warning is not None
    assert "served window of 4096" in warning
    assert "declared 131072" in warning


def test_the_same_prompt_is_silent_when_the_backend_serves_what_the_config_declares():
    config = _config()
    assert check_rag_prompt_window(config, _window(config, served=131072)) is None


def test_a_small_retrieved_prompt_inside_the_served_window_is_silent():
    config = _config(top_k=3, chunk_size=256)
    assert check_rag_prompt_window(config, _window(config, served=4096)) is None


def test_the_document_lanes_are_not_checked_here():
    """They skip PER ITEM against the same window; a config-level warning would double-count."""
    config = _config(LANE_LONG_CONTEXT)
    assert check_rag_prompt_window(config, _window(config, served=4096)) is None
    closed = _config(LANE_CLOSED_BOOK)
    assert check_rag_prompt_window(closed, _window(closed, served=4096)) is None


def test_an_injected_runner_has_no_window_to_check_against():
    assert check_rag_prompt_window(_config(), None) is None
