"""The usable prompt window one run is measured under: min(declared, served), and its provenance.

The declared window is what a config CLAIMS the model can take; the served one is what the backend
answered when asked. Ollama serves `num_ctx` 4096 however large a window the model card advertises,
so pricing a prompt against the declared side alone calls a prompt that the backend will truncate
one that fits. These fixtures pin both binding directions and what each one records.
"""

from llb.backends.prompt_window import PromptWindow
from llb.backends.served_window import BUDGET_SOURCE_DECLARED, BUDGET_SOURCE_SERVED
from llb.core.config import RunConfig

# No roster entry, so nothing but the explicit budget can declare a window -- the fixture states
# the declared side itself instead of depending on what the model manifest happens to price.
UNLISTED_MODEL = "not-in-any-roster:test"


class _Launcher:
    """A started launcher reporting the window it serves (None == the probe could not see one)."""

    def __init__(self, served: int | None):
        self._served = served
        self.reads = 0

    def served_context(self) -> int | None:
        self.reads += 1
        return self._served


def _config(declared: int, max_tokens: int = 256) -> RunConfig:
    return RunConfig(model=UNLISTED_MODEL, context_budget=declared, max_tokens=max_tokens)


def test_the_served_window_binds_when_it_is_smaller_than_the_declared_one():
    window = PromptWindow(_config(declared=32768), launcher=_Launcher(served=4096))

    assert window.resolve().bound_max_model_len == 4096
    assert window.provenance() == {
        "declared_max_model_len": 32768,
        "served_max_model_len": 4096,
        "budget_source": BUDGET_SOURCE_SERVED,
    }


def test_the_declared_window_still_binds_when_the_backend_serves_a_larger_one():
    window = PromptWindow(_config(declared=4096), launcher=_Launcher(served=32768))

    assert window.resolve().bound_max_model_len == 4096
    assert window.provenance() == {
        "declared_max_model_len": 4096,
        "served_max_model_len": 32768,
        "budget_source": BUDGET_SOURCE_DECLARED,
    }


def test_a_probe_that_sees_nothing_falls_back_to_the_declared_window():
    """A miss is not a measurement of "unbounded": the config still bounds the prompt."""
    window = PromptWindow(_config(declared=8192), launcher=_Launcher(served=None))

    assert window.resolve().budget_source == BUDGET_SOURCE_DECLARED
    assert window.provenance()["served_max_model_len"] is None


def test_the_window_resolves_once_and_only_when_something_asks():
    """A probe taken at wiring time would read a backend that is not serving yet."""
    launcher = _Launcher(served=4096)
    window = PromptWindow(_config(declared=32768), launcher=launcher)

    assert window.provenance() is None, "nothing asked, so nothing bound anything"
    assert window.fits(10) is True
    first = window.resolve()
    assert window.resolve() is first, "the probe must not repeat per caller"
    assert launcher.reads == 1


def test_the_bound_window_decides_what_fits():
    """(4096 - 512 headroom - 256 completion) * 3 chars/token = 9,984 usable characters."""
    window = PromptWindow(_config(declared=131072), launcher=_Launcher(served=4096))

    assert window.fits(9_000)
    assert not window.fits(20_000)
