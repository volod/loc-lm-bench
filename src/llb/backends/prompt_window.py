"""The usable prompt window ONE run was measured under, resolved once and only when asked.

`context_budget.resolve_context_budget` answers "how big a prompt fits" for a config plus a probed
backend. This wraps that in the shape a run needs: the backend is not serving yet when a run's
graph is wired (`runner_setup` builds it before `launcher.start()`, and Ollama reports no window at
all until a request has loaded the model), so resolution has to wait for the first caller that
actually needs an answer. Every later caller reuses it, and the run manifest reads `provenance()`
afterwards to record which window bound the run.
"""

from typing import TYPE_CHECKING

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

if TYPE_CHECKING:
    from llb.backends.base import BackendLauncher
    from llb.backends.context_budget import ContextBudget


class PromptWindow:
    """A run's `fits(chars)` predicate plus the provenance of the window it decided against.

    Resolution takes the MINIMUM of the declared window and the one the launcher is serving -- the
    same rule the agent-loop prompt guard is bound by -- so a lane, a loop, and a sweep on one host
    cannot disagree about what fits.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        launcher: "BackendLauncher | None" = None,
        model_spec: ModelSpec | None = None,
        vram_mib: int | None = None,
        ram_mib: int | None = None,
    ):
        self._config = config
        self._launcher = launcher
        self._model_spec = model_spec
        self._vram_mib = vram_mib
        self._ram_mib = ram_mib
        self._budget: "ContextBudget | None" = None

    def resolve(self) -> "ContextBudget":
        """The bound window, probing the live backend the first time and caching it after."""
        if self._budget is None:
            from llb.backends.context_budget import resolve_context_budget
            from llb.backends.served_window import launcher_served_window

            # The launcher is the backend that will answer the prompts, so it is the thing to
            # ask; the HTTP probe is only for a caller that has no launcher to ask. Asking both
            # would just repeat the same failing request when the backend is unreachable.
            served = launcher_served_window(self._launcher) if self._launcher is not None else None
            self._budget = resolve_context_budget(
                self._config,
                model_spec=self._model_spec,
                vram_mib=self._vram_mib,
                ram_mib=self._ram_mib,
                served_max_model_len=served,
                probe=self._launcher is None,
            )
        return self._budget

    def fits(self, context_chars: int) -> bool:
        return self.resolve().fits(context_chars)

    def provenance(self) -> dict[str, object] | None:
        """Which window bound the run, or None when nothing ever asked (nothing was checked)."""
        return self._budget.provenance() if self._budget is not None else None
