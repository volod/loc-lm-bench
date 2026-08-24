"""What a document lane will accept into the prompt, and which window said so.

Part of rag-vs-long-context-ablation. Split out of `sources.py` because it answers a different
question from the lanes there: they decide WHICH documents the prompt carries, this decides whether
those documents fit at all -- and unlike them it has to talk to the live backend to find out.
"""

from typing import TYPE_CHECKING

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

if TYPE_CHECKING:
    from llb.backends.base import BackendLauncher
    from llb.backends.context_budget import ContextBudget


class DocumentWindow:
    """The usable window a document lane checks each item against -- resolved once, on first use.

    Lazily, because the window that decides a skip is the SERVED one and the backend is not
    serving yet when the lane is wired (`runner_setup` builds the graph before `launcher.start()`,
    and Ollama reports no window at all until a request has loaded the model). The first item's fit
    check is the earliest moment a probe can see the truth, so that is when resolution runs; every
    later item reuses the same answer, and the run manifest reads `provenance()` afterwards.

    Resolution is `llb.backends.context_budget`, the same arithmetic and the same MINIMUM-of-
    declared-and-served rule the agent-loop prompt guard is bound by -- so a lane and a loop on one
    host cannot disagree about what fits.
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
        """Which window bound the skips, or None when no item ever asked (nothing was checked)."""
        return self._budget.provenance() if self._budget is not None else None
