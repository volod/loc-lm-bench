"""How big a prompt a served model can hold, and whether a given one fits.

The arithmetic every window guard in the repo rests on, in one place: the char/token conversion,
the DECLARED window a config claims (host planner cap, model card, `max_model_len`,
`context_budget`), and the bind against what the backend is PROBED as serving. It lives beside the
launchers because that is what it is about -- `llb.optimize.tuning_space` prices a RETRIEVED prompt
on top of it, `llb.backends.context_budget` prices an agent step, and both must agree.

Pure and import-light on purpose: no probe, no HTTP, no Optuna. A caller that has a served window
passes it in; one that does not gets the declared answer and a `budget_source` saying so.
"""

from llb.backends.planner.plan import plan_model
from llb.backends.served_window import bind_window
from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

CHARS_PER_TOKEN = 3.0  # UA measured ~0.33 tok/char in real-model validation -> ~3 chars/token

PROMPT_HEADROOM_TOKENS = 512  # system prompt + question + answer headroom


def estimate_context_tokens(config: RunConfig, context_chars: int) -> int:
    """Rough tokens consumed by `context_chars` of context + headroom + the requested completion."""
    return int(context_chars / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS + config.max_tokens


def declared_max_context(
    config: RunConfig, model_spec: ModelSpec | None, vram_mib: int, ram_mib: int
) -> int:
    """What the CONFIG says the model can hold, before any live backend is asked.

    The smallest of: the planner's max context for the host, the model window, the requested
    `max_model_len` cap, and an explicit `context_budget`. 0 means "cannot bound" -- without a
    `model_spec` only an explicit budget can, which is why an unlisted model is not refused.
    """
    ctx = 0
    if model_spec is not None:
        row = plan_model(model_spec, vram_mib, ram_mib)
        ctx = row["ctx_max"] or int(model_spec.get("max_context") or 0)
    if config.max_model_len:
        ctx = min(ctx, config.max_model_len) if ctx else config.max_model_len
    if config.context_budget:
        ctx = min(ctx, config.context_budget) if ctx else config.context_budget
    return ctx


def bound_max_context(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
    served_max_model_len: int | None = None,
) -> tuple[int, str]:
    """The usable window and which side bound it: declared, served, or neither.

    A declared window is what a config CLAIMS; the served one is what the backend answered when
    asked. Ollama serves `num_ctx` 4096 however large a window the model card advertises, so the
    declared side alone prices a prompt the backend would truncate as one that fits. The smaller
    of the two is what actually truncates, so it is the one a fit check must use.
    """
    return bind_window(
        declared_max_context(config, model_spec, vram_mib, ram_mib), served_max_model_len
    )


def fits_context_chars(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
    context_chars: int,
    served_max_model_len: int | None = None,
) -> bool:
    """True if a prompt carrying `context_chars` of context fits the usable window.

    With neither a `model_spec` nor a probed window, only an explicit `context_budget` can bound
    the prompt, so an unknown model on an unreachable backend never silently declares a document
    unusable. A probe changes that on purpose: a served window IS a measurement, and it bounds the
    prompt whether or not the roster prices the model.
    """
    estimated = estimate_context_tokens(config, context_chars)
    if config.context_budget is not None and estimated > config.context_budget:
        return False
    ctx, _source = bound_max_context(config, model_spec, vram_mib, ram_mib, served_max_model_len)
    return ctx <= 0 or estimated <= ctx
