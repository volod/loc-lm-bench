"""The usable prompt window of a served model, and what fits inside it -- resolved ONCE per run.

Two callers ask the same question. The agent loop asks it per step: without this it sends every
prompt it builds and finds out about an over-long one as a backend error, a truncated prompt, or
(worst) a confidently wrong answer read from a shortened transcript. The document lanes of the
context ablation ask it per item, to decide whether a whole document can be laid into the prompt
at all. Both terminate on a prompt that cannot fit as `context_overflow` -- the status already in
the shared taxonomy (`llb.eval.common`) -- so an unusable configuration is a TYPED outcome instead
of a wrong answer.

The arithmetic is not new: `llb.optimize.tuning_space` already resolves a model's DECLARED usable
window (host planner cap, model window, served `max_model_len`, explicit `context_budget`) and
prices a prompt against it. This module wraps that in the seam both callers need -- a `fits(chars)`
predicate, the char budget the `compact` policy measures its trigger share against, and the
provenance of the bound -- and imports the heavy resolution lazily so the module stays importable
(and unit-testable) without the backend/hardware stack.

Live backends can disagree with the declared window (Ollama's default `num_ctx` is 4096 regardless
of a GGUF advertising 131072), and the disagreement is SILENT: the backend truncates and answers.
Resolution therefore takes the MINIMUM of the declared window and a probed `served_max_model_len`,
and records which one bound the budget so a report can say which window a skip was measured
against.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

from llb.backends.served_window import (
    BUDGET_SOURCE_DECLARED,
    BUDGET_SOURCE_FIXED,
    BUDGET_SOURCE_SERVED,
    BUDGET_SOURCE_UNBOUNDED,
    HttpGet,
    bind_window,
    probe_config_served_max_model_len,
)
from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec

DEFAULT_MODELS_MANIFEST = Path("samples/configs/models_uk.yaml")

# A prompt-char budget of 0 means "cannot bound": no model spec, no served cap, no explicit budget.
# Nothing is refused in that state -- an unknown model must never silently declare a prompt
# unusable, the same rule `fits_context_chars` follows.
UNBOUNDED = 0


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """The resolved per-step prompt budget: how many chars fit, and the predicate that decides."""

    max_prompt_chars: int
    fits: Callable[[int], bool]
    declared_max_model_len: int | None = None
    served_max_model_len: int | None = None
    budget_source: str = BUDGET_SOURCE_UNBOUNDED

    @property
    def bounded(self) -> bool:
        return self.max_prompt_chars > UNBOUNDED

    @property
    def bound_max_model_len(self) -> int | None:
        """The window that actually bound the budget -- the smaller of declared and served."""
        if self.budget_source == BUDGET_SOURCE_SERVED:
            return self.served_max_model_len
        if self.budget_source == BUDGET_SOURCE_DECLARED:
            return self.declared_max_model_len
        return None

    def compaction_trigger_chars(self, share: float) -> int:
        """The prompt size at which a `compact` policy folds its older steps into a summary.

        A share of the resolved budget rather than a constant, so the same policy compacts late on
        a 32k window and early on a 4k one instead of being tuned per host.
        """
        return int(self.max_prompt_chars * share) if self.bounded else UNBOUNDED

    def summary_input_cap_chars(self, overhead_chars: int) -> int:
        """The largest summarize INPUT that still fits the window once its template is paid for.

        The summarize call is bounded by the SAME window every controller prompt is, so this is the
        widest cap that keeps it from being refused -- and, unlike the compaction trigger, it does
        not move with `compact_share`. A folded transcript below it is summarized whole.
        """
        return max(1, self.max_prompt_chars - overhead_chars) if self.bounded else UNBOUNDED

    def provenance(self) -> dict[str, object]:
        """Manifest fields naming the declared window, the probed window, and which bound."""
        return {
            "declared_max_model_len": self.declared_max_model_len,
            "served_max_model_len": self.served_max_model_len,
            "budget_source": self.budget_source,
        }


def prompt_tokens(prompt_chars: int) -> int:
    """Project prompt CHARS onto the token estimate every window check in the repo uses.

    The loop measures chars (it has no tokenizer for the served model); `CHARS_PER_TOKEN` is the
    one measured UA conversion the budget arithmetic already rests on, so a reported prompt-token
    column and the guard that refused a prompt are on the same scale.
    """
    from llb.backends.context_fit import CHARS_PER_TOKEN

    return int(prompt_chars / CHARS_PER_TOKEN)


def unbounded_budget() -> ContextBudget:
    """The no-guard budget: every prompt fits. The default for callers with no resolved model."""
    return ContextBudget(
        max_prompt_chars=UNBOUNDED,
        fits=lambda _chars: True,
        budget_source=BUDGET_SOURCE_UNBOUNDED,
    )


def fixed_budget(max_prompt_chars: int) -> ContextBudget:
    """A budget stated directly in prompt chars (the CI seam, and an explicit operator override)."""
    if max_prompt_chars <= UNBOUNDED:
        return unbounded_budget()
    return ContextBudget(
        max_prompt_chars=max_prompt_chars,
        fits=lambda chars: chars <= max_prompt_chars,
        budget_source=BUDGET_SOURCE_FIXED,
    )


def resolve_model_spec(
    model: str, backend: str | None = None, manifest: Path = DEFAULT_MODELS_MANIFEST
) -> ModelSpec | None:
    """Best-effort planning spec for the SERVED artifact `model` (None when the manifest has none).

    A roster entry names its per-backend artifacts under `sources` -- the run is served by an
    Ollama GGUF tag, not by the entry's headline HF repo id -- so the lookup goes through
    `candidate_sources` and returns the spec priced for the artifact that actually runs.

    None is not a failure: without a spec only an explicit `context_budget` / `max_model_len` can
    bound the prompt, so an unlisted model skips nothing instead of skipping everything.
    """
    from llb.backends.prepare.manifest import load_manifest
    from llb.backends.resolver_sources import candidate_sources

    try:
        specs = load_manifest(manifest)
    except (OSError, ValueError):
        return None
    for spec in specs:
        if spec.get("name") == model:
            return spec
        for source_backend, record in candidate_sources(spec):
            if record.get("source") != model:
                continue
            if backend is not None and source_backend != backend:
                continue
            return cast(ModelSpec, {**spec, "backend": source_backend, **record})
    return None


def _declared_window(
    config: RunConfig,
    model_spec: ModelSpec | None,
    vram_mib: int,
    ram_mib: int,
) -> int:
    """The DECLARED usable window before any live backend probe (0 == cannot bound)."""
    from llb.backends.context_fit import declared_max_context

    return declared_max_context(config, model_spec, vram_mib, ram_mib)


def _budget_from_window(
    config: RunConfig,
    window: int,
    *,
    declared_max_model_len: int,
    served_max_model_len: int | None,
    budget_source: str,
) -> ContextBudget:
    from llb.backends.context_fit import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    usable_tokens = window - PROMPT_HEADROOM_TOKENS - config.max_tokens if window else UNBOUNDED
    max_prompt_chars = max(UNBOUNDED, int(usable_tokens * CHARS_PER_TOKEN))

    def fits(prompt_chars: int) -> bool:
        # Bound against the SAME char budget the report prints, so a served-window bind cannot
        # drift from fits_context_chars' declared-only path when model_spec is missing.
        return max_prompt_chars <= UNBOUNDED or prompt_chars <= max_prompt_chars

    return ContextBudget(
        max_prompt_chars=max_prompt_chars,
        fits=fits,
        declared_max_model_len=declared_max_model_len or None,
        served_max_model_len=served_max_model_len,
        budget_source=budget_source,
    )


def resolve_context_budget(
    config: RunConfig,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int | None = None,
    ram_mib: int | None = None,
    served_max_model_len: int | None = None,
    probe: bool = False,
    http_get: HttpGet | None = None,
) -> ContextBudget:
    """Resolve the usable prompt budget once, for the whole run.

    When `probe` is true (or `served_max_model_len` is passed), the budget is the MINIMUM of the
    declared window and the probed served window. A probe miss falls back to the declared window
    and records `budget_source=declared` with `served_max_model_len=None`.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb

    spec = (
        model_spec if model_spec is not None else resolve_model_spec(config.model, config.backend)
    )
    vram = vram_mib if vram_mib is not None else max_vram_mb(detect_gpus())
    ram = ram_mib if ram_mib is not None else detect_ram_mb()
    declared = _declared_window(config, spec, vram, ram)
    probed = served_max_model_len
    if probe and probed is None:
        probed = probe_config_served_max_model_len(config, http_get=http_get)
    window, source = bind_window(declared, probed)
    if source == BUDGET_SOURCE_UNBOUNDED:
        return ContextBudget(
            max_prompt_chars=UNBOUNDED,
            fits=lambda _chars: True,
            declared_max_model_len=declared or None,
            served_max_model_len=probed,
            budget_source=BUDGET_SOURCE_UNBOUNDED,
        )
    # bind_window never returns "fixed"; keep declared/served as-is.
    if source not in (BUDGET_SOURCE_DECLARED, BUDGET_SOURCE_SERVED):
        source = BUDGET_SOURCE_DECLARED if declared > 0 else BUDGET_SOURCE_SERVED
    return _budget_from_window(
        config,
        window,
        declared_max_model_len=declared,
        served_max_model_len=probed,
        budget_source=source,
    )
