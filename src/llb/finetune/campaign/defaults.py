"""Default (real-backend) implementations of the injectable campaign seams."""

from pathlib import Path

from llb.backends.planner.architecture import enrich_arch
from llb.backends.planner.plan import plan_model
from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, JsonObject, ModelPlanRow, ModelSpec
from llb.finetune.campaign.coerce import _model_key
from llb.finetune.campaign.model import CompatFn, EvalFn, PlannerFn, ReclaimFn


def _default_eval_fn(*, limit: int | None) -> EvalFn:
    from llb.executor.runner import run_eval

    def run(config: RunConfig, split: str, _round_run_dir: Path) -> EvalResult:
        return run_eval(config, split=split, limit=limit)

    return run


def _default_planner_fn(model_specs: list[ModelSpec]) -> PlannerFn:
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb

    specs = {_model_key(spec): spec for spec in model_specs}
    vram_mib = max_vram_mb(detect_gpus())
    ram_mib = detect_ram_mb()

    def plan(model: str, config: RunConfig) -> ModelPlanRow:
        spec = specs.get(model) or {
            "name": model,
            "source": model,
            "backend": config.backend,
        }
        return plan_model(enrich_arch(spec), vram_mib=vram_mib, ram_mib=ram_mib)

    return plan


def _default_compat_fn() -> CompatFn:
    """Config-only trainability probe: cheap, and UNKNOWN (never a skip) when unreachable."""
    from llb.finetune.compat import config_compat_probe

    return config_compat_probe


def _default_reclaim_fn() -> ReclaimFn:
    baseline: int | None = None

    def reclaim() -> JsonObject:
        nonlocal baseline
        from llb.backends.hardware import detect_gpus
        from llb.executor.vram import assert_reclaimed, read_baseline

        if not detect_gpus():
            return {"skipped": True, "reason": "no GPU detected"}
        try:
            if baseline is None:
                baseline = read_baseline()
            return dict(assert_reclaimed(baseline))
        except SystemExit as exc:
            return {"skipped": True, "reason": str(exc)}

    return reclaim


def _default_out_dir(config: RunConfig) -> Path:
    _run_id, stamp = new_run_timestamp()
    return config.data_dir / "finetune-campaign" / stamp
