"""vLLM serving knobs shared by the joint-search CLI entry points.

`run-eval` takes a `--config` file plus explicit serving overrides, so a candidate it evaluates is
served at the utilization this host was validated for. A search that could not take them built its
base config with `load_config(None, ...)` and served EVERY vLLM candidate at the `RunConfig`
default, which on a 16 GiB card is not a slower run but a lost candidate: the engine exceeds its
own budget during graph capture and the candidate drops out of the screen. The pre-launch VRAM
guard cannot correct it either -- it derates against OTHER processes' memory, and the default
fraction of a quiet card already reads as free.

The values land on the base `RunConfig`, which is what `candidate_config` derives every screen
cell, every finalist tune, and the public screen from -- so one declaration reaches all of them.
"""

from pathlib import Path
from typing import Any, Optional

from llb.cli.helpers import load_config
from llb.core.config import RunConfig
from llb.optimize.joint_search.constants import DEFAULT_SEARCH_MAX_MODEL_LEN


def search_base_config(
    config: Optional[Path],
    *,
    gpu_memory_utilization: Optional[float] = None,
    **overrides: Any,
) -> RunConfig:
    """The base config for a search: the YAML file, then the explicit overrides on top."""
    fields: dict[str, Any] = dict(overrides)
    fields["gpu_memory_utilization"] = gpu_memory_utilization
    return load_config(config, **{k: v for k, v in fields.items() if v is not None})


def search_max_model_len(base: RunConfig, max_model_len: Optional[int]) -> int:
    """The vLLM context cap every candidate cell is served at.

    Flag, then the config file, then the search default -- so `--config` alone can move it, and a
    search never falls back to a model's native window (128k+ across this roster), which no
    16 GiB card can hold a KV cache for.
    """
    return max_model_len or base.max_model_len or DEFAULT_SEARCH_MAX_MODEL_LEN


def serving_summary(base: RunConfig, max_model_len: int) -> str:
    """One `[joint-search] serving ...` line naming what every vLLM cell will be served at."""
    return f"gpu_memory_utilization={base.gpu_memory_utilization:g} max_model_len={max_model_len}"
