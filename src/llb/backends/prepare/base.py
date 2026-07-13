"""Shared vocabulary for the prepare package: action/backend constants, the download-preflight
tuning knobs, the Ollama pull-timeout env, and the injectable-callable type aliases.

No behavior lives here -- the manifest loader, store/disk probes, planner, fetchers, and the
`prepare_models` orchestrator all import their shared names from this module.
"""

from pathlib import Path
from typing import Callable

from llb.core.contracts import ModelSpec, PreparedModel

ACTION_PULL = "pull"  # ollama pull
ACTION_CACHE = "cache"  # hf snapshot download for vLLM
ACTION_SKIP = "skip"
SUPPORTED_BACKENDS = ("ollama", "vllm")
OLLAMA_PULL_TIMEOUT_ENV = "LLB_OLLAMA_PULL_TIMEOUT_S"
DEFAULT_OLLAMA_PULL_TIMEOUT_S = 1800
MAX_ERROR_DETAIL_CHARS = 400

# Disk preflight: refuse a long download up front when the destination filesystem is too small,
# so a multi-GiB pull does not fail an hour in. It is REUSE-AWARE -- an artifact already in its
# backend store skips the check, since re-running prepare reuses the cache and fetches ~nothing.
DOWNLOAD_SAFETY_FACTOR = 1.15  # tokenizer + safetensors index + packaging beyond the raw weights
DOWNLOAD_HEADROOM_MB = 2048  # leave the store filesystem some slack rather than filling it
MIN_DOWNLOAD_MB = 512  # floor for tiny / unsized artifacts so they never block on a bogus estimate

OllamaPull = Callable[[str], tuple[bool, str]]
HfCache = Callable[[str, str | None, Path | None], tuple[bool, str]]
PrepareProgress = Callable[[PreparedModel], None]
DiskFreeReader = Callable[[Path], int]  # store path -> free MB (0 == unknown, never blocks)
PresentCheck = Callable[[ModelSpec], bool]  # is this artifact already cached in its backend store?
