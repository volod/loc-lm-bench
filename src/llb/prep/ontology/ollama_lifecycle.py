"""Strict Ollama model unloading for sequential local comparison runs."""

import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

OLLAMA_UNLOAD_TIMEOUT_S = 30.0
OLLAMA_UNLOAD_POLL_S = 0.25


def ollama_native_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return f"{parsed.scheme or 'http'}://{parsed.netloc or parsed.path}".rstrip("/")


def resident_models(base_url: str) -> list[str]:
    response = httpx.get(f"{ollama_native_root(base_url)}/api/ps", timeout=5.0)
    response.raise_for_status()
    return [
        str(entry.get("name") or entry.get("model"))
        for entry in response.json().get("models", [])
        if entry.get("name") or entry.get("model")
    ]


def unload_models(
    base_url: str,
    models: list[str] | None = None,
    *,
    timeout_s: float = OLLAMA_UNLOAD_TIMEOUT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Unload selected resident models and block until none remain resident."""
    root = ollama_native_root(base_url)
    targets = models if models is not None else resident_models(base_url)
    for model in targets:
        response = httpx.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10.0,
        )
        response.raise_for_status()
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = [name for name in resident_models(base_url) if name in targets]
        if not remaining:
            return targets
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ollama models did not unload within {timeout_s:.0f}s: {remaining}")
        sleep(OLLAMA_UNLOAD_POLL_S)
