"""Focused ollama eviction implementation."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, cast
from llb.core.config_validation import DEFAULT_OLLAMA_HOST

_LOG = logging.getLogger(__name__)


def evict_ollama(
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    http_get: Callable[[str], dict[str, Any] | None] | None = None,
    http_post: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Ask Ollama to unload every resident model (`keep_alive: 0`). Best-effort; never raises."""
    get = http_get or _http_get_json
    post = http_post or _http_post_json
    base = host.rstrip("/")
    try:
        running = get(f"{base}/api/ps")
    except Exception:
        return
    for entry in (running or {}).get("models", []):
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        try:
            post(f"{base}/api/generate", {"model": name, "keep_alive": 0})
            _LOG.info("[contention] requested Ollama unload of %s (keep_alive=0)", name)
        except Exception:
            continue


def _http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return cast(dict[str, Any], json.loads(resp.read().decode("utf-8", "replace")))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 10.0) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout):
        return
