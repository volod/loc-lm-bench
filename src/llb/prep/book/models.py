"""Is the model a stage is configured with actually available here?

A stage model is written `<backend>:<name>`: `ollama:<tag>` is checked against the running Ollama
daemon's served tags, and `fake:<name>` is the CI stand-in that is always present. Any other backend
is refused by name, so a typo fails the dry run instead of a page twenty minutes in. Later tasks add
a backend here when they wire an engine that needs one.
"""

import os
from collections.abc import Callable

from llb.core import env
from llb.core.config_validation import DEFAULT_OLLAMA_HOST

OLLAMA_BACKEND = "ollama"
FAKE_BACKEND = "fake"
MODEL_BACKENDS = (OLLAMA_BACKEND, FAKE_BACKEND)

OllamaLister = Callable[[str], list[str]]


def _ollama_models(host: str) -> list[str]:
    from llb.backends.ollama import list_models

    return list_models(host)


def model_problem(model: str, ollama_lister: OllamaLister | None = None) -> str | None:
    """Why `model` cannot be used, or None when it is available."""
    backend, sep, name = model.partition(":")
    if not sep or not name:
        return f"model {model!r} must be written <backend>:<name> ({', '.join(MODEL_BACKENDS)})"
    if backend == FAKE_BACKEND:
        return None
    if backend == OLLAMA_BACKEND:
        host = os.environ.get(env.OLLAMA_HOST) or DEFAULT_OLLAMA_HOST
        served = (ollama_lister or _ollama_models)(host)
        if not served:
            return f"model {model!r}: no Ollama daemon answered at {host}, or it serves no model"
        if name not in served and f"{name}:latest" not in served:
            return f"model {model!r} is not served by Ollama at {host}"
        return None
    return f"model {model!r}: unsupported backend {backend!r}; supported: {MODEL_BACKENDS}"
