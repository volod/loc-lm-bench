"""The resolver's availability probes: does a source exist, and can this runtime serve it?

Split out of `resolver` so the decision logic there stays pure: everything that touches the
network, the Hugging Face Hub, or the local Ollama daemon lives here behind injectable callables.
Two questions are answered per candidate source -- whether the artifact EXISTS (HF repo, GGUF file,
Ollama tag) and whether the installed runtime is new enough to serve its architecture
(`runtime_floor`), which is the difference between a missing model and an old host.
"""

import json
import urllib.error
import urllib.request
from typing import Callable

from llb.backends.runtime_floor import RuntimeRequirement, artifact_requirement, runtime_version
from llb.core.config_validation import DEFAULT_OLLAMA_HOST

# Probes: source -> availability signal. Defaults hit HF Hub / Ollama; all injectable.
HfRepoProbe = Callable[[str], bool]  # repo id -> exists
GgufProbe = Callable[[str], bool]  # repo id -> has at least one *.gguf file
OllamaProbe = Callable[[str], bool]  # tag -> pulled locally or in the Ollama library
VersionProbe = Callable[[str], str | None]  # backend -> the version of the runtime that serves it
RequirementProbe = Callable[[str, str], RuntimeRequirement | None]  # (backend, source) -> its floor


def _probe_available(backend: str, source: str, probes: "ResolverProbes") -> bool:
    if backend == "vllm":
        return probes.hf_repo(source)
    if backend == "ollama":
        return probes.ollama_tag(source)
    if backend == "llamacpp":
        return probes.gguf(source)
    return False


class ResolverProbes:
    """The availability probes, defaulting to live HF Hub / Ollama checks.

    Beyond "does the source exist", two of them answer "can the installed runtime serve it at
    all": the running runtime version and what the artifact itself declares it needs. The version
    read is memoized per backend -- one `/api/version` for a whole roster resolution, not one per
    candidate.
    """

    def __init__(
        self,
        hf_repo: HfRepoProbe | None = None,
        gguf: GgufProbe | None = None,
        ollama_tag: OllamaProbe | None = None,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        runtime_version: VersionProbe | None = None,
        artifact_requirement: RequirementProbe | None = None,
    ):
        self.hf_repo = hf_repo or _hf_repo_exists
        self.gguf = gguf or _hf_has_gguf
        self.ollama_tag = ollama_tag or _make_ollama_probe(ollama_host)
        self.runtime_version = _memoized(runtime_version or _make_version_probe(ollama_host))
        self.artifact_requirement = artifact_requirement or _make_requirement_probe(ollama_host)


def _memoized(probe: VersionProbe) -> VersionProbe:
    """One version read per backend per resolution -- a roster resolves many candidates."""
    cache: dict[str, str | None] = {}

    def cached(backend: str) -> str | None:
        if backend not in cache:
            cache[backend] = probe(backend)
        return cache[backend]

    return cached


def _make_version_probe(host: str) -> VersionProbe:
    def probe(backend: str) -> str | None:
        return runtime_version(backend, ollama_host=host)

    return probe


def _make_requirement_probe(host: str) -> RequirementProbe:
    def probe(backend: str, source: str) -> RuntimeRequirement | None:
        return artifact_requirement(backend, source, ollama_host=host)

    return probe


# --- live probes (best-effort; any error -> "not available", never raises) ----------------


def _hf_repo_exists(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi

        return bool(HfApi().repo_exists(repo_id))
    except Exception:
        return False


def _hf_has_gguf(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi

        normalized = repo_id
        for prefix in ("https://huggingface.co/", "huggingface.co/", "hf.co/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        normalized = normalized.split(":", 1)[0]
        files = HfApi().list_repo_files(normalized)
        return any(f.lower().endswith(".gguf") for f in files)
    except Exception:
        return False


def _make_ollama_probe(host: str) -> OllamaProbe:
    def probe(tag: str) -> bool:
        try:
            url = f"{host.rstrip('/')}/api/tags"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get("name", "") for m in body.get("models", [])}
        # Match `llama3.2:3b` and a bare `llama3.2` (Ollama defaults to :latest).
        return tag in names or any(n.split(":", 1)[0] == tag.split(":", 1)[0] for n in names)

    return probe
