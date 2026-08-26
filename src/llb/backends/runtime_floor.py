"""Runtime version floors: an artifact the installed runtime is too OLD to serve.

Availability is not only "does the source exist". An artifact whose architecture a runtime learned
in a LATER release is present, pullable, and unservable: Ollama 0.20 answers the Gemma 4 12B GGUF
with `unknown model architecture: 'gemma4'`, and that reaches a run as a generic backend error --
indistinguishable from a broken daemon, a wrong tag, or an OOM. This module makes it a NAMED skip
carrying the four facts an operator can act on: the runtime, the version it runs, the architecture
it does not implement, and the version that does.

Two signals feed one comparison, so the check works before AND after a pull:

  * the ARTIFACT's own requirement -- Ollama's `/api/show` reports `requires` and the GGUF's
    `general.architecture` for a tag the daemon already holds, so the runtime is the authority
    whenever it knows the artifact;
  * the MANIFEST pin -- `min_runtime_version` / `runtime_arch` on a source record. It is the only
    signal for a tag that is not pulled yet (exactly the host-setup case) and for a raw GGUF, which
    carries no `requires` field at all.

The higher of the two wins, so a pin can never lower what the artifact itself demands. Every probe
is injectable and every comparison degrades to "no skip" when a version does not parse: a floor
check that guesses would ground a runnable model.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from llb.core.config_validation import DEFAULT_OLLAMA_HOST

# The named skip category a resolver candidate carries when the runtime is below an artifact's floor.
RUNTIME_FLOOR_SKIP = "runtime-version-floor"

# What a runtime says when it does not implement an artifact's architecture. Ollama phrases it
# `unknown model architecture: 'gemma4'`; vLLM's transformers path phrases the same fact with the
# architecture in brackets.
_ARCH_ERROR_PATTERNS = (
    re.compile(r"unknown model architecture:?\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(
        r"architectures?\s*\[?['\"]([^'\"]+)['\"]\]?\s*(?:are|is) not supported", re.IGNORECASE
    ),
)

# Injectable seams: backend -> running runtime version, and (backend, source) -> what the artifact
# itself declares. Both answer None for "nothing known", which never produces a skip.
VersionReader = Callable[[str], str | None]
RequirementReader = Callable[[str, str], "RuntimeRequirement | None"]

_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")
_PROBE_TIMEOUT_S = 3.0


def unsupported_architecture(message: str | None) -> str | None:
    """The architecture name a runtime error says it does not implement, or None."""
    if not message:
        return None
    for pattern in _ARCH_ERROR_PATTERNS:
        found = pattern.search(message)
        if found:
            return found.group(1)
    return None


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """The numeric part of a version string (`0.32.15`, `v0.30.5`, `ollama version is 0.20.6`)."""
    if not text:
        return None
    found = _VERSION_RE.search(str(text))
    if not found:
        return None
    return tuple(int(part) for part in found.group(1).split("."))


def _below(running: str | None, floor: str | None) -> bool:
    """True only when BOTH versions parse and the running one is older than the floor."""
    have, need = parse_version(running), parse_version(floor)
    if have is None or need is None:
        return False
    return have < need


@dataclass(frozen=True)
class RuntimeRequirement:
    """What one artifact needs from the runtime that would serve it."""

    backend: str
    source: str
    arch: str | None = None
    min_version: str | None = None

    def merge(self, other: "RuntimeRequirement | None") -> "RuntimeRequirement":
        """Combine a manifest pin with the artifact's own requirement; the HIGHER floor wins."""
        if other is None:
            return self
        floor = self.min_version
        if floor is None or _below(floor, other.min_version):
            floor = other.min_version
        return replace(self, arch=other.arch or self.arch, min_version=floor)


def declared_requirement(
    backend: str, source: str, record: Mapping[str, object]
) -> RuntimeRequirement:
    """The floor a manifest entry pins for one source record (fields may be absent)."""
    arch = record.get("runtime_arch")
    floor = record.get("min_runtime_version")
    return RuntimeRequirement(
        backend=backend,
        source=source,
        arch=str(arch) if arch else None,
        min_version=str(floor) if floor else None,
    )


def floor_reason(requirement: RuntimeRequirement, running_version: str | None) -> str | None:
    """The named skip message for a runtime below this artifact's floor, or None when it is not."""
    if not _below(running_version, requirement.min_version):
        return None
    backend = requirement.backend
    what = f"architecture '{requirement.arch}'" if requirement.arch else "this artifact"
    return (
        f"{backend} {running_version} does not implement {what} -- {requirement.source} needs "
        f"{backend} >= {requirement.min_version} (upgrade the host runtime)"
    )


def source_floor_reason(
    backend: str,
    source: str,
    record: Mapping[str, object],
    *,
    version_reader: "VersionReader",
    requirement_reader: "RequirementReader",
) -> str | None:
    """The named skip for one candidate source, from the manifest pin AND the artifact itself.

    The running version is read FIRST: a runtime nobody can identify cannot be judged too old, so
    an unreachable daemon costs no artifact probe and grounds no model.
    """
    running = version_reader(backend)
    if running is None:
        return None
    requirement = declared_requirement(backend, source, record).merge(
        requirement_reader(backend, source)
    )
    return floor_reason(requirement, running)


def architecture_error(backend: str, source: str, arch: str, hint: str = "") -> str:
    """The named message for a runtime that REPORTED an unknown architecture while serving."""
    tail = f" ({hint})" if hint else ""
    return (
        f"{backend} does not implement model architecture '{arch}' for {source} -- upgrade "
        f"{backend} to the version the artifact requires{tail}"
    )


# --- live probes (best-effort; any error -> "nothing known", never raises) ----------------


def _get_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object] | None:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def ollama_runtime_version(host: str = DEFAULT_OLLAMA_HOST) -> str | None:
    """The running daemon's version from `/api/version`, or None when it cannot be read."""
    body = _get_json(f"{host.rstrip('/')}/api/version")
    version = body.get("version") if body else None
    return str(version) if version else None


def ollama_artifact_requirement(
    source: str, host: str = DEFAULT_OLLAMA_HOST
) -> RuntimeRequirement | None:
    """What a locally held Ollama tag declares it needs (`/api/show`), or None for an unheld tag."""
    body = _get_json(f"{host.rstrip('/')}/api/show", {"model": source})
    if body is None:
        return None
    info = body.get("model_info")
    details = body.get("details")
    arch = (info or {}).get("general.architecture") if isinstance(info, dict) else None
    if not arch and isinstance(details, dict):
        arch = details.get("family")
    requires = body.get("requires")
    if not arch and not requires:
        return None
    return RuntimeRequirement(
        backend="ollama",
        source=source,
        arch=str(arch) if arch else None,
        min_version=str(requires) if requires else None,
    )


def runtime_version(backend: str, *, ollama_host: str = DEFAULT_OLLAMA_HOST) -> str | None:
    """The version of the runtime that would serve this backend, or None when it is unknown."""
    if backend == "ollama":
        return ollama_runtime_version(ollama_host)
    if backend == "vllm":
        try:
            from importlib.metadata import version

            return str(version("vllm"))
        except Exception:
            return None
    return None


def artifact_requirement(
    backend: str, source: str, *, ollama_host: str = DEFAULT_OLLAMA_HOST
) -> RuntimeRequirement | None:
    """The artifact's OWN declared requirement, for backends whose runtime reports one."""
    if backend == "ollama":
        return ollama_artifact_requirement(source, ollama_host)
    return None
