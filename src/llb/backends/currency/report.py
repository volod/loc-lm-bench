"""Comparing the generation the roster carries against the newest one upstream offers.

The register says which generation of a family is `current` for US. This module asks the registries
those artifacts actually come from what is current for THEM, and reports the gap per family --
`current`, `behind`, or `unknown`. It reports only: promoting a generation is an operator decision
with a re-measurement cost attached, and nothing here edits the roster, pulls weights, or claims the
newer generation is better.

Every family produces a row, including a family that is already up to date. Silence is what the
roster already had; a probe whose output an operator has to interpret as "no news" is a probe that
cannot be distinguished from one that did not run.
"""

from dataclasses import dataclass
from typing import Callable

from llb.backends.currency.generations import (
    UpstreamGeneration,
    compile_pattern,
    generation_key,
    newest,
    read_generations,
)
from llb.backends.currency.registries import (
    HUGGINGFACE,
    OLLAMA,
    Fetcher,
    Response,
    hf_models_url,
    live_fetch,
    memoized,
    ollama_library_url,
    parse_hf_models,
    parse_ollama_library,
)
from llb.backends.roster import Family, Register

CURRENT = "current"
BEHIND = "behind"
UNKNOWN = "unknown"
VERDICTS = (CURRENT, BEHIND, UNKNOWN)

_NO_NAMESPACE = "the family declares no namespace for this registry"
_PASSTHROUGH = "namespace `{ns}` is a Hugging Face passthrough, not an Ollama library namespace"
_NO_MATCH = "answered, but offers no artifact under namespace `{ns}`"
_NO_CARRIED = "the roster records no current generation for this family"
_NO_ANSWER = "no registry answered for this family"


@dataclass(frozen=True)
class RegistryReading:
    """What one registry answered for one family, and when the answer arrived."""

    registry: str
    namespace: str
    url: str
    read_at: str
    generations: tuple[UpstreamGeneration, ...] = ()
    error: str | None = None

    @property
    def newest(self) -> UpstreamGeneration | None:
        return newest(self.generations)


@dataclass(frozen=True)
class FamilyCurrency:
    """One family's row: what the roster carries, what upstream offers, and the gap between them."""

    family_id: str
    label: str
    carried: str
    verdict: str
    readings: tuple[RegistryReading, ...]
    upstream: UpstreamGeneration | None = None
    registry: str | None = None
    reason: str | None = None


# --- reading one registry ----------------------------------------------------------------


def _reading(registry: str, namespace: str, url: str, read_at: str, error: str) -> RegistryReading:
    return RegistryReading(
        registry=registry, namespace=namespace, url=url, read_at=read_at, error=error
    )


def _read_names(
    registry: str,
    namespace: str,
    url: str,
    response: Response,
    parse: Callable[[str], tuple[str, ...]],
    *,
    pattern_namespace: str,
    override: str | None,
) -> RegistryReading:
    """One registry answer turned into generations, with every failure kept as a stated reason.

    Three things can go wrong after a response arrives -- an unparseable body, a malformed declared
    pattern, and a namespace nothing under it matched -- and all three become this family's reason
    rather than an exception, because one bad answer must not cost the report every other row.
    """
    if response.error or response.body is None:
        return _reading(registry, namespace, url, response.read_at, response.error or "empty body")
    try:
        generations = read_generations(
            parse(response.body), compile_pattern(pattern_namespace, override)
        )
    except ValueError as exc:
        return _reading(registry, namespace, url, response.read_at, str(exc))
    if not generations:
        return _reading(registry, namespace, url, response.read_at, _NO_MATCH.format(ns=namespace))
    return RegistryReading(registry, namespace, url, response.read_at, generations)


def read_ollama(family: Family, fetch: Fetcher) -> RegistryReading:
    """What the Ollama library offers under this family's namespace."""
    namespace = family.upstream.get("ollama_namespace", "")
    url = ollama_library_url()
    if not namespace:
        return _reading(OLLAMA, namespace, url, "", _NO_NAMESPACE)
    if "/" in namespace:
        # `hf.co/<author>` is a pull-through of Hugging Face, not a library namespace: the library
        # index does not list it, and the Hugging Face reading is the authoritative one for it.
        return _reading(OLLAMA, namespace, url, "", _PASSTHROUGH.format(ns=namespace))
    return _read_names(
        OLLAMA,
        namespace,
        url,
        fetch(url),
        parse_ollama_library,
        pattern_namespace=namespace,
        override=family.upstream.get("generation_pattern"),
    )


def read_huggingface(family: Family, fetch: Fetcher) -> RegistryReading:
    """What the Hugging Face model API offers under this family's author and repo prefix."""
    author = family.upstream.get("hf_author", "")
    prefix = family.upstream.get("hf_prefix", "")
    namespace = f"{author}/{prefix}" if prefix else author
    if not author:
        return _reading(HUGGINGFACE, namespace, "", "", _NO_NAMESPACE)
    url = hf_models_url(author, prefix)
    return _read_names(
        HUGGINGFACE,
        namespace,
        url,
        fetch(url),
        parse_hf_models,
        pattern_namespace=prefix,
        override=family.upstream.get("generation_pattern"),
    )


# --- the verdict -------------------------------------------------------------------------


def _verdict(family: Family, readings: tuple[RegistryReading, ...]) -> FamilyCurrency:
    carried_generation = family.current
    carried = carried_generation.id if carried_generation else ""

    def row(
        verdict: str,
        upstream: UpstreamGeneration | None = None,
        registry: str | None = None,
        reason: str | None = None,
    ) -> FamilyCurrency:
        return FamilyCurrency(
            family_id=family.id,
            label=family.label,
            carried=carried,
            verdict=verdict,
            readings=readings,
            upstream=upstream,
            registry=registry,
            reason=reason,
        )

    answered = [(reading, offered) for reading in readings if (offered := reading.newest)]
    if not answered:
        reasons = "; ".join(
            f"{reading.registry}: {reading.error}" for reading in readings if reading.error
        )
        return row(UNKNOWN, reason=reasons or _NO_ANSWER)

    best, upstream = max(answered, key=lambda pair: pair[1].key)
    carried_key = generation_key(carried)
    if carried_key is None:
        return row(UNKNOWN, upstream, best.registry, _NO_CARRIED)
    return row(BEHIND if upstream.key > carried_key else CURRENT, upstream, best.registry)


def probe_family(family: Family, fetch: Fetcher) -> FamilyCurrency:
    """Read both registries for one family and compare what they offer with what the roster carries."""
    readings = (read_ollama(family, fetch), read_huggingface(family, fetch))
    return _verdict(family, readings)


def probe_register(register: Register, fetch: Fetcher | None = None) -> tuple[FamilyCurrency, ...]:
    """Every registered family's currency row, in register order."""
    reader = memoized(fetch or live_fetch)
    return tuple(probe_family(family, reader) for family in register.families)
