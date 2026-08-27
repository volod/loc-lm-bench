"""Reading a GENERATION id out of an upstream artifact name.

An upstream registry answers with artifact names, not generations: the Ollama library offers
`qwen3.8`, `qwen3.6`, `qwen2.5`; the Hugging Face model API offers `Qwen/Qwen3.8-27B-FP8` and
`Qwen/Qwen3.8-Flash-Next`. All of those are the SAME family, and what a currency check needs from
them is the one number the roster also records: which generation they carry.

The mapping is declared, never inferred from the family id. A family's `upstream` block names the
namespace each registry files it under, and the default pattern reads the version that follows that
namespace -- which is where Qwen, Gemma, and Mistral Small all put it. Families that put it
elsewhere (`MamayLM-Gemma-3-27B-IT-v2.0` carries a Gemma 3 ARCHITECTURE and a MamayLM v2.0
generation, in that order) declare an explicit `generation_pattern` instead, because a probe that
guessed here would report the architecture as the generation and call a current family behind.
"""

import re
from dataclasses import dataclass

# After the namespace: optional separators and an optional `v`, then the version itself.
_AFTER_NAMESPACE = r"[-_. ]*v?"
_VERSION = r"(\d+(?:\.\d+)*)"
# A PARAMETER COUNT sits in the same position as a generation -- `gemma-7b` and
# `Mistral-Small-24B-Instruct-2501` are Gemma 1 and Mistral Small 1, not generations 7 and 24. A
# number a size unit follows is therefore not a generation, and neither is a truncated one (the
# `3` of `qwen3.8b`), which is what the `.`/digit half of this lookahead refuses.
_NOT_A_SIZE = r"(?![bm.\d])"


@dataclass(frozen=True)
class UpstreamGeneration:
    """One generation an upstream registry offers, and the artifact name it was read from."""

    id: str
    key: tuple[int, ...]
    evidence: str

    def __str__(self) -> str:
        return self.id


def generation_key(text: str | None) -> tuple[int, ...] | None:
    """A generation id as a comparable tuple: `v2.0` -> (2, 0), `3.8` -> (3, 8), `4` -> (4,)."""
    if not text:
        return None
    found = re.search(_VERSION, str(text))
    if not found:
        return None
    return tuple(int(part) for part in found.group(1).split("."))


def default_pattern(namespace: str) -> str:
    """The version that directly follows a family's namespace -- the common upstream scheme."""
    return rf"^{re.escape(namespace)}{_AFTER_NAMESPACE}{_VERSION}{_NOT_A_SIZE}"


def compile_pattern(namespace: str | None, override: str | None = None) -> re.Pattern[str] | None:
    """The regex that reads a generation out of one artifact name, or None when nothing declares one.

    An override is a family-declared pattern with one capture group; it wins over the namespace
    default so a family whose upstream names carry the version somewhere other than the front is
    read correctly instead of approximately. A malformed one raises `ValueError`, so the family it
    was declared on degrades to a reported reason rather than taking the whole report down.
    """
    if override:
        try:
            compiled = re.compile(override, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid generation_pattern `{override}` -- {exc}") from None
        if compiled.groups != 1:
            raise ValueError(
                f"generation_pattern `{override}` has {compiled.groups} capture groups -- "
                "exactly one, around the version, is what is read"
            )
        return compiled
    if not namespace:
        return None
    return re.compile(default_pattern(namespace), re.IGNORECASE)


def read_generation(name: str, pattern: re.Pattern[str]) -> UpstreamGeneration | None:
    """The generation one upstream artifact name carries, or None when the name is not one."""
    found = pattern.search(name)
    if not found:
        return None
    raw = found.group(1) if found.groups() else found.group(0)
    key = generation_key(raw)
    if key is None:
        return None
    return UpstreamGeneration(id=raw, key=key, evidence=name)


def read_generations(
    names: tuple[str, ...], pattern: re.Pattern[str] | None
) -> tuple[UpstreamGeneration, ...]:
    """Every generation a registry's answer carries for one family, newest first."""
    if pattern is None:
        return ()
    found = [
        generation
        for generation in (read_generation(name, pattern) for name in names)
        if generation
    ]
    # Newest first, and within one generation the PLAINEST artifact name first: `qwen3.8` is the
    # evidence an operator wants to see cited, not `qwen3.8-flash-next`. The second sort is stable,
    # so the name order survives it.
    by_name = sorted(found, key=lambda gen: (len(gen.evidence), gen.evidence))
    return tuple(sorted(by_name, key=lambda gen: gen.key, reverse=True))


def newest(generations: tuple[UpstreamGeneration, ...]) -> UpstreamGeneration | None:
    """The newest generation in a set of readings, or None when the set is empty."""
    return max(generations, key=lambda gen: gen.key, default=None)
