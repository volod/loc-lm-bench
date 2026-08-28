"""Resolving a RECORDED model identity back to the family generation it was measured on.

Evidence records a model the way the run reached it: an aggregate holds the Ollama tag
(`mistral-small3.1:24b`), a doc table holds the logical roster name (`mistral-small-3.1-24b`) beside
the served artifact, and a vLLM lane holds the Hugging Face repo id. All three name ONE roster
entry, and what a generation swap needs from them is the one field none of them carries: which
generation that entry belongs to.

The register is the only thing that can answer, so the mapping is built out of it rather than
pattern-matched off the string. Every identity a model can be recorded under -- its logical name and
every per-backend source the resolver would serve it from -- becomes a key pointing at that model's
family and generation. A string no key matches is UNRESOLVED, never quietly dropped: an evidence
record naming a model the register cannot place is exactly the case where a swap's cost is
undercounted, and reporting it is the difference between "nothing else is affected" and "nothing
else was recognized".
"""

from dataclasses import dataclass

from llb.backends.resolver_sources import candidate_sources
from llb.backends.roster import Register
from llb.core.contracts.models import ModelSpec


@dataclass(frozen=True)
class ResolvedModel:
    """One recorded identity placed in the register: which entry it names, and where that sits."""

    recorded: str
    model_name: str
    family_id: str
    generation_id: str

    def named(self) -> str:
        """One line naming the entry a recorded identity resolved to."""
        return f"{self.model_name} ({self.family_id} {self.generation_id})"


def _identities(model: ModelSpec) -> set[str]:
    """Every string one roster entry can be RECORDED under: its name and each served source.

    `candidate_sources` is reused rather than reading `sources` here, so the identities this index
    accepts are exactly the artifacts the resolver would serve -- a source the resolver folds in
    (the declared backend's spec-level `source`) is one a run can be measured on, and an index that
    missed it would report that run as unresolved.
    """
    found = {str(model["name"])}
    for _backend, record in candidate_sources(model):
        source = record.get("source")
        if source:
            found.add(str(source))
    return found


class ModelIndex:
    """Every recordable model identity in one register, mapped to the entry it names.

    Case-folded, because the same artifact is written `Qwen/Qwen3.8-27B-FP8` in a manifest and
    `qwen/qwen3.8-27b-fp8` in a hand-typed table, and a lookup that missed on case would report a
    measured row as unresolved. Nothing else is normalized: a quant suffix or a namespace prefix
    changes WHICH artifact is named, so trimming one would resolve a row to a model it never ran.
    """

    def __init__(self, register: Register) -> None:
        self._by_identity: dict[str, ResolvedModel] = {}
        for family in register.families:
            for generation in family.generations:
                for model in generation.models:
                    for identity in _identities(model):
                        self._by_identity.setdefault(
                            identity.casefold(),
                            ResolvedModel(
                                recorded=identity,
                                model_name=str(model["name"]),
                                family_id=family.id,
                                generation_id=generation.id,
                            ),
                        )

    def __len__(self) -> int:
        return len(self._by_identity)

    def resolve(self, recorded: str) -> ResolvedModel | None:
        """The register entry a recorded identity names, or None when nothing carries it."""
        found = self._by_identity.get(str(recorded).strip().casefold())
        if found is None:
            return None
        # The identity as the EVIDENCE spelled it, not as the manifest spells it: a report that
        # rewrote the string is one an operator cannot grep the evidence for.
        return ResolvedModel(
            recorded=str(recorded).strip(),
            model_name=found.model_name,
            family_id=found.family_id,
            generation_id=found.generation_id,
        )

    def known(self) -> tuple[str, ...]:
        """Every identity the index accepts, sorted -- what a report means by "recognized"."""
        return tuple(sorted(self._by_identity))
