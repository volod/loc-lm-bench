"""The candidate roster read as a register of FAMILIES and GENERATIONS, not a list of tags.

A model name says nothing about whether the roster still carries the current release of that family:
`qwen3.6-27b` and `qwen3.8-27b` are two generations of one lane, and a family result is only
readable as a generation comparison when something states which of them is current. The manifest's
`families:` block carries that statement and every model declares the generation it belongs to; this
module joins the two and reports every way the join can be wrong.

Serving decisions are NOT here -- which artifact runs on which host stays with the resolver and the
planner. This module owns identity only: which family, which generation, and whether the register
and the models agree about it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.backends.prepare.manifest import load_families, load_manifest
from llb.core.contracts.models import FamilySpec, ModelSpec

CURRENT = "current"
PREVIOUS = "previous"
STATUSES = (CURRENT, PREVIOUS)

UA_SPECIALIZED = "ua-specialized"
MULTILINGUAL_BASELINE = "multilingual-baseline"
ROLES = (UA_SPECIALIZED, MULTILINGUAL_BASELINE)

ROLE_LABELS = {
    UA_SPECIALIZED: "Ukrainian-specialized",
    MULTILINGUAL_BASELINE: "Multilingual baseline",
}


@dataclass(frozen=True)
class Generation:
    """One generation of a family, with the logical models the roster carries for it."""

    family_id: str
    id: str
    status: str
    label: str
    license: str
    license_url: str
    weights_url: str
    models: tuple[ModelSpec, ...]

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(str(model["name"]) for model in self.models)


@dataclass(frozen=True)
class Family:
    """One family in the roster: what it is in the sweep for, and the generations it carries."""

    id: str
    label: str
    role: str
    focus: str
    upstream: dict[str, str]
    generations: tuple[Generation, ...]

    @property
    def current(self) -> Generation | None:
        return next((gen for gen in self.generations if gen.status == CURRENT), None)

    @property
    def previous(self) -> tuple[Generation, ...]:
        return tuple(gen for gen in self.generations if gen.status == PREVIOUS)


@dataclass(frozen=True)
class Register:
    """The roster as families and generations, beside the models the manifest actually lists."""

    families: tuple[Family, ...]
    models: tuple[ModelSpec, ...]

    @property
    def placed(self) -> tuple[str, ...]:
        """Names of models a generation carries -- everything else fell through the register."""
        return tuple(
            name
            for family in self.families
            for generation in family.generations
            for name in generation.model_names
        )

    @property
    def unplaced(self) -> tuple[ModelSpec, ...]:
        placed = set(self.placed)
        return tuple(model for model in self.models if str(model["name"]) not in placed)

    def family(self, family_id: str) -> Family | None:
        return next((family for family in self.families if family.id == family_id), None)


def _generation(family: FamilySpec, raw: dict[str, Any], models: list[ModelSpec]) -> Generation:
    generation_id = str(raw["id"])
    return Generation(
        family_id=str(family["id"]),
        id=generation_id,
        status=str(raw["status"]),
        label=str(raw.get("label") or f"{family['label']} {generation_id}"),
        license=str(raw.get("license", "")),
        license_url=str(raw.get("license_url", "")),
        weights_url=str(raw.get("weights_url", "")),
        models=tuple(
            model
            for model in models
            if model.get("family") == family["id"] and str(model.get("generation")) == generation_id
        ),
    )


def build_register(families: list[FamilySpec], models: list[ModelSpec]) -> Register:
    """Join the family register with the models, keeping manifest order in both."""
    return Register(
        families=tuple(
            Family(
                id=str(family["id"]),
                label=str(family["label"]),
                role=str(family["role"]),
                focus=str(family.get("focus", "")),
                upstream={str(k): str(v) for k, v in dict(family.get("upstream") or {}).items()},
                generations=tuple(
                    _generation(family, dict(raw), models) for raw in family["generations"]
                ),
            )
            for family in families
        ),
        models=tuple(models),
    )


def load_register(manifest: Path | str) -> Register:
    """Read one candidate-model manifest as a family register."""
    return build_register(load_families(manifest), load_manifest(manifest))


def register_findings(register: Register) -> list[str]:
    """Every disagreement between the family register and the models it claims to describe."""
    known = {family.id: family for family in register.families}
    findings: list[str] = []
    for model in register.unplaced:
        family_id, generation_id = model.get("family"), model.get("generation")
        where = f"model `{model['name']}`"
        if not family_id:
            findings.append(f"{where}: declares no family -- every model belongs to one")
        elif family_id not in known:
            findings.append(f"{where}: family `{family_id}` is not in the family register")
        else:
            findings.append(
                f"{where}: generation `{generation_id}` is not declared by family `{family_id}`"
            )
    for family in register.families:
        findings.extend(_family_findings(family))
    return findings


def _family_findings(family: Family) -> list[str]:
    where = f"family `{family.id}`"
    findings: list[str] = []
    if family.role not in ROLES:
        findings.append(f"{where}: role '{family.role}' is not one of {ROLES}")
    current = [gen for gen in family.generations if gen.status == CURRENT]
    if len(current) != 1:
        findings.append(
            f"{where}: {len(current)} generation(s) marked `{CURRENT}` -- exactly one is current"
        )
    for generation in family.generations:
        findings.extend(_generation_findings(family, generation))
    return findings


def _generation_findings(family: Family, generation: Generation) -> list[str]:
    where = f"family `{family.id}` generation `{generation.id}`"
    findings: list[str] = []
    if generation.status not in STATUSES:
        findings.append(f"{where}: status '{generation.status}' is not one of {STATUSES}")
    if not generation.models:
        findings.append(f"{where}: no model carries it -- retire the generation or carry one")
    if not generation.license or not generation.license_url:
        findings.append(f"{where}: no license recorded -- weights always travel with terms")
    for model in generation.models:
        declared = model.get("license")
        if declared and declared != generation.license:
            findings.append(
                f"model `{model['name']}`: license `{declared}` disagrees with {where} "
                f"(`{generation.license}`)"
            )
    return findings
