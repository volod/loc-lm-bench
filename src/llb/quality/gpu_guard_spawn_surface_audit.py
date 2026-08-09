"""The refusal: an interpreter whose spawn surface the declarations no longer describe.

`llb.quality.gpu_guard_spawn_surface` enumerates the process-starting names this interpreter exposes
and declares how the denial reaches each one. This module compares the two and refuses six shapes of
disagreement -- five about names, one about `multiprocessing`:

- a name the enumeration finds and no declaration covers (the shape a Python upgrade arrives as);
- a declared delegation the interpreter no longer makes, because the callable was rewritten in C or
  now reaches something else;
- a delegation chain that ends outside the seam set, so the name is declared covered by a name that
  is not itself covered;
- a name declared a seam that `spawn_seams()` does not patch, and a seam it patches that no
  declaration names -- the two halves of the set drifting apart;
- a `multiprocessing` start method the interpreter offers and the declarations do not name, or a
  DEFAULT start method that is a declared residual, which is exactly what Python 3.14 does to the
  `fork` default this denial was written on.

A residual stays a residual: nothing here closes one, and a declared residual is silent until the
interpreter makes it the default. The audit is a plain function over an `ObservedSurface`, so a test
drives it against a fabricated interpreter as easily as against this one.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from llb.quality.gpu_guard_spawn_surface import (
    COVERAGE_RESIDUAL,
    COVERAGE_SEAM,
    COVERAGE_THROUGH,
    DECLARED_SPAWN_SURFACE,
    DECLARED_START_METHODS,
    ObservedSurface,
    SpawnCoverage,
)

PROBLEM_UNDECLARED = "undeclared"
PROBLEM_UNPINNED = "unpinned-delegation"
PROBLEM_UNREACHED = "unreached-delegation"
PROBLEM_SEAM_UNPATCHED = "declared-seam-not-patched"
PROBLEM_SEAM_UNDECLARED = "patched-seam-not-declared"
PROBLEM_START_METHOD = "start-method"

_REMEDY = (
    "register a seam for it in gpu_guard_spawn.spawn_seams, or declare it in "
    "gpu_guard_spawn_surface (the entry point it is covered through, or a residual with its "
    "reason) -- and say so in the module docstring's residual list"
)


@dataclass(frozen=True)
class SurfaceFinding:
    """One disagreement between the declared spawn surface and the interpreter running it."""

    name: str
    problem: str
    detail: str

    def __str__(self) -> str:
        return f"{self.name} [{self.problem}]: {self.detail}"


def audit_spawn_surface(
    surface: ObservedSurface | None = None,
    declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE,
    start_methods: Mapping[str, SpawnCoverage] = DECLARED_START_METHODS,
) -> tuple[SurfaceFinding, ...]:
    """Every way this interpreter's spawn surface has moved out from under the declarations."""
    observed = surface if surface is not None else ObservedSurface.read()
    return (
        *_name_findings(observed, declared),
        *_seam_findings(observed, declared),
        *_start_method_findings(observed, declared, start_methods),
    )


def surface_message(findings: Iterable[SurfaceFinding]) -> str:
    """The operator-facing block: what moved, and what closing it means."""
    listed = "\n".join(f"  - {finding}" for finding in findings)
    return f"[gpu-guard] the declared spawn surface no longer describes this interpreter:\n{listed}\n{_REMEDY}"


def absent_declarations(
    surface: ObservedSurface, declared: Mapping[str, SpawnCoverage] = DECLARED_SPAWN_SURFACE
) -> tuple[str, ...]:
    """Declared names this interpreter does not expose -- reported, never refused.

    `fork`, `forkpty`, `posix_spawnp`, and the `*e` exec siblings are POSIX-only and `startfile` is
    Windows-only, so a declaration naming nothing here is ordinarily a host difference rather than a
    stale claim. It is worth SEEING for the same reason the seam builder tolerates it.
    """
    return tuple(name for name in declared if name not in surface.names)


def _name_findings(
    observed: ObservedSurface, declared: Mapping[str, SpawnCoverage]
) -> tuple[SurfaceFinding, ...]:
    findings = (_name_finding(name, observed, declared) for name in observed.names)
    return tuple(finding for finding in findings if finding is not None)


def _name_finding(
    name: str, observed: ObservedSurface, declared: Mapping[str, SpawnCoverage]
) -> SurfaceFinding | None:
    """What is wrong with how one enumerated name is declared, or None when nothing is."""
    coverage = declared.get(name)
    if coverage is None:
        return SurfaceFinding(
            name, PROBLEM_UNDECLARED, "this interpreter exposes it and no declaration covers it"
        )
    if coverage.kind == COVERAGE_SEAM and name not in observed.seams:
        return SurfaceFinding(
            name, PROBLEM_SEAM_UNPATCHED, "declared a seam, but spawn_seams() does not patch it"
        )
    if coverage.kind != COVERAGE_THROUGH or coverage.through is None:
        return None
    return _delegation_finding(name, coverage.through, observed, declared)


def _delegation_finding(
    name: str, target: str, observed: ObservedSurface, declared: Mapping[str, SpawnCoverage]
) -> SurfaceFinding | None:
    if not observed.delegates(name, target):
        return SurfaceFinding(
            name,
            PROBLEM_UNPINNED,
            f"declared covered through {target}, which its implementation no longer resolves "
            f"(rewritten in C, or reaching somewhere else now)",
        )
    if not reaches_a_seam(target, observed, declared):
        return SurfaceFinding(
            name,
            PROBLEM_UNREACHED,
            f"declared covered through {target}, which is not itself covered by a patched seam",
        )
    return None


def _seam_findings(
    observed: ObservedSurface, declared: Mapping[str, SpawnCoverage]
) -> tuple[SurfaceFinding, ...]:
    """A seam the denial installs and the declarations do not name as one."""
    return tuple(
        SurfaceFinding(
            label,
            PROBLEM_SEAM_UNDECLARED,
            "spawn_seams() patches it and no declaration names it a seam",
        )
        for label in observed.seams
        if getattr(declared.get(label), "kind", None) != COVERAGE_SEAM
    )


def _start_method_findings(
    observed: ObservedSurface,
    declared: Mapping[str, SpawnCoverage],
    start_methods: Mapping[str, SpawnCoverage],
) -> tuple[SurfaceFinding, ...]:
    findings = [
        _start_method_finding(method, start_methods.get(method), observed, declared)
        for method in observed.start_methods
    ]
    default = start_methods.get(observed.default_start_method)
    if default is not None and default.kind != COVERAGE_THROUGH:
        findings.append(
            SurfaceFinding(
                f"multiprocessing({observed.default_start_method})",
                PROBLEM_START_METHOD,
                "the interpreter's DEFAULT start method is a declared residual, so every "
                f"multiprocessing child of an unmarked test keeps the device: {default.reason}",
            )
        )
    return tuple(finding for finding in findings if finding is not None)


def _start_method_finding(
    method: str,
    coverage: SpawnCoverage | None,
    observed: ObservedSurface,
    declared: Mapping[str, SpawnCoverage],
) -> SurfaceFinding | None:
    label = f"multiprocessing({method})"
    if coverage is None:
        return SurfaceFinding(
            label, PROBLEM_UNDECLARED, "this interpreter offers it and no declaration covers it"
        )
    if coverage.kind == COVERAGE_RESIDUAL:
        return None
    if coverage.through is None or not reaches_a_seam(coverage.through, observed, declared):
        return SurfaceFinding(
            label,
            PROBLEM_UNREACHED,
            f"declared covered through {coverage.through}, which no patched seam covers",
        )
    return None


def reaches_a_seam(
    target: str, observed: ObservedSurface, declared: Mapping[str, SpawnCoverage]
) -> bool:
    """Walk a delegation chain to the seam that carries it, refusing a cycle or a dead end."""
    seen: set[str] = set()
    while target not in seen:
        if target in observed.seams:
            return True
        seen.add(target)
        coverage = declared.get(target)
        if coverage is None or coverage.kind != COVERAGE_THROUGH or coverage.through is None:
            return False
        target = coverage.through
    return False
