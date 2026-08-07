"""Every committed design that publishes resolvable values: what it cites, and what checks it.

The committed evidence tree is shared: one verbatim copy per CITED run aggregate, pruned down to
what the published values still point at. A refresh that knows about one design therefore prunes to
that design's citations, which is correct exactly while one study uses the resolver and is evidence
loss the moment a second one adopts it -- refreshing through either design would delete the other
study's committed aggregates, and the deletion reads as a clean prune rather than as a retirement.

So the cited set is the UNION over every design that publishes resolvable values, and the designs
are a registry the refresh walks -- the same shape as `AUDITED_DESIGN_PATHS` in
`agentic_policy_change_audit`, for the same reason: two code paths that disagree about which
evidence exists are two answers to one question. A design not in this registry is invisible to the
refresh, so adopting the resolver in a new study means registering it here in the same commit.

Two refusals guard the union rather than trusting the walk:

  - a registered design whose citations this host cannot serve (its run is not under DATA_DIR) stops
    the refresh, because a partial refresh retires the evidence of the studies this host never ran;
  - the write seam itself refuses any set that omits an artifact the registry says is cited, so a
    caller holding one design's citations cannot prune the rest by construction.

An entry also carries what CHECKS its published values, not only what they cite, because those are
two different guarantees and the weaker one is the more convincing. Citations alone make a study's
evidence durable -- its aggregates are committed, pinned, and never pruned by someone else's refresh
-- while saying nothing about whether the numbers the design publishes are the ones those aggregates
hold. A design registered for the first and validated by nothing would ship committed bytes beside
unchecked claims, which reads exactly like the checked case. So registration buys both: the walk
here validates every registered design, and an entry that registers no validation is refused rather
than skipped, since a registry that silently walks past an entry is how the gap would reopen.

The two walks are ONE walk, in both directions. Regenerating the evidence and asking whether the
published values still resolve out of it are the same question asked by the same registry, so a
refresh that answered only the first left the second to a later `make ci` -- an operator who re-ran a
study, committed the new aggregates, and saw a clean manifest path learned on some other day that
the design's numbers are now the old run's, at the point where the fix is a design edit they no
longer have the run context for. `report_published_designs` is therefore the primitive: it collects
what did not resolve instead of raising, `validate_published_designs` is the refusing form over it,
and `refresh_committed_evidence_and_report` runs it across what the refresh just wrote.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llb.bench.agentic_published_value_fixture import write_provenance_fixture

# The registered kinds, named after the design that publishes the values rather than after the
# studies it restates -- one design can publish values measured by several studies.
KIND_CROSSOVER_RESTATEMENT = "compact_crossover_restatement"


class ValidatePublishedValues(Protocol):
    """Resolve one design's published values, refusing any the committed evidence does not state.

    `root` is the repo root the committed aggregates live under, and `data_dir` is the host's run
    root or None -- the validation is authoritative against the committed copies, because CI has no
    runs, and reads the artifacts themselves only as a check on those copies.
    """

    def __call__(self, design_path: Path, *, root: Path, data_dir: Path | None) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishedValueDesign:
    """One registered design: where it lives, what its values cite, and what resolves them.

    Both callables are per design because the values are shaped per design -- the restatement states
    published crossovers per audited study, and the next adopter will state something else. What
    every entry owes the registry is the same two things: the DATA_DIR-relative run artifacts its
    published values resolve against, so the refresh can take the union without knowing the shape,
    and the validation that reads those values back out, so registering a design is what subjects it
    to the check rather than remembering to write a test named after it.
    """

    design_path: str
    cited_artifacts: Callable[[Path], list[str]]
    validate_published_values: ValidatePublishedValues

    def __post_init__(self) -> None:
        # Stated rather than left to the type checker: `None` is the shape an opt-out would take,
        # and an entry that opts out is the one thing this field exists to make impossible.
        if not callable(self.validate_published_values):
            raise ValueError(
                f"the registered design {self.design_path} states no validation of its published "
                "values, so registering it would commit its evidence without ever resolving the "
                "numbers that evidence stands for -- durable bytes beside unchecked claims"
            )


def _crossover_restatement_citations(path: Path) -> list[str]:
    """The run artifacts every published crossover of the restatement design resolves against."""
    from llb.bench.agentic_memory_crossover_restatement_design import (
        load_restatement_design,
        published_crossovers,
    )
    from llb.bench.agentic_published_value_provenance import provenance_pair

    return [
        provenance_pair(
            crossover.get("provenance"),
            where=f"{crossover.get('study_kind')} depth {crossover.get('depth')}",
        )[0]
        for crossover in published_crossovers(load_restatement_design(path))
    ]


def _validate_crossover_restatement(
    design_path: Path, *, root: Path, data_dir: Path | None
) -> None:
    """The restatement design's own validation, which resolves all six published crossovers."""
    from llb.bench.agentic_memory_crossover_restatement_design import (
        load_restatement_design,
        validate_restatement_design,
    )

    validate_restatement_design(load_restatement_design(design_path), root=root, data_dir=data_dir)


# The registry. One entry per design that publishes values resolved through the committed evidence.
PUBLISHED_VALUE_DESIGNS: dict[str, PublishedValueDesign] = {
    KIND_CROSSOVER_RESTATEMENT: PublishedValueDesign(
        design_path="samples/benchmarks/agentic_compact_crossover_restatement_design.json",
        cited_artifacts=_crossover_restatement_citations,
        validate_published_values=_validate_crossover_restatement,
    ),
}


def registered_design_path(kind: str, design: PublishedValueDesign, design_root: Path) -> Path:
    """Where one registered design lives, refusing a registry entry the repo does not carry.

    Refused rather than skipped, in both walks below: a design nobody can read has unknown citations
    and unknown claims, and unknown is not the same as none -- skipping it would prune the evidence
    it cites and pass the validation it should have failed.
    """
    path = design_root / design.design_path
    if not path.is_file():
        raise ValueError(
            f"{kind}: the registered published-value design {design.design_path} does not exist, "
            "so nothing can tell which committed evidence it cites or whether the values it "
            "publishes resolve -- refusing rather than walking past it"
        )
    return path


@dataclass(frozen=True, slots=True)
class UnresolvedDesign:
    """A registered design whose published values the committed evidence does not state.

    Carried rather than raised. An operator who re-ran a study and regenerated the evidence is about
    to edit published numbers, and a walk that stopped at the first refusal would hand them one name
    per re-run of the refresh -- so the walk collects, and whoever wants a refusal builds one.
    """

    kind: str
    design_path: str
    reason: str

    def named(self) -> str:
        """One line naming the design and the value its committed evidence no longer states."""
        return f"{self.kind} ({self.design_path}): {self.reason}"


@dataclass(frozen=True, slots=True)
class PublishedValueReport:
    """One walk of the registry: the designs it read, and the ones whose values did not resolve.

    `walked` is carried for the reason the validation returns it -- a walk that checked nothing reads
    exactly like one that checked everything -- and it is what lets a refresh name the designs its
    clean answer speaks for rather than reporting only that nothing complained.
    """

    walked: tuple[str, ...]
    unresolved: tuple[UnresolvedDesign, ...]


def report_published_designs(
    *, root: Path, design_root: Path | None = None, data_dir: Path | None = None
) -> PublishedValueReport:
    """Resolve every registered design's published values, collecting refusals instead of raising.

    The collecting form is the primitive and the refusing one is built on it, so what CI fails on and
    what a refresh reports are the same walk over the same registry rather than two that can drift.
    """
    walked: list[str] = []
    unresolved: list[UnresolvedDesign] = []
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        walked.append(kind)
        try:
            path = registered_design_path(
                kind, design, design_root if design_root is not None else root
            )
            design.validate_published_values(path, root=root, data_dir=data_dir)
        except ValueError as exc:
            unresolved.append(
                UnresolvedDesign(kind=kind, design_path=design.design_path, reason=str(exc))
            )
    return PublishedValueReport(walked=tuple(walked), unresolved=tuple(unresolved))


def validate_published_designs(
    *, root: Path, design_root: Path | None = None, data_dir: Path | None = None
) -> list[str]:
    """Resolve every registered design's published values, refusing if any of them does not.

    Returns the kinds walked. Returned rather than nothing, because the failure this walk exists to
    prevent is silence: a registry that validated no design passes exactly like one that validated
    them all, so a caller states how many it expected rather than trusting a clean run.

    The refusal names EVERY unresolved design rather than the first, since the walk already has them
    all and a reader fixing one at a time is the same slow loop in CI that it is at the refresh.
    """
    report = report_published_designs(root=root, design_root=design_root, data_dir=data_dir)
    if report.unresolved:
        raise ValueError("; ".join(unresolved.named() for unresolved in report.unresolved))
    return list(report.walked)


def published_citations(design_root: Path) -> dict[str, list[str]]:
    """Every run artifact a registered design cites, mapped to the designs that cite it.

    Mapped rather than collected into a set, because the refresh's refusals are only useful when
    they can name WHOSE evidence a regeneration is about to drop or cannot serve.
    """
    citations: dict[str, list[str]] = {}
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        path = registered_design_path(kind, design, design_root)
        for artifact in design.cited_artifacts(path):
            citing = citations.setdefault(artifact, [])
            if kind not in citing:
                citing.append(kind)
    return citations


def refresh_committed_evidence(
    *, root: Path, data_dir: Path, design_root: Path | None = None
) -> Path:
    """Re-commit the union of the run aggregates every registered design's values resolve against.

    Copied, never typed: the bytes and the digest come out of one read of one file, so a committed
    copy and the pin that makes it falsifiable cannot disagree.

    `design_root` is where the registered designs are read from, defaulting to the evidence root.
    They differ only when the evidence is regenerated into a throwaway tree -- a test that rewrote
    the tracked fixture it was checking would report a pass by having repaired it.
    """
    citations = published_citations(design_root if design_root is not None else root)
    cited: dict[str, bytes] = {}
    for artifact, citing in sorted(citations.items()):
        path = data_dir / artifact
        if not path.is_file():
            raise ValueError(
                f"{', '.join(citing)}: the run artifact {artifact} is not under DATA_DIR on this "
                "host, so its committed copy cannot be regenerated here -- refresh on a host that "
                "holds every cited run, because a partial refresh retires the committed evidence "
                "of the studies this host never ran"
            )
        cited[artifact] = path.read_bytes()
    return write_provenance_fixture(root, cited, cited_by=citations)


def refresh_committed_evidence_and_report(
    *, root: Path, data_dir: Path, design_root: Path | None = None
) -> tuple[Path, PublishedValueReport]:
    """Re-commit the union, then say whether the published values still resolve out of what it wrote.

    Reported, never rolled back. A refresh after a re-run IS the repair flow: the new aggregates are
    what the operator has to commit before they can restate anything, so refusing the write would
    leave them with neither the new evidence nor a way to record it. The write stands and the values
    it no longer states are named while the run is still in front of the operator, instead of
    arriving as a CI failure on a day when the fix is a design edit with no run context behind it.
    """
    manifest = refresh_committed_evidence(root=root, data_dir=data_dir, design_root=design_root)
    return manifest, report_published_designs(root=root, design_root=design_root, data_dir=data_dir)
