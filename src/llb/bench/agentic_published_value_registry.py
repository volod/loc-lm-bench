"""Every committed design that publishes resolvable values, and the refresh that serves them all.

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
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llb.bench.agentic_published_value_fixture import write_provenance_fixture

# The registered kinds, named after the design that publishes the values rather than after the
# studies it restates -- one design can publish values measured by several studies.
KIND_CROSSOVER_RESTATEMENT = "compact_crossover_restatement"


@dataclass(frozen=True, slots=True)
class PublishedValueDesign:
    """One registered design: where it lives, and how to read the artifacts its values cite.

    The reader is per design because the values are shaped per design -- the restatement states
    published crossovers per audited study, and the next adopter will state something else. What
    every reader owes the registry is the same thing: the DATA_DIR-relative run artifacts its
    published values resolve against, so the refresh can take the union without knowing the shape.
    """

    design_path: str
    cited_artifacts: Callable[[Path], list[str]]


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


# The registry. One entry per design that publishes values resolved through the committed evidence.
PUBLISHED_VALUE_DESIGNS: dict[str, PublishedValueDesign] = {
    KIND_CROSSOVER_RESTATEMENT: PublishedValueDesign(
        design_path="samples/benchmarks/agentic_compact_crossover_restatement_design.json",
        cited_artifacts=_crossover_restatement_citations,
    ),
}


def published_citations(design_root: Path) -> dict[str, list[str]]:
    """Every run artifact a registered design cites, mapped to the designs that cite it.

    Mapped rather than collected into a set, because both refusals below are only useful when they
    can name WHOSE evidence a refresh is about to drop or cannot serve.
    """
    citations: dict[str, list[str]] = {}
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        path = design_root / design.design_path
        if not path.is_file():
            raise ValueError(
                f"{kind}: the registered published-value design {design.design_path} does not "
                "exist, so the refresh cannot tell which committed evidence it still cites -- "
                "refusing rather than pruning the copies it would have named"
            )
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
