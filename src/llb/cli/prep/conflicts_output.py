"""Console rendering for `audit-corpus-conflicts`: flag parsing and the summary lines.

Split from the command module so `conflicts.py` stays the flag declaration plus the wiring that
composes the audit; everything an operator READS about the finished audit is assembled here.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.conflicts.claim.probe import PROBE_TIERS
from llb.conflicts.constants import TIER_SEMANTIC
from llb.conflicts.resolution.policy import POLICIES
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:
    from llb.conflicts.models import AuditResult


def projected_policies(value: Optional[str]) -> list[str]:
    """Parse `--project-policy a,b` into an ordered, de-duplicated, validated policy list.

    The first policy named is the delta's baseline and stays the one whose counts sit at the top
    level of the artifact, so the order an operator types is the order the report reads in.
    """
    if value is None:
        return []
    named = [item.strip() for item in value.split(",") if item.strip()]
    if not named:
        raise typer.BadParameter(
            f"--project-policy needs at least one policy; choose from {', '.join(POLICIES)}"
        )
    unknown = [policy for policy in named if policy not in POLICIES]
    if unknown:
        raise typer.BadParameter(
            f"unknown resolution policy {', '.join(repr(p) for p in unknown)}; "
            f"choose from {', '.join(POLICIES)}"
        )
    if len(set(named)) != len(named):
        raise typer.BadParameter(
            f"--project-policy repeats a policy ({value!r}); each column must be a distinct policy"
        )
    return named


def parsed_probe_tiers(value: Optional[str]) -> Optional[tuple[str, ...]]:
    """Parse `--probe-tiers a,b` into the tiers to adjudicate, or None for every declared tier.

    The tier NAMES are not validated here: the probe file declares which tiers exist, so an
    unknown name is the probe's error to report against the tiers it actually carries.
    """
    if value is None:
        return None
    named = [item.strip() for item in value.split(",") if item.strip()]
    if not named:
        raise typer.BadParameter(
            f"--probe-tiers needs at least one tier; known tiers are {', '.join(PROBE_TIERS)}"
        )
    if len(set(named)) != len(named):
        raise typer.BadParameter(f"--probe-tiers repeats a tier ({value!r})")
    return tuple(named)


def echo_projection(projection: JsonObject, coverage: JsonObject) -> None:
    """One line per projected policy, then the delta -- the same reading the report prints."""
    if not projection:
        return
    from llb.conflicts.grouping.census import counted
    from llb.conflicts.report.projection import coverage_sentence, policies_of

    for policy in policies_of(projection):
        counts = projection.get("by_policy", {}).get(policy, projection)
        typer.echo(
            f"[conflicts] to review PROJECTED under policy {policy}: "
            f"{counted(int(counts['review_rows']), 'row')} in "
            f"{counted(int(counts['review_groups']), 'decision group')} "
            "(a projection, not a measurement; resolve-corpus-conflicts measures it)"
        )
    for delta in projection.get("deltas") or []:
        typer.echo(_delta_line(delta))
    # The same precondition the report prints beside the delta, decided by the same helper: a free
    # choice on a corpus with no orderable pair is a property of the ingestion, not of the corpus.
    precondition = coverage_sentence(projection, coverage)
    if precondition:
        typer.echo(f"[conflicts] {precondition}")


def _delta_line(delta: JsonObject) -> str:
    """The cost of one policy CHOICE against the baseline, and where in the corpus it lands."""
    from llb.conflicts.report.projection import moved_share_phrase, signed_delta

    moved = [str(group) for group in delta["moved_groups"]]
    if not int(delta["moved_rows"]):
        where = " -- the choice is free on this corpus"
    elif moved:
        where = f" in {', '.join(moved)}"
    else:
        where = " -- cancelling out inside the groups it moves rows in"
    return (
        f"[conflicts] policy choice {delta['baseline']} -> {delta['policy']}: "
        f"{signed_delta(int(delta['review_rows']))} rows to review{where}; "
        f"moves {moved_share_phrase(delta)}"
    )


def echo_summary(result: "AuditResult", paths: dict[str, Path]) -> None:
    """The whole operator-facing reading of one audit, in the order the report presents it."""
    from llb.conflicts.grouping.census import census_units, finding_census, relation_census

    census = finding_census(result.findings)
    typer.echo(
        f"[conflicts] effort={result.effort} docs={result.n_docs} findings={census['findings']} "
        f"groups={census['groups']} docs_touched={census['documents']} "
        f"doc_pairs={census['document_pairs']} chunk_units={census['chunk_units']}"
    )
    _echo_semantic_threshold(result)
    for relation, row in relation_census(result.findings).items():
        typer.echo(f"[conflicts]   {relation}: {row['findings']} rows {census_units(row)}")
    echo_projection(result.policy_projection, result.governance_coverage)
    _echo_edition_linkage(result.edition_linkage)
    _echo_claim_prefilter(result.claim_prefilter)
    _echo_claim_precision(result.claim_precision)
    if result.needles:
        typer.echo(
            f"[conflicts] needles: {result.needles.get('ambiguous_items')}/"
            f"{result.needles.get('items')} gold items answerable from more than one document"
        )
    typer.echo(f"[conflicts] report: {paths['report']}")


def _echo_semantic_threshold(result: "AuditResult") -> None:
    """The cosine cutoff the semantic tier ran at, and where that number came from."""
    semantic = next((t for t in result.tiers if t.tier == TIER_SEMANTIC), None)
    if semantic is None or "cos_threshold" not in semantic.extra:
        return
    source = semantic.extra.get("cos_threshold_source", "default")
    line = f"[conflicts] cos_threshold={semantic.extra['cos_threshold']:.4f} ({source})"
    null = semantic.extra.get("null_distribution")
    if isinstance(null, dict):
        line += f" q={null.get('resolved_quantile')} over {null['total_pairs']} comparable pairs"
    typer.echo(line)


def _echo_edition_linkage(summary: JsonObject) -> None:
    """The edition-linkage lane's reading, or the reason it did not run."""
    from llb.conflicts.linkage.report import console_lines

    for line in console_lines(summary):
        typer.echo(line)


def _echo_claim_precision(precision: JsonObject) -> None:
    """Either the measured claim-tier precision, or why the run declined to report one."""
    if precision.get("reported"):
        point = precision["returned_budget"]
        typer.echo(
            f"[conflicts] claim-tier precision {point['precision']} at budget {point['budget']} "
            f"(two-way clustered 95% LCB {point['two_way_clustered_lcb']})"
        )
    elif precision:
        typer.echo(f"[conflicts] claim-tier precision not reported: {precision['reason']}")


def _echo_claim_prefilter(prefilter: JsonObject) -> None:
    """The row saving a full-list run measures, or why a capped run cannot measure loss."""
    if not prefilter:
        return
    same = prefilter["same_conflicts"]
    if same["evaluated"]:
        typer.echo(
            "[conflicts] claim prefilter same actionable set: "
            f"cosine {same['cosine_rows_needed']} rows, cross-encoder "
            f"{same['reranked_rows_needed']} rows "
            f"({same['adjudication_calls_saved']} calls saved by the gate, "
            f"{same['conflict_rows_lost']} conflict rows lost)"
        )
    else:
        typer.echo(
            f"[conflicts] claim prefilter adjudicated {prefilter['adjudicated_rows']} of "
            f"{prefilter['candidate_rows']} rows; conflict loss needs a paired full-list run"
        )
