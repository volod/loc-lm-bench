"""The operator's read of the edition-linkage lane: the report section and the console lines.

Every line here says the same thing in two halves -- what the fit ranked, and what it did NOT
decide. The audit's findings are still the tiers'; a probability beside them is a second opinion a
reviewer can sort by, and the report may never let it read as the verdict.
"""

from collections.abc import Sequence

from llb.conflicts.linkage.constants import REPORT_EXAMPLES
from llb.conflicts.linkage.run import VERDICT_RANKED
from llb.core.contracts.common import JsonObject

HEADING = "## Edition linkage"
DECLINED_PREFIX = "The edition-linkage lane did not run"
NOT_A_CONFLICT = (
    "A match probability answers *are these two documents the same document*, never *do these two "
    "documents contradict each other*. No number in this section is a conflict verdict or a "
    "false-positive rate for one."
)


def report_section(summary: JsonObject) -> list[str]:
    """The Markdown section, or nothing at all when the lane never ran."""
    if not summary:
        return []
    if summary.get("declined"):
        return [HEADING, "", f"{DECLINED_PREFIX}: {summary['reason']}.", ""]
    lines = [HEADING, "", NOT_A_CONFLICT, "", *_headline(summary), "", *_recovery_table(summary)]
    lines += _editions_table(summary)
    return lines


def _headline(summary: JsonObject) -> list[str]:
    prior, cut, editions = summary["prior"], summary["cut"], summary["editions"]
    order, decided = summary["ordering"], summary["decisions"]
    fit = summary["linkage"]
    return [
        f"- documents fitted: {summary['n_documents']}, pairs scored: {fit['n_scored_pairs']}",
        f"- prior that two random documents are one document: "
        f"{prior['random_match_probability']:.5f} ({prior['source']}, from "
        f"{prior['settled_pairs']} settled of {prior['total_document_pairs']} document pairs)",
        f"- edition cut: **{cut['cut']:.4f}** -- {cut['source']}",
        f"- edition groups: {editions['groups']} over {editions['documents_grouped']} documents, "
        f"largest {editions['largest_group']}, {editions['with_current_edition']} with a current "
        "edition the governance fields could name",
        f"- the two rankings order {order['discordant_orderings']} of "
        f"{order['comparable_orderings']} comparable relation pairs differently "
        f"(Kendall tau {order['kendall_tau']}); a tier score is a Jaccard for one relation and a "
        "containment for the other, so the ordering it implies ACROSS relations is not a "
        "quantity, which is what the discordance is measuring",
        f"- at the cut the fit matches {decided['fit_matches']} pairs against the thresholds' "
        f"{decided['threshold_duplicates']} duplicates: {decided['agreed']} agreed, "
        f"{len(decided['fit_only'])} fit only, {len(decided['thresholds_only'])} thresholds only",
        f"- verdict: `{summary['verdict']}` -- {summary['statement']}",
        *_untrained(fit),
    ]


def _untrained(fit: JsonObject) -> list[str]:
    levels = list(fit.get("untrained_levels") or ())
    if not levels:
        return []
    return [
        f"- comparison levels the fit could not estimate: {', '.join(f'`{lv}`' for lv in levels)} "
        "(a level nobody observed, reported rather than defaulted)"
    ]


def _recovery_table(summary: JsonObject) -> list[str]:
    """Every relation the current thresholds recover, and where the fit put it."""
    recovered = summary["recovery"]
    rows: Sequence[JsonObject] = recovered["rows"]
    lines = [
        "### What the fit did with every relation the thresholds recover",
        "",
        f"{recovered['scored']} of {recovered['relations']} scored; "
        f"{recovered['above_cut']} clear the cut; {recovered['co_clustered']} were merged into one "
        f"edition; {recovered['unreported_pairs_ranked_higher']} pair(s) the thresholds do not "
        "report outrank one they do.",
        "",
        "| rank | relation | tier score | probability | in one edition | documents |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(rows, key=lambda row: (row["rank"] is None, row["rank"]))
    for row in ordered[:REPORT_EXAMPLES]:
        rank = row["rank"] if row["rank"] is not None else "not scored"
        probability = (
            "--" if row["match_probability"] is None else f"{row['match_probability']:.4f}"
        )
        lines.append(
            f"| {rank} | `{row['relation']}` | {row['tier_score']} | {probability} | "
            f"{'yes' if row['co_clustered'] else 'no'} | "
            f"`{row['doc_pair'][0]}` + `{row['doc_pair'][1]}` |"
        )
    if len(ordered) > REPORT_EXAMPLES:
        lines.append(f"| ... | | | | | {len(ordered) - REPORT_EXAMPLES} more in `linkage/` |")
    lines.append("")
    return lines


def _editions_table(summary: JsonObject) -> list[str]:
    groups = summary.get("edition_groups") or []
    if not groups:
        return []
    lines = [
        "### Proposed edition groups",
        "",
        "A group is a proposal, not an edit: naming the current edition retires nothing.",
        "",
        "| edition | documents | current | ordered by |",
        "| --- | --- | --- | --- |",
    ]
    for group in groups[:REPORT_EXAMPLES]:
        named = ", ".join(f"`{doc}`" for doc in group["current"])
        current = named if group["basis"] else f"{named} (not orderable)"
        lines.append(
            f"| {group['edition_id']} | {group['size']} | {current} | {group['basis'] or '--'} |"
        )
    lines.append("")
    return lines


def console_lines(summary: JsonObject) -> list[str]:
    """The two or three lines an operator reads at the end of a run."""
    if not summary:
        return []
    if summary.get("declined"):
        return [f"[conflicts] edition linkage not run: {summary['reason']}"]
    editions, cut = summary["editions"], summary["cut"]
    lines = [
        f"[conflicts] edition linkage: {summary['linkage']['n_scored_pairs']} pairs scored, "
        f"{editions['groups']} edition group(s) over {editions['documents_grouped']} documents "
        f"at cut {cut['cut']:.4f}",
        f"[conflicts] edition linkage verdict={summary['verdict']} "
        f"({summary['recovery']['scored']}/{summary['recovery']['relations']} reported relations "
        f"scored, {summary['ordering']['discordant_orderings']} discordant orderings)",
    ]
    if summary["verdict"] != VERDICT_RANKED:
        lines.append(f"[conflicts] edition linkage: {summary['statement']}")
    return lines
