"""Render an audit as `findings.jsonl` + `report.md` + `summary.json`.

The Markdown is the operator's read: what the corpus looks like, what each tier cost, and the
worst offenders first. Every count it prints carries the distinct-unit census behind it
(`census.py`), because a row count on a corpus whose conflicts concentrate is a multiple of the
evidence -- and an operator reads that count to decide how much review to fund.

`findings.jsonl` is the machine read and the input a resolution lane consumes -- one JSON object
per claim pair, with both sides' exact offsets. It stays one line per row: the census and the
grouping are rendering, never a filter over the rows.
"""

import json
from pathlib import Path

from llb.conflicts.grouping.census import (
    census_units,
    counted,
    finding_census,
    relation_census,
)
from llb.conflicts.constants import (
    DECIDE_LABEL,
    FINDINGS_FILE,
    GROUPS_FILE,
    REPORT_FILE,
    REVIEW_LABEL,
    SUMMARY_FILE,
    TIER_SEMANTIC,
    TREE_META_FILE,
    is_actionable,
)
from llb.conflicts.grouping.artifact import groups_document
from llb.conflicts.tiers.hashing import sha256_text
from llb.conflicts.models import AuditResult
from llb.conflicts.report.findings import findings_section
from llb.conflicts.report.precision import precision_section
from llb.conflicts.linkage.report import report_section as linkage_section
from llb.conflicts.report.projection import projected_review_lines


def render_report(result: AuditResult) -> str:
    """The operator-facing Markdown report."""
    findings = f"- findings: {len(result.findings)}"
    if result.findings:
        # A count alone reads as N independent conflicts; the units say what it is evidence of.
        findings += f" {census_units(finding_census(result.findings))}"
    lines = [
        "# Corpus conflict audit",
        "",
        f"- corpus: `{result.corpus_root}`",
        f"- effort: `{result.effort}`",
        f"- documents: {result.n_docs}",
        findings,
        *_decide_line(result),
        *_projected_review_line(result),
        "",
    ]
    lines += _relations_section(result)
    lines += _tiers_section(result)
    lines += precision_section(result)
    lines += _needles_section(result)
    lines += linkage_section(result.edition_linkage)
    lines += findings_section(result)
    return "\n".join(lines)


def _decide_line(result: AuditResult) -> list[str]:
    """The headline TO DECIDE count, named apart from the review count it is not.

    An operator reads the headline before anything else and then meets a second number in the
    resolution plan; naming only one of them there is what leaves them reconciling two counts of
    the same group's work. The audit can print only this one -- the other needs a policy.
    """
    if not result.findings:
        return []
    decide = sum(1 for finding in result.findings if is_actionable(finding.relation))
    return [
        f"- {DECIDE_LABEL}: {decide} of {counted(len(result.findings), 'row')} "
        "(every relation but `complementary`). This is NOT the number of human reviews the corpus "
        f"costs: that count is **{REVIEW_LABEL}** (`review_rows` in the resolution `plan.json`), "
        "it depends on the resolution policy, and it can be larger or smaller than this one."
    ]


def _projected_review_line(result: AuditResult) -> list[str]:
    """The opt-in TO REVIEW projection, printed only when an operator named a policy.

    Without `--project-policy` this is nothing at all and the report is byte-identical to a report
    that never heard of the resolution vocabulary. With it, the headline count an operator funds is
    available one command earlier -- labelled as a projection under a NAMED policy every time it
    appears -- and with several named, the delta that says whether the choice costs anything here,
    beside the governance coverage that says whether a non-zero delta was reachable at all.
    """
    if not result.findings:
        return []
    return projected_review_lines(result.policy_projection, result.governance_coverage)


def _relations_section(result: AuditResult) -> list[str]:
    """Per relation, the count AND the distinct units it rests on -- never the count alone."""
    census = relation_census(result.findings)
    if not census:
        return ["No conflicting, duplicated, or subsumed claims were found.", ""]
    lines = [
        "## Relations",
        "",
        "`findings` counts rows; the remaining columns count the evidence those rows rest on. A "
        "relation whose rows collapse into one group is one decision, not N.",
        "",
        "| relation | findings | documents | document pairs | chunk units | groups |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{relation}` | {row['findings']} | {row['documents']} | {row['document_pairs']} "
        f"| {row['chunk_units']} | {row['groups']} |"
        for relation, row in census.items()
    ]
    lines.append("")
    return lines


def _threshold_section(result: AuditResult) -> list[str]:
    """How the semantic cosine cutoff was chosen, and the null distribution behind it.

    An operator comparing two runs needs the ABSOLUTE cosine even when the knob was a quantile;
    a quantile alone is not comparable across corpora, which is the whole reason it exists.
    """
    semantic = next((stat for stat in result.tiers if stat.tier == TIER_SEMANTIC), None)
    if semantic is None or "cos_threshold" not in semantic.extra:
        return []
    source = semantic.extra.get("cos_threshold_source", "default")
    lines = [
        "## Semantic threshold",
        "",
        f"- cosine cutoff: **{float(semantic.extra['cos_threshold']):.4f}** ({source})",
    ]
    null = semantic.extra.get("null_distribution")
    if not isinstance(null, dict):
        lines.append("")
        return lines
    kind = "exhaustive" if null.get("exhaustive") else f"sampled (seed {null.get('seed')})"
    lines += [
        f"- measured over {null.get('n_pairs')} comparable cross-document pairs, {kind}, "
        f"out of {null.get('total_pairs')} total",
        f"- null quantile used: {null.get('resolved_quantile')}",
        f"- rank cutoff: {null.get('selected_rank')}"
        + (
            f" (candidate budget {null['max_candidate_pairs']})"
            if null.get("max_candidate_pairs") is not None
            else ""
        ),
        "- this is a rank cutoff into the corpus's own similarity ordering, NOT a "
        "false-positive rate (see the data-prep known limitation)",
        f"- mean cosine of a comparable pair: {null.get('mean')}",
        "",
        "| quantile | cosine |",
        "| --- | --- |",
    ]
    quantiles = null.get("quantiles") or {}
    lines += [f"| {q} | {value} |" for q, value in quantiles.items()]
    lines.append("")
    return lines


def _tiers_section(result: AuditResult) -> list[str]:
    if not result.tiers:
        return []
    lines = [
        "## Tiers",
        "",
        "| tier | candidate pairs | findings | seconds |",
        "| --- | --- | --- | --- |",
    ]
    for stat in result.tiers:
        lines.append(
            f"| `{stat.tier}` | {stat.candidate_pairs} | {stat.findings} | {stat.seconds:.2f} |"
        )
    lines.append("")
    lines += _threshold_section(result)
    lines.append("")
    semantic = next((s for s in result.tiers if "pruned_fraction" in s.extra), None)
    if semantic is not None:
        extra = semantic.extra
        lines += [
            f"The semantic prefix tree examined {extra.get('pairs_examined')} of "
            f"{extra.get('exhaustive_pairs')} possible chunk pairs "
            f"({float(extra.get('pruned_fraction', 0.0)) * 100:.1f}% pruned) at cosine "
            f">= {extra.get('cos_threshold')}. Pruning is exact: the surviving pairs are the "
            "same ones an all-pairs scan would return.",
            "",
        ]
    return lines


def _needles_section(result: AuditResult) -> list[str]:
    needles = result.needles
    if not needles:
        return []
    lines = [
        "## Needle ambiguity",
        "",
        f"- gold items checked: {needles.get('items')}",
        f"- answerable from more than one document: {needles.get('ambiguous_items')} "
        f"({float(needles.get('non_unique_needle_fraction', 0.0)) * 100:.1f}%)",
    ]
    unlocated = int(needles.get("unlocated_items", 0) or 0)
    if unlocated:
        lines.append(f"- gold spans that matched no chunk (not scored for ambiguity): {unlocated}")
    lines.append("")
    return lines


def _editions_of(result: AuditResult) -> dict[str, str]:
    """Which edition group each document was proposed into, or nothing when the lane did not run."""
    lane = result.edition_linkage_run
    return dict(lane.editions_of) if lane is not None and not lane.declined else {}


def write_audit(out_dir: Path | str, result: AuditResult) -> dict[str, Path]:
    """Persist the audit artifacts; returns their paths by name."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The rows are written once and grouped from the SAME list, so the sidecar's group ids address
    # exactly the rows on disk -- and the resolution lane, grouping those rows in file order,
    # reproduces the ids without needing the sidecar at all.
    rows = result.rows()
    findings_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    findings_path = out / FINDINGS_FILE
    findings_path.write_text(findings_text, encoding="utf-8")
    report_path = out / REPORT_FILE
    report_path.write_text(render_report(result), encoding="utf-8")
    summary_path = out / SUMMARY_FILE
    summary_path.write_text(
        json.dumps(result.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    groups_path = out / GROUPS_FILE
    groups_path.write_text(
        json.dumps(
            groups_document(
                rows,
                findings_sha256=sha256_text(findings_text),
                editions=_editions_of(result),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths = {
        "findings": findings_path,
        "report": report_path,
        "summary": summary_path,
        "groups": groups_path,
    }
    if result.edition_linkage:
        from llb.conflicts.linkage.artifacts import write_edition_linkage

        paths.update(write_edition_linkage(out, result.edition_linkage_run))
    if result.tree_meta:
        tree_path = out / TREE_META_FILE
        tree_path.write_text(
            json.dumps(result.tree_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths["tree_meta"] = tree_path
    return paths
