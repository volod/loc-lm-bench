"""Render an axiom validation run as `report.md` + `violations.jsonl` + `summary.json`.

The Markdown is the operator's read and the sign-off lane's input. Two rules shape it:

- **A zero is a finding, not a blank.** An axiom that found nothing either did not apply here (no
  fact carried its relation) or held everywhere it applied, and those are different measurements.
  Every row states which one it is in words, so nobody reads a silent pass as evidence.
- **Every violation carries its spans.** A reviewer adjudicates a contradiction from the report
  alone; a row that names two facts without their exact evidence sends them back to the corpus.
"""

import json
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.prep.ontology.axioms.constants import (
    EVIDENCE_PREVIEW_CHARS,
    N_REPORT_VIOLATIONS,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    VIOLATIONS_FILENAME,
    AXIOM_EVIDENCE_FILENAME,
)
from llb.prep.ontology.axioms.models import (
    AxiomStat,
    CrosscheckResult,
    LedgerReport,
    ValidationReport,
    Violation,
    ViolationFact,
)


def reading(stat: AxiomStat) -> str:
    """The one-phrase reading of a row, so a zero is never left to the reader to interpret."""
    if stat.checked == 0:
        if stat.unchecked:
            return f"no typed endpoint to check ({stat.unchecked} untyped)"
        return "did not apply here (no fact carries this relation)"
    if stat.violating == 0:
        return f"held on all {stat.checked} units"
    return f"broken on {stat.violating} of {stat.checked} units"


def _subject(stat: AxiomStat) -> str:
    return stat.relation or " + ".join(stat.entity_types)


def _stats_table(stats: list[AxiomStat]) -> list[str]:
    lines = [
        "| axiom | class | relation / types | checked | supporting | violating | unchecked "
        "| rate | reading |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| `{s.axiom_id}` | `{s.kind}` | {_subject(s)} | {s.checked} | {s.supporting} "
        f"| {s.violating} | {s.unchecked} | {s.rate:.3f} | {reading(s)} |"
        for s in stats
    ]
    lines.append("")
    return lines


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= EVIDENCE_PREVIEW_CHARS:
        return flat
    return flat[:EVIDENCE_PREVIEW_CHARS] + "..."


def _fact_line(fact: ViolationFact) -> str:
    span = fact.evidence
    return (
        f"  - `{fact.subject}` -[{fact.relation}]-> `{fact.object}` "
        f"({span.doc_id} {span.char_start}-{span.char_end}): {_preview(span.text)}"
    )


def _violation_lines(violation: Violation) -> list[str]:
    return [
        f"- `{violation.axiom_id}` ({violation.kind}): {violation.detail}",
        *[_fact_line(fact) for fact in violation.facts],
    ]


def _ledger_section(ledger: LedgerReport) -> list[str]:
    lines = [
        f"## Ledger `{ledger.label}`",
        "",
        f"- source: `{ledger.source}`",
        f"- documents: {ledger.n_docs}; entities: {ledger.n_entities}; "
        f"facts: {ledger.n_facts}; distinct relations: {ledger.n_relations}",
        f"- violations: {len(ledger.violations)}",
        "",
        *_stats_table(ledger.stats),
    ]
    if not ledger.violations:
        lines += ["No axiom in this set is broken by this ledger.", ""]
        return lines
    shown = ledger.violations[:N_REPORT_VIOLATIONS]
    lines += [
        f"### Violations ({len(shown)} of {len(ledger.violations)} shown; "
        f"`{VIOLATIONS_FILENAME}` carries all)",
        "",
    ]
    for violation in shown:
        lines += _violation_lines(violation)
    lines.append("")
    return lines


def _crosscheck_section(crosscheck: CrosscheckResult | None) -> list[str]:
    if crosscheck is None:
        return []
    lines = ["## Reasoner cross-check", ""]
    if not crosscheck.ran:
        return lines + [f"Not run: {crosscheck.reason}", ""]
    verdict = "AGREES" if crosscheck.agrees else "DISAGREES"
    lines += [
        f"`owlrl` OWL 2 RL closure over the same ledger and axioms: **{verdict}** with the "
        f"in-repo checker on {', '.join(f'`{k}`' for k in crosscheck.kinds)}.",
        "",
    ]
    if crosscheck.checker_only:
        lines += ["Reported by the checker, not entailed by the reasoner:", ""]
        lines += [f"- `{key}`" for key in crosscheck.checker_only] + [""]
    if crosscheck.reasoner_only:
        lines += ["Entailed by the reasoner, not reported by the checker:", ""]
        lines += [f"- `{key}`" for key in crosscheck.reasoner_only] + [""]
    return lines


def render_report(report: ValidationReport) -> str:
    """The operator-facing Markdown read of one validation run."""
    lines = [
        "# Ontology axiom validation",
        "",
        f"- axioms: `{report.axioms_source}` (version `{report.axioms_version}`)",
        f"- constraints: {report.n_axioms} candidate, {report.n_signed} signed by a reviewer",
        f"- ledgers checked: {len(report.ledgers)}",
        f"- violations: {report.n_violations}",
        "",
        "An axiom is a domain claim, so a violation is a CONTRADICTION IN THE LEDGER to adjudicate, "
        "never a deletion: the graph build is unchanged unless an operator passes "
        "`--refuse-violations`. An unsigned axiom is a candidate; it gates nothing.",
        "",
    ]
    for ledger in report.ledgers:
        lines += _ledger_section(ledger)
    lines += _crosscheck_section(report.crosscheck)
    return "\n".join(lines)


def write_artifacts(report: ValidationReport, out_dir: Path) -> dict[str, str]:
    """Write the run bundle and return the artifact name -> path map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / REPORT_FILENAME, render_report(report) + "\n")
    rows = (
        {"ledger": ledger.label, **violation.model_dump(mode="json")}
        for ledger in report.ledgers
        for violation in ledger.violations
    )
    atomic_write_text(
        out_dir / VIOLATIONS_FILENAME,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    atomic_write_text(
        out_dir / AXIOM_EVIDENCE_FILENAME,
        "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for row in report.evidence
        ),
    )
    atomic_write_text(
        out_dir / SUMMARY_FILENAME,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "report": str(out_dir / REPORT_FILENAME),
        "violations": str(out_dir / VIOLATIONS_FILENAME),
        "evidence": str(out_dir / AXIOM_EVIDENCE_FILENAME),
        "summary": str(out_dir / SUMMARY_FILENAME),
    }
