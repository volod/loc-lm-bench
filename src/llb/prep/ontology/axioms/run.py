"""Production wiring: check extraction ledgers against an axiom set and publish the run.

One run answers one question -- "what does this constraint set find in these corpora?" -- so it
takes several ledgers at once. The per-axiom base rate is only readable against the corpus it was
measured on, and a set that fires on one corpus and not another is the finding, not a discrepancy
to average away.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import new_run_timestamp
from llb.core.paths import PROJECT_ROOT
from llb.graph.ingest import load_extractions
from llb.prep.ontology.axioms.checker import LedgerCheck, check_ledger, collect_evidence
from llb.prep.ontology.axioms.constants import METHOD_DIR
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.loader import load_axioms
from llb.prep.ontology.axioms.models import LedgerReport, ValidationReport, Violation
from llb.prep.ontology.axioms.report import reading, write_artifacts
from llb.prep.ontology.constants import EXTRACTION_FILENAME
from llb.prep.ontology.models import DocExtraction

_LOG = logging.getLogger(__name__)


def bundle_dir(data_dir: Path, method: str = METHOD_DIR, run: str | None = None) -> Path:
    """`$DATA_DIR/<method>/<run>/` -- a fresh timestamped run unless one is named."""
    return Path(data_dir) / method / (run or new_run_timestamp()[1])


def display_path(path: Path) -> str:
    """A path a reader on another host can still place: project-relative when it is."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name


def ledger_label(path: Path) -> str:
    """Name a ledger by its bundle directory, since every bundle's file has the same name."""
    path = Path(path)
    return path.parent.name if path.name == EXTRACTION_FILENAME else path.stem


def resolve_ledger_path(path: Path) -> Path:
    """Accept a draft bundle directory as well as an explicit `extraction.jsonl`."""
    path = Path(path)
    return path / EXTRACTION_FILENAME if path.is_dir() else path


def validate_axioms(
    ledger_paths: list[Path],
    axioms_path: Path,
    *,
    crosscheck: bool = False,
) -> ValidationReport:
    """Check every ledger against the axiom set, optionally holding it to an OWL reasoner."""
    axiom_set = load_axioms(axioms_path)
    checks: list[LedgerCheck] = []
    ledgers: list[Ledger] = []
    for raw in ledger_paths:
        path = resolve_ledger_path(raw)
        ledger = Ledger(load_extractions(path))
        ledgers.append(ledger)
        checks.append(check_ledger(axiom_set, ledger, ledger_label(path), display_path(path)))
    report = ValidationReport(
        axioms_source=display_path(axioms_path),
        axioms_version=axiom_set.version,
        n_axioms=len(axiom_set.axioms),
        n_signed=len(axiom_set.signed),
        ledgers=[check.report for check in checks],
        evidence=collect_evidence(axiom_set, checks),
    )
    if crosscheck:
        from llb.prep.ontology.axioms.crosscheck import crosscheck_report

        report.crosscheck = crosscheck_report(axiom_set, ledgers, report)
    return report


def publish(report: ValidationReport, out_dir: Path) -> dict[str, str]:
    """Write the run bundle and log where it landed."""
    paths = write_artifacts(report, out_dir)
    _LOG.info("[axioms] published %d artifacts -> %s", len(paths), out_dir)
    return paths


def format_summary(report: ValidationReport) -> list[str]:
    """Console lines: the headline per ledger, then the per-axiom reading."""
    lines = [
        f"[axioms] {report.n_axioms} axioms ({report.n_signed} signed) from {report.axioms_source}"
    ]
    for ledger in report.ledgers:
        lines.append(
            f"[axioms] {ledger.label}: {len(ledger.violations)} violations over "
            f"{ledger.n_facts} facts in {ledger.n_docs} documents"
        )
        lines += [
            f"           {stat.axiom_id:<24} {stat.kind:<19} {reading(stat)}"
            for stat in ledger.stats
        ]
    if report.crosscheck is not None:
        state = "agrees" if report.crosscheck.agrees else "DISAGREES"
        detail = report.crosscheck.reason or state
        lines.append(f"[axioms] reasoner cross-check: {detail}")
    return lines


@dataclass(frozen=True)
class BuildAxiomCheck:
    """The build-time read of an axiom set over the extractions a graph is about to be built from.

    The split is the trust boundary, not a convenience: a SIGNED axiom is a decision a named
    reviewer dated, so an operator who asked for `--refuse-violations` gets a refusal on it. A
    CANDIDATE axiom is a proposal nobody has accepted; it is reported and never refuses a build,
    which is the same rule the answer gate applies one layer up.
    """

    report: LedgerReport
    signed_violations: list[Violation]
    candidate_violations: list[Violation]

    def lines(self) -> list[str]:
        """Console lines: what would refuse the build, and what only wants a reviewer."""
        out = []
        for violation in self.signed_violations:
            out.append(f"[axioms] signed axiom broken: {violation.axiom_id}: {violation.detail}")
        if self.candidate_violations:
            broken = sorted({v.axiom_id for v in self.candidate_violations})
            out.append(
                f"[axioms] {len(self.candidate_violations)} violations of UNSIGNED candidate "
                f"axioms ({', '.join(broken)}) -- reported, not refused; sign them first"
            )
        return out


def check_build_inputs(
    extractions: list[DocExtraction], axioms_path: Path, label: str = "build"
) -> BuildAxiomCheck:
    """Check the extractions a graph build is about to consume against an axiom set."""
    axiom_set = load_axioms(axioms_path)
    signed = {axiom.axiom_id for axiom in axiom_set.signed}
    check = check_ledger(axiom_set, Ledger(extractions), label, display_path(axioms_path))
    return BuildAxiomCheck(
        report=check.report,
        signed_violations=[v for v in check.report.violations if v.axiom_id in signed],
        candidate_violations=[v for v in check.report.violations if v.axiom_id not in signed],
    )
