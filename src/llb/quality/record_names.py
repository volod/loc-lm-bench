"""Task records are named `NNNN-<group>-<task-id>.md`, numbered in sequence, and indexed.

A record outlives the plan heading it came from, so its filename is how it is found later. The
sequence number says in which order work closed and makes the directory listing that order; the
group says which capability the work served (`adhoc` for a direct request, never a plan task); the
task id is the plan id, unchanged. The rule is in the planning workflow ("Record file naming"); this
check makes a misnamed, duplicated, skipped, or unindexed record a lint failure:

- every record but `README.md` and `template.md` matches the shape, with a group that is `adhoc` or
  a capability in the spec's registry and a non-empty task id after it;
- the sequence numbers are unique and run from 0001 without a gap -- the next record takes the next
  unused number, and a record is never deleted;
- the records index (`README.md`) links every record.
"""

import argparse
import logging
import re
from pathlib import Path

from llb.core.paths import PROJECT_ROOT
from llb.quality.plan_model import read_registry
from llb.quality.spec_plan_integrity import SPEC_DOC

_LOG = logging.getLogger(__name__)

RECORDS_DIR = Path("docs/impl/records")
INDEX_NAME = "README.md"
TEMPLATE_NAME = "template.md"
ADHOC_GROUP = "adhoc"
SEQUENCE_DIGITS = 4

_NAME = re.compile(rf"^(?P<seq>[0-9]{{{SEQUENCE_DIGITS}}})-(?P<rest>[a-z0-9][a-z0-9-]*)\.md$")


def _group_problem(rest: str, groups: set[str]) -> str | None:
    """Why `<group>-<task-id>` does not start with a known group, or None when it does."""
    for group in sorted(groups, key=len, reverse=True):
        if rest.startswith(f"{group}-") and len(rest) > len(group) + 1:
            return None
    return f"group is not `{ADHOC_GROUP}` or a registered capability id, or the task id is missing"


def record_findings(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every naming, sequence, and index problem in the records directory."""
    records_dir = project_root / RECORDS_DIR
    groups = {capability.identifier for capability in read_registry(project_root / SPEC_DOC)}
    groups.add(ADHOC_GROUP)
    index = (records_dir / INDEX_NAME).read_text(encoding="utf-8")
    findings: list[str] = []
    sequences: dict[int, str] = {}
    for path in sorted(records_dir.glob("*.md")):
        if path.name in (INDEX_NAME, TEMPLATE_NAME):
            continue
        where = f"{RECORDS_DIR / path.name}"
        match = _NAME.match(path.name)
        if match is None:
            findings.append(f"{where}: name is not NNNN-<group>-<task-id>.md")
            continue
        problem = _group_problem(match["rest"], groups)
        if problem:
            findings.append(f"{where}: {problem}")
        number = int(match["seq"])
        if number in sequences:
            findings.append(f"{where}: sequence {match['seq']} is also {sequences[number]}")
        sequences[number] = path.name
        if f"]({path.name})" not in index:
            findings.append(f"{where}: not linked from {RECORDS_DIR / INDEX_NAME}")
    missing = sorted(set(range(1, max(sequences, default=0) + 1)) - set(sequences))
    findings.extend(
        f"{RECORDS_DIR}: sequence {number:0{SEQUENCE_DIGITS}d} is skipped" for number in missing
    )
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="check task-record names, sequence, and index")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    findings = record_findings(args.root)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    if findings:
        _LOG.error(
            "Name records per docs/guides/development/planning-workflow.md#record-file-naming."
        )
    _LOG.info("[record-names] %d finding(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
