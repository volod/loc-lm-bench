"""Refuse a measured result cited by a run directory or a bare run label.

`$DATA_DIR/<method>/<run-id>/` is host-local and temporary: it is gone after a cleanup, absent on a
fresh checkout, and absent on every other GPU host. A bare `<timestamp>-<slug>` run label is no
better -- it is a lookup key into a directory the reader does not have. A page that cites either one
stops being checkable the moment the run is deleted, which is why AGENTS.md ("Citing a measured
result") requires the description, the date, the host, and the numbers instead.

Two forms are refused, both only inside a backticked span so ordinary prose is never scanned:

- a path under `$DATA_DIR/` or `.data/` carrying a run segment (`<8 digits>T...`);
- a bare run label outside a table row, where a label TRAILING a description is the accepted form.

A `$DATA_DIR` TEMPLATE with no run segment (`$DATA_DIR/corpus-conflicts/<run>/`) is fine and is what
documents where a command writes. The guide that DEFINES this rule has to quote the anti-pattern to
state it, so it is exempt by name -- and nothing else is, `plan.md` included: the forward plan has
no reason to name a run either.
"""

import argparse
import logging
import re
from pathlib import Path

from llb.core.paths import PROJECT_ROOT
from llb.quality.doc_links import tracked_docs

_LOG = logging.getLogger(__name__)

_RUN_SEGMENT = r"[0-9]{8}T"
_PATH_CITATION = re.compile(rf"`[^`]*(?:\$DATA_DIR|\.data)/[^`]*{_RUN_SEGMENT}[^`]*`")
_BARE_LABEL = re.compile(rf"`{_RUN_SEGMENT}[^`]*`")
_FENCE = "```"
_TABLE_ROW = "|"

# The guide that STATES the rule must quote the shape it forbids.
RULE_DEFINING_DOCS = (Path("docs/guides/development/heavy-runs-and-evidence.md"),)


def _citations(doc: Path) -> list[tuple[int, str, str]]:
    """Every refused citation in one document, as `(line, form, span)`."""
    found: list[tuple[int, str, str]] = []
    fenced = False
    for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
            continue
        if fenced:
            continue
        paths = _PATH_CITATION.findall(line)
        found.extend((number, "run-directory path", span) for span in paths)
        if line.lstrip().startswith(_TABLE_ROW):
            continue
        found.extend(
            (number, "bare run label", span)
            for span in _BARE_LABEL.findall(line)
            if not any(span in path for path in paths)
        )
    return found


def path_citations(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every refused citation, named `file:line: form -> span`."""
    findings: list[str] = []
    for doc in tracked_docs(project_root):
        if doc.relative_to(project_root) in RULE_DEFINING_DOCS:
            continue
        findings.extend(
            f"{doc.relative_to(project_root)}:{number}: {form} -> {span}"
            for number, form, span in _citations(doc)
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="refuse run-directory and bare-run-label citations"
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    findings = path_citations(args.root)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    if findings:
        _LOG.error("Cite the description, date, host, and numbers instead (AGENTS.md).")
    _LOG.info("[doc-citations] %d run-path citation(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
