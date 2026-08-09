"""Inventory and audit absolute experiment controls that can look like evidence gates."""

import argparse
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llb.core.paths import PROJECT_ROOT, resolve_data_dir
from llb.core.store_generations import generation_timestamp
from llb.quality.acceptance_gate_registry import DECLARATIONS, MAKE_CONFIG

METHOD = "acceptance-gate-audit"
_LOG = logging.getLogger(__name__)

_MAKE_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*\?=\s*([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)
_TYPER_CONTROL = re.compile(
    r"(\w*(?:trials|sample_size|min_chains))\s*:\s*[^=]+=\s*typer\.Option\(\s*([0-9]+)",
    re.MULTILINE,
)


def _is_audited_make_symbol(symbol: str) -> bool:
    return symbol.endswith(
        ("_TRIALS", "_VERIFY_N", "_MIN_ACCEPTED", "_MIN_FINALISTS", "_SEEDS")
    ) or symbol in {"GOLDSET_N", "PIPELINE_TOP_N", "RECOMMEND_MIN_CASES"}


# Inferential controls that must stay derived: an absolute default here would be a gate again.
_FIXED_MAKE_PATTERNS = {
    "VERIFY_N": r"^VERIFY_N\s*\?=\s*[1-9][0-9]*$",
    "CHAIN_VERIFY_N": r"^CHAIN_VERIFY_N\s*\?=\s*[1-9][0-9]*$",
    "CHAIN_MIN_ACCEPTED": r"^CHAIN_MIN_ACCEPTED\s*\?=\s*[1-9][0-9]*$",
    "QUICKSTART_DRAFT_VERIFY_N": r"^QUICKSTART_DRAFT_VERIFY_N\s*\?=\s*[1-9][0-9]*$",
}
_FIXED_CLI_PATTERNS = {
    "src/llb/goldset/verify/cli.py": (
        r'add_argument\(\s*"-n",\s*"--size"[\s\S]{0,160}?default=[0-9]+'
    ),
    "src/llb/goldset/promote_chains.py": (
        r'add_argument\(\s*"--min-chains"[\s\S]{0,160}?default=[0-9]+'
    ),
}
_RETIRED_CONTROLS = [
    {
        "id": "ua-model-roster-long-run",
        "status": "absent",
        "successor_controls": ["JOINT_SEARCH_TRIALS", "JOINT_SEARCH_MIN_FINALISTS"],
        "reading": "resource and structural controls, not inferential gates",
    }
]


def _discover_make_controls(make_text: str) -> list[dict[str, str]]:
    """Every absolute Make assignment this audit is responsible for classifying."""
    return [
        {"location": MAKE_CONFIG, "symbol": symbol, "value": value}
        for symbol, value in _MAKE_ASSIGNMENT.findall(make_text)
        if _is_audited_make_symbol(symbol)
    ]


def _discover_cli_controls(project_root: Path) -> list[dict[str, str]]:
    """Every absolute Typer default that looks like an experiment control."""
    discoveries: list[dict[str, str]] = []
    for path in sorted((project_root / "src" / "llb").rglob("*.py")):
        relative = str(path.relative_to(project_root))
        for symbol, value in _TYPER_CONTROL.findall(path.read_text(encoding="utf-8")):
            discoveries.append({"location": relative, "symbol": symbol, "value": value})
    return discoveries


def _undeclared_findings(
    discoveries: list[dict[str, str]], declared: set[tuple[str, str]]
) -> list[str]:
    """A control nobody classified is the hole this audit exists to close."""
    findings = []
    for row in discoveries:
        if (row["location"], row["symbol"]) in declared:
            continue
        kind = "Make" if row["location"] == MAKE_CONFIG else "CLI"
        where = row["symbol"] if kind == "Make" else f"{row['location']}:{row['symbol']}"
        findings.append(f"unclassified absolute {kind} control: {where}={row['value']}")
    return findings


def _declaration_findings(
    project_root: Path, discovered_values: dict[tuple[str, str], str]
) -> list[str]:
    """Every declared gate still exists, and still carries the default it was declared with."""
    findings: list[str] = []
    for row in DECLARATIONS:
        path = project_root / row.location
        if not path.is_file():
            findings.append(f"declared gate location is missing: {row.location}")
            continue
        text = path.read_text(encoding="utf-8")
        if row.symbol not in text:
            findings.append(f"declared gate symbol is missing: {row.location}:{row.symbol}")
        if row.location == MAKE_CONFIG:
            findings.extend(_make_default_findings(row, text))
        elif row.default != "derived":
            findings.extend(_cli_default_findings(row, discovered_values))
    return findings


def _make_default_findings(row: Any, make_text: str) -> list[str]:
    """One declared Make gate, read against the assignment actually in the config."""
    expected = (
        rf"^{re.escape(row.symbol)}\s*\?=\s*$"
        if row.default == "derived"
        else rf"^{re.escape(row.symbol)}\s*\?=\s*{re.escape(row.default)}$"
    )
    if re.search(expected, make_text, re.MULTILINE):
        return []
    return [f"declared Make default changed: {row.symbol} expected {row.default}"]


def _cli_default_findings(row: Any, discovered_values: dict[tuple[str, str], str]) -> list[str]:
    """One declared CLI gate, read against the default the discovery pass actually found."""
    actual = discovered_values.get((row.location, row.symbol))
    if actual is None or actual == row.default:
        return []
    return [
        f"declared CLI default changed: {row.location}:{row.symbol} "
        f"expected {row.default}, got {actual}"
    ]


def _regained_default_findings(project_root: Path, make_text: str) -> list[str]:
    """A control that was deliberately made inferential must not grow an absolute default again."""
    findings = [
        f"inferential Make control regained an absolute default: {symbol}"
        for symbol, pattern in _FIXED_MAKE_PATTERNS.items()
        if re.search(pattern, make_text, re.MULTILINE)
    ]
    findings.extend(
        f"inferential CLI control regained an absolute default: {location}"
        for location, pattern in _FIXED_CLI_PATTERNS.items()
        if re.search(pattern, (project_root / location).read_text(encoding="utf-8"))
    )
    return findings


def audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the complete inventory and unexplained-control findings."""
    make_text = (project_root / MAKE_CONFIG).read_text(encoding="utf-8")
    discoveries = _discover_make_controls(make_text) + _discover_cli_controls(project_root)
    discovered_values = {(row["location"], row["symbol"]): row["value"] for row in discoveries}
    declared = {(row.location, row.symbol) for row in DECLARATIONS}
    findings = _undeclared_findings(discoveries, declared)
    findings.extend(_declaration_findings(project_root, discovered_values))
    findings.extend(_regained_default_findings(project_root, make_text))
    return {
        "schema_version": 1,
        "method": METHOD,
        "passed": not findings,
        "classifications": sorted({row.classification for row in DECLARATIONS}),
        "declarations": [asdict(row) for row in DECLARATIONS],
        "discoveries": discoveries,
        "findings": findings,
        "retired_controls": _RETIRED_CONTROLS,
    }


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    """Write the machine inventory and a compact human-readable classification report."""
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "inventory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in result["declarations"]:
        classification = row["classification"]
        counts[classification] = counts.get(classification, 0) + 1
    lines = [
        "# Acceptance Gate Audit",
        "",
        f"Status: {'pass' if result['passed'] else 'fail'}",
        "",
        "## Classification Counts",
        "",
        *[f"- {key}: {counts[key]}" for key in sorted(counts)],
        "",
        "The machine-readable source of detail is `inventory.json`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="audit absolute experiment acceptance controls")
    parser.add_argument("--check", action="store_true", help="validate without writing artifacts")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit()
    if not args.check:
        out_dir = args.out_dir or resolve_data_dir() / METHOD / generation_timestamp()
        write_report(result, out_dir)
        _LOG.info("%s", out_dir)
    for finding in result["findings"]:
        _LOG.error("ERROR: %s", finding)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
