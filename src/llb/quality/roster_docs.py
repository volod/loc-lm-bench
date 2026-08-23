"""Publish the model-family register into the docs, instead of restating it there by hand.

A roster table written in prose is correct on the day it is written and wrong on the day a
generation lands: the manifest gains `qwen3.8`, the README still names `Qwen 3.6`, and a reader has
no way to tell which of the two statements the tooling actually runs. So the published tables are
GENERATED from `samples/configs/models_uk.yaml` into marked blocks, and `--check` fails when a block
no longer matches the register -- the same shape as `spec_plan_integrity`, applied to the roster.

Only the marked blocks are owned here. Everything around them is prose a person writes.
"""

import argparse
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from llb.backends.roster import (
    CURRENT,
    MULTILINGUAL_BASELINE,
    ROLE_LABELS,
    UA_SPECIALIZED,
    Generation,
    Register,
    load_register,
    register_findings,
)
from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

ROSTER_MANIFEST = Path("samples/configs/models_uk.yaml")
SYNC_COMMAND = "make sync-model-family-docs"
WRAP_WIDTH = 98


@dataclass(frozen=True)
class DocBlock:
    """One generated block: the document that carries it and the table it holds."""

    doc: Path
    name: str


DOC_BLOCKS = (
    DocBlock(Path("README.md"), "model-families"),
    DocBlock(Path("docs/reference/model-families.md"), "model-roster"),
)


def _begin(name: str) -> str:
    """Kept short on purpose: the docs lint caps a line at 100 characters, markers included."""
    return f"<!-- generated: {name} ({SYNC_COMMAND}) -->"


def _end(name: str) -> str:
    return f"<!-- end generated: {name} -->"


def _link(text: str, url: str) -> str:
    return f"[{text}]({url})" if url else text


def _models_cell(generation: Generation) -> str:
    return ", ".join(f"`{name}`" for name in generation.model_names) or "--"


def _generation_cell(generation: Generation | None) -> str:
    if generation is None:
        return "--"
    return _link(generation.label, generation.weights_url)


def _summary(register: Register) -> str:
    families = register.families
    roles = [family.role for family in families]
    generations = [gen for family in families for gen in family.generations]
    previous = [gen for gen in generations if gen.status != CURRENT]
    sentence = (
        f"The default candidate sweep carries {len(families)} open-weight families "
        f"({roles.count(UA_SPECIALIZED)} Ukrainian-specialized, "
        f"{roles.count(MULTILINGUAL_BASELINE)} multilingual baselines) across "
        f"{len(generations)} generations -- one current per family plus {len(previous)} retained "
        f"for generation comparison -- and {len(register.models)} logical models. "
        f"Comply with the listed license when serving or redistributing."
    )
    return textwrap.fill(sentence, width=WRAP_WIDTH)


def render_model_families(register: Register) -> str:
    """README view: one row per family -- what it answers, what it runs now, what it ran before."""
    rows = [
        "| Family | Role in the sweep | Current generation | Also carried | License |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family in register.families:
        current = family.current
        older = ", ".join(gen.label for gen in family.previous) or "--"
        license_cell = _link(current.license, current.license_url) if current else "--"
        rows.append(
            f"| {family.label} | {ROLE_LABELS.get(family.role, family.role)} | "
            f"{_generation_cell(current)} | {older} | {license_cell} |"
        )
    return "\n".join([_summary(register), "", *rows])


def render_model_roster(register: Register) -> str:
    """Reference view: one row per generation -- status, the models on it, and its terms."""
    rows = [
        "| Family | Generation | Status | Models carried | Weights | License |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for family in register.families:
        for generation in family.generations:
            rows.append(
                f"| {family.label} | {generation.label} | {generation.status} | "
                f"{_models_cell(generation)} | {_link('upstream', generation.weights_url)} | "
                f"{_link(generation.license, generation.license_url)} |"
            )
    return "\n".join(rows)


RENDERERS = {
    "model-families": render_model_families,
    "model-roster": render_model_roster,
}


def render_block(name: str, register: Register) -> str:
    """The full marked block, markers included, exactly as it must appear in the document."""
    body = RENDERERS[name](register)
    return f"{_begin(name)}\n\n{body}\n\n{_end(name)}"


def _block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"<!-- generated: {re.escape(name)}[^>]*-->.*?{re.escape(_end(name))}",
        re.DOTALL,
    )


def sync_findings(register: Register, root: Path = PROJECT_ROOT, write: bool = False) -> list[str]:
    """Every generated block that is missing or stale; rewrite them when `write` is set."""
    findings: list[str] = []
    for block in DOC_BLOCKS:
        path = root / block.doc
        text = path.read_text(encoding="utf-8")
        pattern = _block_pattern(block.name)
        rendered = render_block(block.name, register)
        found = pattern.search(text)
        if not found:
            findings.append(
                f"{block.doc}: no `{block.name}` block -- add the generated markers around the table"
            )
            continue
        if found.group(0) == rendered:
            continue
        if not write:
            findings.append(
                f"{block.doc}: `{block.name}` block is stale against {ROSTER_MANIFEST} "
                f"-- run `{SYNC_COMMAND}`"
            )
            continue
        path.write_text(text.replace(found.group(0), rendered), encoding="utf-8")
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="publish the model-family register into the docs")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--check", action="store_true", help="report stale blocks instead of rewriting them"
    )
    args = parser.parse_args(argv)

    register = load_register(args.manifest or (args.root / ROSTER_MANIFEST))
    findings = register_findings(register)
    if findings:
        for finding in findings:
            _LOG.error("ERROR: %s: %s", ROSTER_MANIFEST, finding)
        _LOG.info("[roster-docs] register is inconsistent -- nothing published")
        return 1
    findings = sync_findings(register, root=args.root, write=not args.check)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    verb = "checked" if args.check else "synced"
    _LOG.info("[roster-docs] %s %d block(s), %d finding(s)", verb, len(DOC_BLOCKS), len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
