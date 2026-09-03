"""The additional members a producer may add to a run bundle, and what each one declares.

`persist_run` used to take `Mapping[str, str]`: any name, any bytes. A study design, a power
analysis, and a Markdown table all went in the same way, so a bundle's extra files were readable
only by whoever remembered what a lane wrote. Every additional member now arrives as one of these,
and each says which registered contract validates it -- or that it is a HUMAN report, the one
declared exemption, because a Markdown table rendered for a person has no machine consumer to
protect.

The two structured constructors cover the whole of the current surface: a lane declares a design
before it measures and writes the analysis it took against that design. Both bodies stay the
lane's own (`llb.study-design` / `llb.study-analysis` are envelopes); what the member fixes is
that a reader can tell which of the two it is holding without knowing the lane.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.runs.rows import encode_record
from llb.core.contracts.common import JsonObject
from llb.core.contracts.run_bundle import STUDY_ANALYSIS_SCHEMA_ID, STUDY_DESIGN_SCHEMA_ID

# The one declared exemption: Markdown written for a person to read. It has no record contract
# because it has no machine consumer -- a reader that parsed it would be reading a rendering.
HUMAN_REPORT = "human-report"
HUMAN_REPORT_SUFFIX = ".md"


@dataclass(frozen=True)
class RunMember:
    """One additional file published inside a run bundle, and what makes it readable.

    `schema_id` names the registered contract the content satisfies, or `HUMAN_REPORT` for the
    declared Markdown exemption. There is no third option: that is what stops a new serializer
    from putting an unreadable payload into a bundle by naming a file.
    """

    name: str
    content: str
    schema_id: str


def study_design(name: str, design: Mapping[str, object]) -> RunMember:
    """A study's declared design, written as `llb.study-design`."""
    return _document(name, STUDY_DESIGN_SCHEMA_ID, design)


def study_analysis(name: str, analysis: Mapping[str, object]) -> RunMember:
    """The reading a study took against its design, written as `llb.study-analysis`."""
    return _document(name, STUDY_ANALYSIS_SCHEMA_ID, analysis)


def human_report(name: str, markdown: str) -> RunMember:
    """A Markdown report for a person -- the declared exemption from record contracts."""
    if not name.endswith(HUMAN_REPORT_SUFFIX):
        raise ValueError(f"a human report must be Markdown: {name!r}")
    return RunMember(name=name, content=markdown, schema_id=HUMAN_REPORT)


def table_report(name: str, title: str, table: str) -> RunMember:
    """The fenced ASCII table every benchmark lane renders beside its structured evidence."""
    return human_report(name, f"# {title}\n\n```text\n{table}\n```\n")


def member_problems(
    members: Sequence[RunMember], registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[str, ...]:
    """Every reason one of these members may not enter a bundle, reported together."""
    problems: list[str] = []
    seen: set[str] = set()
    for member in members:
        if Path(member.name).name != member.name or not member.name:
            problems.append(f"{member.name!r}: an additional member must be a plain file name")
        if member.name in seen:
            problems.append(f"{member.name!r}: declared twice")
        seen.add(member.name)
        if member.schema_id == HUMAN_REPORT:
            if not member.name.endswith(HUMAN_REPORT_SUFFIX):
                problems.append(f"{member.name!r}: the human-report exemption covers Markdown only")
            continue
        try:
            registry.definition(member.schema_id)
        except ValueError:
            problems.append(f"{member.name!r}: unregistered contract {member.schema_id!r}")
    return tuple(problems)


def structured_members(members: Sequence[RunMember]) -> tuple[RunMember, ...]:
    """The members bound to a record contract, which are the ones a dataset describes."""
    return tuple(member for member in members if member.schema_id != HUMAN_REPORT)


def _document(name: str, schema_id: str, body: Mapping[str, object]) -> RunMember:
    """One structured member: the body under the field its envelope declares, identity first."""
    encoded: JsonObject = encode_record(schema_id, body)
    return RunMember(
        name=name,
        content=json.dumps(encoded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        schema_id=schema_id,
    )
