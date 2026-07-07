"""Plain-text rendering for curated inventory coverage slices.

NotebookLM is better at grounding over uploaded source files than at consuming a large JSON blob
inside the chat prompt. This module converts the prompt-01 inventory slice for one document into a
compact text source that can be uploaded beside the staged corpus files and referenced by name from
prompt 02.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llb.fsutil import atomic_write_text
from llb.prep.curation.common import load_json_documents

Formatter = Callable[[Any], list[str]]


@dataclass(frozen=True)
class CoverageTextResult:
    """Result metadata for a rendered coverage-plan text file."""

    path: Path
    documents: int
    cross_document_links: int


@dataclass(frozen=True)
class ListSection:
    """Mapping from one inventory list field to its text heading and row formatter."""

    heading: str
    field: str
    formatter: Formatter


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _quote(value: Any) -> str:
    text = _clean(value)
    return f'"{text}"' if text else '""'


def _format_topic(value: Any) -> list[str]:
    text = _clean(value)
    return [f"- {text}"] if text else []


def _format_entity(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return _format_topic(value)
    return [
        f"- name: {_clean(value.get('name'))}",
        f"  type: {_clean(value.get('type'))}",
        f"  mentions: {_clean(value.get('mentions'))}",
        f"  quote: {_quote(value.get('quote'))}",
    ]


def _format_relation(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return _format_topic(value)
    return [
        f"- subject: {_clean(value.get('subject'))}",
        f"  relation: {_clean(value.get('relation'))}",
        f"  object: {_clean(value.get('object'))}",
        f"  quote: {_quote(value.get('quote'))}",
    ]


def _format_numeric_fact(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return _format_topic(value)
    return [
        f"- fact: {_clean(value.get('fact'))}",
        f"  quote: {_quote(value.get('quote'))}",
    ]


def _format_cross_document(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return _format_topic(value)
    docs = value.get("docs") if isinstance(value.get("docs"), list) else []
    return [
        f"- entity_or_topic: {_clean(value.get('entity_or_topic'))}",
        f"  docs: {', '.join(_clean(doc) for doc in docs if _clean(doc))}",
        f"  note: {_clean(value.get('note'))}",
    ]


_DOCUMENT_SECTIONS = (
    ListSection("Topics", "topics", _format_topic),
    ListSection("Entities", "entities", _format_entity),
    ListSection("Relations", "relations", _format_relation),
    ListSection("Numeric facts", "numeric_facts", _format_numeric_fact),
    ListSection("Sensitive topics", "sensitive_topics", _format_topic),
)


def _render_section(lines: list[str], section: ListSection, rows: Any) -> None:
    values = rows if isinstance(rows, list) else []
    if not values:
        return
    lines.extend(("", f"{section.heading}:"))
    for value in values:
        lines.extend(section.formatter(value))


def _render_document(lines: list[str], raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    lines.extend(("", f"Document: {_clean(raw.get('doc'))}"))
    for section in _DOCUMENT_SECTIONS:
        _render_section(lines, section, raw.get(section.field))


def _coverage_object(path: Path) -> dict[str, Any]:
    values = load_json_documents(path)
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError(f"{path}: expected one coverage-plan JSON object")
    return values[0]


def coverage_plan_to_text(coverage: dict[str, Any]) -> str:
    """Render a prompt-01 inventory slice as a NotebookLM-friendly plain-text source."""
    documents = coverage.get("documents") if isinstance(coverage.get("documents"), list) else []
    cross = (
        coverage.get("cross_document")
        if isinstance(coverage.get("cross_document"), list)
        else []
    )
    lines = [
        "Coverage plan",
        "",
        "Use this source as the coverage map for goldset drafting.",
        "Draft questions only from the uploaded staged document text.",
    ]
    for raw in documents:
        _render_document(lines, raw)
    if cross:
        lines.extend(("", "Cross-document links:"))
        for raw in cross:
            lines.extend(_format_cross_document(raw))
    return "\n".join(lines).rstrip() + "\n"


def default_coverage_text_path(path: Path) -> Path:
    """Default output path for a coverage JSON slice."""
    return path.with_suffix(".txt")


def write_coverage_plan_text(input_path: Path, output_path: Path | None = None) -> CoverageTextResult:
    """Load a coverage JSON slice and atomically write its text rendering."""
    input_path = Path(input_path)
    output = Path(output_path) if output_path is not None else default_coverage_text_path(input_path)
    coverage = _coverage_object(input_path)
    text = coverage_plan_to_text(coverage)
    atomic_write_text(output, text)
    documents = coverage.get("documents") if isinstance(coverage.get("documents"), list) else []
    cross = (
        coverage.get("cross_document")
        if isinstance(coverage.get("cross_document"), list)
        else []
    )
    return CoverageTextResult(path=output, documents=len(documents), cross_document_links=len(cross))
