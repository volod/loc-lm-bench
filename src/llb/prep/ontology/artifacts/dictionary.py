"""Source-backed prompt-dictionary candidate construction."""

from typing import Any

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.artifacts.citations import span_page_refs
from llb.prep.ontology.constants import PROMPT_DICTIONARY_MAX_EXAMPLES
from llb.prep.ontology.models import DocExtraction


def build_prompt_dictionary_candidates(
    extractions: list[DocExtraction], index: dict[str, dict[str, Any]]
) -> list[dict[str, object]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for extraction in extractions:
        for entity in extraction.entities:
            for mention in entity.mentions:
                _add_candidate(
                    rows,
                    kind="entity",
                    term=entity.name,
                    ontology_type=entity.type,
                    span=mention,
                    index=index,
                    aliases=entity.aliases,
                )
        for fact in extraction.facts:
            _add_candidate(
                rows,
                kind="relation",
                term=fact.relation,
                ontology_type="relation",
                span=fact.evidence,
                index=index,
                subject=fact.subject,
                object_=fact.object,
            )
    output: list[dict[str, object]] = []
    for row in rows.values():
        docs = row.pop("_docs")
        row["doc_count"] = len(docs)
        if "subjects" in row:
            row["subjects"] = sorted(row["subjects"])
        if "objects" in row:
            row["objects"] = sorted(row["objects"])
        output.append(row)
    output.sort(
        key=lambda row: (
            -_row_int(row, "support_count"),
            -_row_int(row, "doc_count"),
            str(row["kind"]),
            str(row["term"]).casefold(),
        )
    )
    return output


def _add_candidate(
    rows: dict[tuple[str, str, str], dict[str, Any]],
    *,
    kind: str,
    term: str,
    ontology_type: str,
    span: SourceSpan,
    index: dict[str, dict[str, Any]],
    aliases: list[str] | None = None,
    subject: str | None = None,
    object_: str | None = None,
) -> None:
    clean_term = " ".join(term.split())
    if not clean_term:
        return
    key = (kind, ontology_type, clean_term.casefold())
    row = rows.setdefault(
        key,
        {
            "term": clean_term,
            "kind": kind,
            "ontology_type": ontology_type,
            "support_count": 0,
            "doc_count": 0,
            "aliases": [],
            "examples": [],
        },
    )
    row["support_count"] += 1
    row.setdefault("_docs", set()).add(span.doc_id)
    if subject is not None:
        row.setdefault("subjects", set()).add(subject)
    if object_ is not None:
        row.setdefault("objects", set()).add(object_)
    for alias in aliases or []:
        if alias and alias not in row["aliases"]:
            row["aliases"].append(alias)
    if len(row["examples"]) < PROMPT_DICTIONARY_MAX_EXAMPLES:
        row["examples"].append(
            {
                "doc_id": span.doc_id,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "text": span.text,
                "pdf_pages": span_page_refs(span, index),
            }
        )


def _row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) else 0
