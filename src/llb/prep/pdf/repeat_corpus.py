"""Apply intra-document repeat handling to a whole converted corpus, and carry its labels along.

The conversion lane applies `llb.prep.pdf.repeats` per document while it renders; this module is
the same handling for a corpus that is ALREADY converted -- the common case, since an operator's
`_md` root outlives the source PDFs. It rewrites into a NEW root (never in place), remaps the
page-citation sidecars so `page` chunking still lines up, and remaps a gold set so the same
questions can be scored on the stripped corpus without re-drafting them.

A gold span inside a dropped copy lands on the surviving copy of the identical text; a span that
straddles a rewrite has no image and its item is dropped from the remapped gold set and named in
the report, because scoring an item whose evidence offsets moved would measure the remap, not
retrieval. Every kept span is re-read out of the stripped document and must still equal its
labeled text, so a remap that is off by one character fails loudly instead of scoring garbage.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.prep.pdf.model import PDF_CITATION_SUFFIX
from llb.prep.pdf.repeats import (
    DEFAULT_MIN_REPEATS,
    REPEAT_KEEP,
    RepeatCensus,
    TextEdit,
    remap_span,
    rewrite_repeated_blocks,
    span_rehomed,
)

CORPUS_SUFFIXES = (".md", ".txt")
REPEAT_REPORT_NAME = "repeat_strip.json"


class DocumentRepeats(TypedDict):
    """One document's block-repeat census and the size change its rewrite caused."""

    doc_id: str
    census: RepeatCensus
    chars_before: int
    chars_after: int


class GoldsetRemap(TypedDict):
    """How a gold set survived the rewrite."""

    items: int
    remapped: int
    dropped: list[str]  # ids whose spans straddle a rewrite and can no longer be anchored
    rehomed: list[str]  # remapped ids whose evidence sat on a dropped copy (now on the survivor)


class CorpusRepeatReport(TypedDict):
    """Corpus-level census plus, for a rewriting mode, what the rewrite did."""

    kind: str
    mode: str
    min_repeats: int
    corpus_root: str
    out_root: str | None
    documents: list[DocumentRepeats]
    groups: int
    handled_groups: int
    handled_blocks: int
    chars_before: int
    chars_after: int
    goldset: GoldsetRemap | None


class _Stripped(TypedDict):
    """Per-document rewrite state the goldset and citation remaps both read."""

    text: str
    edits: list[TextEdit]


def strip_corpus_repeats(
    corpus_root: Path | str,
    out_root: Path | str | None = None,
    *,
    mode: str = REPEAT_KEEP,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    goldset: Path | str | None = None,
    goldset_out: Path | str | None = None,
) -> CorpusRepeatReport:
    """Census (and, for a rewriting mode, rewrite) a converted corpus into `out_root`.

    With `mode='keep'` -- or no `out_root` -- nothing is written except the returned report: the
    census is the deliverable and the corpus is left untouched.
    """
    root = Path(corpus_root)
    if not root.is_dir():
        raise ValueError(f"corpus root does not exist: {root}")
    target = Path(out_root) if out_root is not None and mode != REPEAT_KEEP else None
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
    documents: list[DocumentRepeats] = []
    stripped: dict[str, _Stripped] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        doc_id = path.relative_to(root).as_posix()
        if path.suffix.lower() not in CORPUS_SUFFIXES:
            _write_sidecar(path, target, doc_id, stripped)
            continue
        text = path.read_text(encoding="utf-8")
        rewrite = rewrite_repeated_blocks(text, mode=mode, min_repeats=min_repeats)
        stripped[doc_id] = {"text": rewrite.text, "edits": rewrite.edits}
        documents.append(
            {
                "doc_id": doc_id,
                "census": rewrite.census,
                "chars_before": len(text),
                "chars_after": len(rewrite.text),
            }
        )
        _write_doc(target, doc_id, rewrite.text)
    report = _report(root, target, mode, min_repeats, documents)
    report["goldset"] = _remap_goldset(goldset, goldset_out, stripped)
    if target is not None:
        (target / REPEAT_REPORT_NAME).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def format_repeat_report(report: CorpusRepeatReport) -> str:
    """ASCII summary lines for the CLI (AGENTS.md: ASCII-only)."""
    lines = [
        f"[strip-corpus-repeats] mode={report['mode']} min-repeats={report['min_repeats']} "
        f"over {len(report['documents'])} documents",
        f"  repeated block groups: {report['groups']} "
        f"({report['handled_groups']} eligible, {report['handled_blocks']} blocks rewritten), "
        f"{report['chars_before']} -> {report['chars_after']} chars",
    ]
    for document in report["documents"]:
        census = document["census"]
        if not census["groups"]:
            continue
        lines.append(
            f"  {document['doc_id']}: {census['groups']} repeated groups "
            f"({census['handled_groups']} eligible) over {census['blocks']} blocks, "
            f"largest {census['largest_group']} copies, "
            f"{document['chars_before']} -> {document['chars_after']} chars"
        )
    goldset = report["goldset"]
    if goldset is not None:
        dropped = f", dropped {', '.join(goldset['dropped'])}" if goldset["dropped"] else ""
        rehomed = (
            f", {len(goldset['rehomed'])} re-homed onto a survivor" if goldset["rehomed"] else ""
        )
        lines.append(
            f"  goldset: {goldset['remapped']}/{goldset['items']} items remapped{rehomed}{dropped}"
        )
    return "\n".join(lines)


def remap_citation_pages(
    pages: list[dict[str, Any]], edits: list[TextEdit]
) -> list[dict[str, Any]]:
    """Page-citation records whose offsets follow the rewrite; a fully removed page is dropped."""
    remapped: list[dict[str, Any]] = []
    for page in pages:
        text_span = remap_span(edits, int(page["text_start"]), int(page["text_end"]))
        char_span = remap_span(edits, int(page["char_start"]), int(page["char_end"]))
        if text_span is None or char_span is None:
            continue
        remapped.append(
            {
                **page,
                "char_start": char_span[0],
                "char_end": char_span[1],
                "text_start": text_span[0],
                "text_end": text_span[1],
                "n_chars": text_span[1] - text_span[0],
                "blocks": _remap_blocks(page.get("blocks") or [], edits),
            }
        )
    return remapped


def _remap_blocks(blocks: list[dict[str, Any]], edits: list[TextEdit]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for block in blocks:
        start, end = block.get("char_start"), block.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int):
            kept.append(dict(block))
            continue
        span = remap_span(edits, start, end)
        if span is not None:
            kept.append({**block, "char_start": span[0], "char_end": span[1]})
    return kept


def _write_doc(target: Path | None, doc_id: str, text: str) -> None:
    if target is None:
        return
    out_path = target / doc_id
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def _write_sidecar(
    path: Path, target: Path | None, doc_id: str, stripped: dict[str, _Stripped]
) -> None:
    """Copy a non-corpus file into the target, remapping a page-citation sidecar on the way.

    Sidecars are read AFTER their document because the corpus walk is sorted and
    `<doc>.citations.json` sorts after `<doc>.md`; a sidecar whose document was not rewritten
    (or does not exist) is copied through unchanged.
    """
    if target is None:
        return
    out_path = target / doc_id
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not doc_id.endswith(PDF_CITATION_SUFFIX):
        shutil.copy2(path, out_path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    edits = stripped.get(str(payload.get("doc_id", "")), {"text": "", "edits": []})["edits"]
    payload["pages"] = remap_citation_pages(payload.get("pages") or [], edits)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _report(
    root: Path, target: Path | None, mode: str, min_repeats: int, documents: list[DocumentRepeats]
) -> CorpusRepeatReport:
    return {
        "kind": "corpus-repeat-strip",
        "mode": mode,
        "min_repeats": min_repeats,
        "corpus_root": str(root),
        "out_root": str(target) if target is not None else None,
        "documents": documents,
        "groups": sum(document["census"]["groups"] for document in documents),
        "handled_groups": sum(document["census"]["handled_groups"] for document in documents),
        "handled_blocks": sum(document["census"]["handled_blocks"] for document in documents),
        "chars_before": sum(document["chars_before"] for document in documents),
        "chars_after": sum(document["chars_after"] for document in documents),
        "goldset": None,
    }


def _remap_goldset(
    goldset: Path | str | None,
    goldset_out: Path | str | None,
    stripped: dict[str, _Stripped],
) -> GoldsetRemap | None:
    """Rewrite each item's span offsets onto the stripped corpus; drop what cannot be anchored."""
    if goldset is None:
        return None
    items = load_goldset(goldset)
    kept: list[GoldItem] = []
    dropped: list[str] = []
    rehomed: list[str] = []
    for item in items:
        remapped = [_remap_gold_span(span, stripped.get(span.doc_id)) for span in item.source_spans]
        moved = [span for span, _ in remapped if span is not None]
        if len(moved) != len(item.source_spans):
            dropped.append(item.id)
            continue
        if any(was_rehomed for _, was_rehomed in remapped):
            rehomed.append(item.id)
        kept.append(item.model_copy(update={"source_spans": moved}))
    if goldset_out is not None:
        out = Path(goldset_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(item.model_dump_json() + "\n" for item in kept), encoding="utf-8")
    return {"items": len(items), "remapped": len(kept), "dropped": dropped, "rehomed": rehomed}


def _remap_gold_span(
    span: SourceSpan, stripped: _Stripped | None
) -> tuple[SourceSpan | None, bool]:
    """The span's offsets in the stripped document (and whether it was re-homed onto a survivor).

    The offsets are verified against the stripped text itself, so a remap that is off by one
    character reads as unanchorable rather than scoring the wrong words.
    """
    if stripped is None:
        return None, False
    if not stripped["edits"]:
        return span, False
    rehomed = span_rehomed(stripped["edits"], span.char_start, span.char_end)
    moved = remap_span(stripped["edits"], span.char_start, span.char_end)
    if moved is None or stripped["text"][moved[0] : moved[1]] != span.text:
        return None, False
    return span.model_copy(update={"char_start": moved[0], "char_end": moved[1]}), rehomed
