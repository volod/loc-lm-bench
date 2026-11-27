"""Cross-references link only when the inventory names one target."""

import json
import re
from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book_blocks import Block, Region
from llb.core.contracts.data_prep.book_structure import (
    STRUCTURE_KIND,
    STRUCTURE_SCHEMA_ID,
    STRUCTURE_SCHEMA_VERSION,
    DocumentStructure,
    NumberedItem,
    PrintedPage,
)
from llb.core.contracts.data_prep.book_references import BookReferences
from llb.prep.book.project.build import project_corpus
from llb.prep.book.render.catalog import Catalog, Target, build_catalog, canonical
from llb.prep.book.render.grammar import load_grammar, parse_grammar
from llb.prep.book.render.link import link_text
from llb.prep.book.render.names import anchor_map
from llb.prep.book.render.types import VaultFile
from llb.prep.book.structure.text import norm_printed
from llb.prep.book.validate import validate_book
from tests.llb.prep.book.test_render import (
    _adopted,
    _block,
    _config,
    _node,
    _sample,
    _save_page,
    _sha,
    _vault,
    _write_structure,
)

RUNNER = CliRunner()
_PROSE = (
    "See Figs. 2-2 and 2-3, Table 2-1, and Eq. (1-1). "
    "Secs. 1.1 and 1.2.1 sit here. Chapter 1 opens it. See p. 4. "
    "Fig. 1-1 stays plain. See p. 9. Figures of Merit stay words. "
    "Chapter 22.1.5 stays plain.\n"
    "$$\n"
    "Eq. (9-9)\n"
    "$$\n"
)
_INDEX = (
    "orbit, 4\n  elements, 4\nmissing, 9\ndelta ..... 5\nspan, 4-6\n"
    "[{'type': 'text', 'content': 'apsis, 4'}][{'type': 'text', 'content': '  node, 4'}]\n"
)


def test_every_reference_type_resolves_and_the_misses_stay_plain(tmp_path: Path) -> None:
    vault = _vault(_book(tmp_path), tmp_path / "vault")
    body = _content(vault)
    _assert_resolved(body)
    _assert_left_plain(body, _index(vault))
    _assert_lists_and_report(vault)


def _assert_resolved(body: str) -> None:
    assert "Figs. [2-2](" in body
    assert "[2-3](" in body
    assert "Table [2-1](" in body
    assert "Eq. ([1-1](" in body
    assert "Secs. [1.1](" in body
    assert "[1.2.1](" in body
    assert "Chapter [1](" in body
    assert "p. [4](" in body


def _assert_left_plain(body: str, index: str) -> None:
    assert "Fig. 1-1 stays plain" in body
    assert "p. 9" in body
    assert "p. [9]" not in body
    assert "Figures of Merit stay words" in body
    assert "Chapter 22.1.5 stays plain" in body
    assert "Eq. (9-9)" in body
    assert "[4](" in index
    assert "[5](" in index
    assert "missing, 9" in index
    assert "[9](" not in index


def _assert_lists_and_report(vault: Path) -> None:
    for name in ("01_List_of_Figures.md", "02_List_of_Tables.md", "03_List_of_Equations.md"):
        text = (vault / name).read_text(encoding="utf-8")
        assert text.startswith("# ")
        assert "content/" in text
    report = validate_book(vault)
    assert report.ok
    payload = json.loads((vault / "metadata" / "reference_links.json").read_text(encoding="utf-8"))
    assert payload["grammar"] == "default"
    assert "figure 1-1" in payload["ambiguous_keys"]
    assert payload["gaps"] == 2
    assert payload["unresolved"].get("equation", 0) == 0
    assert payload["unresolved"].get("chapter", 0) == 0
    text = (vault / "metadata" / "processing_report.md").read_text(encoding="utf-8")
    assert "Mapping gaps left unlinked: 2." in text
    assert "`figure 1-1`" in text
    assert any("mapping gap" in item for item in report.warnings)


def test_a_split_page_lands_on_the_file_where_it_starts(tmp_path: Path) -> None:
    run = _split_book(tmp_path)
    vault = _vault(run, tmp_path / "vault", section_cap=200)
    notes = sorted((vault / "content").glob("*.md"))
    assert len(notes) == 2
    first = next(path for path in notes if "^p0001" in path.read_text(encoding="utf-8"))
    second = next(path for path in notes if "See p. [7](" in path.read_text(encoding="utf-8"))
    assert first != second
    assert f"{first.name}#^p0001" in second.read_text(encoding="utf-8")
    assert validate_book(vault).ok


def test_a_footer_echo_is_not_a_second_figure() -> None:
    grammar = load_grammar(None)
    single = _figure_structure(("caption", "page-footer"))
    catalog = build_catalog([_anchor_file(single)], single, grammar)
    assert ("figure", canonical("1-1")) in catalog.unique
    doubled = _figure_structure(("caption", "caption"))
    ambiguous = build_catalog([_anchor_file(doubled)], doubled, grammar)
    assert ("figure", canonical("1-1")) in ambiguous.ambiguous
    assert ("figure", canonical("1-1")) not in ambiguous.unique


def test_phrases_that_are_not_citations_stay_untouched() -> None:
    catalog = _phrase_catalog()
    grammar = load_grammar(None)
    linked, events = link_text(
        "From Eqs. (9-4) and (9-5), see p. 4.",
        here="note",
        catalog=catalog,
        grammar=grammar,
        index=False,
    )
    assert "[9-4](note.md#^eq-9-4)" in linked
    assert "[9-5](note.md#^eq-9-5)" in linked
    assert "p. [4](note.md#^p0001)" in linked
    assert [event.outcome for event in events] == ["resolved", "resolved", "resolved"]
    for sample in ("Figures of Merit stay.", "Chapter 22.1.5 stays.", "See [Fig. 9-4](keep.md)."):
        text, found = link_text(sample, here="note", catalog=catalog, grammar=grammar, index=False)
        assert text == sample
        assert found == ()
    math = "$$\nEq. (9-4)\n$$\n"
    assert link_text(math, here="note", catalog=catalog, grammar=grammar, index=False)[1] == ()


def test_a_chapter_range_and_a_scanned_comma_resolve() -> None:
    catalog = _phrase_catalog()
    grammar = load_grammar(None)
    ranged, events = link_text(
        "See Chaps. 3-6 and Chaps. 7\u201310.",
        here="note",
        catalog=catalog,
        grammar=grammar,
        index=False,
    )
    assert "Chaps. [3](note.md#^ch-3)" in ranged
    assert "[6](note.md#^ch-6)" in ranged
    assert "7\u201310" in ranged and "[7]" not in ranged and "[10]" not in ranged
    assert [event.outcome for event in events] == [
        "resolved",
        "resolved",
        "unresolved",
        "unresolved",
    ]
    scanned, found = link_text(
        "See Fig, 1-1 and Eq, (1-1). Fig. 1-1 is one number.",
        here="note",
        catalog=catalog,
        grammar=grammar,
        index=False,
    )
    assert "Fig, [1-1](note.md#^fig-1-1)" in scanned
    assert "Eq, ([1-1](note.md#^eq-1-1))" in scanned
    assert "Fig. [1-1](note.md#^fig-1-1)" in scanned
    assert "[1]" not in scanned
    assert [event.outcome for event in found] == ["resolved", "resolved", "resolved"]
    dotted, dotted_events = link_text(
        "Chapter 22.1.5 stays.",
        here="note",
        catalog=catalog,
        grammar=grammar,
        index=False,
    )
    assert dotted == "Chapter 22.1.5 stays."
    assert dotted_events == ()


def test_a_book_grammar_replaces_the_default_cues(tmp_path: Path) -> None:
    path = tmp_path / "grammar.yaml"
    path.write_text(
        "figure:\n  cues: [plate, plates]\n  number: '\\d+'\n"
        "page:\n  cues: [p.]\n"
        "index:\n  titles: [index]\n",
        encoding="utf-8",
    )
    run = _plate_book(tmp_path, path)
    body = _content(_vault(run, tmp_path / "vault", reference_grammar=path))
    assert "Plate [3](" in body
    assert "Fig. 3" in body and "Fig. [3]" not in body


def test_a_bad_grammar_is_refused(tmp_path: Path) -> None:
    try:
        parse_grammar("nope: [1]\n", "bad")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown key was accepted")
    missing = tmp_path / "missing.yaml"
    run = _sample(tmp_path)
    refused = RUNNER.invoke(
        app,
        [
            "book-ocr",
            "render",
            str(run),
            "--output",
            str(tmp_path / "out"),
            "--reference-grammar",
            str(missing),
        ],
    )
    assert refused.exit_code == 2
    assert "reference grammar" in refused.stderr


_LINK = re.compile(r"\[([^\[\]\n]+)\]\(([^)#\s]+)#\^([A-Za-z0-9-]+)\)")


def test_the_sidecar_names_the_citations_the_vault_links(tmp_path: Path) -> None:
    run = _book(tmp_path)
    vault = _vault(run, tmp_path / "vault")
    out = tmp_path / "corpus"
    project_corpus(run, out, "redistributable", language="en")
    first = _bytes(out)
    again = project_corpus(run, out, "redistributable", language="en")
    assert again.revisions == 0
    assert _bytes(out) == first
    report = json.loads((vault / "metadata" / "reference_links.json").read_text(encoding="utf-8"))
    citations = _sidecars(out)
    resolved = Counter(_identity(out, item) for item in citations if item.outcome == "resolved")
    assert resolved == _vault_links(vault)
    assert {item.key for item in citations if item.reason == "ambiguous"} == set(
        report["ambiguous_keys"]
    )
    assert sum(item.reason == "gap" for item in citations) == report["gaps"]
    assert all(item.target is None for item in citations if item.outcome != "resolved")
    _assert_index(out)


def test_projection_uses_the_vault_grammar(tmp_path: Path) -> None:
    path = tmp_path / "grammar.yaml"
    path.write_text(
        "figure:\n  cues: [plate, plates]\n  number: '\\d+'\n"
        "page:\n  cues: [p.]\n"
        "index:\n  titles: [index]\n",
        encoding="utf-8",
    )
    run = _plate_book(tmp_path, path)
    custom = tmp_path / "custom"
    project_corpus(run, custom, "redistributable", language="en", reference_grammar=path)
    assert _cue_context(custom) == ["Plate 3"]
    plain = tmp_path / "plain"
    project_corpus(run, plain, "redistributable", language="en")
    assert [item.strip() for item in _cue_context(plain)] == ["Fig. 3"]


def _cue_context(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sidecar = path.with_name(path.name + ".references.json")
        record = BookReferences.model_validate_json(sidecar.read_text(encoding="utf-8"))
        for item in record.citations:
            found.append(text[max(0, item.cue_start - 6) : item.cue_end])
    return found


def _bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _sidecars(root: Path) -> list:
    found = []
    for path in sorted(root.glob("*.md.references.json")):
        record = BookReferences.model_validate_json(path.read_text(encoding="utf-8"))
        text = (root / record.doc_id).read_text(encoding="utf-8")
        assert record.hierarchy and record.hierarchy[0].level == "book"
        assert "See Figs" not in path.read_text(encoding="utf-8")
        for item in record.citations:
            assert text[item.cue_start : item.cue_end].strip()
            if item.target is not None:
                target = (root / item.target.doc_id).read_text(encoding="utf-8")
                assert target[item.target.char_start : item.target.char_end].strip()
                _block_span(root, item)
        for entry in record.index:
            assert text[entry.term_start : entry.term_end].strip()
            if entry.parent_start is not None and entry.parent_end is not None:
                assert text[entry.parent_start : entry.parent_end].strip()
        found.extend(record.citations)
    assert found
    return found


def _block_span(root: Path, item) -> tuple[int, str]:
    payload = json.loads(
        (root / Path(item.target.doc_id).with_suffix(".citations.json")).read_text(encoding="utf-8")
    )
    for page in payload["pages"]:
        for block in page["blocks"]:
            if block["block_id"] != item.target.block_id:
                continue
            assert block["char_start"] == item.target.char_start
            assert block["char_end"] == item.target.char_end
            return page["page"], block["vault_anchor"]
    raise AssertionError(item.target.block_id)


def _identity(root: Path, item) -> tuple:
    pdf_page, anchor = _block_span(root, item)
    if item.kind in {"page", "index"}:
        return item.key, ("pdf", pdf_page)
    return item.key, ("anchor", anchor)


def _vault_links(vault: Path) -> Counter:
    counts: Counter = Counter()
    for path in sorted((vault / "content").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        index_note = "orbit," in text
        for match in _LINK.finditer(text):
            token, anchor = match.group(1), match.group(3)
            kind = _link_kind(anchor, index_note)
            number = norm_printed(token) if kind in {"page", "index"} else canonical(token)
            if re.fullmatch(r"p\d+", anchor):
                ident = ("pdf", int(anchor[1:]))
            else:
                ident = ("anchor", anchor)
            counts[(f"{kind} {number}", ident)] += 1
    return counts


def _link_kind(anchor: str, index_note: bool) -> str:
    if anchor.startswith("fig-"):
        return "figure"
    if anchor.startswith("tbl-"):
        return "table"
    if anchor.startswith("eq-"):
        return "equation"
    if anchor.startswith("ch-"):
        return "chapter"
    if anchor.startswith(("sec-", "sub-")):
        return "section"
    if re.fullmatch(r"p\d+", anchor):
        return "index" if index_note else "page"
    raise AssertionError(anchor)


def _assert_index(root: Path) -> None:
    found = {"elements": False, "span": False, "node": False}
    for path in root.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        record = BookReferences.model_validate_json(
            path.with_name(path.name + ".references.json").read_text(encoding="utf-8")
        )
        for entry in record.index:
            _assert_index_entry(text, entry, found)
    assert found["elements"] and found["span"] and found["node"]


def _assert_index_entry(text: str, entry, found: dict[str, bool]) -> None:
    term = text[entry.term_start : entry.term_end]
    if term == "elements":
        _assert_parent(text, entry, "orbit")
        found["elements"] = True
    elif term == "span":
        assert {page.pdf_page for page in entry.pages} == {1, 2, 4}
        found["span"] = True
    elif term == "missing":
        assert entry.unresolved == ["9"] and entry.pages == []
    elif term == "node":
        _assert_parent(text, entry, "apsis")
        found["node"] = True


def _assert_parent(text: str, entry, parent: str) -> None:
    assert entry.parent_start is not None and entry.parent_end is not None
    assert text[entry.parent_start : entry.parent_end] == parent


def _book(tmp: Path) -> Path:
    run = tmp / "run"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n%links\n")
    pages = [1, 2, 3, 4]
    _config(run, pdf, pages)
    blocks = {
        1: [
            _block(1, 0, "heading", "1 Introduction", 40),
            _block(1, 1, "heading", "1.1 Purpose", 120),
            _block(1, 2, "body", _PROSE, 200),
        ],
        2: [
            _block(2, 0, "heading", "1.2.1 Detail", 40),
            _block(2, 1, "caption", "Figure 2-2 Orbit", 120),
            _block(2, 2, "caption", "Figure 2-3 Path", 200),
            _block(2, 3, "caption", "Table 2-1 Costs", 280),
            _block(2, 4, "formula", "E=mc", 360),
            _block(2, 5, "caption", "Figure 1-1 First", 440),
        ],
        3: [_block(3, 0, "caption", "Figure 1-1 Second", 40)],
        4: [_block(4, 0, "body", _INDEX, 40)],
    }
    for pdf_page, page_blocks in blocks.items():
        _save_page(run, 4, page_blocks, [_adopted(item) for item in page_blocks], False)
    chapter = _node("book/chapter-1", "chapter", "1", "Introduction", 1, 3, "p0001-b0000")
    section = _node("book/chapter-1/section-1-1", "section", "1.1", "Purpose", 1, 1, "p0001-b0001")
    subsection = _node(
        "book/chapter-1/subsection-1-2-1", "subsection", "1.2.1", "Detail", 2, 2, "p0002-b0000"
    )
    index = _node("book/index", "chapter", "", "Index", 4, 4, None)
    chapter = chapter.model_copy(update={"children": [section, subsection]})
    root = _node("book", "book", "", "Book", 1, 4, None, [chapter, index])
    _write_structure(
        run,
        _sha(pdf),
        [
            _label(1, "4", "observed"),
            _label(2, "5", "observed"),
            _label(3, "9", "gap"),
            _label(4, "6", "observed"),
        ],
        root,
        [
            _numbered("figure", "2-2", 2, "p0002-b0001"),
            _numbered("figure", "2-3", 2, "p0002-b0002"),
            _numbered("table", "2-1", 2, "p0002-b0003"),
            _numbered("equation", "1-1", 2, "p0002-b0004", "display-number"),
            _numbered("figure", "1-1", 2, "p0002-b0005"),
            _numbered("figure", "1-1", 3, "p0003-b0000"),
        ],
    )
    return run


def _split_book(tmp: Path) -> Path:
    run = tmp / "split"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n%split\n")
    _config(run, pdf, [1])
    blocks = [
        _block(1, 0, "body", "A" * 195, 40),
        _moved(_block(1, 1, "body", "See p. 7.", 200)),
    ]
    _save_page(run, 1, blocks, [_adopted(item) for item in blocks], False)
    root = _node(
        "book",
        "book",
        "",
        "Book",
        1,
        1,
        None,
        [_node("book/chapter-1", "chapter", "1", "Wide", 1, 1, None)],
    )
    _write_structure(run, _sha(pdf), [_label(1, "7", "observed")], root, [])
    return run


def _plate_book(tmp: Path, _grammar: Path) -> Path:
    run = tmp / "plate"
    pdf = run / "source" / "book.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n%plate\n")
    _config(run, pdf, [1])
    blocks = [_block(1, 0, "body", "See Plate 3. See Fig. 3.", 40)]
    _save_page(run, 1, blocks, [_adopted(item) for item in blocks], False)
    root = _node("book", "book", "", "Book", 1, 1, None)
    _write_structure(
        run,
        _sha(pdf),
        [_label(1, "1", "observed")],
        root,
        [_numbered("figure", "3", 1, "p0001-b0000")],
    )
    return run


def _figure_structure(evidence: tuple[str, str]) -> DocumentStructure:
    items = [
        _numbered("figure", "1-1", 1, "p0001-b0000", evidence[0]),
        _numbered("figure", "1-1", 1, "p0001-b0001", evidence[1]),
    ]
    root = _node("book", "book", "", "Book", 1, 1, None)
    return DocumentStructure(
        schema_id=STRUCTURE_SCHEMA_ID,
        schema_version=STRUCTURE_SCHEMA_VERSION,
        kind=STRUCTURE_KIND,
        source_sha256="a" * 64,
        pages=[_label(1, "1", "observed")],
        hierarchy=root,
        inventory=items,
    )


def _anchor_file(structure: DocumentStructure) -> VaultFile:
    names = "\n".join(f"^{name}" for name in anchor_map(structure).values())
    node = structure.hierarchy
    return VaultFile("note", node, [], 1, 1, 0, names + "\n^p0001\n", 0, True)


def _phrase_catalog() -> Catalog:
    return Catalog(
        unique={
            ("equation", "9-4"): Target("eq-9-4", "note"),
            ("equation", "9-5"): Target("eq-9-5", "note"),
            ("equation", "1-1"): Target("eq-1-1", "note"),
            ("figure", "1-1"): Target("fig-1-1", "note"),
            ("chapter", "3"): Target("ch-3", "note"),
            ("chapter", "6"): Target("ch-6", "note"),
        },
        ambiguous=frozenset({("figure", "9-4")}),
        pages={"4": Target("p0001", "note")},
        page_ambiguous=frozenset(),
        gaps=frozenset({"9"}),
        rows={},
        index_stems=frozenset(),
    )


def _content(vault: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((vault / "content").glob("*.md"))
    )


def _index(vault: Path) -> str:
    notes = [
        path
        for path in (vault / "content").glob("*.md")
        if "orbit," in path.read_text(encoding="utf-8")
    ]
    assert len(notes) == 1
    return notes[0].read_text(encoding="utf-8")


def _label(pdf: int, printed: str, status: str) -> PrintedPage:
    return PrintedPage(
        pdf_page=pdf,
        page_id=f"p{pdf:04d}",
        printed=printed,
        scheme="arabic",
        status=status,  # type: ignore[arg-type]
        block_id=None,
        fallback=status == "gap",
    )


def _numbered(
    kind: str, number: str, pdf: int, block_id: str, evidence: str = "caption"
) -> NumberedItem:
    return NumberedItem(
        kind=kind,  # type: ignore[arg-type]
        number=number,
        pdf_page=pdf,
        printed=None,
        block_id=block_id,
        evidence=evidence,  # type: ignore[arg-type]
    )


def _moved(block: Block) -> Block:
    region = Region(x0=400, y0=200, x1=700, y1=228, width=800, height=1200)
    return block.model_copy(update={"region": region})
