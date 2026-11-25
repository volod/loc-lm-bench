"""Front matter, links, and block anchors, read the way a vault host resolves them.

Heading anchors reuse `llb.quality.doc_links`. Block anchors are the `^name`
lines the renderer writes on their own line. Wikilinks and embeds resolve from
the vault root. A relative link resolves from the note that holds it, so a
PDF link may leave the vault.
"""

import re
from pathlib import Path

from llb.prep.book.render.constants import CONTENT_DIRNAME, TOC_FILENAME
from llb.prep.book.validate.types import VaultDoc, VaultLink

_FENCE = "```"
_ANCHOR_LINE = re.compile(r"^\^([A-Za-z][A-Za-z0-9-]*)\s*$")
_PAGE_ANCHOR = re.compile(r"^p(\d+)$")
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_MARKDOWN = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
_EMBED = re.compile(r"!\[\[([^\]|#]+)(#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_WIKI = re.compile(r"(?<!!)\[\[([^\]|#]+)(#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_EXTERNAL = ("http://", "https://", "mailto:")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """The YAML map and the body. A file with no front matter returns an empty map."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 3)
    if end < 0:
        return {}, text
    matter: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        matter[key.strip()] = value.strip()
    return matter, text[end + 5 :]


def matter_int(matter: dict[str, str], key: str) -> int | None:
    """An integer front-matter field, or None when it is missing or not a number."""
    raw = matter.get(key)
    if raw is None:
        return None
    text = raw.strip().strip("'\"")
    if not text.isdigit():
        return None
    return int(text)


def load_docs(book_dir: Path) -> list[VaultDoc]:
    """Every Markdown file under the vault, in path order."""
    docs: list[VaultDoc] = []
    for path in sorted(book_dir.rglob("*.md")):
        if not path.is_file():
            continue
        matter, body = split_front_matter(path.read_text(encoding="utf-8"))
        docs.append(VaultDoc(path.relative_to(book_dir).as_posix(), path, matter, body))
    return docs


def content_docs(docs: list[VaultDoc]) -> list[VaultDoc]:
    """Notes under `content/`, in path order."""
    prefix = f"{CONTENT_DIRNAME}/"
    return [doc for doc in docs if doc.rel.startswith(prefix)]


def toc_doc(docs: list[VaultDoc]) -> VaultDoc | None:
    """The contents note, when the vault has one."""
    for doc in docs:
        if doc.rel == TOC_FILENAME:
            return doc
    return None


def block_anchor_names(body: str) -> list[str]:
    """Block anchors on their own lines, in order, including repeats."""
    names: list[str] = []
    for line in _plain_lines(body):
        matched = _ANCHOR_LINE.match(line)
        if matched:
            names.append(matched.group(1))
    return names


def page_anchor_number(name: str) -> int | None:
    """The PDF page a `p0007` anchor names, or None for any other anchor."""
    matched = _PAGE_ANCHOR.match(name)
    if matched is None:
        return None
    return int(matched.group(1))


def extract_links(doc: VaultDoc, book_dir: Path) -> list[VaultLink]:
    """Every link in one note. Fenced samples are skipped.

    Display math is not a Markdown link. A formula can contain ``](``, which is
    LaTeX, and it can also contain an image or a wikilink the renderer inserted.
    """
    found: list[VaultLink] = []
    for line_no, (line, in_math) in enumerate(_classified_lines(doc.body), 1):
        found.extend(_line_links(doc, book_dir, line_no, line, in_math))
    return found


def _plain_lines(body: str) -> list[str]:
    """Note lines outside fenced samples. Display-math delimiters are not lines."""
    return [line for line, _in_math in _classified_lines(body)]


def _classified_lines(body: str) -> list[tuple[str, bool]]:
    """Those lines, and whether each one sits between ``$$`` delimiters."""
    lines: list[tuple[str, bool]] = []
    fenced = False
    math = False
    for line in body.splitlines():
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.strip() == "$$":
            math = not math
            continue
        lines.append((line, math))
    return lines


def _line_links(
    doc: VaultDoc, book_dir: Path, line_no: int, line: str, in_math: bool
) -> list[VaultLink]:
    found: list[VaultLink] = []
    for match in _IMAGE.finditer(line):
        _keep(found, _relative(doc, line_no, "image", "", match.group(1)))
    if not in_math:
        for match in _MARKDOWN.finditer(line):
            _keep(found, _relative(doc, line_no, "markdown", match.group(1), match.group(2)))
    for match in _EMBED.finditer(line):
        link = _wiki(doc, book_dir, line_no, "embed", "", match.group(1), match.group(2))
        _keep(found, link)
    for match in _WIKI.finditer(line):
        link = _wiki(doc, book_dir, line_no, "wikilink", "", match.group(1), match.group(2))
        _keep(found, link)
    return found


def _keep(found: list[VaultLink], link: VaultLink | None) -> None:
    if link is not None:
        found.append(link)


def _relative(doc: VaultDoc, line_no: int, kind: str, label: str, raw: str) -> VaultLink | None:
    if raw.startswith(_EXTERNAL):
        return None
    target, _, fragment = raw.partition("#")
    located = doc.path if not target else (doc.path.parent / target).resolve()
    return VaultLink(
        doc.rel, doc.path, kind, label, target, fragment, line_no, located, located.is_file()
    )


def _wiki(
    doc: VaultDoc,
    book_dir: Path,
    line_no: int,
    kind: str,
    label: str,
    raw: str,
    fragment_group: str | None,
) -> VaultLink | None:
    target = raw.strip()
    fragment = (fragment_group or "").lstrip("#")
    located = _wiki_path(book_dir, target)
    return VaultLink(
        doc.rel, doc.path, kind, label, target, fragment, line_no, located, located.is_file()
    )


def _wiki_path(book_dir: Path, target: str) -> Path:
    path = Path(target)
    if path.is_absolute() or ".." in path.parts:
        return book_dir / "__missing__"
    direct = book_dir / target
    if direct.is_file():
        return direct.resolve()
    as_note = book_dir / f"{target}.md"
    if as_note.is_file():
        return as_note.resolve()
    if "/" not in target and "\\" not in target:
        matches = [item.resolve() for item in book_dir.rglob(f"{target}.md") if item.is_file()]
        if len(matches) == 1:
            return matches[0]
    return direct.resolve()
