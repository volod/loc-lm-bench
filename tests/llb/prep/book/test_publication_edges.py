"""Failure points and section-cap cases beside the long publication audit."""

from pathlib import Path
import json

import pytest

from llb.prep.book.project.build import project_corpus
from llb.prep.book.render import swap
from llb.prep.book.render.publish import publish
from llb.prep.book.review.markers import marker_layout_size, render_marker
from tests.llb.prep.book._stages import SimulatedKill
from tests.llb.prep.book.test_render import _CAP, _WORDS, _oversized, _vault


def test_swap_restores_previous_vault_before_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    partial = tmp_path / ".vault.partial"
    previous = tmp_path / ".vault.trash"
    publish(vault, {"old.md": b"old"}, {})
    replace = swap.os.replace

    def interrupt(source: Path, target: Path) -> None:
        if source == partial and target == vault:
            raise SimulatedKill()
        replace(source, target)

    monkeypatch.setattr(swap, "_exchange", lambda *_: False)
    with monkeypatch.context() as patch:
        patch.setattr(swap.os, "replace", interrupt)
        with pytest.raises(SimulatedKill):
            publish(vault, {"new.md": b"new"}, {})
    assert not vault.exists()
    assert (previous / "old.md").read_bytes() == b"old"

    # A retry that fails during staging must still leave the old complete vault readable.
    with pytest.raises(FileNotFoundError):
        publish(vault, {"new.md": tmp_path / "missing"}, {})
    assert (vault / "old.md").read_bytes() == b"old"
    assert not partial.exists() and not previous.exists()
    publish(vault, {"new.md": b"new"}, {})
    assert (vault / "new.md").read_bytes() == b"new"
    assert not (vault / "old.md").exists()


def test_oversized_paragraph_projects_each_piece_once(tmp_path: Path) -> None:
    run = _oversized(tmp_path)
    vault = _vault(run, tmp_path / "vault", section_cap=_CAP)
    out = tmp_path / "corpus"
    project_corpus(run, out, "redistributable", section_cap=_CAP)
    projected = {}
    for cite in out.glob("*.citations.json"):
        payload = json.loads(cite.read_bytes())
        text = (out / payload["doc_id"]).read_text()
        for block in payload["pages"][0]["blocks"]:
            piece = text[block["char_start"] : block["char_end"]]
            if piece.startswith("WORD"):
                projected[block["vault_file"]] = piece
    pieces = []
    for note in sorted((vault / "content").glob("*.md")):
        lines = [line for line in note.read_text().splitlines() if line.startswith("WORD")]
        if lines:
            pieces.extend(lines)
            assert projected[f"content/{note.name}"] in lines
    assert " ".join(pieces) == _WORDS
    assert sum(path.read_text().count("WORD") for path in out.glob("*.md")) == _WORDS.count("WORD")


def test_marker_size_does_not_depend_on_the_vault_pdf_path() -> None:
    near = render_marker("unreadable", "#page=3", "p0003-b0001", 2, 3, None)
    far = render_marker(
        "unreadable", "../../scan-with-a-long-name.pdf#page=3", "p0003-b0001", 2, 3, None
    )
    assert marker_layout_size(near) == marker_layout_size(far)
