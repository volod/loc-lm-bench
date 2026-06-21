from pathlib import Path

from llb.goldset.schema import load_goldset
from llb.goldset.validate import validate_items
from llb.prep.gen_rag_items import main

REPO = Path(__file__).resolve().parents[1]


def test_gen_runs_and_validates(tmp_path):
    spec = REPO / "samples" / "rag_items_uk.json"
    assert main(["--spec", str(spec), "--out-dir", str(tmp_path)]) == 0

    items = load_goldset(tmp_path / "goldset" / "sample_rag_items.jsonl")
    assert len(items) == 6
    assert all(item.verified for item in items)
    assert any(item.split == "final" for item in items)
    assert validate_items(items, tmp_path / "corpus")["errors"] == []

    corpus_path = tmp_path / "corpus" / "sample" / "memo.txt"
    corpus_mtime = corpus_path.stat().st_mtime_ns
    assert main(["--spec", str(spec), "--out-dir", str(tmp_path)]) == 0
    assert corpus_path.stat().st_mtime_ns == corpus_mtime
