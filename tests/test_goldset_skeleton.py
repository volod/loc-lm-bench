import json

import pytest

from llb.prep.goldset_skeleton import create_skeleton
from llb.prep.ingest_squad import load_squad_json, squad_to_gold


def test_create_skeleton_writes_importable_squad_template(tmp_path):
    out = create_skeleton(tmp_path, "20260101T000000Z")
    source = out / "squad_goldset.json"
    records = load_squad_json(source)
    _docs, items, skipped = squad_to_gold(records)
    assert len(items) == 1 and skipped == 0
    assert items[0].verified is False
    assert (out / "README.txt").read_text(encoding="ascii").isascii()


def test_create_skeleton_refuses_to_overwrite_run(tmp_path):
    create_skeleton(tmp_path, "same")
    with pytest.raises(FileExistsError):
        create_skeleton(tmp_path, "same")


def test_skeleton_json_is_pretty_and_utf8(tmp_path):
    out = create_skeleton(tmp_path, "run")
    data = json.loads((out / "squad_goldset.json").read_text(encoding="utf-8"))
    assert data["data"][0]["paragraphs"][0]["context"].startswith("Київ")
