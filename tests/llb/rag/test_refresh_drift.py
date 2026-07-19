"""Drift report: old-vs-new retrieval validation, the re-tune flag, and persistence."""

import json

from llb.goldset.schema import GoldItem, SourceSpan
from llb.rag.refresh.diff import ManifestDiff
from llb.rag.refresh.drift import (
    DRIFT_JSON,
    DRIFT_REPORT_MD,
    measure_drift,
    render_drift_report,
    write_drift_report,
)

DOC = "Тарас Шевченко народився у селі Моринці."


class FixedStore:
    """Deterministic retrieval fake: always returns the same ranked chunks."""

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k):
        return self.hits[:k]


def _item(idx: int) -> GoldItem:
    return GoldItem(
        id=f"it-{idx}",
        question="Де народився Шевченко?",
        reference_answer="У Моринцях.",
        source_doc_id="a.md",
        source_spans=[SourceSpan(doc_id="a.md", char_start=0, char_end=len(DOC), text=DOC)],
        provenance="human-authored",
        split="final",
    )


def _hit(doc_id: str):
    return {"doc_id": doc_id, "char_start": 0, "char_end": len(DOC), "text": DOC, "rank": 1}


def test_measure_drift_flags_retune_on_recall_drop():
    items = [_item(0), _item(1)]
    old = FixedStore([_hit("a.md")])  # overlaps the gold span -> recall 1.0
    new = FixedStore([_hit("other.md")])  # misses -> recall 0.0
    drift = measure_drift(old, new, items, k=5, threshold=0.05)
    assert drift.old_recall == 1.0 and drift.new_recall == 0.0
    assert drift.delta_recall == -1.0
    assert drift.retune_recommended


def test_measure_drift_stays_quiet_under_threshold():
    items = [_item(0)]
    store = FixedStore([_hit("a.md")])
    drift = measure_drift(store, store, items, k=5, threshold=0.05)
    assert drift.delta_recall == 0.0 and drift.delta_mrr == 0.0
    assert not drift.retune_recommended


def test_write_drift_report_persists_json_and_markdown(tmp_path):
    items = [_item(0)]
    drift = measure_drift(FixedStore([_hit("a.md")]), FixedStore([_hit("other.md")]), items, k=5)
    diff = ManifestDiff(added=["d.md"], modified=["b.md"], deleted=["c.md"], unchanged=["a.md"])
    json_path, md_path = write_drift_report(tmp_path / "run", diff, drift, "gen-dir")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == DRIFT_JSON and md_path.name == DRIFT_REPORT_MD
    assert payload["diff"] == {"added": 1, "modified": 1, "deleted": 1, "unchanged": 1}
    assert payload["deleted"] == ["c.md"]
    assert payload["retrieval"]["retune_recommended"] is True
    assert payload["retrieval"]["delta"]["recall_at_k"] == -1.0
    report = md_path.read_text(encoding="utf-8")
    assert "RE-TUNE RECOMMENDED" in report
    assert "recall@5" in report


def test_report_without_goldset_notes_the_skip(tmp_path):
    diff = ManifestDiff(added=["d.md"])
    text = render_drift_report(diff, None, None)
    assert "skipped" in text
    json_path, _md_path = write_drift_report(tmp_path, diff, None, None)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "retrieval" not in payload
