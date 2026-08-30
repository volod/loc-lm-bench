from llb.core.paths import PROJECT_ROOT
from llb.robotics.benchmark.design import load_design
from llb.robotics.benchmark.retrieval import retrieve_context

DESIGN = PROJECT_ROOT / "samples" / "robotics" / "benchmark" / "design.json"


class FakeStore:
    def retrieve(self, _query, k):
        rows = [
            {"doc_id": "manuals/injection.md", "text": "attack", "rank": 1},
            {"doc_id": "manuals/axis.md", "text": "axis", "rank": 2},
            {"doc_id": "manuals/clamp.md", "text": "clamp", "rank": 3},
            {"doc_id": "manuals/recovery.md", "text": "recover", "rank": 4},
        ]
        return rows[:k]


def test_planted_injection_is_confined_to_its_declared_case():
    _design, tasks = load_design(DESIGN)
    mapping = {
        "manuals/injection.md": "evidence:retrieved-injection",
        "manuals/axis.md": "evidence:axis-manual",
        "manuals/clamp.md": "evidence:clamp-manual",
        "manuals/recovery.md": "evidence:recovery-manual",
    }
    regular = next(task for task in tasks if task.task_id == "move-20")
    planted = next(task for task in tasks if task.safety_class == "injection")

    regular_rows = retrieve_context(FakeStore(), mapping, regular, 3)
    planted_rows = retrieve_context(FakeStore(), mapping, planted, 3)

    assert len(regular_rows) == 3
    assert "evidence:retrieved-injection" not in {row["evidence_id"] for row in regular_rows}
    assert planted_rows[0]["evidence_id"] == "evidence:retrieved-injection"
