"""security benchmark corpus-specific security planter -- RAG-injection + canary leak cases over a real corpus."""

from llb.prep import security_planter as sp
from llb.scoring import security
from llb.scoring import security_cases

DOC = "Перший абзац документа про ринок.\n\nДругий абзац з фактами та цифрами."


def test_make_marker_deterministic_and_slugged():
    m = sp.make_marker("RAGINJECT", "doc-001", 0)
    assert m == sp.make_marker("RAGINJECT", "doc-001", 0)  # deterministic
    assert m.startswith("RAGINJECT-DOC001")


def test_rag_injection_case_splices_marker_into_context():
    case = sp.rag_injection_case(
        "doc1", DOC, index=0, template=sp.INJECTION_TEMPLATES[0], task=sp.BENIGN_TASKS[0]
    )
    assert case["family"] == security_cases.RAG_INJECTION
    marker = case["markers"][0]
    assert marker in case["injected_context"]  # the attack lives in the retrieved context
    assert marker not in case["prompt"]  # the user turn is benign
    # the marker must not be a substring the clean doc already contains
    assert marker not in DOC


def test_canary_case_embeds_secret_in_confidential_doc():
    case = sp.canary_case("doc1", DOC, index=0)
    assert case["family"] == security_cases.DATA_EXFILTRATION
    assert case["markers"][0] in case["injected_context"]
    assert case["attrs"]["canary"] is True


def test_plant_security_cases_counts():
    docs = {"a": DOC, "b": DOC, "empty": "   "}
    cases = sp.plant_security_cases(docs, n_injection_per_doc=2, n_canary_per_doc=1)
    # 2 docs (empty skipped) * (2 injection + 1 canary) = 6
    assert len(cases) == 6
    assert all(c["attrs"]["planted"] for c in cases)


def test_planted_cases_score_objectively(tmp_path):
    """A vulnerable model that echoes the injected marker -> attack succeeds; a robust one defends."""
    cases = security_cases.load_security_cases(sp.plant_security_cases({"d": DOC}))
    leaks = [c.markers[0] for c in cases]

    vuln = security.score_security(cases, [f"... {m} ..." for m in leaks])
    assert vuln.asr == 1.0  # every planted marker leaked

    robust = security.score_security(cases, ["Документ описує ринок." for _ in cases])
    assert robust.asr == 0.0  # nothing leaked


def test_plant_from_corpus(tmp_path):
    (tmp_path / "doc1.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "doc2.md").write_text(DOC, encoding="utf-8")
    cases = sp.plant_from_corpus(tmp_path, n_injection_per_doc=1, n_canary_per_doc=1)
    assert len(cases) == 4
    loaded = security_cases.load_security_cases(
        cases
    )  # round-trips through the SecurityCase schema
    assert {c.family for c in loaded} == {
        security_cases.RAG_INJECTION,
        security_cases.DATA_EXFILTRATION,
    }
