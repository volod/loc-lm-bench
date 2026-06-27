"""verified-data hardening second-frontier cross-check -- the automated verified-data gate."""

from llb.goldset.schema import GoldItem, SourceSpan
from llb.prep import cross_check

DOC = "Бюджет міста зріс на 15 відсотків минулого року завдяки новим інвестиціям."


def item(
    item_id="i1",
    question="На скільки зріс бюджет?",
    answer="на 15 відсотків",
    span="зріс на 15 відсотків",
):
    start = DOC.find(span)
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer=answer,
        source_doc_id="doc",
        source_spans=[
            SourceSpan(doc_id="doc", char_start=start, char_end=start + len(span), text=span)
        ],
        provenance="frontier-drafted",
        verified=False,
        split="final",
    )


def always_ok(_item, _doc):
    return True, True, "ok"


def always_reject(_item, _doc):
    return False, False, "not supported"


def test_is_grounded_and_non_circular():
    it = item()
    assert cross_check.is_grounded(it, DOC) is True
    assert cross_check.is_grounded(it, "інший текст без цитати") is False
    assert cross_check.is_non_circular(it) is True


def test_non_circular_detects_leaked_answer():
    leaked = item(question="Чи правда, що бюджет зріс на 15 відсотків?", answer="на 15 відсотків")
    assert cross_check.is_non_circular(leaked) is False  # answer is a substring of the question


def test_cross_check_item_passes_with_second_frontier():
    verdict = cross_check.cross_check_item(item(), DOC, always_ok)
    assert verdict.passed is True


def test_cross_check_item_skips_verifier_on_failed_precheck():
    # ungrounded -> deterministic pre-check fails, second frontier never consulted
    called = []

    def spy(it, doc):
        called.append(it.id)
        return True, True, "ok"

    verdict = cross_check.cross_check_item(item(), "geen цитати тут", spy)
    assert verdict.passed is False and verdict.grounded is False
    assert called == []  # verifier skipped


def test_cross_check_item_fails_when_second_frontier_rejects():
    verdict = cross_check.cross_check_item(item(), DOC, always_reject)
    assert verdict.passed is False and verdict.supported is False


def test_cross_check_goldset_report():
    items = [
        item("good"),
        item("bad", question="що це?", answer="абвгд", span="зріс на 15 відсотків"),
    ]
    # 'bad': answer 'абвгд' is not in the doc-grounded span check? span is grounded, but verifier rejects
    report = cross_check.cross_check_goldset(
        items,
        {"doc": DOC},
        lambda it, doc: (it.id == "good", it.id == "good", ""),
    )
    assert report.n_passed == 1 and report.passed_ids == {"good"}
    assert report.to_dict()["n_items"] == 2


def test_second_frontier_verify_parses_json():
    verify = cross_check.second_frontier_verify(
        "m", complete=lambda _: '{"supported": true, "answerable": true, "note": "ok"}'
    )
    assert verify(item(), DOC) == (True, True, "ok")
    bad = cross_check.second_frontier_verify("m", complete=lambda _: "not json")
    assert bad(item(), DOC) == (False, False, "unparseable verifier output")
