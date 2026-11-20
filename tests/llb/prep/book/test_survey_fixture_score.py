"""The synthetic smoke page and the smoke reading scored against it."""

import hashlib
import json

import fitz

from llb.prep.book.survey.fixture import write_fixture
from llb.prep.book.survey.page_content import PAGE_H_PT, PAGE_W_PT, SCAN_DPI, fixture_truth
from llb.prep.book.survey.score import (
    EXACT,
    HYPHEN_MINUS,
    MISSING,
    edit_distance,
    plain_text,
    score_reading,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_is_byte_identical_and_shaped_like_the_scan(tmp_path):
    first, second = write_fixture(tmp_path / "a"), write_fixture(tmp_path / "b")
    assert _digest(first.image) == _digest(second.image)
    assert _digest(first.pdf) == _digest(second.pdf)
    pix = fitz.Pixmap(str(first.image))
    assert (pix.width, pix.height, pix.n) == (
        round(PAGE_W_PT / 72 * SCAN_DPI),
        round(PAGE_H_PT / 72 * SCAN_DPI),
        1,
    )
    doc = fitz.open(first.pdf)
    assert doc.page_count == 1
    assert doc[0].get_text().strip() == ""  # image-only, like the target scan
    assert len(doc[0].get_images()) == 1
    stored = json.loads((tmp_path / "a" / "truth.json").read_text(encoding="utf-8"))
    assert stored["dangerous_tokens"] == fixture_truth().dangerous_tokens


def test_fixture_page_is_degraded_not_clean(tmp_path):
    page = write_fixture(tmp_path)
    pix = fitz.Pixmap(str(page.image))
    samples = pix.samples
    # Grey paper and a gutter shadow: no pure-white background survives, and the gutter edge is
    # darker than the outer margin.
    assert max(samples) < 255
    left = samples[pix.width * 600 : pix.width * 600 + 10]
    right = samples[pix.width * 600 + pix.width - 30 : pix.width * 600 + pix.width - 20]
    assert sum(left) / 10 < sum(right) / 10 - 40


def test_a_perfect_reading_scores_one_and_keeps_every_dangerous_token():
    truth = fixture_truth()
    score = score_reading(truth, truth.reading_text())
    assert score.word_recall == 1.0 and score.char_agreement == 1.0
    assert set(score.dangerous.values()) == {EXACT}


def test_a_hyphen_for_a_minus_is_its_own_class_and_a_lost_sign_is_missing():
    truth = fixture_truth()
    text = truth.reading_text().replace("−3.2", "-3.2").replace("−98.2", "98.2")
    score = score_reading(truth, text)
    assert score.dangerous["−3.2"] == HYPHEN_MINUS
    assert score.dangerous["−98.2"] == MISSING
    assert score.word_recall < 1.0


def test_markup_is_read_through_and_lookalikes_are_not():
    assert plain_text(r"## Head $\alpha$ <b>x</b> | 5\% |") == "Head α x 5%"
    # A hyphen inside LaTeX math typesets as a minus sign; outside math it stays a hyphen.
    assert plain_text(r"thrust $-3.2$ N, \(-1\), see 3-4") == "thrust −3.2 N, −1, see 3-4"
    # Scripts join their base, a command's space is its terminator, and an equation label stays.
    formula = r"$$\[\Delta \mathrm{v} = I _ {sp} g_0 \ln(m_0/m_f) \tag{3-4}\]$$"
    assert plain_text(formula) == "Δv = Isp g0 ln(m0/mf) (3-4)"
    assert plain_text("| a | b |\n|---|:---:|\n| 1 | 2 |") == "a b 1 2"
    truth = fixture_truth()
    increment = truth.reading_text().replace("Δv", "∆v")  # INCREMENT, not Greek Delta
    assert score_reading(truth, increment).dangerous["Δv"] == MISSING


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
