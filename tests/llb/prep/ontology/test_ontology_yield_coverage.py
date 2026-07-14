"""Tests for ontology yield coverage."""

from llb.prep.ontology.coverage import select_seeds
from llb.prep.ontology.coverage_report import coverage_report
from llb.prep.ontology.question_types import classify_question_type
from ontology_yield_helpers import _relation_pool


def test_coverage_target_drafts_more_than_a_small_flat_cap_and_reports_exhaustion():
    pool = _relation_pool(10)
    flat = select_seeds(pool, max_items=3, seed=7)
    covered = select_seeds(pool, max_items=100, seed=7, coverage_target=1)

    assert len(flat) == 3  # the flat cap stops early
    assert len(covered) == 10  # coverage-target spans every relation bucket
    assert len(covered) > len(flat)

    report = coverage_report(pool, covered, coverage_target=1, max_items=100)
    assert report["mode"] == "coverage-target"
    assert report["drafted_seeds"] == 10 and report["seeds_remaining"] == 0
    assert report["exhausted"] is True
    relation = report["strata"]["relation"]
    assert relation["buckets"] == 10 and relation["buckets_drafted"] == 10
    assert relation["buckets_at_target"] == 10 and relation["seeds_remaining"] == 0


def test_flat_cap_report_records_undrafted_seeds_remaining():
    pool = _relation_pool(10)
    flat = select_seeds(pool, max_items=3, seed=7)
    report = coverage_report(pool, flat, coverage_target=None, max_items=3)
    assert report["mode"] == "flat-cap"
    assert report["drafted_seeds"] == 3 and report["seeds_remaining"] == 7
    assert report["exhausted"] is False
    assert report["strata"]["relation"]["seeds_remaining"] == 7


def test_classify_question_type_closed_taxonomy():
    assert classify_question_type("Що таке столиця?", "місто") == "definition"
    assert classify_question_type("У якому році засновано місто?", "1256 рік") == "numeric"
    assert classify_question_type("Скільки відсотків?", "багато") == "numeric"
    assert classify_question_type("Чим відрізняється А від Б?", "усім") == "comparative"
    assert classify_question_type("Як подати заяву?", "через портал") == "procedural"
    assert classify_question_type("Хто керує містом?", "мер") == "factoid"
