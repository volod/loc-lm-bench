"""M7.3 human-assisted RAG prompt-system generation -- corpus, budget, template, review, tuning."""

import json

import pytest

from llb.bench import agentic
from llb.board.data import load_prompt_system_records, prompt_system_comparison
from llb.prompt_system import budget as bud
from llb.prompt_system import corpus as cp
from llb.prompt_system import manifest as psm
from llb.prompt_system import review as rv
from llb.prompt_system import tuning as tn
from llb.prompt_system.pipeline import (
    CANDIDATES_FILE,
    MANIFEST_FILE,
    prepare_prompt_system,
)
from llb.prompt_system.template import (
    GRAPH_INLINE,
    GRAPH_NONE,
    METADATA_FULL,
    METADATA_NONE,
    PromptPackage,
    TemplateFields,
    render_package,
    wrap_complete,
)

SAMPLE_CORPUS = "samples/corpus"

DOCS = {
    "a.md": "Бюджет міста зріс на 15 відсотків. Інвестиції в інфраструктуру збільшились.\n\n"
    "Освітні програми отримали додаткове фінансування цього року для шкіл міста.",
    "b.md": "Інфраструктура транспорту потребує оновлення. Бюджет на дороги обмежений завжди.",
}


# --- corpus prep --------------------------------------------------------------------------


def test_read_corpus_reads_sample():
    docs = cp.read_corpus(SAMPLE_CORPUS)
    assert "ip_regulation_uk.md" in docs and len(docs["ip_regulation_uk.md"]) > 1000


def test_tokenize_terms_drops_short_and_stopwords():
    terms = cp.tokenize_terms("Це бюджет міста для шкіл")
    assert "бюджет" in terms and "міста" in terms
    assert "це" not in terms and "для" not in terms  # stopwords dropped


def test_split_paragraphs_preserves_offsets():
    text = DOCS["a.md"]
    spans = cp.split_paragraphs(text)
    assert len(spans) == 2
    for start, end, body in spans:
        assert text[start:end].strip() == body


def test_select_anthology_preserves_source_spans():
    passages = cp.select_anthology(DOCS, max_passages=5, min_chars=10)
    assert passages
    docs = DOCS
    for p in passages:
        assert docs[p["doc_id"]][p["char_start"] : p["char_end"]].strip() == p["text"]


def test_graph_rag_mapping_links_terms_to_passages():
    pkg = cp.build_corpus_package(DOCS, max_passages=5, min_passage_chars=10, top_terms_k=8)
    assert "бюджет" in pkg.graph_rag_mapping
    # every mapped passage id is a real anthology passage
    ids = {p["passage_id"] for p in pkg.anthology}
    for passage_ids in pkg.graph_rag_mapping.values():
        assert set(passage_ids) <= ids


def test_build_corpus_package_rejects_empty():
    with pytest.raises(ValueError, match="empty corpus"):
        cp.build_corpus_package({})


# --- context budget -----------------------------------------------------------------------


def test_char_ratio_tokenizer_ceils():
    tok = bud.CharRatioTokenizer(chars_per_token=4.0)
    assert tok.count("") == 0 and tok.count("abcd") == 1 and tok.count("abcde") == 2


def test_plan_budget_reserves_and_leaves_remainder():
    b = bud.plan_budget(1000, question_tokens=50, chunk_tokens=400, answer_tokens=200)
    assert b.reserved == 650 and b.prompt_budget == 350


def test_fit_sections_respects_budget_and_reports_drops():
    tok = bud.CharRatioTokenizer(chars_per_token=1.0)  # 1 char == 1 token
    sections = [
        ("anthology", [{"item_id": "p1", "text": "aaaa"}, {"item_id": "p2", "text": "bbbb"}]),
        ("graph", [{"item_id": "g1", "text": "cccc"}]),
    ]
    fit = bud.fit_sections(sections, budget_tokens=5, tokenizer=tok)
    assert fit.used_tokens == 4  # only p1 fits (4 tokens); p2/g1 dropped
    assert fit.kept["anthology"] == [{"item_id": "p1", "text": "aaaa"}]
    dropped = {s["section"]: s["n_dropped"] for s in fit.report["sections"]}
    assert dropped == {"anthology": 1, "graph": 1}


# --- template generation ------------------------------------------------------------------


def _corpus():
    return cp.build_corpus_package(DOCS, max_passages=5, min_passage_chars=10, top_terms_k=8)


def test_render_package_fits_budget_and_embeds_anthology():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=32, chunk_tokens=256)
    pkg = render_package(_corpus(), TemplateFields(anthology_size=3), b, tok)
    assert pkg.used_tokens <= b.prompt_budget
    assert "Джерела" in pkg.additional_prompt
    assert pkg.system_prompt and pkg.fields.anthology_size == 3


def test_template_field_styles_change_output():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    no_meta = render_package(
        _corpus(),
        TemplateFields(metadata_density=METADATA_NONE, graph_reference_style=GRAPH_NONE),
        b,
        tok,
    )
    full = render_package(
        _corpus(),
        TemplateFields(metadata_density=METADATA_FULL, graph_reference_style=GRAPH_INLINE),
        b,
        tok,
    )
    assert "Метадані" not in no_meta.additional_prompt
    assert "Метадані" in full.additional_prompt


def test_template_field_validation():
    with pytest.raises(ValueError, match="metadata_density"):
        TemplateFields(metadata_density="bogus").validate()


def test_prompt_package_apply_and_wrap_complete():
    pkg = PromptPackage(
        system_prompt="SYS",
        additional_prompt="CTX",
        fields=TemplateFields(),
        dropped_context={"budget_tokens": 0, "used_tokens": 0, "sections": []},
    )
    assert pkg.as_prefix() == "SYS\n\nCTX"
    assert pkg.apply("TASK").endswith("TASK") and pkg.apply("TASK").startswith("SYS")
    seen = {}

    def record(prompt):
        seen["p"] = prompt
        return "ok"

    wrapped = wrap_complete(record, pkg)
    assert wrapped("TASK") == "ok" and "SYS" in seen["p"] and "TASK" in seen["p"]


# --- manifest digests ---------------------------------------------------------------------


def test_prompt_system_id_stable_and_field_sensitive():
    corpus = _corpus()
    id_a = psm.prompt_system_id(corpus, TemplateFields(anthology_size=4))
    id_b = psm.prompt_system_id(corpus, TemplateFields(anthology_size=4))
    id_c = psm.prompt_system_id(corpus, TemplateFields(anthology_size=6))
    assert id_a == id_b and id_a != id_c


def test_prompt_system_provenance_records_inputs():
    corpus = _corpus()
    b = bud.plan_budget(4096, question_tokens=10, chunk_tokens=100)
    prov = psm.prompt_system_provenance(corpus, TemplateFields(), b, tokenizer="char-ratio")
    assert prov["context_window"] == 4096 and prov["tokenizer"] == "char-ratio"
    assert prov["prompt_system_id"] and prov["corpus_digest"] and prov["mapping_digest"]


# --- review loop --------------------------------------------------------------------------


def test_review_transitions_and_roundtrip(tmp_path):
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    c = rv.make_candidate(_corpus(), TemplateFields(anthology_size=2), b, tok)
    rv.approve(c, "looks good")
    assert c.status == rv.STATUS_APPROVED and c.note == "looks good"
    path = tmp_path / "candidates.json"
    rv.save_candidates([c], path)
    loaded = rv.load_candidates(path)
    assert loaded[0].prompt_system_id == c.prompt_system_id
    assert loaded[0].status == rv.STATUS_APPROVED
    assert loaded[0].fields.anthology_size == 2


def test_review_revise_creates_new_id():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    corpus = _corpus()
    c = rv.make_candidate(corpus, TemplateFields(anthology_size=2), b, tok)
    revised = rv.revise(c, TemplateFields(anthology_size=4), corpus, b, tok)
    assert revised.status == rv.STATUS_REVISED and revised.prompt_system_id != c.prompt_system_id


# --- tuning loop --------------------------------------------------------------------------


def test_variant_grid_dedupes_and_validates():
    grid = tn.variant_grid(
        TemplateFields(), anthology_sizes=[2, 2, 4], graph_styles=[GRAPH_NONE, GRAPH_INLINE]
    )
    keys = {(f.anthology_size, f.metadata_density, f.graph_reference_style) for f in grid}
    assert len(keys) == len(grid)  # no duplicates
    assert all(f.anthology_size in (2, 4) for f in grid)


def test_generate_candidates_dedupes_by_id():
    tok = bud.CharRatioTokenizer()
    b = bud.plan_budget(8192, question_tokens=0, chunk_tokens=0)
    grid = tn.variant_grid(
        TemplateFields(), anthology_sizes=[2, 4], graph_styles=[GRAPH_NONE, GRAPH_INLINE]
    )
    candidates = tn.generate_candidates(_corpus(), grid, b, tok)
    ids = [c.prompt_system_id for c in candidates]
    assert len(ids) == len(set(ids)) and len(candidates) >= 4


# --- pipeline -----------------------------------------------------------------------------


def test_prepare_prompt_system_writes_artifacts(tmp_path):
    run = prepare_prompt_system(
        SAMPLE_CORPUS, data_dir=tmp_path, context_window=4096, max_passages=6
    )
    assert run.candidates and (run.run_dir / MANIFEST_FILE).exists()
    manifest = json.loads((run.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["method"] == "prompt-system" and manifest["n_candidates"] == len(run.candidates)
    assert manifest["corpus_digest"] and manifest["context_window"] == 4096
    candidates = json.loads((run.run_dir / CANDIDATES_FILE).read_text(encoding="utf-8"))
    assert len(candidates) == len(run.candidates)


# --- benchmark integration: prompt-system board axis --------------------------------------


def _two_tasks():
    return [
        agentic.AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}]),
        agentic.AgenticTask("b", "p", success=[{"kind": "answer_contains", "value": "x"}]),
    ]


def test_run_agentic_records_prompt_system_and_board_compares(tmp_path):
    # ps1: both succeed (answer contains x); ps2: fails (empty answer)
    agentic.run_agentic(
        _two_tasks(),
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":"x"}}',
        prompt_system="ps1",
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    agentic.run_agentic(
        _two_tasks(),
        model="m",
        backend="ollama",
        complete=lambda _: '{"name":"finish","arguments":{"answer":""}}',
        prompt_system="ps2",
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    records = load_prompt_system_records(tmp_path)
    assert {(r.model, r.prompt_system) for r in records} == {("m", "ps1"), ("m", "ps2")}
    rows, table, ids = prompt_system_comparison(tmp_path, "m")
    assert {row["model"] for row in rows} == {"ps1", "ps2"}
    top = next(row for row in rows if row["rank"] == 1)
    assert top["model"] == "ps1" and "policy:" in table
