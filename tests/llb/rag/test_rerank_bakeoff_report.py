"""Reranker bake-off rendering: the ranked table, the cost columns, and the keep-or-swap sentence."""

from _rerank_bakeoff_helpers import BASELINE, CANDIDATE, fake_loader, items, pools

from llb.rag.rerank_bakeoff.lane import run_rerank_bakeoff
from llb.rag.rerank_bakeoff.models import ROW_NO_RERANK
from llb.rag.rerank_bakeoff.report import format_report, render_markdown

GOLD_POSITIONS = [4, 4, 1, 4, 4, 1, 4, 4, 1, 4]


def _report(**overrides):
    kwargs = {
        "corpus_root": "corpus",
        "embedding_model": "intfloat/multilingual-e5-base",
        "chunking": "recursive@800/120",
        "pool_depth": 6,
        "batch_size": 8,
        "candidates": [BASELINE, CANDIDATE],
        "load_scorer": fake_loader({BASELINE: 1100.0, CANDIDATE: 2300.0}),
        "resamples": 200,
    }
    kwargs.update(overrides)
    return run_rerank_bakeoff(items(len(GOLD_POSITIONS)), pools(GOLD_POSITIONS, 6), 3, **kwargs)


def test_ascii_summary_is_ascii_and_names_every_row():
    text = format_report(_report())
    assert text.isascii()
    assert all(model in text for model in (ROW_NO_RERANK, BASELINE, CANDIDATE))
    assert "best (recall@k)" in text and "Verdict:" in text


def test_markdown_pins_the_retrieval_configuration_every_row_shared():
    md = render_markdown(_report())
    assert "`intfloat/multilingual-e5-base`" in md and "`recursive@800/120`" in md
    assert "candidate pool 6" in md and "rerank batch size: 8" in md


def test_markdown_prints_both_bars_and_the_cost_of_the_swap():
    md = render_markdown(_report())
    assert "recall delta vs `BAAI/bge-reranker-v2-m3`".replace("`", "") in md.replace("`", "")
    assert "MRR delta" in md and "first-hit rank" in md
    assert "rerank ms/query" in md and "peak VRAM (MB)" in md
    assert "Cost of running" in md and "of VRAM beside" in md


def test_markdown_states_the_swap_verdict_and_how_to_apply_it():
    md = render_markdown(_report())
    assert "Verdict: SWAP TO" in md and CANDIDATE in md
    assert "RunConfig.reranker" in md


def test_a_declared_budget_is_printed_and_an_undeclared_one_says_the_gate_did_not_run():
    md = render_markdown(_report())
    assert "no generator residency declared" not in md  # no headroom passed -> no budget line
    budget = {
        "total_mb": 16000.0,
        "generator_mb": 9000.0,
        "reserve_mb": 512.0,
        "headroom_mb": 6488.0,
    }
    with_budget = render_markdown(_report(headroom=budget))
    assert "6488 MB left for the reranker" in with_budget
    undeclared = render_markdown(
        _report(
            headroom={
                "total_mb": 16000.0,
                "generator_mb": None,
                "reserve_mb": 512.0,
                "headroom_mb": None,
            }
        )
    )
    assert "no generator residency declared" in undeclared


def test_a_run_without_the_floor_says_so_instead_of_letting_a_lead_read_as_real():
    assert "re-run with `--noise-floor`" in render_markdown(_report())


def test_the_floor_section_renders_when_it_was_measured():
    md = render_markdown(_report(noise_floor=True, noise_floor_replicates=8))
    assert "re-run with `--noise-floor`" not in md and "floor" in md.lower()


def test_skipped_candidates_read_as_declined_not_as_beaten():
    declined = [
        {
            "model": "jinaai/jina-reranker-v2-base-multilingual",
            "family": "jina-reranker-v2",
            "reason": "trust_remote_code_not_opted_in",
            "detail": "needs trust_remote_code",
        }
    ]
    md = render_markdown(_report(skipped=declined))
    assert "## Candidates not scored" in md and "needs trust_remote_code" in md


def test_a_multiline_host_error_stays_inside_its_table_cell():
    """A CUDA assert prints four lines; a raw newline in a cell would end the table there."""
    declined = [
        {
            "model": "Alibaba-NLP/gte-multilingual-reranker-base",
            "family": "gte-multilingual-reranker",
            "reason": "load_failed",
            "detail": "CUDA error: device-side assert triggered\nSearch for `cudaErrorAssert`\n",
        }
    ]
    md = render_markdown(_report(skipped=declined))
    row = next(line for line in md.splitlines() if "gte-multilingual-reranker-base" in line)
    assert row.endswith("|") and "device-side assert triggered Search for" in row


def test_the_floor_names_the_per_row_amplitudes_it_was_read_at():
    """Rows are perturbed at their OWN score scale, so the floor table has to say so."""
    md = render_markdown(_report(noise_floor=True, noise_floor_replicates=8))
    assert "SCALE-MATCHED per row" in md
