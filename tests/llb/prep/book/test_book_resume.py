"""Resume, failure isolation, cache-key changes, and the source checksum on resume."""

import shutil
from pathlib import Path

import pytest

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest, read_config
from llb.prep.book.runner import BookRunError, resume_run, start_run
from llb.prep.book.source import SourceError
from llb.prep.book.stages.fake import FAKE_STAGES
from tests.llb.prep.book._scan_factory import ScanSpec, make_scan
from tests.llb.prep.book._stages import CountingRender, CountingStage, SimulatedKill, registry_with


def _stages(*names: str, **params: dict) -> tuple[BookStageSpec, ...]:
    return tuple(BookStageSpec(name=n, params=params.get(n.replace("-", "_"), {})) for n in names)


def _setup():
    render, counted = CountingRender(), CountingStage("counted")
    return render, counted, registry_with(render, counted)


def _only_run(out_root: Path) -> Path:
    (run,) = list(out_root.iterdir())
    return run


def test_resume_after_a_kill_repeats_no_completed_stage(scan_pdf, out_root):
    render, counted, registry = _setup()
    counted.kill_on = 3
    request = RunRequest(source=scan_pdf, batch_pages=2, stages=_stages("render", "counted"))
    with pytest.raises(SimulatedKill):
        start_run(request, out_root, registry)
    run = _only_run(out_root)
    page3 = run / "pages" / "p0003"
    # The killed attempt left its half-written output only in the partial directory.
    assert not (page3 / "counted").exists()
    assert (page3 / ".counted.partial" / "out.txt").is_file()
    assert not (page3 / ".counted.partial" / "second.txt").exists()
    assert sorted(render.calls) == [1, 2, 3, 4] and sorted(counted.calls) == [1, 2, 3]

    render.calls.clear()
    counted.calls.clear()
    outcome = resume_run(run, registry=registry)
    assert sorted(render.calls) == [5, 6]  # pages 1-4 were rendered before the kill
    assert sorted(counted.calls) == [3, 4, 5, 6]  # the killed page re-ran; 1-2 did not
    assert not (page3 / ".counted.partial").exists()
    assert (page3 / "counted" / "second.txt").is_file()
    assert outcome.manifest.summary["counted"]["done"] == 6
    assert outcome.outcomes == {"cached": 6, "ran": 6}


def test_a_forced_failure_fails_only_its_own_page_and_is_reported(scan_pdf, out_root):
    _, counted, registry = _setup()
    counted.fail_on = {2}
    request = RunRequest(source=scan_pdf, max_attempts=3, stages=_stages("render", "counted"))
    outcome = start_run(request, out_root, registry)
    assert counted.calls[2] == 3  # bounded retry, then recorded
    states = {page.pdf_page: page.stages[1] for page in outcome.manifest.pages}
    assert states[2].status == "failed" and states[2].attempts == 3
    assert "RuntimeError: engine crashed on page 2" in (states[2].error or "")
    assert all(states[p].status == "done" for p in (1, 3, 4, 5, 6))
    assert not list((outcome.run_dir / "pages" / "p0002" / "counted").glob("*.txt"))
    report = (outcome.run_dir / "metadata" / "processing_report.md").read_text()
    assert "p0002 (PDF page 2) `counted` after 3 attempt(s): RuntimeError" in report

    counted.fail_on = set()
    retried = resume_run(outcome.run_dir, registry=registry)
    assert retried.manifest.summary["counted"]["done"] == 6


def test_a_downstream_stage_of_a_failed_stage_is_blocked(scan_pdf, out_root):
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="fake-preprocess", params={"fail_pages": [4]}),
        BookStageSpec(name="fake-ocr"),
    )
    counted = CountingStage("after-preprocess", inputs=("render", "fake-preprocess"))
    registry = registry_with(CountingRender(), *FAKE_STAGES, counted)
    request = RunRequest(source=scan_pdf, stages=(*stages, BookStageSpec(name="after-preprocess")))
    manifest = start_run(request, out_root, registry).manifest
    page4 = {state.stage: state.status for state in manifest.pages[3].stages}
    assert page4 == {
        "render": "done",
        "fake-preprocess": "failed",
        "fake-ocr": "done",  # reads only the render, so it is not blocked
        "after-preprocess": "blocked",
    }
    assert 4 not in counted.calls


def test_a_cache_key_change_reruns_only_the_affected_stage(scan_pdf, out_root):
    render, counted, registry = _setup()
    first = start_run(
        RunRequest(source=scan_pdf, stages=_stages("render", "counted", counted={"k": 1})),
        out_root,
        registry,
    ).run_dir
    render.calls.clear()
    counted.calls.clear()
    second = start_run(
        RunRequest(
            source=scan_pdf, stages=_stages("render", "counted", counted={"k": 2}), reuse_from=first
        ),
        out_root,
        registry,
    )
    assert not render.calls  # every render key matched the earlier run
    assert sorted(counted.calls) == [1, 2, 3, 4, 5, 6]
    assert second.manifest.summary["render"]["reused"] == 6
    assert second.manifest.summary["counted"]["reused"] == 0
    assert read_config(second.run_dir).reuse_from == str(first.resolve())


def test_a_resume_against_a_changed_source_refuses(tmp_path, out_root):
    source_dir = tmp_path / "lib"
    source_dir.mkdir()
    scan = make_scan(source_dir / "scan.pdf")
    _, _, registry = _setup()
    run = start_run(RunRequest(source=scan, stages=_stages("render")), out_root, registry).run_dir
    other = make_scan(tmp_path / "other.pdf", ScanSpec(pages=6, landscape_page=None))
    with pytest.raises(SourceError, match="refusing to mix pages of two documents"):
        resume_run(run, given_source=other, registry=registry)
    shutil.copyfile(other, scan)  # the referenced original itself changed under the symlink
    with pytest.raises(SourceError, match="refusing to mix pages of two documents"):
        resume_run(run, registry=registry)


def test_resume_rechecks_the_recorded_vram_budget(scan_pdf, out_root, monkeypatch):
    _, counted, registry = _setup()
    request = RunRequest(
        source=scan_pdf,
        stages=_stages("render", "counted", counted={"vram_mb": {"reader": 3000}}),
    )
    counted.kill_on = 2
    with pytest.raises(SimulatedKill):
        start_run(request, out_root, registry, host_vram_mb=8000)

    monkeypatch.setattr("llb.prep.book.survey.measure.device_total_mb", lambda: 2000)
    with pytest.raises(BookRunError, match="reader.*needs 3000 MB VRAM"):
        resume_run(_only_run(out_root), registry=registry)
    assert sorted(counted.calls) == [1, 2]


def test_the_render_key_holds_the_resolved_resolution(scan_pdf, out_root):
    render, counted, registry = _setup()
    native = start_run(
        RunRequest(source=scan_pdf, stages=_stages("render", "counted")), out_root, registry
    ).run_dir
    render.calls.clear()
    counted.calls.clear()
    # `150` is what `native` resolves to on this scan: same key, nothing re-runs.
    same = start_run(
        RunRequest(
            source=scan_pdf,
            stages=_stages("render", "counted", render={"dpi": 150}),
            reuse_from=native,
        ),
        out_root,
        registry,
    )
    assert not render.calls and not counted.calls
    assert same.manifest.summary["render"]["reused"] == 6
    # A different resolution re-renders, and the changed image re-runs what read it.
    start_run(
        RunRequest(
            source=scan_pdf,
            stages=_stages("render", "counted", render={"dpi": 100}),
            reuse_from=native,
        ),
        out_root,
        registry,
    )
    assert sorted(render.calls) == sorted(counted.calls) == [1, 2, 3, 4, 5, 6]
