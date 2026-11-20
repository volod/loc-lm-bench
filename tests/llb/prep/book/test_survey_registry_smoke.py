"""The candidate registry's integrity and the smoke orchestrator over injected candidates."""

import os
import sys
import time
from pathlib import Path

import pytest

from llb.prep.book.survey import envs
from llb.prep.book.survey.candidates import (
    CANDIDATES,
    ENVS,
    EXCLUDE_NEWER,
    ROUTE_NONE,
    ROUTE_OLLAMA,
    ROUTE_VENV,
    SMOKE_COMMAND,
    STAGES,
    Candidate,
    Smoke,
)
from llb.prep.book.survey.fixture import write_fixture
from llb.prep.book.survey.report import render_markdown
from llb.prep.book.survey.results import (
    STATUS_FAILED,
    STATUS_NOT_INSTALLED,
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_TIMEOUT,
)
from llb.prep.book.survey.smoke import READING_NAME, run_smoke


def test_registry_rows_are_complete_and_consistent():
    ids = [candidate.id for candidate in CANDIDATES]
    assert len(ids) == len(set(ids))
    for candidate in CANDIDATES:
        assert candidate.stages and set(candidate.stages) <= set(STAGES), candidate.id
        assert candidate.licence and candidate.version and candidate.source, candidate.id
        if candidate.route == ROUTE_NONE:
            assert candidate.blocker and candidate.smoke is None, candidate.id
        else:
            assert candidate.smoke is not None, candidate.id
        if candidate.route == ROUTE_VENV:
            assert candidate.env in ENVS, candidate.id
        if candidate.route == ROUTE_OLLAMA:
            assert candidate.smoke and candidate.smoke.model and candidate.smoke.prompt
    served = {stage for candidate in CANDIDATES for stage in candidate.stages}
    assert served == set(STAGES)


def test_every_environment_pins_exactly_and_resolves_at_the_cutoff(tmp_path):
    for spec in ENVS.values():
        assert all("==" in requirement for requirement in spec.requirements), spec.name
        argv = envs.install_command(spec, tmp_path / "python")
        assert argv[argv.index("--exclude-newer") + 1] == EXCLUDE_NEWER
        assert list(spec.requirements) == argv[-len(spec.requirements) :]
        # A vendor index without upload dates is exempted per package, never globally.
        exempt = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--exclude-newer-package"]
        assert exempt == [f"{package}=false" for package in spec.undated]
        assert ("--index" in argv) == bool(spec.index)


def test_tool_environment_keeps_every_cache_under_the_data_dir(tmp_path):
    env = envs.tool_environment(tmp_path, tmp_path / "bin")
    for variable in (
        "XDG_CACHE_HOME",
        "HF_HOME",
        "MODELSCOPE_CACHE",
        "PADDLE_PDX_CACHE_HOME",
        "MINERU_HOME",
    ):
        assert Path(env[variable]).is_relative_to(tmp_path), variable
    assert env["PATH"].startswith(str(tmp_path / "bin"))
    assert env["MINERU_TELEMETRY_DISABLED"] == "1"


def _fake_env(data_dir: Path) -> None:
    """A survey environment whose interpreter is this test's interpreter."""
    bin_dir = envs.env_dir(data_dir, "fake") / "bin"
    bin_dir.mkdir(parents=True)
    os.symlink(sys.executable, bin_dir / "python")


def _candidate(code: str, **smoke: object) -> Candidate:
    return Candidate(
        id="fake",
        name="fake",
        stages=("ocr",),
        version="0",
        licence="MIT",
        locality="local",
        route=ROUTE_VENV,
        source="-",
        claims=(),
        env="fake",
        smoke=Smoke(
            SMOKE_COMMAND,
            ("{python}", "-c", code, "{out}"),
            reading="out.txt",
            **smoke,  # type: ignore[arg-type]
        ),
    )


@pytest.fixture()
def page(tmp_path):
    return write_fixture(tmp_path / "page")


def test_a_reading_is_scored_and_its_text_kept_out_of_the_result(tmp_path, page):
    _fake_env(tmp_path)
    text = page.truth.reading_text().replace("\n", " ")
    code = f"import sys,pathlib; pathlib.Path(sys.argv[1],'out.txt').write_text({text!r})"
    result = run_smoke(_candidate(code), page, tmp_path / "out", tmp_path)
    assert result.status == STATUS_OK, result.error
    assert result.score["word_recall"] == 1.0
    assert result.score["dangerous_exact"] == result.score["dangerous_total"]
    assert result.reading == READING_NAME
    assert (tmp_path / "out" / READING_NAME).read_text(encoding="utf-8") == text


def test_a_silent_fallback_an_empty_output_and_a_hang_all_fail(tmp_path, page):
    _fake_env(tmp_path)
    write = "import sys,pathlib; pathlib.Path(sys.argv[1],'out.txt').write_text('x')"
    fallback = _candidate(f"{write}; print('falling back to weaker mode')", fail_on="falling back")
    fell_back = run_smoke(fallback, page, tmp_path / "a", tmp_path)
    assert fell_back.status == STATUS_FAILED
    assert fell_back.error == "falling back to weaker mode"
    assert run_smoke(_candidate("pass"), page, tmp_path / "b", tmp_path).status == STATUS_FAILED
    hang = _candidate("import time; time.sleep(30)", timeout_s=1.0)
    assert run_smoke(hang, page, tmp_path / "c", tmp_path).status == STATUS_TIMEOUT


def test_missing_and_blocked_candidates_are_recorded_not_dropped(tmp_path, page):
    missing = run_smoke(_candidate("pass"), page, tmp_path / "x", tmp_path)
    assert missing.status == STATUS_NOT_INSTALLED
    assert "make book-survey-install BOOK_SURVEY_ENVS=fake" in missing.error
    blocked = next(c for c in CANDIDATES if c.route == ROUTE_NONE)
    result = run_smoke(blocked, page, tmp_path / "y", tmp_path)
    assert result.status == STATUS_NOT_RUN and result.error == blocked.blocker
    table = render_markdown(
        {"survey_date": "d", "exclude_newer": "e", "vram_budget_mb": 1}, [missing, result]
    )
    assert "not-installed" in table and "## Not run, failed, or over budget" in table


def test_rescore_recomputes_scores_from_saved_readings_without_running(tmp_path):
    from llb.prep.book.survey.report import PAGES_DIRNAME, rescore_survey, run_survey

    _fake_env(tmp_path)
    survey = tmp_path / "survey"
    page = write_fixture(survey / PAGES_DIRNAME / "synthetic")
    code = "import sys,pathlib; pathlib.Path(sys.argv[1],'out.txt').write_text('nothing right')"
    [first] = run_survey([_candidate(code)], [page], survey, tmp_path)
    assert first.score["word_recall"] < 0.1
    saved = survey / "synthetic" / "fake" / READING_NAME
    saved.write_text(page.truth.reading_text(), encoding="utf-8")
    [rescored] = rescore_survey(survey)
    assert rescored.score["word_recall"] == 1.0 and rescored.status == STATUS_OK


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_an_ollama_reading_records_placement_and_unloads(tmp_path, page, monkeypatch):
    import httpx

    from llb.prep.book.survey import ollama_smoke

    sent = {}

    def post(url, json, timeout):
        sent.update(json)
        time.sleep(0.2)  # the model reads; the placement sampler polls meanwhile
        return _Response({"response": page.truth.reading_text()})

    placement = {"models": [{"name": "m:1", "size": 3 * 2**30, "size_vram": 2 * 2**30}]}
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Response(placement))
    monkeypatch.setattr(ollama_smoke, "PLACEMENT_INTERVAL_S", 0.01)
    reading = ollama_smoke.read_page("m:1", "read", page.image, 5.0, None)
    assert reading.error is None and reading.text == page.truth.reading_text()
    assert reading.vram_resident_mb == 2048.0 and reading.ram_resident_mb == 1024.0
    assert sent["keep_alive"] == 0 and sent["options"]["temperature"] == 0
    assert sent["think"] is False
    assert len(sent["images"]) == 1


def test_an_ollama_transport_failure_is_recorded_not_raised(tmp_path, page, monkeypatch):
    import httpx

    from llb.prep.book.survey import ollama_smoke

    def post(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: _Response({"models": []}))
    reading = ollama_smoke.read_page("m:1", "read", page.image, 5.0, None)
    assert reading.text == "" and "ConnectError" in reading.error
