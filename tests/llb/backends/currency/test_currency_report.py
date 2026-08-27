"""The upstream currency report, reproduced from recorded registry responses with no network.

The recorded cassette in `tests/fixtures/roster_currency/` is a trimmed capture of the two live
registries, so these tests exercise the real HTML and JSON shapes the adapters parse rather than a
hand-shaped stand-in of them.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.backends.currency import Cassette, probe_register, render_text, report_payload
from llb.backends.currency.generations import compile_pattern, generation_key
from llb.backends.currency.registries import Response, hf_models_url, ollama_library_url
from llb.backends.currency.report import BEHIND, CURRENT, UNKNOWN
from llb.backends.roster import load_register
from llb.core.paths import PROJECT_ROOT
from llb.main import app

ROSTER = PROJECT_ROOT / "samples" / "configs" / "models_uk.yaml"
CASSETTE = PROJECT_ROOT / "tests" / "fixtures" / "roster_currency" / "upstream.json"


@pytest.fixture
def rows():
    return probe_register(load_register(ROSTER), Cassette.load(CASSETTE).fetch)


def _row(rows, family_id: str):
    return next(row for row in rows if row.family_id == family_id)


def _row_of(register, family_id: str):
    return next(family for family in register.families if family.id == family_id)


def test_every_registered_family_is_a_row(rows) -> None:
    register = load_register(ROSTER)

    assert [row.family_id for row in rows] == [family.id for family in register.families]
    assert all(row.verdict in {CURRENT, BEHIND, UNKNOWN} for row in rows)


def test_a_newer_upstream_generation_reproduces_a_behind_verdict(rows) -> None:
    mistral = _row(rows, "mistral")

    assert mistral.verdict == BEHIND
    assert mistral.carried == "3.1"
    assert mistral.upstream is not None and mistral.upstream.id == "4"
    assert mistral.upstream.evidence == "Mistral-Small-4-119B-2603"
    assert mistral.registry == "huggingface"


def test_the_newest_upstream_generation_reproduces_a_current_verdict(rows) -> None:
    qwen = _row(rows, "qwen")

    assert qwen.verdict == CURRENT
    assert qwen.carried == "3.8" and qwen.upstream is not None and qwen.upstream.id == "3.8"


def test_a_trailing_version_family_is_read_through_its_declared_pattern(rows) -> None:
    assert _row(rows, "mamaylm").verdict == CURRENT
    lapa = _row(rows, "lapa")

    assert lapa.verdict == BEHIND
    assert lapa.upstream is not None and lapa.upstream.evidence == "lapa-v0.1.3-instruct"


def test_every_reading_carries_the_time_its_response_arrived(rows) -> None:
    answered = [reading for row in rows for reading in row.readings if reading.generations]

    assert answered
    assert all(reading.read_at.endswith("Z") for reading in answered)
    assert all(f"at {reading.read_at}" in render_text(rows) for reading in answered)


def test_a_registry_that_does_not_answer_degrades_that_family_to_unknown() -> None:
    register = load_register(ROSTER)

    def refuse(url: str) -> Response:
        return Response(
            url=url, read_at="2026-08-27T00:00:00Z", error="HTTP 503 Service Unavailable"
        )

    rows = probe_register(register, refuse)

    assert {row.verdict for row in rows} == {UNKNOWN}
    assert all("503" in (row.reason or "") for row in rows)
    assert len(rows) == len(register.families)


def test_one_unreachable_registry_does_not_ground_the_family_the_other_answered(rows) -> None:
    cassette = Cassette.load(CASSETTE)
    cassette.responses.pop(ollama_library_url())

    partial = probe_register(load_register(ROSTER), cassette.fetch)
    gemma = _row(partial, "gemma")

    assert gemma.verdict == CURRENT and gemma.registry == "huggingface"
    assert any("not in the recorded responses" in (r.error or "") for r in gemma.readings)


def test_an_unparseable_registry_body_is_a_reason_not_a_crash() -> None:
    register = load_register(ROSTER)
    url = hf_models_url("Qwen", "Qwen")

    def garbled(asked: str) -> Response:
        body = "<html>not json</html>" if asked == url else None
        return Response(
            url=asked, read_at="2026-08-27T00:00:00Z", body=body, error=None if body else "no"
        )

    qwen = _row(probe_register(register, garbled), "qwen")

    assert qwen.verdict == UNKNOWN
    assert "not JSON" in (qwen.reason or "")


def test_the_report_renders_a_row_for_a_current_family_and_tallies_every_verdict(rows) -> None:
    text = render_text(rows)

    assert "[currency] qwen" in text and CURRENT in text
    payload = report_payload(rows)
    assert sum(payload["counts"].values()) == len(rows)
    assert json.dumps(payload)  # the JSON shape is serializable as-is


def test_cli_replays_a_cassette_and_strict_flags_a_behind_family(tmp_path: Path) -> None:
    argv = ["check-model-currency", "--manifest", str(ROSTER), "--replay", str(CASSETTE)]

    report = CliRunner().invoke(app, argv)
    strict = CliRunner().invoke(app, [*argv, "--family", "mistral", "--strict"])
    single = CliRunner().invoke(app, [*argv, "--family", "qwen", "--strict", "--json"])

    assert report.exit_code == 0, report.output
    assert "[currency] 5 families:" in report.output
    assert strict.exit_code == 1, strict.output
    assert single.exit_code == 0, single.output
    assert json.loads(single.output)["counts"][CURRENT] == 1


def test_cli_rejects_a_family_the_register_does_not_carry() -> None:
    result = CliRunner().invoke(
        app,
        [
            "check-model-currency",
            "--manifest",
            str(ROSTER),
            "--replay",
            str(CASSETTE),
            "--family",
            "llama",
        ],
    )

    assert result.exit_code != 0
    assert "no such family" in result.output


def test_recording_a_run_writes_a_cassette_that_replays_identically(tmp_path: Path) -> None:
    register = load_register(ROSTER)
    recorded = Cassette()
    live = Cassette.load(CASSETTE)

    first = probe_register(register, recorded.recording(live.fetch))
    target = tmp_path / "cassette.json"
    recorded.save(target)
    replayed = probe_register(register, Cassette.load(target).fetch)

    assert [(row.family_id, row.verdict) for row in first] == [
        (row.family_id, row.verdict) for row in replayed
    ]


def test_every_registered_family_declares_a_namespace_a_registry_can_be_asked_for() -> None:
    """A family with no readable `upstream` block can only ever report `unknown`."""
    for family in load_register(ROSTER).families:
        upstream = family.upstream
        ollama = upstream.get("ollama_namespace", "")
        readable = upstream.get("hf_author") or (ollama and "/" not in ollama)

        assert readable, f"family `{family.id}` declares no namespace any registry can be asked for"
        pattern = compile_pattern(
            upstream.get("hf_prefix") or ollama, upstream.get("generation_pattern")
        )
        assert pattern is not None  # a declared pattern that does not compile raises here
        # The carried generation must itself be comparable, or the row can only be `unknown`.
        assert generation_key(family.current.id) is not None


def test_a_malformed_declared_pattern_is_a_reason_on_that_family_alone() -> None:
    register = load_register(ROSTER)
    broken = replace(
        _row_of(register, "qwen"),
        upstream={"hf_author": "Qwen", "hf_prefix": "Qwen", "generation_pattern": "(unclosed"},
    )
    register = type(register)(
        families=tuple(broken if f.id == "qwen" else f for f in register.families),
        models=register.models,
    )

    rows = probe_register(register, Cassette.load(CASSETTE).fetch)

    assert _row(rows, "qwen").verdict == UNKNOWN
    assert "invalid generation_pattern" in (_row(rows, "qwen").reason or "")
    assert _row(rows, "gemma").verdict == CURRENT


def test_a_pattern_with_more_than_one_capture_group_is_refused() -> None:
    with pytest.raises(ValueError, match="capture groups"):
        compile_pattern("qwen", r"(a)(\d+)")
