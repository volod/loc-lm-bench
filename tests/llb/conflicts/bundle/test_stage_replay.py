"""Reading a finished audit's stage attribution back out of the bundle it wrote.

The attribution names the ONE knob that lost an orderable document pair, and it is decided from the
per-document chunk accounting the semantic tier folded plus the corpus's own ordering fields --
neither of which survives the run today. Re-reading a bundle therefore means rebuilding the store,
and a store rebuilt since answers a different question while looking like the same recompute. These
tests pin the record that ends that:

1. equality -- the attribution recomputed from `summary.json` plus the bundle's own rows is the one
   the run recorded, at every stage a pair can be lost at;
2. independence -- the replay reads neither store nor corpus, so it survives both being gone;
3. refusal -- a bundle written before the record answers "not recomputable" rather than a stage
   recomputed against something else;
4. the absence that carries meaning -- a run below the semantic tier records NO chunk accounting,
   and an empty one would say the opposite.

Everything goes through a JSON round-trip, because the record's whole purpose is to be read off
disk months later.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import STAGE_INPUTS_FIELD, TIER_HASH, TIER_SEMANTIC
from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.bundle.document_index import DocumentInterner, DocumentNaming
from llb.conflicts.governance.stage import LOST_PAIR_FIELD
from llb.conflicts.governance.stage_rule import (
    STAGE_CANDIDATES,
    STAGE_CHUNKING,
    STAGE_CLAIM_FLOOR,
    STAGE_DUPLICATE_COLLAPSE,
    STAGE_EFFORT,
)

from llb.conflicts.models import AuditResult
from llb.conflicts.report.stage_replay import replay_line, replay_report
from llb.conflicts.bundle.record import (
    CANDIDATES_KEY,
    CHUNKS_KEY,
    DOCUMENTS_KEY,
    DOC_ID_KEY,
    EXCLUSIONS_KEY,
    NO_RECORD_REASON,
    SCHEMA_KEY,
    STAGE_INPUTS_SCHEMA_VERSION,
    recorded_inputs,
)
from llb.conflicts.bundle.stage_replay import (
    recorded_attribution,
    replay_attribution,
    replay_entry,
)

from tests.llb.conflicts.conflict_helpers import (
    BODY_TOKENS,
    FAKE_COS_THRESHOLD,
    dated_body,
    dated_corpus,
    store_over,
)

SHORT = "коротка примітка"

# Three corpus shapes, read five ways below -- one per stage a pair can stop at.
UNRELATED = {
    "old.md": ("2021-01-01", dated_body(0)),
    "new.md": ("2024-01-01", dated_body(500)),
}
BELOW_THE_FLOOR = {
    "old.md": ("2021-01-01", dated_body(0)),
    "new.md": ("2024-01-01", SHORT),
}
WITH_A_COPY = {
    "handbook.md": ("2026-01-01", f"{dated_body(0, BODY_TOKENS - 2)} {dated_body(900, 2)}"),
    "policy.md": ("2024-01-01", dated_body(0)),
    "policy_copy.md": ("2024-01-01", dated_body(0)),
}


def _audit(
    root: Path,
    documents: dict[str, tuple[str, str]],
    *,
    effort: str = TIER_SEMANTIC,
    without: str = "",
) -> AuditResult:
    """One real audit over a throwaway dated corpus, with a store only from the semantic tier up."""
    corpus = dated_corpus(root, documents)
    store = store_over(corpus, without=without) if effort == TIER_SEMANTIC else None
    return run_audit(
        corpus, AuditParams(effort=effort, cos_threshold=FAKE_COS_THRESHOLD), store=store
    )


def _bundle(result: AuditResult) -> tuple[dict, list]:
    """The two files a replay reads, through JSON exactly as `write_audit` puts them on disk."""
    return (
        json.loads(json.dumps(result.summary(), ensure_ascii=False)),
        json.loads(json.dumps(result.rows(), ensure_ascii=False)),
    )


# --- equality: the bundle answers what the run answered ----------------------------------------


@pytest.mark.parametrize(
    ("documents", "options", "stage"),
    [
        (UNRELATED, {"effort": TIER_HASH}, STAGE_EFFORT),
        (UNRELATED, {"without": "new.md"}, STAGE_CHUNKING),
        (UNRELATED, {}, STAGE_CANDIDATES),
        (BELOW_THE_FLOOR, {}, STAGE_CLAIM_FLOOR),
        (WITH_A_COPY, {"without": "policy_copy.md"}, STAGE_DUPLICATE_COLLAPSE),
    ],
)
def test_the_recompute_from_the_bundle_equals_the_live_run(tmp_path, documents, options, stage):
    """Every stage, recomputed from what the run wrote down instead of from what it read."""
    result = _audit(tmp_path / "corpus", documents, **options)
    summary, rows = _bundle(result)

    replay = replay_attribution(summary, rows)

    assert replay.recomputable and replay.reason == ""
    assert replay.stage == stage
    assert replay.attribution == recorded_attribution(summary), "the pair and its reason too"
    assert replay.attribution[LOST_PAIR_FIELD] == result.governance_coverage[LOST_PAIR_FIELD]


def test_a_run_that_lost_nothing_replays_as_nothing_lost(tmp_path):
    """Recomputable and empty is a real answer, and the opposite of unanswerable."""
    body = dated_body(0)
    result = _audit(
        tmp_path / "corpus",
        {"old.md": ("2021-01-01", body), "new.md": ("2024-01-01", body)},
    )
    summary, rows = _bundle(result)

    replay = replay_attribution(summary, rows)

    assert result.governance_coverage["orderable_pairs"] == 1, "the pair was returned"
    assert (replay.recomputable, replay.attribution, replay.stage) == (True, {}, None)
    assert replay_entry("run", "summary.json", summary, rows)["agrees"] is True


def test_an_undated_corpus_replays_as_nothing_to_attribute(tmp_path):
    """No orderable pair exists, so the replay invents no stage -- the same silence the run kept."""
    result = _audit(
        tmp_path / "corpus",
        {"a.md": ("2021-01-01", dated_body(0)), "b.md": ("2021-01-01", dated_body(500))},
    )
    summary, rows = _bundle(result)

    replay = replay_attribution(summary, rows)

    assert summary["governance_coverage"]["orderable_document_pairs"] == 0
    assert (replay.recomputable, replay.attribution) == (True, {})
    assert recorded_attribution(summary) == {}


# --- independence: neither the store nor the corpus is read ------------------------------------


def test_the_replay_survives_the_store_and_the_corpus_being_gone(tmp_path):
    """The whole point: a bundle is re-read months later, when both inputs have moved on.

    The same corpus audited against a complete store loses its pair at candidate selection instead,
    so a recompute that reached for today's store would quietly answer about today's store.
    """
    corpus_root = tmp_path / "corpus"
    stale = _audit(corpus_root, UNRELATED, without="new.md")
    summary, rows = _bundle(stale)
    rebuilt = _audit(corpus_root, UNRELATED)
    shutil.rmtree(corpus_root)

    replay = replay_attribution(summary, rows)

    assert rebuilt.governance_coverage[LOST_PAIR_FIELD]["stage"] == STAGE_CANDIDATES
    assert replay.stage == STAGE_CHUNKING, "the bundle keeps the answer its own run gave"
    assert replay.lost["documents"] == ["new.md", "old.md"]


def test_the_record_carries_counts_and_ordering_fields_only(tmp_path):
    """No chunk text and no ordinals: the record says what a document REACHED, never what it said."""
    result = _audit(tmp_path / "corpus", WITH_A_COPY, without="policy_copy.md")
    summary, _ = _bundle(result)
    record = summary[STAGE_INPUTS_FIELD]

    assert set(record) == {SCHEMA_KEY, DOCUMENTS_KEY, CHUNKS_KEY, EXCLUSIONS_KEY, CANDIDATES_KEY}
    assert set(record[CHUNKS_KEY]) == {"stored", "comparable", "copies"}
    assert all(
        isinstance(count, int)
        for name in ("stored", "comparable")
        for count in record[CHUNKS_KEY][name].values()
    )
    copies = recorded_inputs(record).chunks.copies
    assert set(copies) == {"policy.md", "policy_copy.md"}
    assert [entry[DOC_ID_KEY] for entry in record[DOCUMENTS_KEY]] == sorted(WITH_A_COPY)
    assert {key for entry in record[DOCUMENTS_KEY] for key in entry} == {
        DOC_ID_KEY,
        "effective_date",
    }


def test_the_chunk_accounting_round_trips_through_the_record():
    """`stored` / `comparable` / `copies` back out as they went in, copies included."""
    doc_ids = ["a.md", "b.md", "kept.md"]
    chunks = DocumentChunks(
        stored={"a.md": 3, "kept.md": 2},
        comparable={"a.md": 3},
        copies={"b.md": ("kept.md",), "kept.md": ("b.md",)},
    )
    written = json.loads(json.dumps(chunks.payload(DocumentInterner(doc_ids))))

    assert DocumentChunks.from_payload(written, DocumentNaming.by_position(doc_ids)) == chunks


# --- refusal: what a bundle that cannot answer says --------------------------------------------


def test_a_bundle_without_the_record_is_not_recomputable_rather_than_wrong(tmp_path):
    """The degradation an archive of older runs has to take: no answer beats an answer from a
    store that has been rebuilt since. The run's own attribution is still readable beside it."""
    result = _audit(tmp_path / "corpus", UNRELATED, without="new.md")
    summary, rows = _bundle(result)
    summary.pop(STAGE_INPUTS_FIELD)

    replay = replay_attribution(summary, rows)
    entry = replay_entry("older-run", "summary.json", summary, rows)

    assert (replay.recomputable, replay.attribution, replay.stage) == (False, {}, None)
    assert replay.reason == NO_RECORD_REASON
    assert entry["agrees"] is None, "an unanswerable bundle disagrees with nothing"
    assert entry["recorded"]["stage"] == STAGE_CHUNKING
    assert entry["recomputed"] is None


def test_a_record_from_a_newer_audit_is_refused_rather_than_read_halfway(tmp_path):
    """A field this build does not know about could be exactly the one the stage now turns on."""
    result = _audit(tmp_path / "corpus", UNRELATED, without="new.md")
    summary, rows = _bundle(result)
    summary[STAGE_INPUTS_FIELD][SCHEMA_KEY] = STAGE_INPUTS_SCHEMA_VERSION + 1

    replay = replay_attribution(summary, rows)

    assert not replay.recomputable
    assert str(STAGE_INPUTS_SCHEMA_VERSION + 1) in replay.reason


def test_a_run_below_the_semantic_tier_records_no_chunk_accounting_at_all(tmp_path):
    """Absent is the `effort` reading; an empty accounting would name CHUNKING on every pair."""
    result = _audit(tmp_path / "corpus", UNRELATED, effort=TIER_HASH)
    summary, rows = _bundle(result)

    assert CHUNKS_KEY not in summary[STAGE_INPUTS_FIELD]
    assert replay_attribution(summary, rows).stage == STAGE_EFFORT

    summary[STAGE_INPUTS_FIELD][CHUNKS_KEY] = {"stored": {}, "comparable": {}, "copies": {}}
    assert replay_attribution(summary, rows).stage == STAGE_CHUNKING, "which is the opposite claim"


# --- the operator's read -----------------------------------------------------------------------


def test_the_report_puts_both_readings_on_one_row(tmp_path):
    """A rule change is scored on the bundles where the two readings PART, so both are printed."""
    result = _audit(tmp_path / "corpus", UNRELATED, without="new.md")
    summary, rows = _bundle(result)
    entries = [replay_entry("kept", "a/summary.json", summary, rows)]
    stripped = dict(summary)
    stripped.pop(STAGE_INPUTS_FIELD)
    entries.append(replay_entry("older", "b/summary.json", stripped, rows))

    report = replay_report(entries)

    assert (
        "| `kept` | CHUNKING (`new.md` + `old.md`) | CHUNKING (`new.md` + `old.md`) | yes |"
        in report
    )
    assert "| `older` | CHUNKING (`new.md` + `old.md`) | not recomputable | -- |" in report
    assert "- bundles read: 2" in report
    assert "- recomputed stage differs from the recorded one: 0" in report
    assert "- not recomputable (no per-document record): 1" in report
    assert f"- {NO_RECORD_REASON} -- 1 of 1" in report, "counted per reason, explained once"
    assert "agrees with the run" in replay_line(entries[0])
    assert NO_RECORD_REASON in replay_line(entries[1])


def test_the_cli_reads_a_run_directory_and_writes_both_artifacts(tmp_path):
    from typer.testing import CliRunner

    from llb.cli.app import app
    from llb.conflicts.report.render import write_audit
    from llb.main import main  # noqa: F401  -- registers every command module

    run_dir = tmp_path / "audit"
    write_audit(run_dir, _audit(tmp_path / "corpus", UNRELATED, without="new.md"))
    out_dir = tmp_path / "stage"

    invoked = CliRunner().invoke(
        app, ["recompute-conflict-stage", "--run", str(run_dir), "--out", str(out_dir)]
    )

    assert invoked.exit_code == 0, invoked.output
    data = json.loads((out_dir / "stage.json").read_text(encoding="utf-8"))
    assert data[0]["recomputable"] and data[0]["agrees"] is True
    assert data[0]["recomputed"]["stage"] == STAGE_CHUNKING
    assert "CHUNKING" in (out_dir / "stage.md").read_text(encoding="utf-8")
