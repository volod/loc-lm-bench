"""Calibrate the claim-tier adjudicator against frozen labels before quoting its precision.

A precision figure computed from a model's own verdicts is only as good as the model. The audit
therefore adjudicates a COMMITTED probe -- section pairs of the planted fixture whose relation is
fixed by construction, half actionable and half complementary -- with the same prompt and the same
endpoint it uses on the operator's corpus, and reports precision only when the model agrees with
those labels at a lower bound that clears `MIN_ADJUDICATOR_ACCURACY_LCB`.

The probe stores document ids and heading lines rather than passages, so the passages are the exact
corpus bytes at run time: a fixture edit that moves the text cannot silently leave the probe
asserting a label about a passage that no longer exists.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from llb.conflicts.claim_prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import REL_COMPLEMENTARY
from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.interval_stats import wilson_interval
from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_project_path
from llb.prep.frontier_telemetry import LLMComplete

DEFAULT_CALIBRATION_PROBE = "samples/corpora/conflicts_uk_v1/adjudicator_probe.json"
# The adjudicator must agree with the frozen labels at this Wilson 95% lower bound before any
# precision measured with it is printed. Same gate the independent-null research lane applies.
MIN_ADJUDICATOR_ACCURACY_LCB = 0.60


@dataclass(frozen=True)
class ProbePair:
    """One frozen-label passage pair: what the adjudicator sees and what it should answer."""

    pair_id: str
    left_text: str
    right_text: str
    relation: str
    actionable: bool


@dataclass(frozen=True)
class CalibrationProbe:
    """The committed probe, resolved against the fixture corpus it addresses."""

    probe_id: str
    corpus_root: str
    pairs: list[ProbePair]


def _sections(text: str, body_offset: int) -> dict[str, str]:
    """Split a document body into `heading line -> heading + section text`."""
    sections: dict[str, str] = {}
    heading = ""
    buffer: list[str] = []
    for line in text[body_offset:].splitlines():
        if line.startswith("#"):
            if heading:
                sections[heading] = "\n".join([heading, *buffer]).strip()
            if line.strip() in sections:
                raise SystemExit(f"[conflicts] probe corpus repeats the heading {line.strip()!r}")
            heading, buffer = line.strip(), []
            continue
        buffer.append(line)
    if heading:
        sections[heading] = "\n".join([heading, *buffer]).strip()
    return sections


def _passage(sections_by_doc: dict[str, dict[str, str]], side: JsonObject) -> str:
    doc_id, heading = str(side["doc_id"]), str(side["heading"])
    sections = sections_by_doc.get(doc_id)
    if sections is None:
        raise SystemExit(f"[conflicts] calibration probe names an unknown document {doc_id!r}")
    if heading not in sections:
        raise SystemExit(f"[conflicts] {doc_id} has no heading {heading!r} for the probe")
    return sections[heading]


def load_calibration_probe(path: Path | str | None = None) -> CalibrationProbe:
    """Read the probe and resolve each side to the exact section bytes it names."""
    probe_path = resolve_project_path(path if path is not None else DEFAULT_CALIBRATION_PROBE)
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    corpus_root = resolve_project_path(str(payload["corpus"]))
    sections_by_doc = {
        doc.doc_id: _sections(doc.text, doc.body_offset) for doc in load_corpus_docs(corpus_root)
    }
    pairs = [
        ProbePair(
            pair_id=str(entry["pair_id"]),
            left_text=_passage(sections_by_doc, entry["a"]),
            right_text=_passage(sections_by_doc, entry["b"]),
            relation=str(entry["relation"]),
            actionable=bool(entry["actionable"]),
        )
        for entry in payload["pairs"]
    ]
    if not pairs:
        raise SystemExit(f"[conflicts] calibration probe {probe_path} has no pairs")
    return CalibrationProbe(
        probe_id=str(payload["probe_id"]), corpus_root=str(payload["corpus"]), pairs=pairs
    )


def _probe_verdict(pair: ProbePair, complete: LLMComplete) -> JsonObject:
    record: JsonObject = {
        "pair_id": pair.pair_id,
        "expected_relation": pair.relation,
        "expected_actionable": pair.actionable,
    }
    try:
        parsed = parse_adjudication(complete(adjudication_prompt(pair.left_text, pair.right_text)))
    except (AdjudicationError, RuntimeError) as exc:
        return {**record, "parsed": False, "relation": None, "error": str(exc)[:200]}
    actionable = parsed["relation"] != REL_COMPLEMENTARY
    return {
        **record,
        "parsed": True,
        "relation": parsed["relation"],
        "actionable": actionable,
        "agrees": actionable == pair.actionable,
    }


def calibrate_adjudicator(probe: CalibrationProbe, complete: LLMComplete) -> JsonObject:
    """Agreement between this adjudicator and the probe's frozen actionable/complementary labels.

    Agreement is measured on the ACTIONABLE binary rather than on the exact relation: a duplicate
    reported as `subsumes` still sends the operator to the same decision, while a conflict reported
    as `complementary` is the error a precision figure would hide.
    """
    verdicts = [_probe_verdict(pair, complete) for pair in probe.pairs]
    parsed = [verdict for verdict in verdicts if verdict["parsed"]]
    unparsed = len(verdicts) - len(parsed)
    agreements = sum(bool(verdict["agrees"]) for verdict in parsed)
    lower, upper = wilson_interval(agreements, len(parsed))
    positives = [verdict for verdict in parsed if verdict["expected_actionable"]]
    negatives = [verdict for verdict in parsed if not verdict["expected_actionable"]]
    calibrated = bool(parsed) and not unparsed and lower >= MIN_ADJUDICATOR_ACCURACY_LCB
    return {
        "probe_id": probe.probe_id,
        "probe_corpus": probe.corpus_root,
        "probe_pairs": len(verdicts),
        "parsed_pairs": len(parsed),
        "unparsed_pairs": unparsed,
        "agreements": agreements,
        "accuracy": round(agreements / len(parsed), 6) if parsed else 0.0,
        "accuracy_wilson_95": [round(lower, 6), round(upper, 6)],
        "labelled_actionable": len(positives),
        "labelled_complementary": len(negatives),
        "recall_on_actionable": round(
            sum(bool(verdict["actionable"]) for verdict in positives) / len(positives), 6
        )
        if positives
        else None,
        "specificity_on_complementary": round(
            sum(not verdict["actionable"] for verdict in negatives) / len(negatives), 6
        )
        if negatives
        else None,
        "min_accuracy_lcb": MIN_ADJUDICATOR_ACCURACY_LCB,
        "calibrated": calibrated,
        "verdicts": verdicts,
    }
