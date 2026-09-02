"""The frozen adjudicator-calibration probe: its file format and how it resolves to passages.

The probe is committed rather than generated, and it addresses passages by document id and
heading line rather than by copied text, so the passages an adjudicator sees are the exact corpus
bytes at run time: a fixture edit that moves the text fails the run loudly instead of leaving a
frozen label attached to a passage that no longer exists.

It is TIERED, because one difficulty cannot do two jobs. The `base` tier is the floor -- plainly
changed deadlines against plainly different subjects -- and answers "is this adjudicator broken?".
The `hard` tier is the separator: pairs whose actionable/complementary split is arguable on a
shallow reading and determinate on a close one, which is what tells two working adjudicators apart.
Each tier names its own corpus, so a tier can be sharpened without editing the planted detector
fixture the other one is drawn from.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from llb.conflicts.corpus import load_corpus_docs
from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_project_path

DEFAULT_CALIBRATION_PROBE = "samples/corpora/conflicts_uk_v1/adjudicator_probe.json"

BASE_TIER = "base"
HARD_TIER = "hard"
# The order tiers are reported in: easiest first, so a report reads as a difficulty ladder.
PROBE_TIERS = (BASE_TIER, HARD_TIER)


@dataclass(frozen=True)
class ProbePair:
    """One frozen-label passage pair: what the adjudicator sees and what it should answer."""

    pair_id: str
    tier: str
    left_text: str
    right_text: str
    relation: str
    actionable: bool


@dataclass(frozen=True)
class CalibrationProbe:
    """The committed probe, resolved against the fixture corpora its tiers address."""

    probe_id: str
    corpora: dict[str, str]
    pairs: list[ProbePair]

    @property
    def tiers(self) -> tuple[str, ...]:
        """Every tier the probe carries pairs for, easiest first."""
        present = {pair.tier for pair in self.pairs}
        ordered = [tier for tier in PROBE_TIERS if tier in present]
        return tuple(ordered + sorted(present - set(ordered)))

    def pairs_of(self, tier: str) -> list[ProbePair]:
        return [pair for pair in self.pairs if pair.tier == tier]


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


def _corpus_sections(corpus: str) -> dict[str, dict[str, str]]:
    """Every document of one probe corpus, split into its heading-addressed sections."""
    return {
        doc.doc_id: _sections(doc.text, doc.body_offset)
        for doc in load_corpus_docs(resolve_project_path(corpus))
    }


def _passage(sections_by_doc: dict[str, dict[str, str]], side: JsonObject) -> str:
    doc_id, heading = str(side["doc_id"]), str(side["heading"])
    sections = sections_by_doc.get(doc_id)
    if sections is None:
        raise SystemExit(f"[conflicts] calibration probe names an unknown document {doc_id!r}")
    if heading not in sections:
        raise SystemExit(f"[conflicts] {doc_id} has no heading {heading!r} for the probe")
    return sections[heading]


def _declared_tiers(payload: JsonObject) -> list[JsonObject]:
    """The probe's tier blocks, reading a single-tier probe as one `base` tier."""
    declared = payload.get("tiers")
    if isinstance(declared, list) and declared:
        return [dict(block) for block in declared]
    return [{"tier": BASE_TIER, "corpus": payload["corpus"], "pairs": payload["pairs"]}]


def _tier_pairs(block: JsonObject) -> list[ProbePair]:
    tier = str(block["tier"])
    sections_by_doc = _corpus_sections(str(block["corpus"]))
    return [
        ProbePair(
            pair_id=str(entry["pair_id"]),
            tier=tier,
            left_text=_passage(sections_by_doc, entry["a"]),
            right_text=_passage(sections_by_doc, entry["b"]),
            relation=str(entry["relation"]),
            actionable=bool(entry["actionable"]),
        )
        for entry in block["pairs"]
    ]


def load_calibration_probe(
    path: Path | str | None = None, tiers: tuple[str, ...] | None = None
) -> CalibrationProbe:
    """Read the probe and resolve each side to the exact section bytes it names.

    `tiers` keeps only the named tiers, which is how a run pays for the floor alone when the
    separator would tell it nothing new.
    """
    probe_path = resolve_project_path(path if path is not None else DEFAULT_CALIBRATION_PROBE)
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    blocks = _declared_tiers(payload)
    declared = [str(block["tier"]) for block in blocks]
    if len(set(declared)) != len(declared):
        raise SystemExit(f"[conflicts] calibration probe {probe_path} repeats a tier name")
    wanted = set(tiers) if tiers is not None else set(declared)
    unknown = sorted(wanted - set(declared))
    if unknown:
        raise SystemExit(
            f"[conflicts] calibration probe {probe_path} has no tier {', '.join(unknown)}; "
            f"it carries {', '.join(declared)}"
        )
    kept = [block for block in blocks if str(block["tier"]) in wanted]
    pairs = [pair for block in kept for pair in _tier_pairs(block)]
    if not pairs:
        raise SystemExit(f"[conflicts] calibration probe {probe_path} has no pairs")
    identifiers = [pair.pair_id for pair in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit(f"[conflicts] calibration probe {probe_path} repeats a pair id")
    return CalibrationProbe(
        probe_id=str(payload["probe_id"]),
        corpora={str(block["tier"]): str(block["corpus"]) for block in kept},
        pairs=pairs,
    )
