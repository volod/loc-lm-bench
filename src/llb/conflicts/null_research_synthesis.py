"""In-support control synthesis: build the control bank out of the target corpus's own structure.

Every earlier control bank was COLLECTED (another corpus) or EDITED (a traced counterfactual). Both
fail for the same measured reason: a collected bank sits outside the target's covariate support, so
no weight can correct it, and an edited passage is adjudicated as a real conflict, so the family is
planted positives. This module takes the remaining option -- GENERATE a passage that keeps the
source's genre, length, and register while asserting something about a different subject, then let
the relation verifier decide whether the generated claim is actually unrelated.

The generated bank is only a null for the claims the verifier clears: a passage the model returns is
a candidate, never evidence. What survives is embedded with the target's own encoder and becomes a
control population that lives inside the target's support by construction.
"""

from dataclasses import dataclass
import time

from llb.conflicts.claim_prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import MIN_CLAIM_TOKENS, REL_COMPLEMENTARY
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research_geometry import CorpusGeometry, EmbedTexts
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.prep.frontier_parsing import parse_json_block
from llb.prep.frontier_telemetry import LLMComplete

SYNTHESIS_DOMAINS = ("generated_a", "generated_b")
# A generated passage shorter than the audit's own claim floor could not have entered the comparable
# population in the first place, so it cannot stand in for one of its rows.
MIN_SYNTHESIZED_TOKENS = MIN_CLAIM_TOKENS
# A bank whose members the verifier mostly rejects is not a cheap bank with some waste: it is the
# counterfactual failure again, so the lane refuses it rather than reporting the survivors.
MIN_VERIFIED_YIELD = 0.5
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class SynthesizedControl:
    """One generated candidate control claim, resolved back to the source it was written from."""

    dataset: str
    doc_id: str
    source_ordinal: int
    source_text: str
    text: str
    domain: str

    def record(self, ordinal: int) -> ChunkRecord:
        """The generated passage as a chunk record the geometry helpers can read."""
        return {
            "doc_id": f"{self.dataset}::generated::{self.doc_id}",
            "chunk_id": f"generated-{self.dataset}-{ordinal}",
            "char_start": 0,
            "char_end": len(self.text),
            "text": self.text,
        }


def synthesis_prompt(source_text: str) -> str:
    """One source passage -> a same-shaped passage about a different subject."""
    return f"""\
You write control passages for a document-collection audit.

Source passage:
\"\"\"
{source_text.strip()}
\"\"\"

Write ONE new passage in the same language, of similar length, and with the same document style as
the source (same register, same use of headings, numbering, dates, and figures where the source has
them).

Rules:
- The new passage must be about a DIFFERENT subject than the source.
- It must not restate, summarize, contradict, narrow, or generalize anything the source asserts.
- Someone comparing the two passages must be able to say they simply discuss different matters.
- Invent plausible details freely; the passage is a control, not a factual record.

Reply with JSON only, no prose and no code fence:
{{"passage": "<the new passage>"}}"""


class SynthesisError(ValueError):
    """The completion did not carry a usable control passage."""


def parse_synthesis(completion: str) -> str:
    """Parse a generated passage; raises `SynthesisError` when it is missing or too short."""
    try:
        payload = parse_json_block(completion)
    except Exception as exc:  # noqa: BLE001 -- any malformed completion is one failure mode
        raise SynthesisError(f"completion was not JSON: {exc}") from exc
    passage = payload.get("passage") if isinstance(payload, dict) else None
    text = passage.strip() if isinstance(passage, str) else ""
    if len(text.split()) < MIN_SYNTHESIZED_TOKENS:
        raise SynthesisError(f"passage has fewer than {MIN_SYNTHESIZED_TOKENS} tokens")
    return text


def source_samples(corpus: CorpusGeometry, *, per_document: int) -> list[tuple[int, str]]:
    """A deterministic, evenly spaced slice of each document's comparable chunks."""
    by_document: dict[str, list[int]] = {}
    for ordinal in corpus.allowed:
        by_document.setdefault(corpus.chunks[ordinal]["doc_id"], []).append(ordinal)
    samples: list[tuple[int, str]] = []
    for doc_id, ordinals in sorted(by_document.items()):
        step = max(1, len(ordinals) // per_document)
        samples.extend((ordinal, doc_id) for ordinal in ordinals[::step][:per_document])
    return samples


def generate_controls(
    corpus: CorpusGeometry, complete: LLMComplete, *, per_document: int
) -> tuple[list[SynthesizedControl], int]:
    """Ask the model for one control passage per sampled source chunk; count refusals."""
    controls: list[SynthesizedControl] = []
    unusable = 0
    for position, (ordinal, doc_id) in enumerate(source_samples(corpus, per_document=per_document)):
        source_text = corpus.chunks[ordinal]["text"]
        try:
            text = parse_synthesis(complete(synthesis_prompt(source_text)))
        except (SynthesisError, RuntimeError):
            unusable += 1
            continue
        controls.append(
            SynthesizedControl(
                dataset=corpus.name,
                doc_id=doc_id,
                source_ordinal=ordinal,
                source_text=source_text,
                text=text,
                domain=SYNTHESIS_DOMAINS[position % len(SYNTHESIS_DOMAINS)],
            )
        )
    return controls, unusable


def verify_controls(
    controls: list[SynthesizedControl], complete: LLMComplete
) -> list[tuple[SynthesizedControl, JsonObject]]:
    """Adjudicate every generated passage against its own source on the claim-tier prompt."""
    verified: list[tuple[SynthesizedControl, JsonObject]] = []
    for control in controls:
        try:
            relation: str | None = parse_adjudication(
                complete(adjudication_prompt(control.source_text, control.text))
            )["relation"]
            parsed = True
        except (AdjudicationError, RuntimeError):
            relation, parsed = None, False
        verified.append(
            (
                control,
                {
                    "dataset": control.dataset,
                    "doc_id": control.doc_id,
                    "source_ordinal": control.source_ordinal,
                    "generated_tokens": len(control.text.split()),
                    "relation": relation,
                    "parsed": parsed,
                    "conflicting": bool(parsed and relation != REL_COMPLEMENTARY),
                    "retained": bool(parsed and relation == REL_COMPLEMENTARY),
                },
            )
        )
    return verified


def synthetic_geometry(
    name: str, controls: list[SynthesizedControl], embed: EmbedTexts
) -> CorpusGeometry:
    """A control population addressed exactly like a corpus, without a store behind it.

    Only the fields the balancing lanes read are populated: chunk records for the structural
    covariates, raw vectors for the encoder covariates and the cosine scores, and `allowed` for the
    row order. Pair statistics belong to a corpus under audit and stay empty here on purpose.
    """
    vectors = VectorSet.from_any(embed([control.text for control in controls]))
    return CorpusGeometry(
        name=name,
        corpus_root="",
        store_dir="",
        embedding_model="",
        corpus_fingerprint="",
        chunks=[control.record(ordinal) for ordinal, control in enumerate(controls)],
        raw_vectors=vectors,
        vectors=vectors,
        allowed=list(range(len(controls))),
        observed_similarities=[],
        document_maxima={},
        center_mean=None,
        excluded={},
    )


def scale_projection(elapsed_seconds: float, retained: int, required_units: int) -> JsonObject:
    """What the measured generation rate says about reaching the required control-bank size."""
    per_claim = elapsed_seconds / retained if retained else 0.0
    rate = SECONDS_PER_HOUR / per_claim if per_claim > 0.0 else 0.0
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "retained_claims": retained,
        "seconds_per_retained_claim": round(per_claim, 3),
        "verified_claims_per_hour": round(rate, 3),
        "required_independent_units": required_units,
        "hours_to_required_units": round(required_units / rate, 1) if rate > 0.0 else None,
        "years_to_required_units": round(required_units / rate / 24.0 / 365.0, 2)
        if rate > 0.0
        else None,
    }


@dataclass(frozen=True)
class SynthesizedBank:
    """One corpus's generated control bank: its geometries, what survived, and how it went."""

    geometries: dict[str, CorpusGeometry]
    retained: list[SynthesizedControl]
    payload: JsonObject


def synthesize_bank(
    corpus: CorpusGeometry,
    complete: LLMComplete,
    embed: EmbedTexts,
    *,
    per_document: int,
    required_units: int,
) -> SynthesizedBank:
    """Generate, verify, and embed one corpus's in-support control bank."""
    started = time.monotonic()
    generated, unusable = generate_controls(corpus, complete, per_document=per_document)
    verified = verify_controls(generated, complete)
    elapsed = time.monotonic() - started
    retained = [control for control, verdict in verified if verdict["retained"]]
    banks = _domain_geometries(corpus.name, retained, embed)
    return SynthesizedBank(
        geometries=banks,
        retained=retained,
        payload=_bank_payload(
            verified,
            banks,
            attempted=len(generated) + unusable,
            unusable=unusable,
            elapsed=elapsed,
            required_units=required_units,
        ),
    )


def _domain_geometries(
    name: str, retained: list[SynthesizedControl], embed: EmbedTexts
) -> dict[str, CorpusGeometry]:
    """One geometry per generation domain, so balancing can hold a domain out."""
    return {
        domain: synthetic_geometry(
            f"{name}_{domain}",
            [control for control in retained if control.domain == domain],
            embed,
        )
        for domain in SYNTHESIS_DOMAINS
        if any(control.domain == domain for control in retained)
    }


def _bank_payload(
    verified: list[tuple[SynthesizedControl, JsonObject]],
    banks: dict[str, CorpusGeometry],
    *,
    attempted: int,
    unusable: int,
    elapsed: float,
    required_units: int,
) -> JsonObject:
    """What the bank cost, what the verifier made of it, and what its size can support."""
    retained = sum(bool(verdict["retained"]) for _, verdict in verified)
    yield_rate = retained / attempted if attempted else 0.0
    lower, _ = wilson_interval(retained, attempted)
    return {
        "sampled_sources": attempted,
        "unusable_completions": unusable,
        "generated_claims": len(verified),
        "conflicting_claims": sum(bool(verdict["conflicting"]) for _, verdict in verified),
        "retained_claims": retained,
        "verified_yield": round(yield_rate, 6),
        "verified_yield_wilson_lcb": round(lower, 6),
        "relations": {
            relation: sum(verdict["relation"] == relation for _, verdict in verified)
            for relation in sorted(
                {str(verdict["relation"]) for _, verdict in verified if verdict["parsed"]}
            )
        },
        "domains": {domain: len(bank.allowed) for domain, bank in banks.items()},
        "yield_sufficient": yield_rate >= MIN_VERIFIED_YIELD,
        "scale": scale_projection(elapsed, retained, required_units),
        "verdicts": [verdict for _, verdict in verified],
    }
