"""Assembling a verified control bank into the geometries and the record the lanes read.

The claims themselves are generated and verified in `null_research_synthesis`; this module turns
what survived into corpus-shaped geometries (one per generation domain, so balancing can hold a
domain out), and into the payload that states what the bank cost, what the verifier made of it,
and what its size can support.
"""

import time
from dataclasses import dataclass

from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research.geometry import CorpusGeometry, EmbedTexts
from llb.conflicts.null_research.controls.synthesis import (
    MIN_VERIFIED_YIELD,
    SYNTHESIS_DOMAINS,
    SynthesizedControl,
    generate_controls,
    verify_controls,
)
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.prep.frontier.telemetry import LLMComplete

SECONDS_PER_HOUR = 3600.0


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
