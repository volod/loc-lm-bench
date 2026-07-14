"""Tests for ontology full flow yield."""

import json
from llb.goldset.chains import load_chains, validate_chains
from llb.goldset.schema import load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ontology.constants import (
    PDF_ONTOLOGY_REPORT_FILENAME,
)
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig, EndpointPlan
from llb.prep.ontology.pipeline.run import draft_goldset
from ontology_yield_helpers import CHAIN_DOC, FakeEmbedder, _chain_endpoint


def test_full_flow_multi_hop_adds_multi_span_items_and_labels(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    out = tmp_path / "bundle"

    config = EndpointConfig(kind="local", model="fake")
    result = draft_goldset(
        corpus,
        EndpointPlan.single(config),
        completers=EndpointCompleters.single(_chain_endpoint),
        max_items=50,
        coverage_target=2,
        multi_hop=True,
        chains=True,
        out_dir=out,
    )

    multi_hop_items = [
        item for item in result.items if result.item_labels[item.id].question_type == "multi-hop"
    ]
    assert multi_hop_items, "expected at least one multi-hop chain item"
    assert all(len(item.source_spans) >= 2 for item in multi_hop_items)

    # the emitted bundle self-validates (multi-hop spans are exact)
    loaded = load_goldset(out / "goldset.jsonl")
    assert validate_items(loaded, out / "corpus")["errors"] == []
    loaded_chains = load_chains(out / "chains.jsonl")
    assert validate_chains(loaded_chains, out / "corpus")["errors"] == []

    report = json.loads((out / PDF_ONTOLOGY_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["coverage_matrix"]["mode"] == "coverage-target"
    assert report["question_type_distribution"].get("multi-hop", 0) >= 1

    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["stages"]["multi_hop_items"] >= 1
    assert prov["stages"]["chains"] >= 1
    assert "seed_coverage" in prov


def test_full_flow_dedup_against_prior_bundle_drops_repeats(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"

    config = EndpointConfig(kind="local", model="fake")
    plan = EndpointPlan.single(config)
    completers = EndpointCompleters.single(_chain_endpoint)
    draft_goldset(
        corpus,
        plan,
        completers=completers,
        max_items=50,
        out_dir=first,
    )
    result = draft_goldset(
        corpus,
        plan,
        completers=completers,
        max_items=50,
        out_dir=second,
        dedup_against=[first],
        dedup_embedder=FakeEmbedder(),
    )

    # the same corpus + seed reproduces the same questions, all near-duplicates of the first bundle
    assert result.dedup_report is not None
    assert result.dedup_report["dropped"] >= 1
    assert result.items == [] or len(result.items) < len(load_goldset(first / "goldset.jsonl"))
    prov = json.loads((second / "provenance.json").read_text(encoding="utf-8"))
    assert prov["dedup"]["prior_bundles"] == [str(first)]
