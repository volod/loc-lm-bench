"""Knowledge-tree prompt candidate tests over the committed miniature graph fixture."""

import json

import pytest

from llb.prompt_system.budget import CharRatioTokenizer
from llb.prompt_system.knowledge_tree_render import MAX_TREE_DEPTH, render_knowledge_tree
from llb.prompt_system.knowledge_tree_source import load_knowledge_tree_source
from llb.prompt_system.pipeline import MANIFEST_FILE, prepare_prompt_system
from llb.prompt_system.template import GRAPH_NONE, METADATA_NONE

SAMPLE_GRAPH = "samples/prompt_system/ip_regulation_uk/graph"
SAMPLE_CORPUS = "samples/corpus"


@pytest.mark.parametrize("depth", range(1, MAX_TREE_DEPTH + 1))
@pytest.mark.parametrize("budget", [32, 64, 128, 256])
def test_rendered_tree_respects_every_depth_budget(depth, budget):
    tokenizer = CharRatioTokenizer()
    source = load_knowledge_tree_source(graph_dir=SAMPLE_GRAPH)

    rendered = render_knowledge_tree(
        source,
        depth=depth,
        budget_tokens=budget,
        tokenizer=tokenizer,
    )

    assert rendered.used_tokens == tokenizer.count(rendered.text)
    assert rendered.used_tokens <= budget
    assert rendered.depth == depth
    assert rendered.source_digest


def test_graph_tree_reuses_communities_and_typed_vocabulary():
    source = load_knowledge_tree_source(graph_dir=SAMPLE_GRAPH)
    rendered = render_knowledge_tree(
        source,
        depth=3,
        budget_tokens=512,
        tokenizer=CharRatioTokenizer(),
    )

    assert "Entity type LAW" in rendered.text
    assert "Community 0" in rendered.text
    assert "право інтелектуальної власності" in rendered.text


def test_prepare_adds_tree_variants_beside_no_tree_controls(tmp_path):
    run = prepare_prompt_system(
        SAMPLE_CORPUS,
        out_dir=tmp_path,
        context_window=4096,
        max_passages=2,
        anthology_sizes=[1],
        graph_styles=[GRAPH_NONE],
        metadata_densities=[METADATA_NONE],
        graph_dir=SAMPLE_GRAPH,
        knowledge_tree_depths=[1, 2],
        knowledge_tree_budgets=[64],
    )

    controls = [candidate for candidate in run.candidates if not candidate.knowledge_tree]
    trees = [candidate for candidate in run.candidates if candidate.knowledge_tree]
    assert len(controls) == 1
    assert {(c.fields.knowledge_tree_depth, c.fields.knowledge_tree_budget) for c in trees} == {
        (1, 64),
        (2, 64),
    }
    assert all(candidate.used_tokens <= run.budget.prompt_budget for candidate in run.candidates)
    assert len({candidate.prompt_system_id for candidate in run.candidates}) == 3
    assert {candidate.knowledge_tree["baseline_prompt_system_id"] for candidate in trees} == {
        controls[0].prompt_system_id
    }
    assert {candidate.knowledge_tree["requested_budget_tokens"] for candidate in trees} == {64}

    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["knowledge_tree_source"]["kind"] == "graph-store"
    assert {item["knowledge_tree_depth"] for item in manifest["candidates"]} == {0, 1, 2}


def test_tree_source_requires_existing_graph_artifacts(tmp_path):
    with pytest.raises(ValueError, match="missing nodes.jsonl"):
        load_knowledge_tree_source(graph_dir=tmp_path)


def test_tree_source_builds_existing_ontology_bundle_in_memory(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_dir.joinpath("doc.md").write_text("Alpha links beta.", encoding="utf-8")
    ontology = {
        "entity_types": [{"name": "CONCEPT", "count": 2, "confidence": 1.0, "examples": ["Alpha"]}],
        "relation_types": [
            {"name": "links", "count": 1, "confidence": 1.0, "examples": ["Alpha -> beta"]}
        ],
    }
    extraction = {
        "doc_id": "doc.md",
        "entities": [
            {
                "name": "Alpha",
                "type": "CONCEPT",
                "mentions": [{"doc_id": "doc.md", "char_start": 0, "char_end": 5, "text": "Alpha"}],
            },
            {
                "name": "beta",
                "type": "CONCEPT",
                "mentions": [
                    {"doc_id": "doc.md", "char_start": 12, "char_end": 16, "text": "beta"}
                ],
            },
        ],
        "facts": [
            {
                "subject": "Alpha",
                "relation": "links",
                "object": "beta",
                "evidence": {
                    "doc_id": "doc.md",
                    "char_start": 0,
                    "char_end": 17,
                    "text": "Alpha links beta.",
                },
            }
        ],
    }
    tmp_path.joinpath("ontology.json").write_text(json.dumps(ontology), encoding="utf-8")
    tmp_path.joinpath("extraction.jsonl").write_text(json.dumps(extraction), encoding="utf-8")

    source = load_knowledge_tree_source(ontology_bundle=tmp_path)

    assert source.source_kind == "ontology-bundle"
    assert source.entity_types == [("CONCEPT", 2)]
    assert source.relation_types == [("links", 1)]
    assert len(source.graph.nodes) == 2 and len(source.graph.community_members()) == 1
