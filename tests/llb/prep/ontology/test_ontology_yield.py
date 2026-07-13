"""draft-yield-quality-max: coverage targets, multi-hop paths, dedup, and question-type labels.

Every LLM/embedder call is an injected fake, so coverage-target sampling, 2-hop graph-path walking,
multi-span multi-hop grounding, near-duplicate suppression, and the per-question-type report are all
exercised deterministically with no server, key, or GPU.
"""

import json

from llb.goldset.chains import load_chains, validate_chains
from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.goldset.validate import validate_items
from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph
from llb.prep.ontology.artifacts import write_calibration_artifacts
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROVENANCE_KIND,
    QUESTION_TYPE_MULTI_HOP,
)
from llb.prep.ontology.coverage import coverage_report, select_seeds
from llb.prep.ontology.dedup import NearDuplicateFilter, load_prior_questions
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig, EndpointPlan
from llb.prep.ontology.chains import build_chain_items
from llb.prep.ontology.graph_paths import walk_chain_paths, walk_two_hop_paths
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    DraftSeed,
    Entity,
    ItemLabels,
    OntologyCandidate,
)
from llb.prep.ontology.multi_hop import build_multi_hop_items, draft_multi_hop
from llb.prep.ontology.pipeline.run import draft_goldset
from llb.prep.ontology.question_types import classify_question_type
from llb.prep.ontology.refine import refine_drafts_labeled

CHAIN_DOC = "# Мережа\n\nAlpha керує Beta. Beta належить Gamma.\n"


def _span(doc_id: str, char_start: int, text: str) -> SourceSpan:
    return SourceSpan(
        doc_id=doc_id, char_start=char_start, char_end=char_start + len(text), text=text
    )


def _relation_pool(n: int) -> list[DraftSeed]:
    """A pool of `n` seeds, each in a DISTINCT relation bucket (shared doc/section/difficulty)."""
    return [
        DraftSeed(
            doc_id="a.md",
            kind="fact",
            section_title="S",
            difficulty="medium",
            strata={"doc": "a.md", "section": "S", "difficulty": "medium", "relation": f"r{i}"},
            evidence=_span("a.md", 0, "abc"),
        )
        for i in range(n)
    ]


# --- coverage-target sampling + exhaustion report --------------------------------------------


def test_coverage_target_drafts_more_than_a_small_flat_cap_and_reports_exhaustion():
    pool = _relation_pool(10)
    flat = select_seeds(pool, max_items=3, seed=7)
    covered = select_seeds(pool, max_items=100, seed=7, coverage_target=1)

    assert len(flat) == 3  # the flat cap stops early
    assert len(covered) == 10  # coverage-target spans every relation bucket
    assert len(covered) > len(flat)

    report = coverage_report(pool, covered, coverage_target=1, max_items=100)
    assert report["mode"] == "coverage-target"
    assert report["drafted_seeds"] == 10 and report["seeds_remaining"] == 0
    assert report["exhausted"] is True
    relation = report["strata"]["relation"]
    assert relation["buckets"] == 10 and relation["buckets_drafted"] == 10
    assert relation["buckets_at_target"] == 10 and relation["seeds_remaining"] == 0


def test_flat_cap_report_records_undrafted_seeds_remaining():
    pool = _relation_pool(10)
    flat = select_seeds(pool, max_items=3, seed=7)
    report = coverage_report(pool, flat, coverage_target=None, max_items=3)
    assert report["mode"] == "flat-cap"
    assert report["drafted_seeds"] == 3 and report["seeds_remaining"] == 7
    assert report["exhausted"] is False
    assert report["strata"]["relation"]["seeds_remaining"] == 7


# --- question-type taxonomy ------------------------------------------------------------------


def test_classify_question_type_closed_taxonomy():
    assert classify_question_type("Що таке столиця?", "місто") == "definition"
    assert classify_question_type("У якому році засновано місто?", "1256 рік") == "numeric"
    assert classify_question_type("Скільки відсотків?", "багато") == "numeric"
    assert classify_question_type("Чим відрізняється А від Б?", "усім") == "comparative"
    assert classify_question_type("Як подати заяву?", "через портал") == "procedural"
    assert classify_question_type("Хто керує містом?", "мер") == "factoid"


# --- multi-hop graph-path seeds --------------------------------------------------------------


def _chain_graph() -> KnowledgeGraph:
    e1 = CHAIN_DOC.index("Alpha керує Beta")
    e2 = CHAIN_DOC.index("Beta належить Gamma")
    m1: GraphMention = {
        "doc_id": "chain.md",
        "char_start": e1,
        "char_end": e1 + len("Alpha керує Beta"),
        "text": "Alpha керує Beta",
        "section_title": "Мережа",
    }
    m2: GraphMention = {
        "doc_id": "chain.md",
        "char_start": e2,
        "char_end": e2 + len("Beta належить Gamma"),
        "text": "Beta належить Gamma",
        "section_title": "Мережа",
    }
    nodes = [
        GraphNode(node_id=0, name="Alpha", type="ORG", confidence=1.0),
        GraphNode(node_id=1, name="Beta", type="ORG", confidence=1.0),
        GraphNode(node_id=2, name="Gamma", type="ORG", confidence=1.0),
    ]
    edges = [
        GraphEdge(edge_id=0, src=0, dst=1, relation="керує", evidence=m1),
        GraphEdge(edge_id=1, src=1, dst=2, relation="належить", evidence=m2),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


def test_walk_two_hop_paths_builds_a_distinct_span_chain():
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.start == "Alpha" and seed.bridge == "Beta" and seed.end == "Gamma"
    assert [step.relation for step in seed.steps] == ["керує", "належить"]
    # the two hops cite DISTINCT spans -> a genuine multi-span question
    keys = {(s.evidence.char_start, s.evidence.char_end) for s in seed.steps}
    assert len(keys) == 2


def test_walk_two_hop_paths_skips_when_no_bridge_node():
    # two edges that do not share a middle node -> no 2-hop path
    m: GraphMention = {
        "doc_id": "d.md",
        "char_start": 0,
        "char_end": 3,
        "text": "abc",
        "section_title": "S",
    }
    graph = KnowledgeGraph(
        nodes=[GraphNode(node_id=i, name=f"n{i}", type="X", confidence=0.0) for i in range(4)],
        edges=[
            GraphEdge(edge_id=0, src=0, dst=1, relation="r", evidence=m),
            GraphEdge(edge_id=1, src=2, dst=3, relation="r", evidence=m),
        ],
    )
    assert walk_two_hop_paths(graph, max_paths=10) == []


def test_walk_chain_paths_fills_from_shared_topic_facts():
    graph = _chain_graph()
    graph.edges[1] = GraphEdge(
        edge_id=1,
        src=0,
        dst=2,
        relation="підтримує",
        evidence=graph.edges[1].evidence,
    )

    assert walk_two_hop_paths(graph, max_paths=10) == []
    seeds = walk_chain_paths(graph, max_paths=10)
    assert len(seeds) == 1
    assert seeds[0].bridge == "Alpha"
    assert [step.relation for step in seeds[0].steps] == ["керує", "підтримує"]


def test_build_multi_hop_items_carries_two_spans_and_validates(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)

    drafts = draft_multi_hop(
        lambda _p: json.dumps(
            {
                "question": "Кому належить компанія, якою керує Alpha?",
                "reference_answer": "Кінцевою організацією є Gamma.",
            }
        ),
        docs,
        seeds,
    )
    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert len(items) == 1
    item = items[0]
    assert len(item.source_spans) >= 2  # >= 2 grounded spans
    assert labels[item.id].question_type == QUESTION_TYPE_MULTI_HOP
    assert labels[item.id].difficulty == "hard"
    # span-exact validation against the copied corpus passes
    report = validate_items(items, corpus)
    assert report["errors"] == []


def test_build_multi_hop_items_drops_draft_without_question():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    items, labels = build_multi_hop_items(docs, seeds, [{"reference_answer": "Gamma"}])
    assert items == [] and labels == {}


def test_build_multi_hop_items_drops_reference_without_bridge_or_end_entity():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    drafts = [
        {
            "question": "Кому належить компанія, якою керує Alpha?",
            "reference_answer": "Відповідь не називає сутність із ланцюжка.",
        }
    ]

    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert items == [] and labels == {}


def test_build_multi_hop_items_accepts_reference_containing_bridge_entity():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    drafts = [
        {
            "question": "Яка проміжна організація поєднує Alpha та Gamma?",
            "reference_answer": "Проміжною організацією є Beta.",
        }
    ]

    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert len(items) == 1
    assert items[0].reference_answer == "Проміжною організацією є Beta."
    assert labels[items[0].id].question_type == QUESTION_TYPE_MULTI_HOP


def test_build_chain_items_carries_ordered_grounded_steps(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)

    chains = build_chain_items(docs, seeds)

    assert len(chains) == 1
    chain = chains[0]
    assert [step.order for step in chain.steps] == [1, 2]
    assert chain.steps[0].dependency_note == ""
    assert chain.steps[1].dependency_note
    assert chain.steps[0].question.startswith("Який факт")
    assert chain.steps[1].question.startswith("З урахуванням")
    assert validate_chains(chains, corpus)["errors"] == []


# --- near-duplicate suppression --------------------------------------------------------------


class FakeEmbedder:
    """Deterministic per-text unit vectors: identical text -> identical vector (cos 1),
    distinct text -> orthogonal vector (cos 0)."""

    def __init__(self) -> None:
        self._basis: dict[str, int] = {}

    def _index(self, text: str) -> int:
        return self._basis.setdefault(text, len(self._basis))

    def embed(self, texts: list[str]) -> list[list[float]]:
        indices = [self._index(text) for text in texts]
        width = len(self._basis)
        return [[1.0 if i == idx else 0.0 for i in range(width)] for idx in indices]


def _item(item_id: str, question: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer="x",
        source_doc_id="a.md",
        source_spans=[_span("a.md", 0, "abc")],
        provenance=PROVENANCE_KIND,
        split="final",
    )


def test_near_duplicate_filter_drops_paraphrase_of_prior_question():
    prior = ["Яка столиця України?"]
    items = [
        _item("dup", "Яка столиця України?"),  # exact prior -> dropped
        _item("keep", "Яка висота вежі?"),  # distinct -> kept
    ]
    kept, report = NearDuplicateFilter(prior, FakeEmbedder(), threshold=0.9).filter(items)
    assert [item.id for item in kept] == ["keep"]
    assert report["dropped"] == 1 and report["dropped_ids"] == ["dup"]
    assert report["prior_questions"] == 1


def test_near_duplicate_filter_no_prior_keeps_all():
    items = [_item("a", "q1"), _item("b", "q2")]
    kept, report = NearDuplicateFilter([], FakeEmbedder()).filter(items)
    assert kept == items and report["dropped"] == 0


def test_load_prior_questions_reads_prior_bundle_goldsets(tmp_path):
    bundle = tmp_path / "prior"
    bundle.mkdir()
    from llb.goldset.schema import dump_goldset

    dump_goldset(
        [_item("q1", "Питання одне?"), _item("q2", "Питання два?")], bundle / "goldset.jsonl"
    )
    assert load_prior_questions([bundle]) == ["Питання одне?", "Питання два?"]
    assert load_prior_questions([tmp_path / "missing"]) == []  # missing bundle skipped


# --- refine labels + per-question-type retrieval fraction ------------------------------------


def test_refine_labeled_tags_question_type_and_difficulty():
    docs = [DocRecord(doc_id="a.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що таке ця організація?",
            "reference_answer": "Організацією є Gamma.",
            "answer_span": "Gamma",
            "difficulty": "easy",
        }
    ]
    items, labels = refine_drafts_labeled(docs, drafts)
    assert len(items) == 1
    label = labels[items[0].id]
    assert label.question_type == "definition"
    assert label.difficulty == "easy"  # carried from the seed via the draft dict


class FakeNeedleRetriever:
    def __init__(self, hits: dict[str, list[dict[str, object]]]):
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return self._hits.get(question, [])[:k]


def test_calibration_report_labels_needles_and_reports_per_type_fraction(tmp_path):
    out = tmp_path / "bundle"
    corpus = out / "corpus"
    corpus.mkdir(parents=True)
    doc_id = "a.md"
    text = "Alpha керує Beta."
    (corpus / doc_id).write_text(text, encoding="utf-8")
    citation = {
        "kind": "pdf-citations",
        "source": "s.pdf",
        "doc_id": doc_id,
        "parser": "test",
        "pages": [
            {"page": 1, "text_start": 0, "text_end": len(text), "parser": "test", "blocks": []}
        ],
    }
    (corpus / "a.citations.json").write_text(
        json.dumps(citation, ensure_ascii=False), encoding="utf-8"
    )

    hit = GoldItem(
        id="q1",
        question="Хто керує Beta?",
        reference_answer="Alpha",
        source_doc_id=doc_id,
        source_spans=[_span(doc_id, 0, "Alpha")],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    miss = GoldItem(
        id="q2",
        question="Що таке Beta?",
        reference_answer="Beta",
        source_doc_id=doc_id,
        source_spans=[_span(doc_id, text.index("Beta"), "Beta")],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    labels = {
        "q1": ItemLabels(question_type="factoid", difficulty="easy"),
        "q2": ItemLabels(question_type="definition", difficulty="easy"),
    }
    extraction = DocExtraction(
        doc_id=doc_id,
        entities=[Entity(name="Alpha", type="ORG", mentions=[_span(doc_id, 0, "Alpha")])],
    )
    retriever = FakeNeedleRetriever(
        {hit.question: [{"doc_id": doc_id, "char_start": 0, "char_end": 5, "text": "Alpha"}]}
    )

    report = write_calibration_artifacts(
        out,
        [DocRecord(doc_id=doc_id, text=text, sha256="x", n_chars=len(text))],
        [extraction],
        OntologyCandidate(),
        [hit, miss],
        elapsed_s=0.0,
        settings={},
        retrieval_store=retriever,
        retrieval_k=3,
        item_labels=labels,
        coverage_matrix={"mode": "coverage-target"},
        dedup_report={"dropped": 2},
    )

    rows = [
        json.loads(line)
        for line in (out / NEEDLE_GOLDSET_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["id"]: row for row in rows}
    assert by_id["q1"]["question_type"] == "factoid"
    assert by_id["q2"]["question_type"] == "definition"

    assert report["question_type_distribution"] == {"definition": 1, "factoid": 1}
    assert report["coverage_matrix"] == {"mode": "coverage-target"}
    assert report["dedup"] == {"dropped": 2}
    per_type = report["retrieval_unique_needle_fraction_by_question_type"]
    assert per_type["factoid"]["retrievable_fraction"] == 1.0
    assert per_type["definition"]["retrievable_fraction"] == 0.0


# --- full flow: multi-hop + coverage-target + dedup over a fake endpoint ----------------------


def _chain_extraction_json() -> str:
    return json.dumps(
        {
            "entities": [
                {"name": "Alpha", "type": "ORG", "mentions": ["Alpha"]},
                {"name": "Beta", "type": "ORG", "mentions": ["Beta"]},
                {"name": "Gamma", "type": "ORG", "mentions": ["Gamma"]},
            ],
            "facts": [
                {
                    "subject": "Alpha",
                    "relation": "керує",
                    "object": "Beta",
                    "evidence": "Alpha керує Beta",
                },
                {
                    "subject": "Beta",
                    "relation": "належить",
                    "object": "Gamma",
                    "evidence": "Beta належить Gamma",
                },
            ],
        }
    )


def _chain_endpoint(prompt: str) -> str:
    if "будує онтологію" in prompt:
        return _chain_extraction_json()
    if "багатокрокових (multi-hop)" in prompt:
        return json.dumps(
            {
                "question": "Кому належить компанія, якою керує Alpha?",
                "reference_answer": "Кінцевою організацією є Gamma.",
            }
        )
    if "укладач набору запитань" in prompt:
        return json.dumps(
            {
                "question": "Яка організація згадана поряд з Alpha?",
                "reference_answer": "Згаданою організацією є Beta.",
                "answer_span": "Beta",
            }
        )
    return "{}"


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
