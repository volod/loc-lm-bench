# Knowledge-Graph Construction And Graph-Retrieval Methods

Which published methods for building a knowledge graph and retrieving over it this project takes as
MEASURED candidates, which it declines, and why. This page records decisions and the reasoning
behind them. It holds no measurement: every reading it relies on is linked to the current-docs page
that holds it, and every candidate is a task in [the plan](../impl/plan.md) that is adopted only on
its own paired reading. The contract the candidates must meet is
[graph retrieval and ontology](spec.md#graph-retrieval-and-ontology).

The survey started from an operator-supplied comparison of seven methods: Wikontic, AutoOntic,
Microsoft GraphRAG, KGGen, HippoRAG, OntoGPT (SPIRES), and classical closed extraction (GLiNER,
REBEL). Each is read below against what this project's own evidence says the graph lane lacks, not
against its published benchmark.

## What the evidence says the lane lacks

- **Multi-hop evidence stops at the budget, not at the ranking knobs.** At a fixed budget the whole
  fusion grid sits in a narrow band, and most of the misses are hops the question already ranks just
  below the cut ([retrieval budget
  evidence](../impl/current/graphrag-backend/retrieval-budget-evidence.md#is-the-both-hops-ceiling-a-budget-or-a-query-problem)).
  Both graph strategies rank a NODE by a coarse level (a hop distance or a community), so all of a
  node's spans tie and the cut lands inside the tie.
- **Hard node merging costs recall.** Every cut loose enough to merge a meaningful number of nodes
  lost recall, more the looser it got, and the first false merge joined two numbered annexes that
  differ in one digit ([entity resolution
  evidence](../impl/current/graphrag-backend/entity-resolution-evidence.md#is-fragmentation-costing-the-graph-lane-recall-2026-08-21)).
- **The relation vocabulary is a long tail.** Induced relations run to nearly one surface per fact,
  so most relation-scoped axioms have no population on a given corpus
  ([ontology axiom layer](../impl/current/robustness-ontology-backends.md#ontology-axiom-layer)).
- **Communities track edge density.** Label propagation fragments a sparse extraction into many
  small communities, and the narrative layer is only as rich as the facts that connect entities
  ([graph ontology schema](graph-ontology-schema.md)).
- **Extraction is one generative call per window,** and a quote the model paraphrases is dropped at
  grounding. That cost scales with the corpus, and a 1,067-page book is the largest corpus the
  project has planned.
- **A recovered book's projection carries its cross-references only as words.** The vault links
  `see Table 4-2` to the table; the retrieval corpus does not
  ([cross-references](../impl/current/book-recovery/structure-and-rendering.md#cross-references),
  [corpus projection](../impl/current/book-recovery/corpus-projection.md)).

## The methods

| Method | What it contributes | Decision | Plan task |
| --- | --- | --- | --- |
| HippoRAG (github.com/OSU-NLP-Group/HippoRAG) | Personalized PageRank from the question's linked seeds ranks passages through the graph; dense-embedding synonymy edges connect near-identical phrases instead of merging them | **Candidate.** It targets both the tie-block ranking and the hard-merge negative. The method is reimplemented over the stored graph on SciPy sparse matrices; the package pins its own Python and LLM stack | `graph-seeded-propagation-ranking` |
| Microsoft GraphRAG (github.com/microsoft/graphrag) | Hierarchical Leiden communities, then recursive LLM summaries for global sensemaking queries | **Leiden is a candidate** for the narrative layer, from `graspologic-native`. **Summaries as retrieval context are declined**: an abstraction cannot be span-scored, and the summaries already ship as a tagged diagnostic. Its exact-name node merge is what the builder already does | `graph-community-detection-candidates` |
| KGGen (github.com/stair-lab/kg-gen) | Embedding- and LLM-guided clustering of synonymous nodes and edges across chunks | **Node clustering declined**: the hard-merge reading is negative on this project's corpora, and over-merging is the risk the method itself names. **Edge clustering is taken up** through the linkage seam rather than through LLM calls | `relation-predicate-linkage-lane` |
| AutoOntic (github.com/screemix/AutoOntic, arXiv:2611.15467) | Relation clustering that separates true synonyms from subtypes, predicate-direction verification, and domain/range inference | **The direction rule and the subtype rule are taken up** in the predicate lane: an inverse is proposed as an inverse and a subtype is a non-match. **Domain/range induction as an axiom source is declined**: an axiom is business logic that a human signs, and the checker already proposes candidates with their evidence | `relation-predicate-linkage-lane` |
| Wikontic (github.com/screemix/Wikontic) | Open extraction validated against the Wikidata type hierarchy and its domain/range constraints, with type-aware merging inside one class | **Declined.** Grounding against a public knowledge base sends corpus terms off the host or needs a local dump, and the method reports dropping facts the knowledge base does not cover, which is worst in exactly the niche domains this project serves. Entity type is already a linkage comparison column | -- |
| OntoGPT / SPIRES (github.com/monarch-initiative/ontogpt) | Recursive LLM interrogation along a LinkML schema template, with terms grounded to ontology identifiers | **Declined.** There is no ontology identifier space for these Ukrainian domains to ground to. The closed vocabulary injected into the extraction prompt is already a schema-guided prompt, and the Turtle axiom file is already the reviewer-readable schema | -- |
| GLiNER (github.com/urchade/GLiNER) | A compact encoder that classifies spans against a label set given at run time | **Candidate extractor**, from an Apache-2.0 multilingual checkpoint, with the closed vocabulary as its labels. Its spans are exact by construction. Its Ukrainian coverage is not documented, so the task measures it | `graph-closed-ie-extraction-adapter` |
| REBEL (github.com/Babelscape/rebel) | Sequence-to-sequence extraction of triples over a fixed relation inventory | **Declined.** Its relation inventory is external and fixed, while this project induces relations per corpus. It generates strings that would need re-grounding, and its published weights carry a non-commercial licence | -- |

## The printed reference graph

The survey's methods all extract a graph from prose. A recovered technical book offers a graph that
needs no extraction: the author already drew it. Sentences cite tables, equations, figures,
sections, and pages by number. The index maps the book's own terms to the pages that treat them, so
it is an entity inventory curated by the book's editor. This is ontology-validated construction in
Wikontic's sense, with the book itself as the ontology. Every edge is a reference the page prints,
so it adds nothing the book does not say, and it costs no model call.

It answers a weakness that is specific to technical text. A numeric table or a display equation
embeds poorly and sits in its own projected document, while the sentence that needs it embeds well.
A vector lane therefore retrieves the citing sentence without the evidence it cites, and one
`cites` hop delivers it. The projection records those references beside each document
([corpus projection](../impl/current/book-recovery/corpus-projection.md#printed-references)).
The [printed-reference graph reading](../impl/current/book-recovery/corpus-projection.md#printed-reference-graph-reading)
compares the graph and vector lanes over the same projection.

## What would change these decisions

- **A corpus of named entities in running prose** where fragmentation measurably costs the graph
  lane would reopen hard node clustering. The entity-resolution evidence names that corpus as the
  one its reading does not cover.
- **A local ontology with Ukrainian labels that covers a corpus's domain,** for example a Wikidata
  subset or a domain ontology checked for coverage, would reopen external-ontology typing without
  egress.
- **Community summaries whose every sentence grounds to a span** would reopen summaries as retrieval
  context, because they could then be span-scored.
- **A relation extractor whose inventory can be the corpus's induced relations, and whose weights
  carry a licence the operator accepts,** would reopen closed relation extraction.
