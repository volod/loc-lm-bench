# Knowledge-Graph Ontology + GraphRAG Scope (M6 proposal -- MH.2 sign-off has been provided).

Signed off 2026-06-26 by volod; scope accepted.

Status: **APPROVED**

This is the Milestone 6 proposal for the human gate **MH.2** (Milestone H). The GraphRAG build
already exists and is tested -- this document is the *accountable acceptance* of the constrained
ontology + the retrieval scope it operates under. Recording the dated sign-off line above is what
authorizes graph runs to produce a PUBLISHED HEADLINE; until then graph boards rank
**objective-only** (recall@k / MRR / objective correctness rank regardless of this gate -- see
[`plan.md`](../impl/plan.md) and [`current.md`](../impl/current.md)). MH.5 (the stratified
human sample-verify of any `verified=true` item) remains a separate gate.

- Executable form -- the closed node vocabulary:
  [`src/llb/prep/ontology/entity_types.py`](../../src/llb/prep/ontology/entity_types.py) (the 13
  canonical types, the synonym map, `normalize_entity_type`).
- Executable form -- extraction + grounding:
  [`src/llb/prep/ontology/extract.py`](../../src/llb/prep/ontology/extract.py) (the vocabulary is
  injected into `extraction_prompt` and enforced in `_entities`) +
  [`src/llb/prep/ontology/grounding.py`](../../src/llb/prep/ontology/grounding.py) (exact spans) +
  [`src/llb/prep/ontology/spacy_adapter.py`](../../src/llb/prep/ontology/spacy_adapter.py)
  (the opt-in spaCy NER path, mapped through the same normalizer).
- Executable form -- the constrained ontology:
  [`src/llb/prep/ontology/induce.py`](../../src/llb/prep/ontology/induce.py) +
  [`src/llb/prep/ontology/constants.py`](../../src/llb/prep/ontology/constants.py) (caps +
  confidence weights).
- Executable form -- the graph backend:
  [`src/llb/graph/`](../../src/llb/graph/) (`build.py`, `community.py`, `retrieval.py`, `store.py`)
  + the `retrieval_backend` / `retrieval_strategy` / `graph_khop_depth` fields in
  [`src/llb/config.py`](../../src/llb/config.py).
- Tests: [`tests/test_graph.py`](../../tests/test_graph.py).
- Spec basis: `docs/design/spec.md` (GraphRAG / knowledge-graph retrieval), the M4.4
  extraction-reuse constraint, and the source-span grounding premise (M1.3).

The ontology is REUSED from the M4.4 extraction (no second extraction framework). Free-form RAG
answer quality is scored by the gated judge (M3.8) exactly as on the FAISS board, only when the
judge is trusted.

---

## 1. Node-type vocabulary (the closed entity-type set)

Entity nodes carry one type from a CLOSED, 13-type vocabulary (`entity_types.py`) -- an
OntoNotes-derived granularity adapted to Ukrainian benchmark corpora. Beyond the generic
PERSON/ORG/LOC it adds the types that make a FACT more granular: `LAW` (codes/treaties/conventions),
`WORK` (intellectual-property + creative objects), `DURATION` (a period, distinct from a `DATE`
point), plus `NORP`, `PRODUCT`, `MONEY`, `QUANTITY`.

| Type | Meaning | Example (IP corpus) |
|------|---------|---------------------|
| `PERSON` | a person, incl. role-holders (author, owner) | автор, виконавець, власник |
| `NORP` | nationality / ethnic / religious / political group | (none in this corpus) |
| `ORG` | organization, authority, body, court | національний орган інтелектуальної власності, митні органи |
| `LOC` | place, country, jurisdiction | Україна |
| `LAW` | law, code, treaty, convention, agreement | Цивільний кодекс України, Бернська конвенція, Угода TRIPS |
| `WORK` | IP / creative object: invention, trademark, design, work, software | патент, винахід, торговельна марка |
| `PRODUCT` | product, good, technology, service | (none in this corpus) |
| `EVENT` | a named event or process | (none in this corpus) |
| `DATE` | a calendar date / point in time | (a date, e.g. дата подання заявки) |
| `DURATION` | a length of time / period | двадцять років, десять років |
| `MONEY` | a monetary amount | (none in this corpus) |
| `QUANTITY` | a measurement with a unit / a percent | (none in this corpus) |
| `MISC` | fallback: abstract concept, right, term | авторське право, комерційна таємниця |

The vocabulary is **enforced, not merely suggested**: it is injected into `extraction_prompt`, AND
every emitted type (LLM or spaCy) is passed through `normalize_entity_type`, so synonyms collapse to
their canonical type (`GPE`/`FAC` -> `LOC`, `WORK_OF_ART`/`PATENT` -> `WORK`, `TREATY` -> `LAW`,
`TIME` -> `DATE`, `PERCENT`/`CARDINAL` -> `QUANTITY`, ...) and any out-of-vocabulary label becomes
`MISC` -- the schema can never silently expand. A fact endpoint that is not a recognized entity
becomes a lightweight `MISC` fact-only node, so no grounded fact is dropped.

**To extend further** (e.g. split a `TREATY` out of `LAW`, or add `ROLE`), edit the one module
`entity_types.py` -- the prompt, the normalizer, and these docs/tests all read from it.

## 2. Relationship types (induced + capped)

Relationship (edge) types are NOT a fixed list -- they are INDUCED from the corpus's
subject-relation-object facts and then CONSTRAINED, so the schema stays small + reviewable
(`induce.py`, caps in `constants.py`):

- `MAX_ENTITY_TYPES = 24`, `MAX_RELATION_TYPES = 32` -- hard caps on the type set.
- `MIN_TYPE_COUNT = 1` -- types below this support are dropped (hapax filter; raise to prune noise).
- `confidence = 0.5 * norm(count) + 0.5 * norm(doc_frequency)` (`CONFIDENCE_COUNT_WEIGHT` /
  `CONFIDENCE_DOCFREQ_WEIGHT`) -- a type spread across documents outranks one concentrated in one.
- `ONTOLOGY_CONSTRAINT_MIN_CONFIDENCE = 0.5`, `N_CONSTRAINT_TYPES = 8` -- only high-confidence types
  are carried back into the drafting prompt as explicit constraints.

The induced `confidence` rides onto every node as a typed property and is recorded in the run.
**Decision for you:** the cap sizes, the `MIN_TYPE_COUNT` hapax floor, and the confidence blend.

## 3. Extraction + grounding constraints

- Every entity mention and every fact evidence MUST be an exact-grounded `SourceSpan` (`doc_id` +
  char offsets + exact text). A quote that does not ground to the document is DROPPED
  (`ground_quote`): a node/edge can never point at text that is not there.
- Nodes carry the containing `section_title` (from the document's heading/paragraph segmentation)
  and the detected `community_id` as typed properties.
- Extraction is LLM-by-default via the M4.4 endpoint adapter (local, no corpus egress); a spaCy
  `uk_core_news` adapter is an opt-in alternative. Long documents are windowed + merged; grounding
  always runs against the FULL text so offsets stay exact.

## 4. GraphRAG retrieval scope (`retrieval_strategy`)

The graph backend (`--retrieval-backend graph`) exposes TWO span-preserving strategies behind one
store; both serialize node mentions + edge evidence WITH their offsets, so every returned context
scores on the existing source-span metric (M1.3):

- **`local_khop`** -- entity-link the question to seed nodes, expand `graph_khop_depth` hops
  (`DEFAULT_KHOP_DEPTH = 2`) via a DuckDB recursive CTE, serialize the subgraph. The multi-hop
  "connect these facts" path.
- **`global_community`** -- map the question to its communities (offline label propagation,
  `COMMUNITY_SEED = 13`), serialize each community's member nodes/edges. The NARRATIVE layer for
  corpus-level theme/trend questions. Seed/community counts: `DEFAULT_N_SEED_NODES = 5`,
  `DEFAULT_N_COMMUNITIES = 2`.
- **Diagnostic-summary rule (part of the scope you approve):** an optional LLM one-paragraph
  community summary is recorded as a TAGGED DIAGNOSTIC artifact and is NEVER returned by retrieval
  / never span-scored -- the un-grounded abstraction never enters the metric (the same
  recorded-but-not-ranked discipline as `--score-semantic`).

Store choice: DuckDB (already a dependency; the abandoned Kuzu pick was dropped). No graph-analytics
dependency -- communities are detected once offline, the engine only does k-hop + `WHERE
community_id`.

## 5. Worked example (over `samples/corpus/ip_regulation_uk.md`)

Building the graph from a representative M4.4 extraction over the committed IP-regulation document
(`llb build-graph` over its `extraction.jsonl`) yields:

- **23 nodes, 12 edges, 11 communities.**
- **Entity types induced (the granular vocabulary at work):** `WORK` (count 5, confidence 1.0:
  патент, винахід, корисна модель, промисловий зразок, торговельна марка), `LAW` (count 4,
  confidence 0.9: Цивільний кодекс України, Бернська/Паризька конвенція, Угода TRIPS), `DURATION`
  (count 2: двадцять років, десять років), `ORG` (count 2), `MISC` (count 2: авторське право,
  комерційна таємниця), `LOC` (count 1). The same corpus under the OLD flat set produced one
  undifferentiated `MISC` (count 10) -- the new types are what make the facts granular.
- **Relationship types induced:** `охороняє` (3), `діє` (2), `стосується` (2), then `видає`,
  `виникає з`, `відповідає`, `зупиняють`, `є учасницею` (1 each) -- all under the caps.
- **A coherent, typed community** (the patent cluster): `патент[WORK], винахід[WORK], корисна
  модель[WORK], промисловий зразок[WORK], двадцять років[DURATION], умовам новизни[MISC]`.

Retrieval (offset-bearing, scores on the span metric):

- `local_khop` "Скільки діє патент?" -> seeds `патент`, expands, returns the edge fact
  `"...діє двадцять років від дати подання заявки"` (the answer) among the top chunks.
- `global_community` "Що захищає патент?" -> the patent community, serialized as
  `"Патентне право охороняє винаходи, корисні моделі та промислові зразки"` +
  `"...діє двадцять років..."` + `"Винахід має відповідати умовам новизни"`.

**Known property to weigh at sign-off:** community granularity tracks EDGE DENSITY. A sparse
extraction (few cross-linking facts) fragments into many small communities, so the
`global_community` narrative layer is only as rich as the facts that connect entities -- richer
extraction (more SRO facts) yields more meaningful narrative communities. This is the lever behind
the optional "graph-vs-FAISS comparison" residual in `plan.md`.

---

## Sign-off (MH.2) -- how to approve this schema

This is the human gate. Nothing about it requires running a GPU.

1. **Read** this document plus the executable form linked above (the graph engine is short). Run
   the tests if you want to see the behavior:

   ```
   make test                                  # full suite, or:
   .venv/bin/python -m pytest tests/test_graph.py -q
   ```

   To regenerate the worked-example numbers over your own corpus, run `llb build-graph` (from an
   M4.4 `prepare-goldset-draft` bundle, or `--corpus-root <dir> --extract-model <id>` to extract
   fresh) and inspect `nodes.jsonl` / `graph_meta.json` under the graph dir.

2. **Confirm or adjust the decisions** that are genuinely yours:
   - the **node-type vocabulary** (Section 1) -- the 13-type closed set; extend or split a type
     (edit `entity_types.py`)?
   - the **relationship caps + hapax floor + confidence blend** (Section 2)
     (edit the constants in `prep/ontology/constants.py`);
   - the **extraction + grounding constraints** (Section 3) -- the exact-span requirement is
     non-negotiable for the span metric; confirm the extractor choice (LLM vs spaCy);
   - the **retrieval scope** (Section 4) -- both strategies, the k-hop depth, and the rule that any
     LLM community summary stays a tagged diagnostic (never span-scored).

   The tests assert against the named constants, not hardcoded values, so they follow your change.

3. **Confirm the dependent corpus facts** (MH.2 item 3): which graph corpus is real vs synthetic
   (the two are reported on separate boards and never merged).

4. **Record the sign-off.** Fill the dated line at the top of this file, e.g.
   `Signed off 2026-06-__ by <name>; the 13-type closed node vocabulary + caps + GraphRAG scope
   accepted as proposed (or: split TREATY out of LAW; MIN_TYPE_COUNT -> 2).` and flip Status to
   **APPROVED**.

Until step 4 is recorded, the GraphRAG backend is a committed, tested build -- but it stays
un-trusted for HEADLINE use, and graph runs report objective-only.
