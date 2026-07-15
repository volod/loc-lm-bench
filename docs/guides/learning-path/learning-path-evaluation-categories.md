# Learning path -- LLM evaluation categories and knowledge-graph retrieval

Detailed documentation **and** a learning path for the evaluation categories beyond RAG question
answering: security, tool use, agents, text analysis, summarization, structured output,
conversation analysis, reliability -- plus serving robustness, hardware breadth, and
knowledge-graph retrieval.

It expands the forward-looking stage of the [main learning path](learning-path.md) and sits next
to the [LLM security learning path](learning-path-security.md), which is the deep dive for the
security category. This guide teaches the *concepts* behind each category, names the *essential
papers and manuals*, and points at *where each capability lives or attaches in the code*. The
project's design rationale lives in the [design spec](../../design/spec.md), what already exists in
the [current state](../../impl/current.md), and the sequenced roadmap in the
[forward plan](../../impl/plan.md).

## Who this is for

You have completed (or can follow) the foundational stages of the
[main learning path](learning-path.md): you understand the RAG flow, local serving, the gated
judge, isolation, and the manifest/board. You are now picking up a new evaluation category or the
graph retrieval backend and want the design and the background reading in one place.

## How to read it

Each capability below has the same five-part shape so you can connect theory to code:

- **What it measures** -- the one-paragraph definition of the capability and its headline metric.
- **Design choices** -- the settled choices (do not re-litigate them); knowing them stops you
  re-deriving the design.
- **How to understand it** -- the mental model and the gotchas.
- **Learn** -- essential papers and manuals, ordered easiest-first.
- **In this repo** -- the module that exists or where the capability attaches.

---

## Design principles for adding any evaluation category

Before any single category, internalize the methodology. Every category obeys these, and most
design questions answer themselves once you know them:

1. **Each category is its own non-comparable board.** A security attack-success-rate is not
   comparable to a question-answering correctness score, so each capability renders on its own
   leaderboard and is never cross-ranked with another. The code seam is the ranking *tier* guard
   in `src/llb/scoring/aggregate.py` -- a security tier, a tool-use tier, an agentic tier, a
   text-analysis tier all join the existing public-screen / private-eval split. Study that guard
   first; every new category copies its shape.
2. **Objective metric first, model-judge second.** The headline of every category is an *objective*
   check against STRUCTURED ground-truth labels (planted by the synthetic-corpus generator, which
   is never also the judge). An LLM judge enters only for free-form quality, and only once
   it has been *calibrated against humans* (see the
   [human-in-the-loop evaluation manual](../human-tooling/human-in-the-loop-evaluation.md)).
   Until then, objective correctness ranks alone.
3. **The verified-data gate.** Every gold/eval item is AI-drafted, a *second* frontier model
   cross-checks it (grounding, non-circularity), and a human spot-verifies a stratified SAMPLE
   before any item is allowed to score a model. The cross-check is pipeline code; only the sample
   verification is human.
4. **One isolation contract.** Every category run goes through the per-cell isolation primitive
   (one process per cell, a PID-attributed VRAM-reclaim gate, a capped thermal cooldown), so a
   longer agentic or tool loop cannot bias the next measurement. You reuse `executor/isolation.py`;
   you do not write new isolation.
5. **No blended composite until every part has a confidence interval.** A weighted blend across
   quality, reliability, security, agentic, tooling, and efficiency is recorded but NOT activated
   as a headline until each component carries an uncertainty estimate. Until then, each category
   reports its own Pareto front and confidence intervals.

> The recurring lesson: these are **task families layered on one substrate**, not new platforms.
> Before reaching for a new framework, check whether `eval/`, `scoring/`, `executor/`, and the
> board already do it. The project's philosophy is *reuse over rebuild*.

---

## Long-document and multi-hop orchestration

### What it measures
Whether a model can answer over content larger than its context window (long-document
comprehension) and over questions whose evidence is spread across several retrievals (multi-hop
reasoning). These are the orchestration *substrates* the harder categories build on, not metrics
themselves.

### Design choices
- Two reusable graph templates, both following the single-call template's pure-node-closure shape:
  **map-reduce** and **multi-hop**. Every node closure / parser is pure and unit-testable without
  the graph library; only the `build_*_graph` function imports it.
- The multi-hop controller is deliberately the foundation that the agentic category later grows
  into tool calls + a sandboxed execution node.

### How to understand it
- **Map-reduce** is the long-document pattern: split a document into overlapping segments, MAP a
  partial answer over each, then REDUCE the partials into one answer. Segments that find nothing
  emit an explicit "no information" marker the reduce step drops. Reach for it when the document is
  longer than the context window.
- **Multi-hop** is `retrieve -> controller -> {retrieve again | answer}` with a conditional edge,
  bounded by a max-hops limit, deduping retrieved chunks across hops. The controller's decision --
  "do I have enough to answer, or do I need another retrieval?" -- is exactly the decision an agent
  makes; that is why it is the agentic substrate.
- Record **trajectory length** (number of hops) and model-call / token counts as an efficiency
  signal -- the same numbers become the agentic efficiency metric later.

### Learn
- Long-context behavior: **Lost in the Middle** (Liu et al. 2023) --
  <https://arxiv.org/abs/2307.03172> -- why naive long-context degrades and why map-reduce helps.
- Map-reduce summarization, the practical pattern: the LangChain
  [summarization tutorial](https://python.langchain.com/docs/tutorials/summarization/).
- Multi-hop reasoning: **HotpotQA** (Yang et al. 2018) -- <https://arxiv.org/abs/1809.09600> -- the
  canonical multi-hop QA dataset, and **Self-Ask** (Press et al. 2022)
  <https://arxiv.org/abs/2210.03350> for the "ask a follow-up" controller idea.

### In this repo
`src/llb/eval/{common,map_reduce,multi_hop}.py`. The shared status
taxonomy, refusal markers, and context formatting live in `eval/common.py` and are reused by all
three templates.

---

## Structured scoring of text analysis

### What it measures
How well a model extracts the structured content of a document -- atomic facts, named entities,
topics, directional trends, risks, decisions, internal contradictions -- plus its free-form
narrative, inferred insight, and long-document comprehension. The objective headline is the mean
F1 over the objective sub-tasks; the free-form sub-tasks are owned by the calibrated judge with an
objective floor.

### Design choices
- The matching engine is **planted-label-identity matching + pinned-embedder cosine** -- explicitly
  NOT lemmatization and NOT LLM-entailment. Each planted label is a structured record with a stable
  id, a canonical surface value plus accepted aliases, grounding offsets, and kind-specific
  attributes (a trend's subject + direction, a contradiction's paired span ids).
- Thresholds (named constants, no magic numbers): full-credit cosine `0.85`, partial-credit band
  `[0.70, 0.85)` worth `0.5`. These were reviewed and accepted; see the signed-off
  [text-analysis scoring schema](../../design/text-analysis-schema.md).

### How to understand it
- A **sub-task is the unit of credit.** OBJECTIVE sub-tasks (fact / entity / topic / trend / risk /
  decision / contradiction) are scored by the matcher alone; JUDGED sub-tasks (narrative / insight
  / long-document) get an objective *floor* but their headline is the model judge -- and only when
  the judge is trusted, otherwise they fall back to the floor.
- **Matching is greedy one-to-one** per sub-task: each prediction and each label is used at most
  once, highest-credit pairs first. Then `recall = matched credit / planted labels`,
  `precision = matched credit / predictions`, `f1 = harmonic mean`. UNMATCHED predictions are false
  positives, which is precisely what penalizes a model that hallucinates extra extractions.
- **Why embedding cosine, not n-gram overlap.** Ukrainian is morphologically rich; the same fact
  appears in many surface forms. Identity matching plus a semantic cosine fallback credits
  paraphrase and inflection without the brittleness of lexical overlap and without the
  unaccountability of asking an LLM "are these equivalent?".
- **Real vs synthetic stays separate.** A synthetic document with planted labels is checkable; a
  real document is not. Results from the two are reported separately and never merged.

### Learn
- Embedding-based similarity: [sentence-transformers semantic textual
  similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
  -- the cosine basis the matcher uses.
- Why semantic beats lexical for evaluation: **BERTScore** (Zhang et al. 2019) --
  <https://arxiv.org/abs/1904.09675> -- and, for contrast, ROUGE
  ([Lin 2004](https://aclanthology.org/W04-1013/)) so you know what you are *not* using and why.
- Extraction quality framing: precision / recall / F1 for information extraction --
  <https://en.wikipedia.org/wiki/F-score>.

### In this repo
`src/llb/scoring/text_analysis.py` (the matcher), the `PlantedLabelRecord` / `SubtaskScore`
contracts in `src/llb/core/contracts/benchmarks.py`, and the signed-off design at
[`docs/design/text-analysis-schema.md`](../../design/text-analysis-schema.md). The cosine similarity
is injected, so the engine is pure and unit-tested without the embedder.

---

## LLM security and robustness evaluation

### What it measures
How often an adversarial input achieves a prohibited objective -- the **attack-success-rate (ASR,
lower is better)** -- across prompt-injection, jailbreak, instruction-hierarchy violation,
unsafe-content generation, tool-abuse, RAG-injection (malicious instructions hidden in retrieved
chunks), and data-exfiltration (corpus-secret / canary leakage); plus **refusal-appropriateness**
so a model is not rewarded for over-refusing benign Ukrainian prompts.

### Design choices
- **Hybrid sourcing.** Reuse and Ukrainian-adapt public adversarial sets (JailbreakBench /
  HarmBench / AdvBench) for the generic attack families; use the synthetic-corpus planter for the
  corpus-specific RAG-injection and canary-exfiltration families (no public equivalent). Every
  malicious instruction and canary is a structured label.
- **Objective detector per family** (planted-instruction-followed / canary-leaked) -> per-case
  binary outcome -> ASR. The unsafe-content family adds the calibrated judge for borderline quality
  only -- no new safety classifier. Renders under a dedicated security tier.

### How to understand it
This is its own deep topic. Rather than repeat it, work the
[LLM security learning path](learning-path-security.md): threat modeling, the instruction
hierarchy, injection vs jailbreak, leakage and canaries, destructive-action containment, bias and
censorship, benchmark metrics, and an eight-session syllabus with a capstone. The one principle to
carry over: **pair every adversarial case with a benign control** -- a defense that drives ASR to
zero by refusing everything has failed the usefulness requirement, so you must measure clean task
success on the same task set.

### Learn
- [OWASP Top 10 for LLM
  Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/),
[MITRE ATLAS](https://atlas.mitre.org/), [NIST GenAI
Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence).
- **The Instruction Hierarchy** (Wallace et al. 2024) -- <https://arxiv.org/abs/2404.13208>.
- [JailbreakBench](https://jailbreakbench.github.io/), [HarmBench](https://www.harmbench.org/), and
  AdvBench (in [Zou et al. 2023](https://arxiv.org/abs/2307.15043)) -- seed material to
  deduplicate, pin, license-check, and Ukrainian-adapt.

### In this repo
Planned. Reuses `src/llb/eval/`, the planter in `src/llb/prep/frontier.py`,
`executor/isolation.py`, the manifest, and confidence-interval reporting under a new security tier.
Fully objective -- no human dependency -- so it parallelizes with the tool-use category.

---

## Tool use, function calling, and MCP evaluation

### What it measures
Objective function-call correctness over a fixed tool catalog: tool-selection accuracy,
argument-exactness (schema-valid + value match), no-hallucinated-tool rate, and well-formed-call
rate. Scored **call-only** -- the emitted call is validated, the tool is NOT executed (execution
belongs to the agentic category).

### Design choices
- **Adapt the Berkeley Function-Calling Leaderboard.** Reuse its cases (Ukrainian-adapted, OpenAI
  tool/function-calling JSON schema) and serve the SAME catalog over the official Model Context
  Protocol Python SDK server -- so both native function-calling and the MCP transport run from one
  source.
- Validate against the catalog schema with Pydantic (the project's existing validation layer; no
  new schema dependency). **Backend capability is recorded, not assumed**: tool-calling support
  varies by backend, so record per-candidate capability and never cross-rank tool-capable vs
  text-only candidates. Renders under a dedicated tool-use tier.

### How to understand it
- A **tool / function** is a JSON-schema declaration (name, description, typed parameters). The
  model emits a *call* -- a `(name, arguments)` object -- which your code validates against the
  schema. **Call-only** means you check that object; you do not run the function. That keeps the
  category purely objective and deterministic.
- Argument-exactness has two layers: *schema-valid* (the arguments conform to the parameter schema)
  and *value match* (the values equal the expected ones). Score and report them separately.
- **MCP (Model Context Protocol)** is a standard client/server protocol for exposing tools to a
  model. Serving one catalog over both native function-calling and MCP lets you measure the same
  model under two transports without authoring two case sets.

### Learn
- [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling) -- the
  request/response shape every backend mimics.
- **Gorilla** (Patil et al. 2023) -- <https://arxiv.org/abs/2305.15334> -- the paper behind the
  Berkeley Function-Calling Leaderboard; then the
  [leaderboard itself](https://gorilla.cs.berkeley.edu/leaderboard.html) for the case taxonomy
  (simple / multiple / parallel / relevance) you adapt.
- **ToolLLM** (Qin et al. 2023) -- <https://arxiv.org/abs/2307.16789> -- broader tool-use eval.
- [Model Context Protocol](https://modelcontextprotocol.io/) +
  [Python SDK](https://github.com/modelcontextprotocol/python-sdk) -- the server quickstart.
- [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/) -- how a tool schema
  becomes a validator.

### In this repo
Planned. A tool-call parse/validate layer over `src/llb/backends/openai_client.py` (which already
speaks tools). Fully objective -- no human dependency.

---

## Agentic workflow evaluation

### What it measures
Multi-step task completion in a sandboxed tool environment, scored by objective **task success**
(completion-rate) read from final environment state, with the calibrated judge scoring only
trajectory quality where a deterministic check cannot. Efficiency = trajectory length + tool-call
count.

### Design choices
- The agentic loop is the **multi-hop template extended** with tool calls + a controller node, so
  it depends on the orchestration substrate and the text-analysis schema above.
- **Custom deterministic tool-world.** A small in-memory environment (mock files / DB + search over
  the Ukrainian corpus + a calculator), with tools EXECUTED in-sandbox -- not an external agent
  benchmark -- to keep it lightweight and Ukrainian-native.
- **One fixed agent harness** (the project's graph library). Ranking the *model* under one harness
  is in scope; comparing agent *frameworks* against each other is research scope and stays
  deferred. Renders under a dedicated agentic tier.

### How to understand it
- An **agent** is a model in a loop that observes, decides on an action (a tool call), executes it,
  and feeds the result back -- until it answers or hits a step bound. The objective score is the
  final **environment state**: did the planted-label assertions hold after the run? That is why the
  tool-world must be deterministic -- the same actions must always reach the same state.
- **Execution here, validation-only in the tool-use category.** Success depends on *chains* of
  calls whose later steps consume earlier results, so you must actually run them against the mock
  world; you cannot judge a chain from a single static call.
- **Capability vs safety are different axes.** This category measures *can the model accomplish the
  legitimate task*; the security category measures *can a hostile input make it do the wrong
  thing*. Keep them on separate tiers.

### Learn
- **ReAct** (Yao et al. 2022) -- <https://arxiv.org/abs/2210.03629> -- the reason+act loop the
  controller implements.
- **Toolformer** (Schick et al. 2023) -- <https://arxiv.org/abs/2302.04761> -- models learning to
  call tools.
- **tau-bench** (Yao et al. 2024) -- <https://arxiv.org/abs/2406.12045> -- a tool-agent eval with
  deterministic state assertions; study its design even though we build our own world.
- **AgentBench** (Liu et al. 2023) -- <https://arxiv.org/abs/2308.03688> -- multi-environment agent
  eval (the comparison-axis scope we defer).
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) -- conditional edges, state, and the
  recursion limit that bounds the loop.

### In this repo
Planned, on top of `src/llb/eval/multi_hop.py`. Objective completion-rate is independent of the
judge; trajectory-quality scoring waits on judge calibration.

---

## Summarization, structured output, conversation analysis, and reliability

Four more capabilities, each a task family with its own schema.

### Summarization
- **What it measures:** reference *coverage* via pinned-embedder cosine (not ROUGE) + judge
  *faithfulness*.
- **How to understand it:** coverage (did the summary include the reference points?) and
  faithfulness (did it invent anything?) are separate axes -- a fluent hallucination scores high
  coverage but must fail faithfulness. ROUGE measures n-gram overlap and is brittle for
  morphologically rich Ukrainian; embedding cosine captures meaning.
- **Learn:** **SummEval** (Fabbri et al. 2021) -- <https://arxiv.org/abs/2007.12626> -- why
  human-correlated metrics beat ROUGE; faithfulness via the
  [DeepEval metrics docs](https://docs.confident-ai.com/) (the same judge engine the project uses).

### Structured output
- **What it measures:** objective JSON-schema conformance + field accuracy, validated with Pydantic.
- **How to understand it:** two failure modes -- malformed (parse fails) and well-formed-but-wrong
  (parses, wrong values). Score them separately; do not let a parseable-but-incorrect object count
  as success.
- **Learn:** [Pydantic JSON schema](https://docs.pydantic.dev/latest/concepts/json_schema/) and the
  [JSON Schema concept guide](https://json-schema.org/understanding-json-schema/).

### Conversation (chat-period) analysis
- **What it measures:** the text-analysis capability over chat logs; it reuses the planted-label
  schema and matcher above.
- **How to understand it:** depends on the synthetic-corpus planted labels; real-corpus and
  synthetic results are reported separately, never merged, because only the synthetic side is
  checkable.

### Reliability
- **What it measures:** aggregate the existing typed failure taxonomy (empty / malformed / refusal
  / timeout / context-truncation / retrieval-miss / backend-crash / out-of-memory / judge-failure)
  into a first-class reliability score with a confidence interval.
- **How to understand it:** these statuses already exist per case in `eval/common.py`; reliability
  *counts and normalizes* them -- reuse, not new instrumentation.
- **Learn:** [Google SRE -- Service Level
  Objectives](https://sre.google/sre-book/service-level-objectives/)
  for the mindset of turning failure counts into a defensible score.

### In this repo
Planned. Summarization / structured output are objective and can ship early; conversation analysis
depends on the text-analysis schema; reliability aggregates `eval/common.py`. Each renders under
its own tier.

---

## Serving robustness and extraction-pipeline hardening

Small but real engineering, grouped by the two domains it touches: **GPU serving** and the
**data-preparation (extraction) pipeline**. These ride along with whichever category first
exercises the host or the draft pipeline.

### GPU serving robustness
- **Sliding-window KV cache.** Some model families (Gemma 3/4) use sliding-window attention on most
  layers, so the KV cache grows with the *window*, not the full context; modeling it as full
  attention over-reserves VRAM at long context. Also: let a cached `config.json` *override* curated
  architecture fields, not only fill gaps. Learn: the model card + sliding-window attention
  ([Longformer, Beltagy et al. 2020](https://arxiv.org/abs/2004.05150)).
- **Multi-GPU and architecture-derived abort headroom.** The VRAM-contention guard reads one GPU;
  read all of them, and derive the KV abort headroom from the served-context architecture instead
  of a fixed floor.
- **Sampler kernel pinning.** When a bundled sampling kernel fails to build for a GPU architecture,
  auto-pin a host-compatible version, record the chosen sampler in the manifest, and re-run the
  preflight on a driver change without a full rebuild. Learn:
  [flashinfer](https://github.com/flashinfer-ai/flashinfer).
- **Partial GPU/CPU offload.** Exercise a real partial layer split on an oversized GGUF (only the
  all-on-GPU path is confirmed). Learn: the
  [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) and the
  n-gpu-layers offload flag.

### Extraction-pipeline hardening (feeds the verified-data gate + graph construction)
- **Second-frontier cross-check** as pipeline code -- a *different* frontier model checks each
  drafted item for grounding and non-circularity before the human sample-verify. This IS the
  verified-data gate.
- **Native NER / coreference adapter.** A Stanza / spaCy `uk_core_news` plug-in implementing the
  existing extraction-adapter protocol (only the LLM adapter ships). Learn:
  [spaCy uk](https://spacy.io/models/uk), [Stanza](https://stanfordnlp.github.io/stanza/),
  [lang-uk](https://github.com/lang-uk).
- **Long-document chunking for extraction** instead of one truncated call.
- **Richer ontology-type confidence** than raw frequency; carry induced types into the drafting
  prompt as explicit constraints (today they inform coverage strata only).

### In this repo
`src/llb/backends/{planner,preflight,llamacpp}.py`, `src/llb/executor/contention.py`, and
`src/llb/prep/ontology/` (extract / induce / draft stages).

---

## Platform and hardware-matrix breadth

### What it measures
Infrastructure breadth: the same logical model base across multiple serving backends (vLLM /
Ollama / llama.cpp), power-aware telemetry, multiple vector stores (Chroma / Qdrant / LanceDB)
behind the RAG-store seam (FAISS stays the default), and generated serving configs for adding
per-GPU-class rows as operators get access to new hosts.

### How to understand it
These are *generalizations of existing seams*, not new science: per-source quant metadata is the
seam for multi-backend; the RAG-store interface (`rag/store.py`) is the seam for multi-vector-store;
the KV-cache-aware planner generalizes to other GPU classes; and `run-eval --telemetry` records
mean power plus quality-per-watt when `nvidia-smi` is reachable. Use the
[platform matrix guide](../benchmarking/platform-matrix.md) for the runnable backend/power flow.

### Learn
- Vector stores: [Chroma](https://docs.trychroma.com/),
  [Qdrant](https://qdrant.tech/documentation/),
  [LanceDB](https://lancedb.github.io/lancedb/) -- compare the *interface*, since the seam is what
  matters here.
- Efficiency reporting: [MLPerf Inference](https://mlcommons.org/benchmarks/inference/) -- how
  performance-per-resource is reported defensibly.

### In this repo
`src/llb/rag/store.py` (vector-store seam), `src/llb/backends/` (multi-backend),
`backends/planner.py` (GPU classes), and `backends/telemetry.py` (power telemetry).

---

## Knowledge-graph retrieval (GraphRAG)

### What it measures
A second retrieval backend with TWO span-preserving strategies: **local k-hop** answers "connect
these facts" / multi-hop questions a single chunk cannot, and **global community** answers
corpus-level theme / trend / narrative questions no single chunk or neighborhood contains. It is an
ADDED backend behind the existing retrieval seam -- vector search stays the default -- so the same
eval, scoring, isolation, and board score it unchanged, and runs record the backend + strategy so
graph-vs-vector and local-vs-global results are comparable.

### Design choices
- **Store: DuckDB if it covers narratives, else NetworkX (was Kuzu).** Kuzu -- the originally chosen
  embedded property graph -- was abandoned 2025-10 (repo archived, sponsor acqui-hired), so the store
  choice reopened. For a MEDIUM corpus the graph is small and RAM-resident, so reuse beats adopting a
  graph DB, and the NARRATIVE (community) layer decides which reuse store: **DuckDB** (already a dep
  -- node/edge tables, k-hop via recursive CTEs / the DuckPGQ extension) is the DEFAULT when
  community detection can run once offline and ride as a `community_id` column; otherwise fall back to
  **NetworkX + the existing FAISS** (in-memory graph, native Leiden community detection, FAISS for
  entity-link vectors). The "no servers, single desktop, reproducible" ethos is the binding
  constraint either way.
- **Construction reuses the existing extraction.** Feed already-extracted entities / relations /
  subject-relation-object facts (with source spans) into the graph -- no second extraction
  framework. Carry the induced ontology-type confidence, section, and community id as typed node/edge
  properties. Extraction defaults to a local model (no corpus egress), frontier opt-in.
- **The narrative layer stays grounded.** A community serializes as its MEMBER nodes/edges WITH
  offsets, so the span metric still applies; any LLM community SUMMARY is a tagged DIAGNOSTIC
  artifact (recorded, never span-scored) -- the same discipline as the recorded-but-not-ranked
  semantic-similarity signal -- so an un-grounded abstraction never enters the metric.
- Swappable via a `--retrieval-backend graph` flag (+ `retrieval_strategy`). Needs the
  human-signed-off ontology schema and GraphRAG backend scope (text-analysis sign-off,
  which now also covers the narrative layer).

### How to understand it
- **Vector RAG vs graph RAG.** Vector RAG retrieves *chunks similar to the question*. Graph RAG
  retrieves a *connected subgraph of entities and relations* around the question's entities -- so
  it answers questions no single chunk contains. They are complementary; this project keeps vector
  search and *adds* the graph.
- **Property graph.** Nodes (entities) and edges (relations) carry typed properties. You query it
  declaratively -- with **Cypher** (the language Neo4j popularized) on a graph DB, or with recursive
  SQL / a graph library on the reuse stores; the model is the same regardless of store.
- **Source-span preservation is the hard constraint.** Every node/edge keeps its `doc_id` + char
  offsets, so when a retrieved subgraph is serialized into the prompt, the source-span retrieval
  metric still scores it. That is *why* construction reuses the grounded extraction pipeline instead
  of a black-box graph builder that would lose the offsets -- and *why* the narrative layer
  serializes member nodes (which carry offsets), not an un-grounded summary.
- **Local retrieval = entity-link + k-hop expand + serialize.** Map the question's mentions to graph
  nodes (entity linking), walk k edges out (k-hop expansion), then linearize that subgraph into text
  context.
- **Global retrieval = community detect + select + serialize.** Partition the graph into communities
  (Leiden), map the question to its relevant communities, then serialize each community's member
  nodes/edges. This is the "global vs local" distinction from Microsoft GraphRAG -- global is what
  answers corpus-level narrative questions -- kept grounded here by serializing members, not
  summaries.

### Learn
- **Microsoft GraphRAG** (Edge et al. 2024) -- <https://arxiv.org/abs/2404.16130> + the
  [project site](https://microsoft.github.io/graphrag/) -- the reference design and the "global vs
  local" query distinction.
- **LightRAG** (Guo et al. 2024) -- <https://arxiv.org/abs/2410.05779> -- a lighter graph+vector
  hybrid, close in spirit to "add a graph behind the seam".
- Reuse stores (single box): [DuckDB recursive
  CTEs](https://duckdb.org/docs/sql/query_syntax/with.html) + the
  [DuckPGQ](https://duckpgq.org/) property-graph extension, and
  [NetworkX](https://networkx.org/documentation/stable/) (`ego_graph` for k-hop). The Cypher model is
  still worth learning as background -- the [Kuzu Cypher tutorial](https://docs.kuzudb.com/cypher/)
  remains a readable primer even though Kuzu itself is no longer maintained (fork: Ladybug).
- Community detection (the narrative layer): **Leiden** (Traag et al. 2019) --
  <https://www.nature.com/articles/s41598-019-41695-z> -- via
  [`igraph`](https://python.igraph.org/) / [`leidenalg`](https://leidenalg.readthedocs.io/), and
  Microsoft GraphRAG's "global" query above for how communities answer corpus-level questions.
- Relation extraction (the artifact the graph ingests): **REBEL** (Huguet Cabot & Navigli 2021) --
  <https://aclanthology.org/2021.findings-emnlp.204/>.
- Entity linking background: the
  [spaCy entity linking](https://spacy.io/usage/linguistic-features#entity-linking) concept page.

### In this repo
Planned. A reuse-first store (DuckDB or NetworkX+FAISS) behind `src/llb/rag/store.py`, ingesting
`src/llb/prep/ontology/` extraction, with local-k-hop + global-community retrieval; the eval graph,
scoring, isolation, and board are reused unchanged.

---

## Time-boxed syllabus

About 2-4 hours per session; each pairs reading with a concrete repo action. Assumes the main
learning path is done.

- - **1. The methodology** ("Design principles" above + the ranking-tier guard in
- `scoring/aggregate.py`): Trace how the public-screen / private-eval tiers stay separate; sketch
- where a security tier slots in.
- - **2. Orchestration** (Long-document + multi-hop section; HotpotQA + Lost-in-the-Middle): Read
- `eval/{map_reduce,multi_hop}.py`; run their tests; trace one multi-hop controller decision.
- - **3. Text-analysis scoring** (The [scoring schema](../../design/text-analysis-schema.md) + worked
- example): Run `pytest tests/llb/scoring/test_text_analysis.py -q`; change the full-credit
- threshold and watch credit move.
- - **4. Security** (The [security learning path](learning-path-security.md), sessions 1-3): Sketch
- one RAG-injection case with a planted canary + a benign control + an objective detector.
- - **5. Tool use** (Tool-use section; OpenAI function calling + Gorilla/BFCL + MCP): Define one
- tool schema in Pydantic; hand-write an expected `(name, args)` and a validator.
- - **6. Agents** (Agentic section; ReAct + tau-bench): Design a 3-step task over a mock file/DB
- world with a final-state assertion.
- - **7. The rest of the taxonomy** (Summarization/structured/conversation/reliability section;
- SummEval + BERTScore): Write a coverage check with pinned-embedder cosine; list the reliability
- statuses in `eval/common.py`.
- - **8. Knowledge-graph RAG** (GraphRAG section; the GraphRAG paper + the Leiden paper): Build a
- tiny graph (2-3 nodes/edges) in NetworkX or DuckDB, run a 1-hop expansion keeping `doc_id`+offsets
- on each node, then group those nodes into a community and serialize its members (the narrative
- layer) -- still offset-bearing.

By session 8 you can place any capability on the right tier, name its objective metric and its
verified-data gate, and explain why graph retrieval preserves source spans. The deepest single
source remains the [design spec](../../design/spec.md); the
[current implementation](../../impl/current.md) maps each current module to its behavior; and
the [human-in-the-loop evaluation manual](../human-tooling/human-in-the-loop-evaluation.md)
covers the human gates that make the judged metrics trustworthy across every category above.
