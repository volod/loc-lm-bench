# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

The any-corpus autopipeline that turns a mixed `txt`/`md`/`pdf` directory into a validated RAG
index plus a resumable, unverified draft bundle is now shipped (`llb ingest-corpus`,
`make quickstart-corpus`, `prepare-goldset-draft --resume`; see [data prep](current/data-prep.md)).
The tasks below build the rest of the corpus-to-recommendation spine on that foundation.

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land to `make ci` green with
  committed fixtures, injected fakes, and deterministic harnesses. A few carry a heavy real-model
  run for durable evidence; those runs are deterministic and execute on the CUDA host without any
  human judgment, so they stay in this section.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance without a
  human in the loop: the deliverable *is* human judgment (verification-gate reviews, drafting
  oversight, measured reviewer throughput) or requires a human authorization (egress consent +
  API spend). An agent can still build the supporting code and unit tests; the marked **human
  step** is what gates completion.

Task numbers are stable ids and never change; every task carries an explicit `Dependencies` line,
and the recommended build order within each section follows those lines. Two dependencies cross the
section boundary and are called out because they are **blocked by human work**:

- **Agent task 8 (`context-policy-bench`) is BLOCKED BY human task 7 (`chain-goldset-generation`).**
  Task 8 scores a *verified* chain fixture, and only the human review gate in task 7 can produce
  one. Task 8's code (context-assembly + fake-endpoint tests) can be written earlier, but its
  acceptance run cannot pass until task 7's human-accepted chains exist.
- **Agent task 11 (`verification-gate-adjudication`) depends on the CODE of human task 3
  (`verify-cli-throughput`).** Both extend `src/llb/goldset/verify.py` and `verify_session.py`;
  land task 3's code first to avoid a merge conflict. Task 3's *human throughput evidence* does
  **not** block task 11 -- only the shared code surface does.

The retrieval-quality cluster (12-16) gives the query-and-rerank side the same tune-and-demonstrate
discipline chunk-side tuning already has (the Optuna tuner searches strategy/size/overlap/mode/
`top_k`, the sweep grids `top_k`, and task 10 adds strategies). Together 12 + 14 + 15 cover the
Ukrainian-language retrieval stack end to end: dense + BM25/sparse + metadata hybrid with
inflection-aware lemmatization (12), a measured embedder ranking over BGE-M3 / multilingual-e5 /
the lang-uk model with an opt-in Cohere API row for open corpora (14), and query-side
normalization -- casefold, apostrophes, transliteration, typo tolerance, aliases/glossary --
that never mutates the stored corpus text (15). Every knob these tasks add must land
in `compare-retrieval`, the sweep grid, or the tuner search space so task 6's miss analysis can cite
it as evidence-backed. Within the cluster only 13 has an ordering preference (it reranks the pool 12
fuses, so it pays off most after 12); 14, 15, and 16 stand alone. Task 17 adds the governance
remainder -- per-chunk `language`/`date`/`version`/`ACL` metadata, permission-aware retrieval, and
the reindex/deletion/rollback policy (measured shortfall and scope decision recorded in
[RAG core](current/rag-core.md) and [product decisions](current/scope-boundaries.md)).

## Agent Implementation Tasks

These land to `make ci` green with fixtures, fakes, and deterministic harnesses. Recommended
sequence: **9 first (PRIORITIZED)** -- the multi-service external drafting lane (per-service
setup, prompts, and the `curate-drafts` merge/dedup/filter step; see
[data prep](current/data-prep.md) external-draft curation) is fully documented and curated, and
the grounded-JSONL import is its one missing executable piece for full-document needle realism.
Then 6 (its probe mode reuses the shipped durable-eval-runner), the independent lot
(10, 12, 14, 15, 16) in any order, 13 after 12, 17's ACL-filter half after 12's metadata-filter
seam (its governance fields stand alone), 11 after task 3's code, and 8 last (blocked by
human task 7). The durable-eval-runner (retry + `cases.progress.jsonl` journal + `--resume` +
bounded backend relaunch + `manifest.durability` counters) is now shipped; see
[RAG core](current/rag-core.md) durability section.

### 6. miss-analysis-recommendations

- Dependencies: none. The probe mode reuses the shipped durable-eval-runner's `--resume`
  (`llb run-eval --resume`; see [RAG core](current/rag-core.md) durability section) so a probe
  campaign survives a flap. Soft-consumes the extra per-case signals from tasks 12-16 when present
  (richer recommendations), but is not blocked by them -- it ships with the existing knobs.
- User-visible outcome: after any run or sweep, one command explains the wrong answers: each
  miss classified as retrieval miss (gold span absent from context), generation miss (evidence
  present, answer wrong), refusal, format/scoring artifact, or judge disagreement; misses
  clustered by document, topic, and question type; and ranked, evidence-backed recommendations
  (raise or lower `top_k`, change chunking, add prompt-system dictionary terms, try the named
  alternative model) that `llb recommend` folds into its summary.
- Scope boundary: in scope -- `src/llb/board/miss_analysis.py` plus `llb analyze-misses`,
  consuming per-case `scores.jsonl`, retrieved spans, typed statuses, and judge diagnostics
  from finalized run bundles; run bundles do not yet persist per-case retrieved spans
  (`retrieval_pairs` stay in-process in `src/llb/executor/cases.py` and `scores.jsonl` carries
  only `retrieval_hit`/`first_hit_rank`), so this task first adds an additive per-case
  retrieved-spans record to the bundle -- the miss classifier's span overlap and the
  observability-trace checklist item both need it; a bounded probe mode that re-runs only the
  miss subset at
  alternative retrieval depths to confirm or reject the retrieval hypothesis; a misses section
  in the `recommend` summary sourced from prompt templates like the existing report prose. Out
  of scope -- automatic re-tuning (the Optuna tuner owns search), mutating run bundles.
- Data and artifact paths: `$DATA_DIR/miss-analysis/<timestamp>/{report.md,misses.jsonl}`;
  `$DATA_DIR/recommend/summary.md` gains the misses section when an analysis exists.
- Execution path: `llb analyze-misses --run-dir <run>` and
  `make analyze-misses RUN_DIR=<run>`; probe mode
  `llb analyze-misses --run-dir <run> --probe-top-k 3,8`; unit tests over a synthetic scored
  bundle covering every miss class.
- Acceptance gates: the classifier separates retrieval misses from generation misses using span
  overlap on the synthetic bundle with zero cross-class leakage; on the committed-fixture
  sweep, every recommendation line names its numeric evidence; the probe mode is resumable via
  the `durable-eval-runner`; `make ci` green.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) recommendation
  section; [`docs/guides/mlflow-analysis.md`](../guides/mlflow-analysis.md).

### 8. context-policy-bench

- Dependencies: **BLOCKED BY human task 7 (`chain-goldset-generation`)** -- the acceptance run
  needs the verified `chains.jsonl` committed fixture that only task 7's human review gate can
  emit. The code (multi-hop substrate reuse, prompt-system role packages, context-assembly unit
  tests over a fake endpoint) can be built ahead of task 7; it just cannot pass its acceptance run
  until the verified chains exist.
- User-visible outcome: for one model and a verified chain set, a ranked comparison of
  context-management policies -- fresh retrieval per step, accumulated full history, summarized
  history, and staged role/system-prompt sequences (for example librarian -> analyst ->
  answerer built from prompt-system packages) -- with per-step and final-answer correctness,
  ending in a written recommendation on which harness and context policy improves scores and
  how to sequence system prompts for better answers.
- Scope boundary: in scope -- `src/llb/bench/chain_context.py` reusing the multi-hop
  retrieve/controller/answer substrate (`src/llb/eval/multi_hop.py`), prompt-system packages
  for role prompts (see [extended workflows](current/extended-workflows.md)), and the category
  persistence and board machinery; the policy is the row label and the model is fixed,
  mirroring the harness-comparison discipline; a recommendation block sourced from prompt
  templates naming the winning policy and its per-step evidence. Out of scope -- new agentic
  frameworks (settled in [product decisions](current/scope-boundaries.md)), cross-model
  blending in one board.
- Data and artifact paths: run bundles under `$DATA_DIR/chain-context/<timestamp>/`; board
  section rows beside the harness comparison; recommend summary gains a context-policy line
  when bundles exist.
- Execution path:
  `llb bench-chain-context --chains <chains.jsonl> --model <m> --backend <b>
  --policies fresh,history,summary,roles`; a make target with the standard MODEL/BACKEND
  variables; unit tests over a fake endpoint asserting the exact context assembled per policy
  per step.
- Acceptance gates: context-assembly unit tests pass for all four policies; a run over the
  committed chain fixture produces per-step and final scores with bootstrap CIs and ranks
  policies for the fixed model; provenance records policy, prompt-system ids, and chain set
  digest; verified-data stamping matches the category suite rules; `make ci` green.
- Documentation target: [extended workflows](current/extended-workflows.md);
  [`docs/guides/prompt-system-rag.md`](../guides/prompt-system-rag.md).

### 9. external-draft-import

- Dependencies: none (committed open-data fixture + sidecar, no network). **PRIORITIZED: first in
  the agent build order** -- the external multi-service drafting lane (per-service setup, prompt
  pack, and the shipped `curate-drafts` merge/dedup/filter step; see
  [data prep](current/data-prep.md) external-draft curation) is complete up to this import, and
  the grounded-JSONL lane is what lifts external drafts from context-sized SQuAD docs to
  full-document needle realism. Stands beside human task 2 as the other non-default drafting
  source; shares the shipped question-type labels (`src/llb/prep/ontology/question_types.py`).
- User-visible outcome: an operator who drafted test data with an external AI provider service
  (Claude Projects, NotebookLM, ChatGPT Projects) on **open** corpus data imports it into a
  standard draft bundle with one command that re-grounds every quote against the local corpus,
  computes exact `source_spans`, carries `question_type`/`difficulty` labels, records the
  external service and model in provenance, and enforces the data-classification sidecar -- so
  externally drafted goldsets flow through the same cross-check + human verification gate as
  local drafts. Today the only supported external path is the SQuAD-shaped `make ingest-squad`,
  which stamps `provenance: public-reused`, discards question-type labels, hashes each context
  into its own corpus doc (losing full-document needle realism), and cannot read the grounded
  JSONL or the required provenance sidecar. The manual workflow, artifact shapes, and prompts
  already exist in
  [`docs/guides/external-ai-service-artifacts.md`](../guides/external-ai-service-artifacts.md),
  [`docs/design/external-draft-contract.md`](../design/external-draft-contract.md), and
  [`docs/guides/external-service-prompts/`](../guides/external-service-prompts/README.md); this
  task makes the grounded-JSONL lane (contract Artifact B) executable.
- Scope boundary: in scope -- `llb import-external-draft --artifact <file> --corpus-root <dir>
  --sidecar <external_provenance.json>` reading contract Artifact B rows (`quote` +
  `source_doc_id`), re-grounding each `quote` as an exact substring of the named corpus doc
  (drop + report non-verbatim rows exactly as the frontier drafter does in
  `src/llb/prep/frontier.py`), emitting a canonical draft bundle
  (`goldset.jsonl` with `provenance: frontier-drafted`, `verified: false`, exact `source_spans`;
  verbatim `corpus/`; `provenance.json` carrying the external service, model, export date, and
  `data_classification`); a hard refusal when the sidecar is absent or
  `data_classification != "open"`; `question_type`/`difficulty` recorded in item provenance
  without changing the `GoldItem` schema (shipped labeling in
  `src/llb/prep/ontology/question_types.py`); a `grounded` kind in `llb curate-drafts` so
  multi-service Artifact B exports merge/dedup/filter through the same curation step the other
  artifact kinds already have. Out of scope --
  network calls to any provider (the operator exports by hand), the chain artifact (belongs to
  `chain-goldset-generation`), changing the security-case loader, making external drafting a
  default. Reuse `src/llb/prep/frontier.py` re-grounding, `src/llb/prep/curation/` (lenient
  loading, quote repair, dedup engine, report), `src/llb/prep/ingest_squad.py` bundle
  writing, and `src/llb/goldset/validate.py`.
- Data and artifact paths: input under `$DATA_DIR/external-drafts/<service>-<YYYYMMDD>/`
  (artifact files + `external_provenance.json`); output a standard bundle under
  `$DATA_DIR/prepare-goldset/<timestamp>/`; a small committed fixture under
  `samples/external-drafts/` (one open-data artifact + sidecar) for tests.
- Execution path: `llb import-external-draft --artifact <file> --corpus-root <dir> --sidecar
  <sidecar>`; `make import-external-draft ARTIFACT=<file> CORPUS=<dir> SIDECAR=<sidecar>`; then
  the existing `validate-goldset` -> `cross-check-goldset` -> `verify-*` chain. Unit tests use
  the committed fixture; no network.
- Acceptance gates: `make ci` green; a row whose `quote` is not a verbatim substring is dropped
  and counted, never mis-grounded; a missing or non-open sidecar aborts with a clear message and
  writes no bundle; the emitted bundle passes `make validate-goldset`; provenance records the
  service, model, and classification; `question_type`/`difficulty` survive into item provenance.
- Documentation target: [data prep](current/data-prep.md) external-draft lane;
  [`docs/guides/external-ai-service-artifacts.md`](../guides/external-ai-service-artifacts.md)
  (flip Artifact B from "target shape" to "works today").

### 10. corpus-chunking-strategies

- Dependencies: none. Picks up the "new chunking strategies" item the shipped autopipeline held
  out of scope; independent of the rest.
- User-visible outcome: the RAG store gains chunking strategies suited to mixed real-world corpora
  and demonstrated (not assumed) to help retrieval: a PDF page/citation-aware strategy that keeps
  chunk boundaries on page-sidecar spans, a heading-hierarchy (layout-aware) strategy that carries
  the full breadcrumb, and a late-chunking / propositional strategy -- each selectable as a
  `--strategy` value, offset-exact, and ranked against the existing strategies on the same gold set.
  This is the "new chunking strategies" item the shipped any-corpus autopipeline held out of scope
  (see [data prep](current/data-prep.md)); the current set is
  `fixed | sentence | recursive | markdown | semantic` (`src/llb/rag/chunking.py`).
- Scope boundary: in scope -- extend `STRATEGIES` in `src/llb/rag/chunking.py` with the new
  strategies, each returning `(start, end, metadata)` spans anchored to `doc_id` + character offsets
  so `validate-goldset` and source-span scoring keep working; the page-aware strategy reads the
  `pdf-<digest>.citations.json` sidecars produced by the PDF/`ingest-corpus` lanes; a
  `compare-retrieval` row per new strategy so the best chunker is DEMONSTRATED per corpus; the RAG
  build grid gains the strategies behind a flag. Out of scope -- new embedding models, changing the
  source-span gold contract, changing the retrieval scorer. Reuse `src/llb/rag/chunking.py`,
  `src/llb/rag/compare.py`, and the citation sidecars.
- Data and artifact paths: FAISS stores per strategy under `$DATA_DIR/llb/rag/<strategy>/`; a
  `compare-retrieval` report gains the new strategy rows; a small page-sidecar fixture under
  `samples/` for the page-aware chunker unit tests.
- Execution path: `make build-index CHUNK_STRATEGY=<name>`;
  `python -m llb.rag.chunking --corpus-root <dir> --strategy <name> --size <n> --overlap <n>`;
  `make compare-retrieval`; unit tests assert offset-exactness and page-boundary alignment on the
  fixture.
- Acceptance gates: `make ci` green; every new strategy's chunks resolve to their exact source text
  (offset round-trip test); the page-aware strategy never splits across a page-sidecar boundary on
  the fixture; a `compare-retrieval` run ranks the new strategies against `markdown`/`semantic` on
  the committed gold set with recall@k and MRR.
- Documentation target: [RAG core](current/rag-core.md) chunking section;
  [`docs/guides/run-rag-core.md`](../guides/run-rag-core.md).

### 11. verification-gate-adjudication

- Dependencies: land after **task 3's `verify.py`/`verify_session.py` code** (both tasks extend the
  same modules; sequencing avoids a merge conflict). Task 3's human throughput evidence does not
  block this task. Otherwise independent -- all acceptance gates use synthetic reviewed fixtures.
- User-visible outcome: the human verification gate supports more than one annotator and richer
  acceptance rules: a stratified sample can be assigned to N reviewers, inter-annotator agreement
  (Cohen's/Fleiss' kappa) is reported, disagreements route to an adjudication pass, and acceptance
  arithmetic becomes configurable (per-stratum thresholds and confidence-weighted acceptance) rather
  than a single global tolerance. This is the "changes to the verification gate" item the shipped
  any-corpus autopipeline held out of scope (see [data prep](current/data-prep.md)), plus the
  multi-annotator / acceptance-arithmetic carve-outs of
  [`verify-cli-throughput`](#3-verify-cli-throughput).
- Scope boundary: in scope -- extend `src/llb/goldset/verify.py` (stratification, sampling,
  acceptance arithmetic, ledger emission) and `src/llb/goldset/verify_session.py` with a reviewer
  id on worksheet rows, an agreement report, an adjudication worksheet drawn from disagreements, and
  per-stratum / confidence-weighted acceptance thresholds; keep the accepted-ledger-through-adoption
  invariant (never hand-edit `verified`) and the CSV worksheet backward compatible (new optional
  columns only). Out of scope -- a web UI, changing the `GoldItem` schema, judge calibration.
  Reuse the existing `verify-sample`/`verify-review`/`verify-accept` command surface and the
  stratified sampler.
- Data and artifact paths: multi-reviewer worksheets and an `agreement.json` report beside the draft
  bundle's worksheets; an `adjudication.csv` worksheet; the accepted ledger under `accepted/` as
  today.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_ANNOTATORS=<k>`;
  `make verify-review VERIFY_WS=<per-reviewer-ws>`;
  `make verify-adjudicate BUNDLE=<draft>`;
  `make verify-accept BUNDLE=<draft> VERIFY_ACCEPT_POLICY=<per-stratum|weighted>`; unit tests for
  agreement math, adjudication draw, and each acceptance policy.
- Acceptance gates: `make ci` green; agreement statistics match a hand-computed fixture; adjudication
  draws exactly the disagreement rows and carries prior decisions forward; each acceptance policy is
  unit-tested against a synthetic reviewed sample; a reused id can still never certify changed
  content (adoption-through-ledger test preserved).
- Documentation target: [data prep](current/data-prep.md) verification gate;
  [`docs/guides/verification-tooling.md`](../guides/verification-tooling.md) and
  [`docs/guides/human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md).

### 12. hybrid-retrieval-uk

- Dependencies: none. Base of the retrieval cluster -- task 13 reranks the pool this task fuses,
  and its fusion knobs feed task 6 recommendations.
- User-visible outcome: retrieval gains the full hybrid shape Ukrainian enterprise corpora need --
  dense E5 plus lexical BM25 fused with reciprocal-rank fusion, plus a chunk-metadata filter seam
  -- so exact surnames, article/law numbers, codes, abbreviations, and mixed Ukrainian-English
  terminology stop losing to semantic-only search, and Ukrainian inflection (a genitive
  "начальника служби" query vs the nominative corpus form) stops defeating the lexical side;
  `compare-retrieval` demonstrates (not assumes) per corpus whether hybrid beats dense-only, what
  lemmatization adds, and how much recall headroom perfect document routing would buy; the
  sweep/tuner can grid the fusion knobs.
  Today every store is dense-only cosine (`src/llb/rag/store.py`, `src/llb/rag/vector_index.py`).
- Scope boundary: in scope -- a lexical index built beside the vector index at `build-index` time
  (pure-Python BM25, in-repo or `rank-bm25` behind the same optional-extra pattern as `[rag]`)
  over the same offset-exact chunks; Ukrainian-aware token normalization on the lexical side only
  (casefold, apostrophe-variant unification U+2019/U+02BC/`'`, punctuation strip), with opt-in
  lemmatization via `pymorphy3` + `pymorphy3-dicts-uk` (cases/inflection collapse to lemmas at
  index and query time; the stored chunk text is never altered); RRF fusion inside
  `RagStore.retrieve`
  driven by `fusion_candidates` and `fusion_weight` in `RunConfig`, so every dense `VectorIndex`
  backend (FAISS/Chroma/Qdrant/LanceDB) gains hybrid identically; a metadata filter seam over the
  fields chunks already carry (`doc_id`, markdown breadcrumb/section, PDF page range from the
  citation sidecars) applied before fusion; an oracle-doc-filter diagnostic row in
  `compare-retrieval` (candidates restricted to each gold item's `source_doc_id`) quantifying the
  recall headroom a document router would buy; a `compare-retrieval` row set (dense vs hybrid vs
  hybrid+lemmas) and tuner/sweep axes for the fusion knobs. Out of scope -- server-side hybrid
  features of any vector DB (fusion stays local and backend-neutral), new embedding models
  (task 14), query rewriting and typo tolerance (task 15), a learned document router (the oracle
  row only measures the headroom). Reuse `src/llb/rag/store.py`,
  `src/llb/rag/compare.py`, and `src/llb/optimize/tuner.py:suggest_overrides`.
- Data and artifact paths: the lexical index persists beside the FAISS artifacts in the store
  directory (`$DATA_DIR/llb/rag/`) and joins the store fingerprint; hybrid rows in the existing
  compare-retrieval report; a small exact-term goldset subset (codes, surnames, numbers) recorded
  under `samples/` for the lexical-win regression.
- Execution path: `make build-index RETRIEVAL_MODE=hybrid` /
  `llb build-index --retrieval-mode hybrid`; `llb run-eval --retrieval-mode hybrid
  --fusion-weight <w>`; `make compare-retrieval` gains the hybrid row;
  `llb sweep --rag-grid fusion_weight=...`; unit tests cover tokenizer normalization, BM25
  determinism, and RRF ordering against a fake dense index.
- Acceptance gates: `make ci` green; on the committed goldset hybrid `recall@10` is
  equal-or-better than the dense baseline and strictly better on the exact-term subset; the
  report shows the lemmatization on/off delta and the oracle-doc-filter headroom row; chunk
  offsets stay exact end-to-end and stored chunk text is byte-identical with lemmatization on; a
  store built without the lexical index refuses
  `--retrieval-mode hybrid` with a clear message; sweep cells fingerprint the fusion knobs so grid
  points resume independently.
- Documentation target: [RAG core](current/rag-core.md) retrieval store and sweep sections;
  [`docs/guides/run-rag-core.md`](../guides/run-rag-core.md).

### 13. rerank-context-order

- Dependencies: soft-follows task 12 (it reranks the fused candidate pool and is most valuable
  after hybrid; it also works over dense-only retrieval). Its reranker/order knobs feed task 6
  recommendations. The heavy real-reranker validation run executes on the CUDA host, no human
  judgment.
- User-visible outcome: a mechanism to tune what happens between retrieval and generation: an
  optional local cross-encoder reranker (retrieve `rerank_candidates`, rerank, keep `top_k`)
  measured for top-k precision gain against its own latency cost; a context-order policy
  (`rank | reverse_rank`, best-first vs best-last) applied when kept chunks are laid into the
  prompt; and `llb probe-context-position` -- a lost-in-the-middle probe that places the gold
  chunk at head/middle/tail among real distractors at fixed k and reports per-model position
  sensitivity -- ending in a per-model ordering recommendation.
- Scope boundary: in scope -- a reranker seam `src/llb/rag/rerank.py` (default candidate
  `BAAI/bge-reranker-v2-m3`, multilingual) behind `RunConfig` fields, off by default, fed by any
  retrieval backend including hybrid; the ordering policy applied at `format_context`
  (`src/llb/eval/common.py`) and recorded in the manifest; per-stage latency (retrieve vs rerank
  vs generate) in run telemetry; pre/post-rerank `recall@k`/MRR through the existing
  `evaluate_retrieval`; tuner/sweep axes for reranker on/off and candidate depth. Out of scope --
  API rerankers (egress policy), training or fine-tuning rerankers, chain-level context policies
  (task 8 owns multi-step chains; this task owns single-turn chunk ordering), changing the
  retrieval-metrics contract.
- Data and artifact paths: probe reports under
  `$DATA_DIR/context-position/<timestamp>/{report.md,cases.jsonl}`; the manifest gains reranker
  model, candidate depth, ordering policy, and per-stage latency fields.
- Execution path: `llb run-eval --reranker <hf-id> --rerank-candidates 30 --context-order rank`;
  `make compare-retrieval RERANKER=<hf-id>`;
  `llb probe-context-position --model <m> --backend <b> --k <k>`; unit tests drive an injected
  fake cross-encoder asserting candidate flow, the kept set, and exact context ordering per
  policy.
- Acceptance gates: `make ci` green with the fake cross-encoder; a real `bge-reranker-v2-m3` run
  over the committed goldset reports post-rerank MRR uplift-or-tie plus measured reranker latency
  (heavy, on the CUDA host, outside quick CI); the position probe emits per-position accuracy with
  bootstrap CIs and names the recommended ordering for the probed model; every knob lands in the
  manifest/fingerprint so sweeps and task 6's miss analysis can recommend "enable reranker" with
  numeric evidence.
- Documentation target: [RAG core](current/rag-core.md);
  [evaluation rigor](current/rigor-board-judge.md) for the position probe.

### 14. embedding-bakeoff-uk

- Dependencies: none. Heavy store builds run on the CUDA host (deterministic, no human judgment).
  Its recommended-embedder result feeds task 6.
- User-visible outcome: an evidence-backed answer to "which embedder for Ukrainian?":
  `llb compare-embeddings` builds one store per candidate embedding model over the same corpus and
  chunking and ranks candidates on `recall@k`/MRR plus embed throughput, index size, and device
  fit, ending in a written recommendation the operator applies via the existing
  `RunConfig.embedding_model`. Default local candidates: `intfloat/multilingual-e5-base` (current
  default), `intfloat/multilingual-e5-large`, `BAAI/bge-m3` (dense lane), and
  `lang-uk/ukr-paraphrase-multilingual-mpnet-base` (a paraphrase/STS-tuned model that may lose to
  retrieval-tuned E5/BGE -- exactly why the ranking must be measured, not assumed). For **open**
  corpora an operator can additionally opt a Cohere multilingual row into the same report.
- Scope boundary: in scope -- per-family query/passage conventions in `Embedder`
  (`src/llb/rag/embedding.py` already handles E5 prefixes; add the BGE and mpnet conventions);
  store fingerprints carrying the embedding model so a mismatched store/run refuses; a comparison
  report reusing `evaluate_retrieval` over the committed goldset; an opt-in
  `--api-model cohere/embed-multilingual-v3.0` lane behind the litellm conventions of
  `src/llb/prep/frontier.py` -- embedding a corpus via API is full corpus egress, so the lane
  requires the interactive consent prompt naming the corpus and destination, honors `--max-usd`,
  maps `input_type` search_query/search_document to the query/passage seam, and is refused for
  anything but explicitly open corpora (policy as recorded in
  [product decisions](current/scope-boundaries.md)). The drafting-side pinned-E5
  seams (ontology dedup, semantic scoring, retrieval-uniqueness annotation) stay pinned for
  reproducibility and are explicitly not switched by this task. Out of scope -- BGE-M3
  sparse/multi-vector output (the lexical lane belongs to task 12), making an API embedder the
  default or usable as `RunConfig.embedding_model` for scored runs (the API row is bake-off
  evidence only; scored retrieval stays local), embedder fine-tuning.
- Data and artifact paths: `$DATA_DIR/compare-embeddings/<timestamp>/report.md` plus one store per
  candidate under `$DATA_DIR/compare-embeddings/<timestamp>/stores/<model-slug>/`; the report is
  the durable evidence recorded in current docs.
- Execution path: `make compare-embeddings` /
  `llb compare-embeddings --corpus <dir> --goldset <jsonl> --models <id,id,...>` (heavy store
  builds stay outside quick CI); `make build-index EMBEDDING_MODEL=<winner>` then a normal
  `run-eval`; unit tests with a fake embedder assert store isolation, fingerprint refusal, and
  report shape.
- Acceptance gates: `make ci` green; the report ranks all four local candidates on the committed
  fixture with recall@10, MRR, embed time, and dimension/size; the Cohere row appears only after
  explicit consent and never without it (unit-tested via an injected fake litellm client, no
  network in CI); an eval run against a non-default-embedder
  store completes with the embedder recorded in the manifest, and a store/query embedder mismatch
  aborts with a clear message; current docs record the winner for the 16 GB host.
- Documentation target: [RAG core](current/rag-core.md) retrieval store;
  [platform matrix](current/platform-vector-matrix.md).

### 15. uk-query-processing

- Dependencies: none (the glossary derives from a shipped `prompt_dictionary_candidates.jsonl`
  draft artifact). Its A/B deltas feed task 6.
- User-visible outcome: an opt-in query lane between the user question and retrieval that
  measurably helps Ukrainian queries while never touching the stored corpus text: deterministic
  normalization (matching-side casefold, apostrophe-variant unification, a small transliteration
  table for Latin-typed Ukrainian terms), corpus-vocabulary typo tolerance, alias/glossary
  expansion (including surzhyk and transliterated variants of domain terms) sourced from the
  shipped `prompt_dictionary_candidates.jsonl` draft
  artifact, and an optional logged local-LLM query rewrite -- with an A/B report proving each
  step's retrieval delta before anyone turns it on by default.
- Scope boundary: in scope -- `src/llb/rag/query_prep.py` as a pure, unit-testable pipeline of
  named steps; a deterministic typo-tolerance step: build the token vocabulary from the indexed
  corpus, correct a query token that is ABSENT from that vocabulary to its nearest in-vocabulary
  token within Damerau-Levenshtein distance 1 (2 for tokens over 8 chars), never alter a token
  the corpus already contains, and log every correction; a glossary builder that turns a draft
  bundle's dictionary candidates into alias
  expansions, with room for hand-added surzhyk/transliteration aliases in the same artifact;
  the LLM rewrite through the existing endpoint seam, off by default, recording both
  original and rewritten query per case; an A/B mode in `validate-retrieval`/`compare-retrieval`
  reporting per-step `recall@k`/MRR deltas. Out of scope -- mutating corpus or chunk text
  (original word forms stay untouched; index-side token normalization belongs to task 12),
  multi-turn conversational rewriting (task 8), learned/ML spell-correction models (the
  edit-distance step is the deterministic ceiling this project needs).
- Data and artifact paths: a `query_glossary.json` artifact derived from a draft bundle and
  referenced by path in `RunConfig`; A/B rows in the compare/validate reports; original and
  processed query per case in run bundles.
- Execution path: `llb run-eval --query-prep normalize,typos,glossary`;
  `llb validate-retrieval --query-prep <steps> --query-prep-ab`;
  `llb build-query-glossary --bundle <draft> --out <json>`; unit tests: apostrophe unification,
  transliteration-table round-trips, typo correction that never touches in-vocabulary tokens,
  deterministic alias expansion, and exact no-op when the lane
  is off.
- Acceptance gates: `make ci` green; the A/B report on the committed goldset attributes a
  per-step delta (positive, zero, or negative -- the mechanism reports it honestly); the raw query
  is always preserved in logs; the LLM-rewrite step never runs without its explicit flag; current
  docs record the measured deltas for the fixture.
- Documentation target: [RAG core](current/rag-core.md); [data prep](current/data-prep.md) for the
  glossary artifact provenance.

### 16. groundedness-citation-metrics

- Dependencies: none. Enriches the per-case signals tasks 6 and 8 consume. The one manual
  groundedness/abstention run executes on the CUDA host (deterministic, no human judgment).
- User-visible outcome: answer-side RAG quality beyond reference-answer overlap: a cited-answer
  mode whose prompt requires `[i]` chunk citations for factual claims, scored for citation
  validity (the cited chunk actually contains the supporting span) and hallucinated-citation rate;
  a deterministic groundedness signal (fraction of answer content supported by the retrieved
  context via span/overlap matching, with the calibration-gated judge as an optional secondary);
  and insufficient-context probes -- gold questions re-run with their gold evidence excluded from
  retrieval -- where correct behavior is explicit abstention, scored as abstention accuracy. All
  three become additive columns in run bundles, the board, and `recommend`.
- Scope boundary: in scope -- `src/llb/scoring/groundedness.py`; an `eval.rag.cited_answer`
  template in the prompt registry (`src/llb/prompts/registry.py`) reusing the numbered-chunk
  format `format_context` already emits; probe construction by excluding gold-span chunks at
  retrieval time for a sampled subset; abstention detection reusing the refusal /
  insufficient-data markers from the shared taxonomy while keeping "correct abstention on a
  probe" distinct from "refusal on a scoreable case". Out of scope -- a RAGAS dependency,
  frontier judges (egress policy), changing the headline objective (new metrics stay separate
  columns until a ranking policy explicitly adopts them), chain scoring (task 8).
- Data and artifact paths: additive per-case fields in `scores.jsonl`/`scores.parquet` (citation
  validity, groundedness fraction, probe flag, abstention outcome); aggregate columns in board
  and `recommend` summaries when present.
- Execution path: `llb run-eval --cited-answers --score-groundedness`;
  `llb run-eval --insufficient-context-probes <n>`; unit tests over synthetic contexts/answers
  covering fully-supported, partially-supported, unsupported-claim, invalid-citation, and
  correct-abstention cases.
- Acceptance gates: `make ci` green; the deterministic scorer separates fully-supported from
  injected-unsupported answers on the synthetic fixture with zero cross-class leakage; a citation
  pointing at a chunk that lacks the claimed span is always flagged; probe cases never enter plain
  correctness aggregates; one manual run on the committed goldset records per-model groundedness
  and abstention accuracy in current docs.
- Documentation target: [RAG core](current/rag-core.md) scoring;
  [evaluation rigor](current/rigor-board-judge.md).

### 17. corpus-governance-metadata

- Dependencies: the ACL-filter half soft-follows task 12 (it applies through the same chunk-metadata
  filter seam); the governance fields and reindex policy stand alone.
- User-visible outcome: corpus ingestion and the RAG store gain governance metadata and a lifecycle
  policy: every `corpus_manifest.json` entry and chunk record carries `language`,
  `version`/`effective_date` when the source provides one, `ingestion_time`, `source_system`, and
  an optional ACL label; retrieval can filter candidates by ACL label before anything reaches the
  model; and `ingest-corpus`/`build-index` gain deletion propagation (a source removed from the
  corpus root drops out of the next build and the manifest diff says so), stale-store detection
  (store fingerprint vs corpus manifest), and a documented rollback unit (immutable store
  directories).
- Scope boundary: in scope -- additive optional governance fields on `corpus_manifest.json`,
  `ChunkRecord.metadata`, and `store_meta.json` (passthrough text derives `language` from an
  operator-supplied default or a cheap detector; PDF lanes inherit from the conversion manifest);
  an ACL-filter argument through the task 12 metadata-filter seam with a refusal guarantee (a
  query scoped to an ACL label never receives an out-of-scope chunk); stale/deleted-doc detection
  comparing the store fingerprint against the corpus manifest with a clear rebuild message. Out of
  scope -- runtime prompt-injection filtering and output PII filters (decision recorded in
  [product decisions](current/scope-boundaries.md)), a permissions backend or user identity model
  (the ACL label is a plain string tag; enforcement policy belongs to the embedding application),
  mutating stored chunk text or offsets.
- Data and artifact paths: governance fields inline in `corpus_manifest.json`, `chunks.jsonl`, and
  `store_meta.json`; a small mixed-ACL, mixed-language fixture under `samples/` for the filter,
  deletion, and staleness tests.
- Execution path: `llb ingest-corpus --default-language uk --acl-label <tag>`;
  `llb build-index` (staleness check against the corpus manifest);
  `llb run-eval --acl <tag>` once the task 12 filter seam exists; unit tests cover field
  propagation end-to-end, ACL filtering, deletion propagation, and the stale-store refusal.
- Acceptance gates: `make ci` green; every chunk built from the fixture carries its governance
  fields through retrieval into the returned chunk records; an ACL-scoped retrieval never returns
  an out-of-scope chunk (unit-tested); removing a source document and re-ingesting drops its chunks
  from the next build with the removal recorded in the manifest; a store older than its corpus
  manifest refuses with a rebuild message; stored chunk text and offsets stay byte-identical.
- Documentation target: [data prep](current/data-prep.md) ingestion;
  [RAG core](current/rag-core.md) retrieval store.

## Human-Assisted Tasks

Each task's code and unit tests are agent-buildable; the marked **human step** is what gates
completion. Tasks 3 and 7 also gate agent work: land task 3's code before agent task 11, and
finish task 7's human review before agent task 8's acceptance run.

### 1. draft-yield-quality-max -- residual empirical acceptance

- Dependencies: none (uses the shipped drafting knobs). Human step: the acceptance evidence below
  needs a local drafter model and a human reviewer and cannot run in CI; the optional multi-hop
  hardening is agent-buildable and unit-tested.
- Context: coverage-target sampling (`--coverage-target`), 2-hop graph-path multi-hop drafting
  (`--multi-hop`), pinned-E5 prior-bundle near-duplicate suppression (`--dedup-against`), and the
  closed question-type + difficulty labels are implementable and unit-covered (module map, report
  fields, and command reference in
  [robust backends and ontology drafting](current/robustness-ontology-backends.md) and
  [data prep](current/data-prep.md)). What remains is the heavy manual acceptance evidence and one
  optional quality hardening, both needing a local drafter model and human review (out of CI):
- Acceptance evidence (human to-do): on the local quickstart PDF corpus, draft once with
  `DRAFT_COVERAGE_TARGET=<n>` and once with the 180-cap default over the same corpus/model, run a
  `make verify-sample VERIFY_N=40` review of each, and record in [data prep](current/data-prep.md)
  whether the coverage-target bundle keeps more citation-valid needles at an equal-or-better accept
  rate, with the retrieval-unique needle fraction per question type. This is the "keeps more needles
  at equal-or-better accept rate" gate that cannot run in CI.
- Optional quality hardening (agent-buildable): multi-hop items ground the two hop-evidence spans
  but leave the reference answer free-text. Require the multi-hop reference answer to be (or
  contain) the verbatim bridge/end-entity span so the answer itself is span-checkable, and extend
  the multi-hop unit tests to assert it -- a free-text answer can drift from the chain even when the
  evidence spans hold.
- Documentation target: [data prep](current/data-prep.md) and
  [robust backends and ontology drafting](current/robustness-ontology-backends.md).

### 2. frontier-ua-draft-lane

- Dependencies: none (code reuses `src/llb/prep/frontier.py`). Human step: the real-frontier
  2-document probe requires **human egress consent and API spend** under the recorded egress policy;
  the code and all fake-completer tests are agent-buildable without any network call.
- User-visible outcome: for the best Ukrainian question quality and completeness, an operator
  can opt a draft run into a best-of-breed external API (litellm-routed) for extraction,
  drafting, or both, with an explicit consent gate, a hard budget cap, per-call cost telemetry
  in provenance, and a side-by-side local-vs-frontier yield and quality report over the same
  seeds. Frontier cross-check exists (`make cross-check-goldset CROSS_CHECK_MODEL=`); the draft
  lane does not.
- Scope boundary: in scope -- a frontier endpoint option for the ontology pipeline reusing the
  litellm conventions in `src/llb/prep/frontier.py` behind the same endpoint seam as Ollama and
  vLLM drafting (`src/llb/prep/ontology/endpoint.py`); `--max-usd` and `--max-calls` guards
  that abort cleanly and record the reason; an interactive egress consent prompt naming the
  corpus and destination (policy stays as recorded in
  [product decisions](current/scope-boundaries.md)); a `llb draft-compare` command that drafts
  the same bounded seed subset locally and via frontier and reports kept-yield, gate results,
  and verify-sample accept rate. Out of scope -- making frontier the default, egress for
  scoring or judging, retries of the egress policy discussion.
- Data and artifact paths: `provenance.json` gains `endpoint.cost_usd`, call counts, and
  latency; comparison reports under `$DATA_DIR/draft-compare/<timestamp>/`.
- Execution path:
  `make prepare-goldset-draft DRAFT_ENDPOINT=frontier DRAFT_FRONTIER_MODEL=<litellm-id>
  DRAFT_MAX_USD=<n>`; `llb draft-compare --corpus-root <dir> --seeds <n> --frontier-model <id>
  --local-model <model>`; unit tests use an injected fake litellm completer.
- Acceptance gates: no network call happens without the flag plus consent (unit-tested via the
  injected completer); the budget guard aborts mid-draft and the bundle remains inspectable
  with the abort recorded; a 2-document probe against a real frontier model passes bundle gates
  with parse rate at least matching the local drafter; the comparison report ranks both lanes
  on kept-yield and accept rate.
- Documentation target: [data prep](current/data-prep.md) frontier lane notes;
  [`docs/guides/goldset-from-scratch.md`](../guides/goldset-from-scratch.md).

### 3. verify-cli-throughput

- Dependencies: none. **Blocks agent task 11** (that task extends the same
  `verify.py`/`verify_session.py` surface -- land this task's code first). Human step: the
  "materially more items per hour" outcome needs a recorded human 40-item review pass with measured
  throughput; the CLI code and unit tests are agent-buildable.
- User-visible outcome: a human reviewer clears materially more items per hour in the terminal
  review session: a confidence-ordered queue (cross-check verdict and `retrieval_rank` decide
  order), on-card evidence with PDF page citations, accept-with-edit that re-grounds an edited
  answer span immediately, additive sample enlargement that never re-shows decided rows,
  session stats with an items-per-hour ETA, and coded rejection reasons exported for draft
  feedback.
- Scope boundary: in scope -- extend `src/llb/goldset/verify.py` and
  `src/llb/goldset/verify_session.py`; keep the worksheet CSV shape backward compatible (new
  optional columns only); a `verify-sample` merge mode that draws additional stratified rows
  while carrying prior decisions forward; a rejection-reason summary artifact the drafting
  pipeline can read to tighten prompts. Out of scope -- a web UI, multi-annotator merging,
  changes to acceptance arithmetic.
- Data and artifact paths: worksheets under the draft bundle as today; a
  `rejection_reasons.json` summary beside the accepted ledger.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_MERGE=1`;
  `make verify-review VERIFY_WS=<ws> VERIFY_ORDER=confidence`; unit tests for merge
  idempotence, ordering, and re-grounding of edited spans.
- Acceptance gates: `make ci` green including session golden-path tests; merging a larger
  sample adds only new rows and preserves every decided row byte-for-byte; an edited answer
  that no longer matches its span is blocked until re-grounded; manual evidence -- one recorded
  40-item review pass on the quickstart draft with the measured items-per-hour noted in the
  current docs.
- Documentation target: [data prep](current/data-prep.md) verification gate;
  [`docs/guides/human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md)
  and [`docs/guides/verification-tooling.md`](../guides/verification-tooling.md).

### 5. security-corpus-probes

- Dependencies: the shipped ontology artifacts (`ontology.json`, `extraction.jsonl`) from a
  local-drafter bundle. Human step: derived security cases must clear the human
  `verify-sample`/`verify-review`/`verify-accept` gate before any headline/composite use; the
  generator and unit tests are agent-buildable.
- User-visible outcome: the security tier gains corpus-specific cases derived from the target
  corpus itself: prohibited-topic denial-guard probes built from the corpus ontology's
  sensitive topics and entities, benign near-boundary controls that catch over-refusal on
  legitimate corpus questions, and matched-pair bias probes (entity or group swapped, behavior
  fixed) scored for decision consistency -- all flowing through the existing detectors,
  cross-language grouping, and the human verification gate before any headline use.
- Scope boundary: in scope -- `llb derive-security-cases --bundle <draft-bundle>` reading
  `ontology.json` and `extraction.jsonl`, generating cases locally through the same drafter
  endpoint seam; emitted cases reuse the committed case schema (`lang`, `attrs.vector`,
  `xlang_group`) so `bench-security`, `cross_language_consistency`, and refusal-appropriateness
  work unchanged (see [category suite](current/category-benchmark-suite.md) and the Ukrainian
  security adaptation in [evaluation rigor](current/rigor-board-judge.md)); a bias-pair
  consistency metric alongside ASR reusing the matched-group machinery. Out of scope -- new
  detector kinds, safety classifiers, ranking unverified case sets.
- Data and artifact paths: derived cases under `$DATA_DIR/security-derive/<timestamp>/cases.json`
  with per-case grounding spans back to the corpus; a small derived-and-verified sample
  committed under `samples/` for regression.
- Execution path: `llb derive-security-cases --bundle <draft> --out <cases.json>`;
  `make bench-security SECURITY_CASES=<cases.json> MODEL=<m> BACKEND=<b>`; verification through
  the existing `verify-sample`/`verify-review`/`verify-accept` path.
- Acceptance gates: every generated probe cites a corpus topic or entity with an exact span
  (unit-tested); benign controls feed refusal-appropriateness only, never ASR; a full
  `bench-security` run over derived quickstart-corpus cases reports per-family ASR plus
  bias-pair consistency with bootstrap CIs; unverified derived sets are rejected from
  composite/headline paths.
- Documentation target: [category suite](current/category-benchmark-suite.md) security section;
  [`docs/guides/learning-path-security.md`](../guides/learning-path-security.md).

### 7. chain-goldset-generation

- Dependencies: none (reuses the shipped graph-path walker `src/llb/prep/ontology/graph_paths.py`).
  **Blocks agent task 8** (`context-policy-bench` scores this task's verified chain fixture). Human
  step: >= 10 chains must be human-reviewed and accepted into the committed fixture; the schema,
  drafting, and validation code are agent-buildable.
- User-visible outcome: the draft pipeline also emits chain-of-questions test sets: ordered 2-4
  step sequences in which each step supplies more specific context for the topic (topic
  overview -> narrowing detail -> exact fact), every step carrying its own reference answer and
  exact source spans, validated and human-verified with the same discipline as flat items.
- Scope boundary: in scope -- a `ChainItem` schema in `src/llb/goldset/chains.py` whose steps
  embed `GoldItem`-compatible question/answer/span fields plus a chain id, step order, and a
  dependency note describing what the previous step establishes; chain drafting seeded from
  knowledge-graph paths and heading hierarchies (reuses the shipped graph-path walker
  `src/llb/prep/ontology/graph_paths.py`);
  span-exact validation via an extended `validate-goldset`; verify-session support rendering a
  chain as one card with per-step checks. Out of scope -- the scoring runner
  (`context-policy-bench`), multi-annotator flows.
- Data and artifact paths: draft bundles gain `chains.jsonl`; a committed fixture
  `samples/goldsets/<name>/chains.jsonl` with verified chains for smoke tests.
- Execution path: `make prepare-goldset-draft DRAFT_CHAINS=1`;
  `llb validate-goldset --chains <chains.jsonl> --corpus <dir>`; chains flow through
  `verify-sample`/`verify-review`/`verify-accept` unchanged at the command level.
- Acceptance gates: every step of every chain passes span-exact validation; steps within a
  chain reference distinct spans and the final step's answer is not answerable from step-1
  context alone (checked by the retrieval-uniqueness filter applied per step); at least 10
  chains reviewed and accepted into the committed fixture; `make ci` green.
- Documentation target: [data prep](current/data-prep.md).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it. Place it under
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; either way give it a `Dependencies` line and
mark any cross-section block explicitly.

Each task entry must include:

- Dependencies: prerequisite tasks (by number/id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
