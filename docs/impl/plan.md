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
`top_k`, the sweep grids `top_k`, and task 10 adds strategies). Together 12 + 15 cover the
Ukrainian-language retrieval stack: dense + BM25/sparse + metadata hybrid with
inflection-aware lemmatization (12) and query-side
normalization -- casefold, apostrophes, transliteration, typo tolerance, aliases/glossary --
that never mutates the stored corpus text (15). The measured Ukrainian embedder ranking that
underpins both (`llb compare-embeddings` over BGE-M3 / multilingual-e5 / the lang-uk model plus an
opt-in Cohere API row for open corpora) is now shipped; see [RAG core](current/rag-core.md)
retrieval store. Every knob these tasks add must land
in `compare-retrieval`, the sweep grid, or the tuner search space so task 6's miss analysis can cite
it as evidence-backed. Within the cluster only 13 has an ordering preference (it reranks the pool 12
fuses, so it pays off most after 12); 15 and 16 stand alone. Task 17 adds the governance
remainder -- per-chunk `language`/`date`/`version`/`ACL` metadata, permission-aware retrieval, and
the reindex/deletion/rollback policy (measured shortfall and scope decision recorded in
[RAG core](current/rag-core.md) and [product decisions](current/scope-boundaries.md)). The
strategy-independent page/section join that links every chunk back to its origin file, page, and
heading -- from chunk char offsets to the PDF citation sidecars -- is now shipped
(`src/llb/rag/page_metadata.py`; see [RAG core](current/rag-core.md) retrieval store); tasks 10 and
12 reuse its sidecar loader and the `pages`/`source_pdf`/`headers` fields it attaches.

The fine-tuning cluster (18-22) extends the spine one step past recommendation: from naming the
best base model to naming the best *adapted* model for the operator's corpus, with the whole loop
drivable by an agent end to end. Task 18 builds the single-model loop (contamination-guarded
export, injectable trainer seam, round report) and owns the invariants every later task reuses;
19 runs that loop across a multi-model roster with feasibility-aware scheduling and ranks models
by measured tunability; 20 gives adapters a registry and lifecycle so every tuned board number
stays traceable and servable; 21 adds budgeted per-model hyperparameter search that never leaves
the tuning split; 22 (optional) distills the roster's best local teacher into smaller students --
all local, no egress. Ordering inside the cluster: 18 first, then 19; 20 lands beside 19; 21 and
22 after 19.

## Agent Implementation Tasks

These land to `make ci` green with fixtures, fakes, and deterministic harnesses. The two
prioritized Ukrainian-RAG-quality foundations are both shipped: the measured embedder ranking
(`llb compare-embeddings`; see [RAG core](current/rag-core.md) retrieval store) that replaces the
assumed default embedder with evidence, and the page/section join
(`src/llb/rag/page_metadata.py`) that links every chunk back to its origin file; tasks 10 and 12
build on the latter's sidecar loader and fields, and 12/15 build on the measured embedder result.
The external multi-service drafting lane is also shipped end to end -- both the `curate-drafts`
merge/dedup/filter step and the grounded-JSONL `import-external-draft` lane for full-document needle
realism (see [data prep](current/data-prep.md) grounded-JSONL import).
Recommended sequence: **6 first** (its probe mode reuses the shipped durable-eval-runner), then
the independent lot (10, 12, 15, 16) in any order, 13 after 12, 17's ACL-filter half after 12's
metadata-filter seam (its governance fields stand alone), 11 after task 3's code, 18 after 6
(its miss-targeted export consumes task 6's miss clusters; the export/guard/trainer code stands
alone), 19-22 after 18 (the fine-tuning cluster reuses 18's trainer seam and contamination
guard; 20 beside 19, 21 and 22 after 19), and 8 last (blocked by human task 7). The
durable-eval-runner (retry + `cases.progress.jsonl` journal +
`--resume` + bounded backend relaunch + `manifest.durability` counters) is now shipped; see
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
  section; [`docs/guides/benchmarking/mlflow-analysis.md`](../guides/benchmarking/mlflow-analysis.md).

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
  [`docs/guides/benchmarking/prompt-system-rag.md`](../guides/benchmarking/prompt-system-rag.md).

### 10. corpus-chunking-strategies

- Dependencies: soft-follows the shipped page-metadata join (`src/llb/rag/page_metadata.py`): the
  page-aware strategy reuses its sidecar page-span loader; the boundary alignment and new
  strategies are this task's own. Picks up the "new
  chunking strategies" item the shipped autopipeline held out of scope; independent of the rest.
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
  so `validate-goldset` and source-span scoring keep working; the page-aware strategy aligns chunk
  boundaries on the page spans exposed by the shipped page-metadata sidecar loader
  (`src/llb/rag/page_metadata.py`); a
  `compare-retrieval` row per new strategy so the best chunker is DEMONSTRATED per corpus; the RAG
  build grid gains the strategies behind a flag. Out of scope -- new embedding models, changing the
  source-span gold contract, changing the retrieval scorer, attaching page metadata to existing
  strategies' chunks (the shipped page-metadata join owns that). Reuse `src/llb/rag/chunking.py`,
  `src/llb/rag/compare.py`, and `src/llb/rag/page_metadata.py` (the shipped sidecar loader).
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
  [`docs/guides/benchmarking/run-rag-core.md`](../guides/benchmarking/run-rag-core.md).

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
  [`docs/guides/human-tooling/verification-tooling.md`](../guides/human-tooling/verification-tooling.md)
  and
  [`docs/guides/human-tooling/human-in-the-loop-evaluation.md`](../guides/human-tooling/human-in-the-loop-evaluation.md).

### 12. hybrid-retrieval-uk

- Dependencies: soft-follows the shipped page-metadata join (the metadata filter seam filters over
  the page/section fields `src/llb/rag/page_metadata.py` attaches; the lexical/fusion half stands
  alone). Base of the retrieval cluster --
  task 13 reranks the pool this task fuses, and its fusion knobs feed task 6 recommendations.
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
  backend (FAISS/Chroma/Qdrant/LanceDB) gains hybrid identically; a metadata filter seam over
  `doc_id` plus the section breadcrumb and PDF page range the shipped page-metadata join attaches,
  applied before fusion; an oracle-doc-filter diagnostic row in
  `compare-retrieval` (candidates restricted to each gold item's `source_doc_id`) quantifying the
  recall headroom a document router would buy; a `compare-retrieval` row set (dense vs hybrid vs
  hybrid+lemmas) and tuner/sweep axes for the fusion knobs. Out of scope -- server-side hybrid
  features of any vector DB (fusion stays local and backend-neutral), new embedding models
  (the shipped `compare-embeddings` bake-off owns embedder selection; see
  [RAG core](current/rag-core.md) retrieval store), query rewriting and typo tolerance (task 15),
  a learned document router (the oracle
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
  [`docs/guides/benchmarking/run-rag-core.md`](../guides/benchmarking/run-rag-core.md).

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
- Data and artifact paths: additive per-case fields in `scores.jsonl` (citation
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

### 18. local-model-self-improvement-loop

- Dependencies: soft-follows task 6 (`miss-analysis-recommendations`): when a miss analysis
  exists, the training-set export targets and weights the miss clusters; without one it falls
  back to the whole tuning split, so task 6 improves this task but does not block it. Reuses
  the shipped split discipline (`src/llb/goldset/splits.py` -- calibration/tuning/final are
  disjoint by seeded assignment precisely so tuning can never leak into the final leaderboard
  number), the durable-eval-runner (per-round resume), and the board/recommend machinery. The
  heavy fine-tune + re-eval rounds execute seeded on the CUDA host with no human judgment --
  the same heavy-run discipline as tasks 13 and 16.
- User-visible outcome: the benchmark closes its loop from measurement to improvement: one
  command turns a scored run into a measurably better *local* model. It exports a
  contamination-guarded training set from the tuning split (SFT records in the exact prompt
  shape the eval sends; optional preference pairs built from the model's own scored misses),
  LoRA/QLoRA fine-tunes the local model, re-evaluates the adapter as a new board row through
  the unchanged eval runner, and iterates rounds until the gain disappears -- ending with a
  per-round report (base vs tuned on the held-out final split, bootstrap CIs) and an explicit
  accept/reject verdict for the adapter. Task 6's evidence-backed "model X fails on cluster Y"
  becomes "model X + adapter-`<digest>` passes, with the round-by-round proof".
- Scope boundary: in scope -- `src/llb/finetune/dataset.py`: a deterministic export from a
  finalized run bundle plus its goldset -- SFT records (question + retrieved context ->
  reference answer, reusing the eval's own prompt templates so train and eval formats cannot
  drift) drawn ONLY from tuning-split items, optional DPO preference pairs (the model's scored
  wrong answer = rejected, the reference = chosen) from task 6's `misses.jsonl` when present,
  and a `dataset_manifest.json` recording item ids, split provenance, and a content digest;
  `src/llb/finetune/trainer.py`: seeded LoRA/QLoRA behind an injectable trainer seam (real
  implementation via a new `[finetune]` optional extra -- peft/trl -- following the existing
  extras pattern; CI drives a fake trainer), emitting an adapter directory plus
  `adapter_manifest.json` (base model id, dataset digest, hyperparameters, seed, loss curve);
  the contamination guard as the non-negotiable invariant: `run-eval` refuses to score a
  model+adapter whose recorded dataset digest intersects calibration/final item ids --
  extending the split discipline from configs to weights -- and a tuned model is barred from
  judging its own answers, mirroring the recorded planter != judge guard
  (`src/llb/prep/frontier.py`); adapter serving through the existing backend seam (vLLM LoRA
  modules directly; a merge-to-GGUF lane for ollama/llama.cpp), with base model + adapter
  digest recorded in the run manifest; an orchestrator `llb self-improve` chaining
  run-eval -> analyze-misses -> export -> fine-tune -> re-eval per round, stopping on a
  CI-overlapping delta or the round budget, resumable mid-campaign; tuned board rows labeled
  `<model>+adapter-<digest>` beside the base row, and a self-improvement section in the
  `recommend` summary when rounds exist. Out of scope -- full-parameter training, RLHF/online
  RL, training or altering the judge or the embedder, frontier-API distillation (egress; the
  frontier lane belongs to human task 2), auto-adopting an adapter as the recommended default
  (the verdict is reported; adoption stays an operator decision), changing the `GoldItem`
  schema or the split assignment.
- Data and artifact paths:
  `$DATA_DIR/self-improve/<timestamp>/round-<i>/{dataset/,adapter/,run/,report.md}` with
  `dataset_manifest.json` and `adapter_manifest.json` as above; a synthetic scored-bundle +
  goldset fixture and a poisoned adapter-manifest fixture under `samples/` for the export and
  guard tests.
- Execution path: `llb export-finetune-set --run-dir <run> --goldset <gs> --out <dir>`;
  `llb finetune-adapter --dataset <dir> --model <m> --seed <s>` (heavy, CUDA host, outside
  quick CI); `llb self-improve --model <m> --backend <b> --rounds 2` and
  `make self-improve MODEL=<m> BACKEND=<b>`; unit tests cover split discipline of the export
  (it can never emit a calibration/final id), preference-pair construction from the synthetic
  miss fixture, the contamination guard's refusal on the poisoned manifest, fake-trainer loop
  wiring including the stop rule, and per-round resume.
- Acceptance gates: `make ci` green with the fake trainer; the split-leakage test proves zero
  calibration/final items in any export; the contamination guard blocks the poisoned-manifest
  fixture with a clear message naming the offending ids; one real seeded QLoRA round on the
  CUDA host over the committed goldset records base vs tuned final-split scores with bootstrap
  CIs in current docs -- reported honestly (gain, tie, or regression are all acceptable
  evidence; the mechanism must not require a gain to land); the tuned row shows no security-tier
  regression (`bench-security` delta recorded beside the correctness delta); provenance chains
  adapter -> dataset digest -> source run so every tuned board number traces to its exact
  training data.
- Documentation target: [extended workflows](current/extended-workflows.md) self-improvement
  section; a new operator guide
  [`docs/guides/benchmarking/self-improvement-loop.md`](../guides/benchmarking/self-improvement-loop.md).

### 19. finetune-campaign-multi-model

- Dependencies: follows task 18 (`local-model-self-improvement-loop`) -- the campaign reuses
  its dataset export, injectable trainer seam, contamination guard, and per-round report; land
  18's code first. Soft-follows task 6 (per-model miss-targeted exports when an analysis
  exists). Reuses the feasibility planner (`src/llb/backends/planner.py` -- can this model run
  on THIS host, and at what context), the VRAM reclaim gate (`src/llb/executor/vram.py` -- the
  sequential-execution contract between roster entries), and the durable-runner journal pattern
  for campaign resume. The heavy campaign executes seeded on the CUDA host, no human judgment.
- User-visible outcome: one command fine-tunes and re-evaluates a whole roster of local models
  over the same corpus goldset, answering a question the base-model leaderboard cannot: which
  model is the best pick for this corpus *after* adaptation. Sequential VRAM-safe scheduling,
  per-model rounds, and a campaign report ranking roster models by measured tunability --
  final-split gain with bootstrap CIs against training wall-clock and peak VRAM -- beside
  base-vs-adapted board rows, so `recommend` can name the best adapted model with evidence.
- Scope boundary: in scope -- `src/llb/finetune/campaign.py` orchestrating task 18's loop per
  roster entry from an explicit `--models` list (there is no single roster variable today; the
  campaign defines the roster shape and `make` passes it through); the model-independent SFT
  export computed once per campaign and shared across entries while preference pairs stay
  per-model (each model's own scored misses); feasibility-aware scheduling -- a roster entry
  the planner rejects, or whose trainer cannot fit beside the serving stack, is skipped with
  the recorded reason, never crashed into; VRAM reclaim enforced between entries exactly as
  between eval runs; a `campaign.progress.jsonl` journal with `--resume` that never re-trains
  a finished entry; adapted rows labeled per task 18 beside every base row; a tunability
  section in the `recommend` summary when a campaign report exists. Out of scope -- parallel
  or multi-GPU training (the sequential single-host contract stands), cross-model weight
  merging or model soups, changing the contamination guard (task 18 owns it), frontier or API
  teachers (egress).
- Data and artifact paths:
  `$DATA_DIR/finetune-campaign/<timestamp>/{campaign.progress.jsonl,report.md}` plus
  `<model>/round-<i>/` per entry reusing task 18's round layout; the shared SFT export stored
  once under the campaign root with its `dataset_manifest.json`.
- Execution path: `llb finetune-campaign --models <m1,m2,...> --backend <b> --rounds <n>` and
  `make finetune-campaign MODELS=<csv> BACKEND=<b>`; unit tests drive the fake trainer plus a
  fake planner -- scheduling order, skip-with-reason on an infeasible entry, journal resume
  mid-roster, shared-export reuse (byte-identical dataset digest across entries), and the
  ranking math in the report.
- Acceptance gates: `make ci` green with the fakes; resuming a half-finished campaign
  re-trains nothing already journaled (unit-tested); an infeasible roster entry appears in the
  report with its planner reason; every adapted row passes task 18's contamination guard; one
  real campaign over at least two quickstart-scale models on the CUDA host records the
  tunability ranking with CIs in current docs -- honestly (a roster where no model gains is
  acceptable evidence).
- Documentation target: [extended workflows](current/extended-workflows.md) self-improvement
  section; the self-improvement-loop guide gains the campaign chapter.

### 20. adapter-registry-lifecycle

- Dependencies: follows task 18 (registers the adapters its rounds emit and extends its
  manifests); task 19 soft-consumes the registry when present (campaign rounds auto-register).
  Land after 18, beside 19. All gates run on committed fixtures -- no heavy run.
- User-visible outcome: adapters become first-class, traceable artifacts instead of loose
  directories: a local registry lists every adapter with its base model, dataset digest,
  source run, and eval evidence; staleness is detected (the goldset or corpus changed since
  training, so the recorded evidence no longer describes the present benchmark) and stamped,
  never silently ignored; one command serves any registered adapter through the existing
  backends; and superseded adapters can be garbage-collected without ever deleting one a run
  bundle still cites. Board and `recommend` cite only registered adapters, so every tuned
  number stays reproducible.
- Scope boundary: in scope -- `src/llb/finetune/registry.py` over an append-only
  `$DATA_DIR/adapters/registry.jsonl` (id = adapter digest; entries record base model id,
  dataset digest, goldset/corpus digests, source run path, and an eval summary); automatic
  registration on a successful task 18/19 round; a staleness check comparing recorded digests
  against the current goldset/corpus with the verdict shown in `llb list-adapters`;
  `llb serve-adapter --adapter <id> --backend vllm|ollama|llamacpp` wiring the existing
  backend seam (vLLM LoRA modules directly; the merge-to-GGUF lane for ollama/llama.cpp, with
  the merge recorded as a registry event); `run-eval` resolving adapter ids through the
  registry so the contamination guard reads recorded digests, not operator-supplied ones; GC
  that refuses to delete an adapter referenced by any run bundle unless forced. Out of scope
  -- remote registries or hubs, uploading adapters anywhere (egress), automatic retraining on
  staleness (report only), a long-running serving daemon.
- Data and artifact paths: `$DATA_DIR/adapters/registry.jsonl` plus the adapter directories it
  indexes; a committed registry fixture with a stale entry and a poisoned-digest entry under
  `samples/` for the lifecycle tests.
- Execution path: `llb list-adapters`; `llb serve-adapter --adapter <id> --backend <b>`;
  `llb gc-adapters [--force]`; unit tests -- registry round-trip, staleness flip when the
  goldset digest changes, guard resolution through the registry (the poisoned entry is
  refused), GC refusal on a cited adapter, merge-event recording via a fake backend.
- Acceptance gates: `make ci` green; a stale adapter is always stamped before its row can
  render on the board; the contamination guard rejects the poisoned-digest fixture with a
  message naming the intersecting ids; GC never deletes a cited adapter without `--force`
  (unit-tested); serving smoke passes against the fake backend for all three backends.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide's serving section.

### 21. finetune-hparam-search

- Dependencies: follows task 18 (searches over its injectable trainer seam); reuses the Optuna
  study conventions of `src/llb/optimize/tuner.py` (the `[track]` extra already pins optuna) --
  seeded study, pruned infeasible points, bounded budget -- and feeds task 19 (a recorded best
  config becomes that model's campaign default). The real bounded search runs on the CUDA host.
- User-visible outcome: fine-tuning stops guessing hyperparameters: per model, a budgeted
  Optuna search over the LoRA space (rank, alpha, learning rate, epochs, target modules) finds
  the best configuration -- scored on a held-out dev slice carved from the tuning split ONLY,
  so neither calibration nor final ever influences the search -- and records it beside the
  adapter artifacts for tasks 18/19 to consume as defaults.
- Scope boundary: in scope -- `src/llb/finetune/hparam_search.py` reusing the tuner's study
  conventions; a seeded dev-slice split *within* the tuning split (train sub-slice vs dev
  sub-slice, disjointness unit-tested), extending the recorded split discipline from
  configuration knobs to hyperparameters; `--max-trials` / `--max-hours` budget guards that
  abort cleanly with the study resumable; an `hparams_manifest.json` per model (best config,
  study seed, trial table, objective values) consumed by the task 18 trainer as defaults;
  fake-trainer trials in CI with a synthetic objective. Out of scope -- searching on the final
  or calibration split (forbidden by the guard), full-parameter tuning, a second tuner
  framework, per-trial human review.
- Data and artifact paths: `$DATA_DIR/finetune-hparams/<model>/<timestamp>/` with the study
  journal and `hparams_manifest.json`; the manifest path recorded in `adapter_manifest.json`
  when a searched config trained the adapter.
- Execution path: `llb finetune-hparams --model <m> --dataset <dir> --max-trials 8` and
  `make finetune-hparams MODEL=<m>`; unit tests -- dev-slice disjointness (no calibration or
  final id can enter a trial), deterministic fake-trainer study given the seed, budget abort
  plus resume, and manifest consumption by the trainer.
- Acceptance gates: `make ci` green with the fake trainer; the dev-slice test proves zero
  calibration/final leakage into any trial; an aborted study resumes without repeating
  finished trials; one real bounded search (at most 8 trials) on the CUDA host records a
  per-model best config and its dev-slice objective in current docs; a task 18 round trained
  with the manifest config reproduces the recorded configuration in its provenance.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide's tuning appendix.

### 22. local-distillation-lane (optional)

- Dependencies: follows task 18 (trainer seam and contamination guard; registry integration
  through task 20 when present) and soft-follows task 19 (the campaign report names the natural
  teacher -- the roster's best adapted model). Local-only, so no egress question arises. The
  heavy distillation run executes on the CUDA host.
- User-visible outcome: the roster's strongest local model teaches the smaller ones: the
  teacher answers tuning-split questions with retrieved context, its answers are quality-gated
  deterministically against the reference answers (only an answer scoring at or above the gate
  becomes a training target -- teacher misses are dropped, never invented into data), and the
  student is fine-tuned on the accepted set through the task 18 trainer. The report compares
  student-distilled against student-SFT-on-references over the same items, so distillation must
  demonstrate its value, not assume it.
- Scope boundary: in scope -- `src/llb/finetune/distill.py`: teacher generation through the
  existing backend seam over tuning-split items only; the deterministic quality gate reusing
  the existing correctness scorers; identity guards -- teacher != student, and the
  calibration-gated judge is never the teacher (extending the recorded planter != judge rule
  in `src/llb/prep/frontier.py`); distilled adapters flow through the same contamination guard
  and (when task 20 exists) registry as every other adapter; the paired
  distilled-vs-reference-SFT comparison in the report. Out of scope -- frontier or API
  teachers (egress; human task 2's lane is drafting-only), logit or soft-label distillation
  across tokenizers (text-level SFT only), training or improving the teacher itself, chain or
  agentic trace distillation (task 8 owns chain evaluation).
- Data and artifact paths: `$DATA_DIR/distill/<timestamp>/` holding `teacher_outputs.jsonl`,
  the accepted `dataset/`, the student `adapter/`, and `report.md`; the dataset manifest
  records the teacher id, gate threshold, and per-item gate scores.
- Execution path: `llb distill --teacher <m1> --student <m2> --backend <b> --gate <t>` and
  `make distill TEACHER=<m1> STUDENT=<m2> BACKEND=<b>`; unit tests with a fake teacher
  endpoint and the fake trainer -- gate exclusion of below-threshold answers, identity-guard
  refusals, tuning-split discipline, and the paired-comparison report math.
- Acceptance gates: `make ci` green with the fakes; a below-gate teacher answer can never
  reach the training set (unit-tested); teacher == student and judge == teacher both refuse
  with clear messages; one real CUDA run distills the campaign's best teacher into one smaller
  student and records the distilled-vs-reference-SFT delta with CIs honestly (no-gain is
  acceptable evidence); the distilled adapter passes the contamination guard and registers
  like any other.
- Documentation target: [extended workflows](current/extended-workflows.md); the
  self-improvement-loop guide.

### embedding-bakeoff-full-corpus

- Dependencies: the shipped `llb compare-embeddings` bake-off (see
  [RAG core](current/rag-core.md) Embedder Conventions And Bake-off). Heavy store builds run on the
  CUDA host (deterministic, no human judgment).
- Why this is forward work: the committed durable evidence ranks the four local candidates on the
  tiny `samples/goldsets/ip_regulation_uk` fixture (8 items / 10 chunks), where recall@10 SATURATES
  (e5-base, e5-large, and bge-m3 all hit 1.000) so the winner is decided only by an MRR + throughput
  tie-break, and the reported `chunks/s` is cold-load-dominated rather than steady-state. The
  recommendation "e5-base for the 16 GB host" is therefore under-discriminated.
- User-visible outcome: a bake-off report over a REAL full Ukrainian corpus (e.g. the quickstart PDF
  corpus index) at a larger `k`, where recall@k actually separates the candidates and embed
  throughput reflects steady state, yielding a confidently-ranked embedder recommendation the
  operator can trust before pinning `RunConfig.embedding_model`.
- Scope boundary: in scope -- run `make compare-embeddings` on a full-corpus goldset, record the
  ranked table + winner in [RAG core](current/rag-core.md) and
  [platform matrix](current/platform-vector-matrix.md), and note the per-candidate index size / VRAM
  fit on the 16 GB host. Out of scope -- new bake-off code (the command is shipped), an API row
  unless the corpus is explicitly open, changing the pinned drafting-side E5 seams.
- Data and artifact paths: `$DATA_DIR/compare-embeddings/<timestamp>/report.md` plus the per-model
  stores; the durable table lands in current docs.
- Execution path: `make compare-embeddings GOLDSET=<full-corpus goldset> RAG_K=20` on the CUDA host
  (outside quick CI); then `make build-index EMBEDDING_MODEL=<winner>`.
- Acceptance gates: the report ranks all four local candidates with a NON-saturated recall@k spread
  on the full corpus; current docs record the discriminated winner and its index size / device fit.
- Documentation target: [RAG core](current/rag-core.md) Embedder Conventions And Bake-off;
  [platform matrix](current/platform-vector-matrix.md).

### external-import-needle-parity (optional)

- Dependencies: the shipped grounded-JSONL import (see [data prep](current/data-prep.md)
  grounded-JSONL import) and the shipped `prepare-goldset-draft --retrieval-index-dir` needle-rank
  annotation. Agent-buildable with the committed `samples/external-drafts` fixture; no network.
- Why this is forward work: `import-external-draft` records each item's `question_type`/`difficulty`
  in `item_provenance.jsonl`, but -- unlike the local ontology lane -- it does NOT annotate imported
  items with `retrieval_rank` against a full-corpus index, so a reviewer verifying an externally
  imported bundle loses the confidence-ordering + retrieval-uniqueness signal local drafts carry.
- User-visible outcome: an optional `--retrieval-index-dir`/`--retrieval-k` on
  `import-external-draft` (and `--drop-nonretrievable-needles`) that annotates each imported item
  with its gold-span retrieval rank, so external and local drafts reach the verify gate with the
  same per-item signal.
- Scope boundary: in scope -- reuse the shipped needle-rank annotator
  (`src/llb/prep/ontology/needles.py`) over the imported items and record the rank in item
  provenance; a verify-worksheet column when present. Out of scope -- changing the `GoldItem`
  schema, building the index (the operator points at an existing one).
- Data and artifact paths: `retrieval_rank` added to `item_provenance.jsonl`; no bundle-shape
  change otherwise.
- Execution path: `make import-external-draft ARTIFACT= CORPUS= SIDECAR= RETRIEVAL_INDEX_DIR=`;
  unit tests over the committed fixture + a fake retriever.
- Acceptance gates: `make ci` green; imported items carry `retrieval_rank` when an index is given
  and the lane is an exact no-op when it is not; a non-retrievable item is dropped only under the
  explicit flag.
- Documentation target: [data prep](current/data-prep.md) grounded-JSONL import.

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
  [`docs/guides/data-prep/goldset-from-scratch.md`](../guides/data-prep/goldset-from-scratch.md).

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
  [`docs/guides/human-tooling/human-in-the-loop-evaluation.md`](../guides/human-tooling/human-in-the-loop-evaluation.md)
  and [`docs/guides/human-tooling/verification-tooling.md`](../guides/human-tooling/verification-tooling.md).

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
  [`docs/guides/learning-path/learning-path-security.md`](../guides/learning-path/learning-path-security.md).

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
