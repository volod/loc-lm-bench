# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

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
and the recommended build order within each section follows those lines.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md). For tasks that depend on scoring or judging, the calibrated
local-judge baseline and tuning/sweep behavior live in
[evaluation rigor](current/rigor-board-judge.md); the prompt-system package flow and other
extended workflows live in [extended workflows](current/extended-workflows.md).

Every task below carries an explicit `Agent status` line with one of four markers:

- **CLEAR** -- agent-buildable to `make ci` green with fixtures/fakes; no run evidence, no human
  gate.
- **RUN NEEDED** -- agent-buildable, but acceptance requires a heavy deterministic run; every dev
  box is a proper CUDA host, so the agent executes these runs itself on the current machine.
- **BLOCKED BY HUMAN** -- the acceptance gate consumes an artifact only a human step can produce.
- **HUMAN-GATED** -- the deliverable itself is human judgment or authorization; supporting code and
  unit tests are agent-buildable.

## Agent Implementation Tasks

Add new agent-buildable work here per [Adding Future Tasks](#adding-future-tasks).

### embedder-decision-on-a-resolvable-item-set

The embedder choice is undecidable on the item sets the repo currently has, and the paired lane
says so precisely: on the accepted converted-PDF ledger 36 of 40 questions are TIED between the
leader and the incumbent, so the 95% paired interval spans `[-0.050, +0.150]` and only a
consistent ~4-question gap could ever clear zero; on the committed fixture the baseline already
retrieves 0.980, leaving 5 questions of headroom for any candidate to win
([RAG core](current/rag-core.md#the-recommendation-re-read-with-paired-uncertainty)). Both sets are
at their ceiling, which is a property of the QUESTIONS, not of the encoders. Build an item set that
can decide it: predeclare a minimum detectable recall gain and the split size it needs, then
assemble a ledger enriched with questions the incumbent currently MISSES (mine the per-item vectors
in `report.json` for baseline zeros, plus domain-term and morphology-heavy questions the general E5
encoder is expected to fail), accept it through the verification gate, and re-run the bake-off on
it. Record whether any candidate then separates -- a recorded "still undecidable at n=N" is a valid
outcome and is what would justify closing the question.

- Agent status: BLOCKED BY HUMAN
- Dependencies: the paired bake-off lane, the verdict, and `report.json` are current behavior
  ([RAG core](current/rag-core.md#paired-uncertainty-and-the-adopt-or-retain-verdict)). Human step
  that gates completion: a reviewer accepts the enriched question set through
  `make verify-review` / `make verify-accept`, since an unaccepted ledger cannot settle a default.
- User-visible outcome: either a measured embedder swap an operator can adopt, or a recorded
  statement of how many questions a decision would need -- instead of a permanently open ranking.
- Scope boundary: in scope -- the power target, the enriched ledger, its acceptance, and one
  re-run per corpus with the verdict restated. Out of scope -- new candidates, fine-tuning (that
  is `ua-embedder-domain-finetune`), and widening the verdict bar beyond recall@k.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout.
- Execution path: `make compare-embeddings GOLDSET=<accepted-enriched-goldset> NOISE_FLOOR=1` on
  the CUDA host; no new CI coverage.
- Acceptance gates: the predeclared minimum detectable gain and split size are written down BEFORE
  retrieval; every candidate row reports its paired interval on the accepted ledger; the verdict is
  recorded as adopt, retain, or explicitly undecidable at the reached sample size.
- Documentation target: the embedder bake-off evidence in [RAG core](current/rag-core.md) and the
  recommendation line in [platform matrix](current/platform-vector-matrix.md#embedding-bake-off).

### embedder-first-hit-rank-adoption-bar (optional)

The bake-off adopts on recall@k alone, and one candidate is already separated on the OTHER metric:
on the accepted converted-PDF ledger `BAAI/bge-m3` beats the incumbent by +0.064 MRR
`[+0.008, +0.137]` while its recall delta spans zero -- it ranks the same evidence earlier without
finding more of it ([RAG core](current/rag-core.md#the-recommendation-re-read-with-paired-uncertainty)).
Whether that is worth adopting depends on a downstream fact the retrieval table cannot see: at a
small `top_k`, or under a cross-encoder reranker that only re-sorts what it is given, first-hit
rank is the binding constraint, while at k=10 with a generous context budget it is nearly free.
Measure it end to end -- score answer quality at the shipped `top_k` and at a small one, with and
without the reranker, on both embedders -- then either add a second, explicitly-scoped adoption bar
to `decide_verdict` or record that recall@k stays the sole bar and why.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `decide_verdict` in `src/llb/rag/embedding_bakeoff_uncertainty.py`, the
  reranking lane ([RAG core](current/rag-core.md)), and `compare-answer-quality` for the end-to-end
  read.
- User-visible outcome: the operator learns whether an encoder that only ranks better is worth its
  cost on their retrieval configuration, instead of that gain being silently discarded.
- Scope boundary: in scope -- the top_k / reranker sweep on both embedders, the verdict-bar
  decision, and its implementation if the measurement supports one. Out of scope -- new candidates
  and changing the recall@k bar itself.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout plus a
  `$DATA_DIR/run-eval/` bundle per scored configuration.
- Execution path: `make compare-embeddings` for the retrieval side, then `make run-eval` /
  `make compare-answer-quality` per (embedder, top_k, reranker) cell on the CUDA host; CI covers a
  second adoption bar over fake vectors if one is added.
- Acceptance gates: `make ci` green; the report states the answer-side delta per cell with paired
  intervals and an explicit keep-or-extend decision on the adoption bar.
- Documentation target: the embedder bake-off evidence in [RAG core](current/rag-core.md).

### vector-store-bake-off-paired-uncertainty (optional)

`compare-vector-stores` still ranks backends on a point estimate plus the measurement floor, the
one reading the embedder bake-off already carries: its `best (recall@k)` line is label order when
the backends tie ([platform matrix](current/platform-vector-matrix.md#embedding-bake-off)), and
nothing states how large a backend difference the item set could even resolve. Give it the same
paired lane -- per-item metric vectors against a baseline backend, shared resample index sets, the
delta interval and win/loss/tie ledger per row, and an adopt-or-retain verdict -- so a backend swap
is decided the same way an embedder swap now is.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `src/llb/rag/embedding_bakeoff_uncertainty.py` wholesale (it takes
  metric vectors, not embedder rows) and the store seam in `src/llb/cli/rag/compare_stores.py`.
- User-visible outcome: the operator learns whether a vector-backend difference is real or is the
  order the labels happened to sort in.
- Scope boundary: in scope -- the paired columns, the verdict, and a re-run on both scored
  corpora. Out of scope -- new backends and any change to the retrieval metrics.
- Data and artifact paths: the existing `$DATA_DIR/compare-vector-stores/<run>/` layout.
- Execution path: `make compare-vector-stores NOISE_FLOOR=1` on the CUDA host; CI covers the
  interval columns over fake stores.
- Acceptance gates: `make ci` green; every backend row carries a paired delta interval against the
  baseline backend and the report states adopt or retain.
- Documentation target: the vector-store section of
  [platform matrix](current/platform-vector-matrix.md).

### graph-lane-score-ties (optional)

The graph lane's own recall is decided by tie order for two thirds of the questions it is scored
on: its link-relevance scores saturate into long exact-tie blocks, the rank-k cut falls inside one
for 68 of 95 items (33 of 35 on the multi-hop slice), and that is what sets the fusion sweep's
whole measurement floor at `+/-0.021` recall@10 overall and `+/-0.043` on the focus slice
([GraphRAG](current/graphrag-backend.md#the-sweep-re-read-against-its-measurement-floor)). The
ranking is reproducible -- `_rank_dedup` breaks ties on `(doc_id, char_start, char_end)` -- but a
document id is not a relevance signal, so every graph-only row is quoted to three decimals it has
not earned. Find out whether the tie blocks are reducible: measure how much of each tie block is
one relevance value versus rounding, and if a finer signal exists (edge weight, hop distance,
mention count, community rank as a continuous term rather than a bucket) score the lane with it and
re-measure the floor.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the scoring in `src/llb/graph/linking.py` / `src/llb/graph/retrieval.py`
  and the floor lane (`compare-graph-fusion --noise-floor`) to read the result.
- User-visible outcome: either graph-only rows whose recall reflects relevance instead of document
  order, or a recorded finding that the lane's evidence is inherently tied and its rows must be
  read at the floor's precision.
- Scope boundary: in scope -- the tie-block census, one finer relevance signal if the census
  supports one, and the before/after floor. Out of scope -- changing the graph schema, the fusion
  mechanics, and the RRF rank basis (fused rows already report a zero band).
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: `make compare-graph-fusion NOISE_FLOOR=1 SPLIT=` before and after, on the CUDA
  host; CI covers the scoring change over the committed graph fixtures.
- Acceptance gates: `make ci` green; the report states the fragile count and floor per graph row
  before and after, and whether any recorded graph-row verdict changes.
- Documentation target: the retrieval-strategies section of
  [GraphRAG](current/graphrag-backend.md) and the floor table in [RAG core](current/rag-core.md).

### chunker-bake-off-under-the-size-cap (optional)

Re-run the seven-strategy chunker bake-off now that `size` is a hard cap on every strategy. The
recorded winner (`sentence`, +0.022 recall@10 over `recursive`) was scored on stores that still
contained oversized units, and the unit-packing strategies are exactly the ones the cap changes:
their chunk counts rise and their long table/heading spans are now split
([RAG core](current/rag-core.md#chunking-strategies)). The ranking may hold, invert, or collapse
into a tie, and the current recommendation cannot say which. Score the same accepted goldset at
the same k and record whether the `sentence` recommendation survives. A second reason to re-run:
those stores also predate exact-duplicate chunk collapse, which changes the chunk counts per
strategy and, on a furniture-heavy corpus, the ranking itself -- it moved the goods rows and drove
that corpus's floor to zero ([RAG core](current/rag-core.md#duplicate-chunk-collapse)).

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `make compare-retrieval` unchanged, with `NOISE_FLOOR=1` so a changed
  row can be read against the corpus's own floor
  ([RAG core](current/rag-core.md#measurement-floor---noise-floor)).
- User-visible outcome: the per-corpus chunker recommendation rests on stores that respect the
  `size` the operator asked for.
- Scope boundary: in scope -- the re-run, the updated table, and an explicit keep-or-change
  verdict on the `sentence` recommendation. Out of scope -- new strategies and tuning `size`.
- Data and artifact paths: the existing per-strategy stores under `$DATA_DIR/llb/rag/<strategy>/`.
- Execution path: `make compare-retrieval CHUNK_STRATEGIES=sentence,recursive,page,heading,late,
  markdown,semantic GOLDSET=<quickstart accepted goldset> NOISE_FLOOR=1` on the CUDA host; no new
  CI coverage.
- Acceptance gates: `make ci` green; the report covers all seven strategies at the recorded k and
  states whether the recorded winner still wins by more than the measurement floor.
- Documentation target: the chunking-strategies evidence in [RAG core](current/rag-core.md).

### multihop-both-hops-ceiling

Every fused row measured so far -- every weight, both depths, both identity policies -- retrieves
BOTH hops for at most 3 of 35 two-hop questions (`all-spans@10` <= 0.086), while single-hop recall
moves freely between 0.686 and 0.800
([GraphRAG](current/graphrag-backend.md#span-identity-evidence)). That ceiling is invariant to
every ranking knob the lane exposes, which means it is probably not a ranking problem: either the
second hop's chunk is not retrievable for the question's own wording (a query problem, addressable
by decomposition), or it is not reachable at k=10 at all (a budget problem). Diagnose which:
measure `all-spans@k` as a function of k (say 10 / 25 / 50) on the same items, and measure the
per-hop retrievability of each labeled span when queried on its own. Record which of the two
explanations the corpus supports, because they lead to opposite fixes.

A third lead is already measured and worth folding into the k sweep: shrinking the CHUNK moves the
ceiling where no ranking knob could, the vector baseline's multi-hop `all-spans@10` running
0.057 -> 0.086 -> 0.114 as the chunking goes from `recursive@800/120` to `sentence@200` to
`recursive@200/30`
([GraphRAG](current/graphrag-backend.md#does-the-pin-survive-a-smaller-chunk-size)) -- which points
at the budget explanation, since k=10 buys more distinct spans when a span is smaller. Overall
recall falls at the same time, so treat it as a diagnostic, not a recommendation.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `all_spans_at_k` / `span_coverage_at_k` in `src/llb/rag/retrieval.py`,
  the sweep lane, and the existing query-decomposition step in
  [RAG core](current/rag-core.md#query-side-processing).
- User-visible outcome: the operator learns whether multi-hop evidence coverage is limited by the
  retrieval budget or by the query, instead of tuning ranking knobs that provably cannot move it.
- Scope boundary: in scope -- the k sweep, the per-hop probe, and a written diagnosis. Out of scope
  -- building a new decomposition strategy before the diagnosis names it, and any ranking-policy
  change.
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: `make compare-graph-fusion RAG_K=<k>` per budget plus a per-hop retrieval probe
  on the CUDA host; CI covers the per-hop probe over fake lane stores.
- Acceptance gates: `make ci` green; the report carries `all-spans@k` per budget and the per-hop
  hit rate, and states which explanation the measurement supports.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence).

### answer-side-span-coverage-metric

The retrieval side distinguishes "carried one hop" from "carried both" (`span_coverage_at_k` /
`all_spans_at_k`, [RAG core](current/rag-core.md#retrieval-metrics)); the ANSWER side has no such
distinction. `objective_score` is reference-answer token F1, so a two-hop answer that states one
fact fluently and omits the other scores roughly half -- the same value a vague answer touching
both facts gets. Every multi-hop answer-quality verdict therefore rests on a metric that cannot
say whether the model used both hops, which is precisely the question the lane exists to ask
([GraphRAG](current/graphrag-backend.md#answer-quality-evidence)). Build the answer-side
counterpart: per gold span, decide whether the ANSWER carries that span's content (lemma and
numeral overlap against the span text, thresholded and Ukrainian-aware, reusing the correctness
tokenizer), then report `answer_span_coverage` and `answer_all_spans` beside the objective and let
the multi-hop verdict read them.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the scoring tokenizer in `src/llb/scoring/correctness.py`, the
  multi-span retrieval metrics in `src/llb/rag/retrieval.py`, and the per-slice comparison in
  `src/llb/eval/answer_quality/`.
- User-visible outcome: the operator learns whether a multi-hop answer actually contains BOTH
  facts, instead of inferring it from a token-overlap score that a half-answer can earn.
- Scope boundary: in scope -- the answer-side coverage metric, its per-case columns, and wiring it
  into the answer-quality slices and verdict as an additive signal. Out of scope -- replacing
  `objective_score` as the leaderboard ranking metric, judge re-calibration, and any change to the
  retrieval metrics.
- Data and artifact paths: additive per-case columns in the standard `$DATA_DIR/run-eval/` bundles;
  comparison under the existing `$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.
- Execution path: `make compare-answer-quality` on the drafted multi-hop bundle; CI covers the
  metric on committed two-span fixtures (both facts, one fact, neither, paraphrased).
- Acceptance gates: `make ci` green; on single-span items the answer-side coverage agrees with the
  existing exact/contains signals; the multi-hop re-run reports the new columns with paired
  intervals and states whether they change the recorded verdict.
- Documentation target: [RAG core](current/rag-core.md#scoring) and the answer-quality evidence
  subsection of [GraphRAG](current/graphrag-backend.md#answer-quality-evidence).

### fusion-answer-quality-second-model (optional)

Repeat the end-to-end answer-quality comparison on a second roster model. Whether extra retrieved
evidence converts into a better answer is a property of the MODEL, not only of the retrieval lane:
a measured coverage gain that one model ignores may be exactly what a stronger (or more
instruction-following) model needs, and a single-model result cannot separate "fusion does not
help answers" from "this model does not use the extra hop". The lane, its verdict vocabulary, and
the drafted-grounding rules are current behavior
([GraphRAG](current/graphrag-backend.md#answer-quality-evidence)).

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `compare-answer-quality` as-is with a different `MODEL`; the matched
  stores and drafted bundle already exist.
- User-visible outcome: the operator learns whether the retrieval-only finding is a property of
  the corpus and the fusion lane, or of the one model that was scored.
- Scope boundary: in scope -- one more model, the same lanes/splits/seed, and a two-model
  comparison of the per-slice deltas. Out of scope -- any ranking-policy change, model selection,
  and re-tuning the graph weight per model.
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.
- Execution path: `make compare-answer-quality MODEL=<second-roster-model> SPLIT=final,tuning,
  calibration ANSWER_QUALITY_LANES=vector,<best-exact-row>,<best-overlap-row> INCLUDE_DRAFTED=1`
  -- the same three lanes the first model was scored on, so the two models are compared row for
  row; no new CI coverage.
- Acceptance gates: `make ci` green; both models score the identical item set at the same seed;
  the report states whether the two models agree on the verdict per lane, including whether the
  factoid cost of the overlap row reproduces.
- Documentation target: the answer-quality evidence subsection of
  [GraphRAG](current/graphrag-backend.md#answer-quality-evidence).

### retrieved-document-long-context-lane

The measured long-context lane is oracle-grounded -- it reads the item's own gold `doc_id`s, so it
sizes a CEILING and cannot be adopted
([product decisions](current/scope-boundaries.md#context-ablation-lanes-stay-diagnostic)). Add the
shippable sibling: a `retrieved_document` context strategy that takes the top-ranked RETRIEVED
chunk's document (no gold label), lays that whole document into the prompt under the same budget
check and `context_overflow` skip rule, and reports beside the existing lanes. The measured
oracle-versus-rag gap (+0.142 / +0.080 objective on two roster models) then splits into the part
an operator can actually capture by widening the unit of retrieval from a chunk to its document,
and the part that was pure oracle advantage.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the context-source seam, `fits_context_chars`, and the comparison in
  [RAG core](current/rag-core.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation).
- User-visible outcome: the operator learns whether "retrieve the chunk, send the document" is a
  real configuration worth shipping, or whether the long-context gain was the gold label all along.
- Scope boundary: in scope -- the strategy, its document-selection rule (top-1 versus top-k
  distinct documents), the budget/skip path, and a four-lane comparison. Out of scope -- changing
  the chunker, the ranking policy, and context-window extension tricks.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies CONTEXT_LANES=closed_book,rag,retrieved_document,long_context`;
  CI covers document selection and the skip rule over fake lane stores.
- Acceptance gates: `make ci` green; the three existing lanes reproduce their current rows exactly;
  a heavy run on both scored roster models reports the four-lane table with paired intervals and an
  explicit adopt-or-reject verdict for the new lane.
- Documentation target: the context-ablation section of [RAG core](current/rag-core.md) and the
  diagnostic-lane boundary in [product decisions](current/scope-boundaries.md).

### closed-book-decoding-stability (optional)

A closed-book score is a noisier measurement than a grounded one: two identical invocations of the
same lane on the same 82 items differed on 11 answers and moved the lane mean 0.160 -> 0.153, while
the `rag` and `long_context` lanes were byte-identical
([RAG core](current/rag-core.md#context-ablation-evidence)). An ungrounded prompt leaves a much
flatter next-token distribution, so kernel-level nondeterminism flips tokens. The drift stayed well
inside the uplift interval and changed no verdict, but a contamination rate quoted to one decimal
place is currently over-stated precision. Measure it: repeat the closed-book lane N times at a
fixed seed, report the between-repeat spread of the lane mean and of the contamination rate, and
either quote the ablation's closed-book numbers with that spread or make the lane reproducible
(pinned sampler / seeded backend options) if the backend allows it.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `compare-context-strategies` with a repeated `closed_book` lane and the
  existing paired-bootstrap reporting.
- User-visible outcome: the operator knows how much of a closed-book delta is measurement noise
  before reading it as parametric knowledge.
- Scope boundary: in scope -- repeat runs, the spread statistic, and whichever of the two remedies
  the measurement supports. Out of scope -- changing the objective metric and swapping backends.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: N repeats of the closed-book lane on the committed UA fixture on the CUDA host;
  CI covers the spread statistic over committed fixture rows.
- Acceptance gates: `make ci` green; the report states the between-repeat spread of the lane mean
  and the contamination rate, and the ablation docs quote closed-book numbers accordingly.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

### context-ablation-question-type-slices (optional)

The context ablation slices by question type, but the committed UA fixture ships no
`needle_items.jsonl` sidecar, so every heavy run so far reported ONE pooled number per lane
([RAG core](current/rag-core.md#context-ablation-evidence)). Pooling hides the question the lane is
most useful for: retrieval almost certainly pays for itself unevenly -- a factoid whose answer is
one span versus a comparative or numeric question whose evidence is scattered. Run the ablation on
a gold set that HAS the sidecar (the quickstart-PDF accepted goldset, or a drafted multi-hop
bundle) so the uplift and the long-context delta are reported per slice, and record which slices
retrieval fails to pay for.

- Agent status: RUN NEEDED
- Dependencies: none. The slicing is already wired; this needs a labeled item set and the run.
  Question-type labels come from the needle sidecar
  ([data prep](current/data-prep.md)).
- User-visible outcome: the operator learns WHICH questions retrieval pays for on their corpus,
  instead of one pooled average over a mixed set.
- Scope boundary: in scope -- the run, the per-slice reading, and a verdict per slice. Out of
  scope -- new metrics, new lanes, and any ranking-policy change.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies GOLDSET=<sidecar-bearing goldset> CORPUS=<dir>`
  on the CUDA host; no new CI coverage.
- Acceptance gates: `make ci` green; the report carries a non-empty slice table with paired
  intervals per slice and states which slices the uplift interval fails to clear zero on.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

### table-aware-chunking

Add a `table` strategy to `src/llb/rag/chunking/`: chunk boundaries never split a markdown table
row, a table that fits `size` stays one chunk carrying its nearest heading breadcrumb, and an
oversized table splits between row blocks with the header row's offsets recorded as additive
`metadata.table_header_span` -- chunk text stays a verbatim corpus slice with exact offsets.
Non-table text routes through the `recursive` splitter. Extend `compare-retrieval` with a
per-question-type breakdown (joined from `item_provenance.jsonl` when the sidecar exists) so the
numeric and comparative slices -- where tables carry the answers in converted Ukrainian PDF
corpora -- are scored beside the aggregate.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the chunking dispatch seam (`chunk_spans`), the markdown table output
  of the PDF conversion lane ([data prep](current/data-prep.md)), and the question-type taxonomy
  in the draft sidecars.
- User-visible outcome: numeric and comparative questions whose evidence lives in tables stop
  losing recall to mid-table chunk cuts, and the per-type breakdown shows exactly which question
  slice a chunking change helps or hurts.
- Scope boundary: in scope -- the strategy, tuner registration behind `--extended-chunkers`, and
  the per-type `compare-retrieval` breakdown. Out of scope -- cell-level table QA, corpus text
  rewriting, and HTML tables.
- Data and artifact paths: per-strategy stores under the existing comparison layout
  `$DATA_DIR/llb/rag/<strategy>/`; no new roots.
- Execution path: `make build-index CHUNK_STRATEGY=table`; `make compare-retrieval
  CHUNK_STRATEGIES=table,recursive,sentence GOLDSET=<gs>`; CI covers offset round-trips and
  row-boundary alignment on a committed markdown-table fixture.
- Acceptance gates: `make ci` green; every chunk stays offset-exact under `validate-goldset`;
  a heavy comparison over the quickstart accepted goldset reports aggregate plus numeric-slice
  recall@10 / MRR against `recursive` and `sentence`.
- Documentation target: [RAG core](current/rag-core.md) chunking strategies and the
  [data prep](current/data-prep.md) chunking list.

### ua-embedder-domain-finetune

Fine-tune the pinned multilingual E5 embedder on the operator's corpus: export contrastive
(question, gold-chunk) pairs from tuning-split gold items only (positives are chunks overlapping
the item's gold spans; hard negatives come from the BM25 lexical index), train with a
sentence-transformers contrastive objective behind lazy imports, and emit a tuned-embedder
directory whose manifest records the base model, dataset digest, item ids, and split counts. A
split guard refuses pairs from calibration or final ids (the `assert_tuning_only` discipline from
the LoRA hparam search). `compare-embeddings` accepts the tuned directory as a candidate so
uplift is measured by the standard source-span metric on the held-out final split, and the
store/query embedder fingerprint guard keeps a tuned-embedder store from being queried by any
other encoder.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the embedder conventions and bake-off in
  [RAG core](current/rag-core.md#embedder-conventions-and-bake-off), the lexical index for hard
  negatives, and the split-guard pattern in `src/llb/finetune/hparam_search/`.
- User-visible outcome: a corpus-adapted Ukrainian retriever the operator can adopt with measured
  final-split evidence, closing the recall gap on domain terms the general E5 encoder misses.
- Scope boundary: in scope -- pair export, the trainer, the manifest, bake-off integration, and
  the split guard. Out of scope -- cross-encoder (reranker) fine-tuning, generation-model
  fine-tuning (owned by the existing finetune lane), and hosted training.
- Data and artifact paths: pair datasets and tuned models under
  `$DATA_DIR/finetune-embedder/<model-slug>/<timestamp>/`; evaluation through the existing
  `$DATA_DIR/compare-embeddings/` layout.
- Execution path: `make finetune-embedder GOLDSET=<gs> CORPUS=<dir>` then
  `make compare-embeddings` with the tuned directory added as a candidate; CI uses a fake trainer
  plus the hashed-BoW embedder pattern from the curation tests, no GPU.
- Acceptance gates: `make ci` green; the guard refuses a pair set naming calibration/final ids;
  a heavy CUDA run trains on the quickstart tuning split and reports tuned-vs-base recall@10 /
  MRR on the held-out final split, where the adopt-or-keep-base verdict is the bake-off's own
  paired one -- the tuned row must clear zero against the base encoder, not merely outrank it
  ([RAG core](current/rag-core.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
- Documentation target: [RAG core](current/rag-core.md) embedder section and
  [extended workflows](current/extended-workflows.md) for the trainer lane.

### ua-model-roster-long-run (optional)

Confirm the refreshed-roster ranking at research scale: run at least 10 multi-objective trials
per viable addition, use a tuning screen of at least 8 cases, score the full held-out final split,
and add the public Ukrainian screen tracks before making a default-model adoption decision. Report
bootstrap uncertainty and quality/latency Pareto tradeoffs so a small-sample rank reversal cannot
silently change the recommended model.

- Agent status: RUN NEEDED
- Dependencies: use the roster/runtime behavior in
  [platform matrix](current/platform-vector-matrix.md#ukrainian-model-roster-refresh) and the
  bounded baseline in [evaluation rigor](current/rigor-board-judge.md#joint-model--config-search).
- User-visible outcome: a stable refreshed-roster recommendation with uncertainty, public-task
  coverage, and an explicit adopt-or-retain verdict.
- Scope boundary: in scope -- larger private joint search, public-screen lanes, uncertainty, and
  the adoption verdict. Out of scope -- model fine-tuning and hosted/API-only candidates.
- Data and artifact paths: `$DATA_DIR/joint-search/<run>/`, `$DATA_DIR/screen/`, and the matching
  current-doc evidence section.
- Execution path: run `make joint-search` on a CUDA host with the refreshed candidates and full
  final split, then run the public screen for both finalists.
- Acceptance gates: `make ci` green; at least 10 trials per finalist; no final-split leakage into
  tuning; confidence-aware ranking; explicit quality-versus-latency recommendation.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) host evidence.

### normalize-casefold-dense-lane-cost (optional)

Normalization casefolds the whole query, but the dense encoder is case-sensitive: on the 82-item
final split the `normalize`-only lane retrieves WORSE than no mitigation at all under keyboard
noise (0.9268 -> 0.9024 recall@10), even though casefolding is supposed to be the safe half of
the lane ([evaluation rigor](current/rigor-board-judge.md#ukrainian-query-robustness-benchmark)).
The split noise classes sharpen the diagnosis: on the `apostrophe_variant` class the `normalize`
lane loses an item (0.9756 -> 0.9634 recall@10) even though the affected-items table shows all 6
perturbed questions retrieved perfectly in every lane -- the loss is on questions the noise class
never touched, so it is the mitigation step acting on an otherwise clean query, not a failed
repair. Casefolding is a lexical-side convention that the dense side never asked for. Measure
whether the processed query should stay cased on the dense lane while the lexical lane keeps the
folded text -- the `retrieve_queries` seam already carries separate dense and lexical text.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the split dense/lexical query seam in
  `src/llb/rag/query_prep/retrieval.py` and `RagStore.retrieve_queries`.
- User-visible outcome: the operator stops paying dense recall for a normalization step whose
  only job was to make matching safer.
- Scope boundary: in scope -- routing case-preserved text to the dense lane, the A/B, and the
  adopt-or-reject verdict. Out of scope -- changing the embedder, the lexical normalization, and
  the transliteration table.
- Data and artifact paths: `$DATA_DIR/query-robustness/<run>/`.
- Execution path: `make bench-query-robustness` on the CUDA host with and without the change; CI
  covers the dense/lexical text split over a fake store.
- Acceptance gates: `make ci` green; the report shows the `normalize` lane no longer retrieving
  below the `off` lane on any noise class, or records that casefolding is not the cause.
- Documentation target: [RAG core](current/rag-core.md) query-side processing and the robustness
  evidence in [evaluation rigor](current/rigor-board-judge.md).

### restoration-constraint-threshold-sweep (optional)

The restoration constraints ship with three unswept design constants: the surface-compatibility
budget (exact, `SURFACE_MAX_DISTANCE = 0`), the short-token cutoff that locks length and refuses
ties (`AMBIGUOUS_TOKEN_MAX_CHARS = 4`), and the ranking order that puts morphology ahead of local
context ([RAG core](current/rag-core.md#query-side-processing)). Each was chosen to be
conservative, and nothing measures what the conservatism costs: a budget of 1 would admit a token
that was BOTH transliterated and mistyped, and a cutoff of 3 or 5 moves how many short words stay
untouched. Sweep them on a corpus where the typo lane is not saturated, report retrieval and the
edit-precision audit per setting, and pin each value with evidence or expose it.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `select_restoration` in `src/llb/rag/query_prep/restore.py` and the
  robustness lanes.
- User-visible outcome: the operator knows whether the safe defaults are costing recoverable
  recall, instead of trusting three hand-picked constants.
- Scope boundary: in scope -- the sweep, the per-setting edit audit, and a pin-or-expose verdict
  per constant. Out of scope -- new constraint signals and a learned ranker.
- Data and artifact paths: `$DATA_DIR/query-robustness/<run>/`.
- Execution path: `make bench-query-robustness` per setting on the CUDA host; CI covers each
  setting's selection decisions over committed candidate fixtures.
- Acceptance gates: `make ci` green; the report states recall and the share of corrections a human
  reading of the audit calls wrong, per setting, with an explicit verdict per constant.
- Documentation target: [RAG core](current/rag-core.md) query-side processing.

### fusion-routing-calibration-power (optional)

Increase the sidecar-free routing calibration's statistical power before reconsidering its
production defaults. The first held-out measurement cannot separate its positive retrieval deltas
from zero; see the compact result and frozen-policy diagnostics in
[GraphRAG](current/graphrag-backend.md#sidecar-free-heuristic-calibration). Assemble a larger,
independent multi-span tuning/final ledger, declare its minimum detectable gain and split sizes
before retrieval, then repeat the frozen-policy workflow without widening the threshold grid.

- Agent status: BLOCKED BY HUMAN
- Dependencies: `multihop-ledger-human-acceptance` must provide a non-empty accepted multi-span
  ledger. Human step that gates completion: accept enough additional genuinely multi-span
  questions to meet the predeclared split sizes.
- User-visible outcome: operators can distinguish a useful sidecar-free route from a sparse-win
  artifact before changing the fallback defaults.
- Scope boundary: in scope -- a prospective power target, disjoint tuning/final splits, and one
  repeat of the existing deterministic calibration. Out of scope -- widening the threshold grid,
  a learned router, and selecting on final.
- Data and artifact paths: a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` calibration over
  the accepted ledger.
- Execution path: run `make calibrate-fusion-routing` with the predeclared splits, then run the
  masked `make compare-graph-fusion` reproduction for the frozen policy on each split.
- Acceptance gates: route precision/recall and paired retrieval intervals meet the predeclared
  power target; a threshold changes only if the tuning gain clears zero without single-span
  regression and the untouched final split confirms the same direction.
- Documentation target: the sidecar-free calibration subsections of
  [RAG core](current/rag-core.md) and [GraphRAG](current/graphrag-backend.md).

### conflict-null-model-research

**Research task** -- the answer is not known in advance, and a negative result is a valid outcome
that must be recorded rather than worked around.

Find a defensible independent null for corpus-conflict detection, so the semantic tier can report
a real false-positive rate instead of a rank cutoff. The current calibration measures the
similarity distribution of the corpus's own comparable cross-document pairs, which contains
whatever genuine duplicates the corpus has; with the pair space enumerated exactly the null and
the observed population are the same set, empirical FDR is identically 1.000 at every threshold,
and a budget of `N` returns exactly `N` pairs by construction (measured; see
[data prep](current/data-prep.md#known-limitation-there-is-no-independent-null)). Every downstream
question an operator asks -- "is this pair worth reading?", "did tightening the threshold remove
noise or evidence?", "is this corpus dirtier than that one?" -- currently has no statistical
answer.

Candidate approaches to evaluate, cheapest first; none is known to work:

- **Cross-corpus null.** Score chunks of the target corpus against chunks of an unrelated Ukrainian
  corpus. Pairs across corpus boundaries are unrelated by construction. Risk: a domain/register
  shift makes the null too easy, understating the threshold.
- **Within-document permutation.** Destroy the semantic relationship while preserving the corpus's
  marginal geometry -- shuffle tokens or sentences within a chunk before embedding. Risk: sentence
  encoders are partly bag-of-words, so a shuffled chunk may stay close to its original and the null
  lands too high.
- **Held-out-document null.** Bootstrap over document pairs, using the fact that most DOCUMENT
  pairs share no content, to estimate a per-document-pair rather than per-chunk-pair null. Risk:
  document pairs are few, so the tail is unresolvable on a small corpus -- the same saturation
  problem already measured for chunk-pair sampling.
- **Labelled calibration set.** Use the committed `samples/corpora/conflicts_uk_v1/` planted
  relations as ground truth to fit a threshold with a real measured precision/recall curve, then
  test whether that transfers to the quickstart corpora. Risk: seven planted pairs is a very small
  fit set, and the fixture uses a hashed-BoW fake embedder in CI.

- Agent status: RUN NEEDED
- Dependencies: the calibrated threshold and the enumerated distribution are current behavior
  ([data prep](current/data-prep.md#corpus-calibrated-cosine-threshold---max-candidate-pairs)).
  Reuse `estimate_null_distribution`, `VectorSet.cross_group_similarities`, and the planted-relation
  fixture. The comparable set excludes structurally repeated metadata blocks; use the measured
  post-filter population in [data prep](current/data-prep.md#what-the-semantic-tier-excludes-and-why).
- User-visible outcome: either a null the audit can quote a real false-positive rate against, or a
  recorded finding that cosine over sentence-encoder chunk vectors cannot support one -- which
  would justify moving threshold selection to the claim tier's measured precision instead.
- Scope boundary: in scope -- constructing and comparing candidate nulls, measuring each against
  the planted fixture and both quickstart corpora, and a written verdict per approach. Out of
  scope -- changing the relation vocabulary or the tier order, and shipping any new default before
  a null demonstrably beats the rank cutoff.
- Data and artifact paths: comparison under `$DATA_DIR/corpus-conflicts/null-research/<run>/`;
  no new committed fixtures unless an approach earns one.
- Execution path: a research harness invoked per null model over both quickstart stores plus the
  fixture; CI covers each null constructor deterministically over committed vectors, with the
  heavy corpus comparison run on the CUDA host.
- Acceptance gates: each candidate null is measured on the planted fixture, where the true relation
  labels are known, and reports precision/recall at its resolved threshold; an approach is adopted
  only if it beats the current rank cutoff on the fixture AND its resolved threshold recovers the
  claim-bearing HR swept baseline without flooding goods. If none does, the negative result is
  [product decisions](current/scope-boundaries.md) and the rank-cutoff framing stays.
- Documentation target: the corpus-hygiene known-limitation section of
  [data prep](current/data-prep.md), and [product decisions](current/scope-boundaries.md) for the
  adopt-or-reject verdict.

## Human-Assisted Tasks

Add new human-gated work here per [Adding Future Tasks](#adding-future-tasks) when acceptance
requires human judgment or authorization.

### embedding-clustered chunk merging (optional)

The measured near-duplicate residue is real but not text-reachable: on the goods corpus 20.7% of
the exact-collapsed chunks have a neighbour at cosine >= 0.99, and the `normalized` collapse tier
merges 26 of those 13105 pairs
([RAG core](current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers)). Only an
embedding-side merge can reach the rest, which the collapse lane deliberately does not do because a
false merge silently deletes a distinct passage from the index. Decide it with a measured
false-merge rate instead of by assumption: cluster the survivors by cosine at several thresholds,
sample the merges for human reading (the residue report's pair sampler already renders them), and
score retrieval per threshold against the corpus's own measurement floor. Adopt only if a threshold
lowers the fragile count without a recall regression AND its sampled false-merge rate is acceptable
on a corpus whose facts differ by one number.

- Agent status: HUMAN-GATED
- Dependencies: none. Reuse `measure_duplicate_residue` in `src/llb/rag/duplicate_residue.py` for
  the clustering and the sampler, and `collapse_duplicate_chunks` for the merge itself. Human step
  that gates completion: a reviewer reads the sampled merges at the candidate threshold and calls
  the false-merge rate acceptable or not.
- User-visible outcome: either an embedding-side merge tier with a measured false-merge rate, or a
  recorded finding that the residue must be left in the index.
- Scope boundary: in scope -- the clustering, the sampled review, the per-threshold retrieval run,
  and the adopt-or-reject verdict. Out of scope -- learned merge policies and corpus rewriting.
- Data and artifact paths: `$DATA_DIR/retrieval-noise-floor/<run>/`.
- Execution path: `make measure-duplicate-residue` per threshold, then `make compare-retrieval
  CHUNK_STRATEGIES=sentence,recursive NOISE_FLOOR=1` per candidate on the CUDA host.
- Acceptance gates: `make ci` green; the report carries the per-threshold recall against the floor,
  the fragile count, and the human's false-merge reading.
- Documentation target:
  [RAG core](current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers).

### goods-fusion-weight-accepted-ledger

Settle the goods-corpus fusion-weight verdict on an item set someone accepted. The recorded
verdict ("the BM25 side costs recall at w=0.5, pin `FUSION_WEIGHT=0.7`") was measured on a
verified 44-item quickstart-PDF accepted goldset that is no longer on disk, and the lexical-row
re-read could not reproduce it: on the SAME corpus at the SAME chunking, the 95-item drafted
goldset inverts it -- fusion ADDS recall at w=0.5 (+0.021, +0.053 with lemmas, against a
+/-0.000 floor) and w=0.7 is the worst of the three weights for the best row
([RAG core](current/rag-core.md#lexical-row-re-read-of-the-fusion-weight-verdict)). The pin is
already withdrawn; what remains is deciding whether the recorded verdict was an artifact of its
item set or of the drafting, which only an accepted ledger over that corpus can answer.

- Agent status: HUMAN-GATED
- Dependencies: none in code -- `make compare-retrieval HYBRID=1 NOISE_FLOOR=1` and the
  verification gate are both current behavior. Human step that gates completion: a reviewer
  accepts (or rejects) enough of the drafted goods questions through
  `make verify-review` / `make verify-accept` to produce an accepted goods ledger.
- User-visible outcome: an operator on a converted-PDF Ukrainian corpus gets a fusion-weight
  recommendation backed by human-reviewed questions, instead of two drafted-versus-vanished item
  sets that disagree.
- Scope boundary: in scope -- worksheet review, the accepted ledger, one re-run per recorded
  weight with the `lexical` row, and a keep-or-change verdict on the recorded table. Out of scope
  -- widening the weight grid, new fusion mechanics, and changing the shipped defaults before the
  accepted run supports it.
- Data and artifact paths: the drafted bundle under
  `$DATA_DIR/graph-vector-fusion-multihop/goods-draft/` plus a new
  `$DATA_DIR/lexical-row-reread/goods-accepted-w<weight>/` per weight.
- Execution path: `make verify-review` then `make verify-accept` over the goods draft, then
  `make compare-retrieval CONFIG=<cfg> GOLDSET=<accepted>/goldset.jsonl SPLIT= HYBRID=1
  NOISE_FLOOR=1` at `FUSION_WEIGHT=0.5,0.6,0.7`.
- Acceptance gates: every worksheet row has a decision; the three weights are scored on the
  identical accepted item set with the `lexical` row and the corpus's floor; the recorded table is
  restated as reproduced, corrected, or retired.
- Documentation target: the hybrid-retrieval evidence section of
  [RAG core](current/rag-core.md#hybrid-retrieval-dense--bm25--rrf).

### multihop-ledger-human-acceptance

Accept (or reject) the drafted multi-hop retrieval slice through the verification gate, then re-run
BOTH draft-grounded lanes on the accepted ledger -- the fusion sweep and the end-to-end
answer-quality comparison -- so the graph-weight verdict rests on human-reviewed questions instead
of drafted ones. The drafted set, its worksheet, the matched vector/graph stores, and the measured
draft-grounded sweep plus answer-quality comparison are current behavior in
[GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence); every drafted multi-hop item
is span-exact and Ukrainian-gated by construction, but only a reviewer can say whether a
shared-bridge question genuinely needs both facts.

- Agent status: HUMAN-GATED
- Dependencies: the drafting, sweep, and store lanes are current behavior. Human step that gates
  completion: a reviewer decides `accept`/`reject` for every row of the multi-hop worksheet --
  specifically whether the question is answerable ONLY with both cited spans -- and signs off on
  the resulting accepted ledger.
- User-visible outcome: a graph-weight recommendation for multi-hop retrieval backed by a
  human-accepted ledger, or a recorded finding that shared-bridge drafting does not produce
  genuine multi-hop questions and the slice must come from another source.
- Scope boundary: in scope -- worksheet review, `verify-accept`, re-running the sweep over BOTH
  span-identity policies on the accepted ledger, and the adopt-or-reject verdict per knob --
  including whether `graph_fusion_span_identity` flips from `exact` to `overlap` as the shipped
  default, which is currently gated only by the drafted ledger
  ([GraphRAG](current/graphrag-backend.md#span-identity-evidence)). Out of scope -- graph schema
  changes and fusion mechanics (the candidate-depth and span-identity verdicts are current
  behavior in [GraphRAG](current/graphrag-backend.md#candidate-depth-evidence)).
- Data and artifact paths: the existing drafted bundle and worksheet plus a new
  `$DATA_DIR/graph-vector-fusion-multihop/<run>/` sweep over `accepted/goldset.jsonl` and its
  `answer-quality/` comparison.
- Execution path: the stratified worksheet is already drawn beside the bundle, so start at
  `make verify-review VERIFY_WS=<worksheet>`, then `make verify-accept VERIFY_WS=<worksheet>
  BUNDLE=<multi-hop-bundle>`, then `make compare-graph-fusion GOLDSET=<accepted>/goldset.jsonl
  GRAPH_FUSION_CANDIDATES=k,50 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap`,
  then `make compare-answer-quality GOLDSET=<accepted>/goldset.jsonl FUSION_COMPARISON=<that
  sweep>/comparison.json` -- WITHOUT `INCLUDE_DRAFTED`, since an accepted ledger no longer needs
  the drafted-grounding escape.
- Acceptance gates: every worksheet row has a decision; the accepted ledger keeps a non-empty
  multi-hop slice; the re-run sweep reports the same rows with paired intervals and the human
  records the adopt-or-reject verdict per graph strategy and per span-identity policy; the
  answer-quality comparison re-runs on the accepted ledger with `grounding: verified`.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence).

### corpus-conflict-resolution-review

Review the unresolved semantic conflict candidates through the workbench, then feed the accepted
ledger back into the resolver and repeat the retrieval plus verified answer-quality comparison.
The resolver behavior and the reason semantic candidates have no automatic suppression authority
are current behavior in
[data prep](current/data-prep.md#corpus-conflict-resolution-corpus-conflict-resolution).

- Agent status: HUMAN-GATED
- Dependencies: the resolution lane is current behavior. Human step that gates completion: an
  authorized corpus reviewer chooses `keep_both`, `drop_a`, or `drop_b` for every escalated row
  and signs off on the resulting suppression directives before application.
- User-visible outcome: an accepted or rejected suppression policy backed by reviewed conflict
  labels and a repeatable effect report, instead of adopting semantic similarity candidates as
  deletions.
- Scope boundary: in scope -- workbench review, accepted-ledger application, the same before/after
  metrics, and an adopt-or-revert decision. Out of scope -- changing detector thresholds,
  rewriting source text, or adding the resolver to auto-rag before the reviewed run supports it.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/resolution_review.jsonl`,
  `plan.json`, `conflict_overlay.json`, and `effect.md`; no new artifact root.
- Execution path: `make review-workbench REVIEW_PATH=<resolution-review-jsonl>`, then
  `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<resolution-review-jsonl>
  APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>` and repeat the fixed verified objective run.
- Acceptance gates: every review row has a decision; the regenerated plan has no unresolved
  records; rollback still restores the exact baseline; the human accepts only if retrieval and
  verified objective metrics do not regress.
- Documentation target: the resolution evidence subsection of [data prep](current/data-prep.md)
  and the conflict adapter notes in [review workbench](current/review-workbench.md).

### frontier-judge-authorization

Authorize the frontier scorer lane against real providers. The report tooling is current behavior
([frontier judge agreement and cost report](current/rigor-board-judge.md#frontier-judge-agreement-and-cost-report));
what remains is entirely the human authorization and the judgment it produces.

- Agent status: HUMAN-GATED human_decision: panding
- Command: once a real Anthropic key is in `.env` (~$0.40, 86 items):

  ```bash
  make frontier-judge-agreement \
    FRONTIER_JUDGE_MODELS=anthropic/claude-sonnet-4-5 \
    FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=1.00
  ```

- Dependencies: the agreement lane is current behavior; it runs on the 86-row calibration
  worksheet `calibration/ua_squad_postedited_v1.csv` (every row carries both a human and a local
  judge rating). Human step that gates completion: the operator puts a real Anthropic / OpenAI /
  Google key in `.env` (all three are currently blank placeholders, so no live run is possible),
  approves the per-run spend cap, and signs off on the resulting report.
- User-visible outcome: a decision record stating whether each frontier judge is trusted for
  autonomous gates on Ukrainian data, plus default budget caps derived from measured
  cost-per-item rather than from a guess.
- Scope boundary: in scope -- running the existing lane on the committed UA fixture, reviewing
  the rho and cost tables, recording an accept/reject per provider, and landing the resolved caps
  in the sample configs. Out of scope -- sending any private corpus to a provider, changing the
  headline-ranking policy, and any further report tooling.
- Data and artifact paths: `$DATA_DIR/frontier-judge/<run>/`; fixture is
  `samples/goldsets/ua_squad_postedited_v1/`.
- Execution path: `make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<id>[,<id>...]
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`; needs live provider access and spend, so it
  stays outside CI entirely.
- Acceptance gates: the report carries a non-`n/a` rho per provider against both references and a
  priced cost-per-item with cap math; the human replaces `human_decision: pending` with an accept
  or reject per provider; the accepted caps land in the sample configs with the decision recorded.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) judge section and
  [product decisions](current/scope-boundaries.md) for the trust decision per provider.

### frontier-judge-retrieved-context-agreement

Optional. Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Agent status: HUMAN-GATED
- Dependencies: blocked by `frontier-judge-authorization` (needs the same provider keys and
  spend). Reuse the agreement lane in `src/llb/scoring/frontier_agreement/` and the store-loading
  seam used by the context-position probe.
- User-visible outcome: evidence for whether frontier-judge trust measured on oracle context
  transfers to the noisy contexts a real scored run produces.
- Scope boundary: in scope -- the retrieved-context source, the two-mode comparison, and the
  delta. Out of scope -- changing the default grounding of the authorization lane before the
  comparison says it should.
- Data and artifact paths: no new roots; a second grounding mode inside the existing
  `$DATA_DIR/frontier-judge/<run>/` bundle.
- Execution path: `make frontier-judge-agreement` with a grounding-mode knob plus a built store;
  CI covers mode selection over a fake store and fake completers.
- Acceptance gates: `make ci` green; the gold-span mode reproduces the current numbers exactly; a
  live run reports rho under both modes with the delta called out.
- Documentation target: the frontier-judge agreement subsection of
  [evaluation rigor](current/rigor-board-judge.md).

### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Agent status: HUMAN-GATED
- Dependencies: the autonomous lane is current behavior ([Auto-RAG](current/auto-rag.md));
  assisted review uses the [review workbench](current/review-workbench.md). Human step that gates
  completion: the operator
  performs both
  runs, reviews gated records in the workbench, measures their own throughput against the legacy
  per-flow sessions, and accepts or rejects the recommendation bundles.
- User-visible outcome: recorded evidence that the autonomous lane produces an acceptable
  recommendation without human action, and that the assisted lane's unified workbench is at least
  as fast and less error-prone than the legacy TUIs.
- Scope boundary: in scope -- the two runs, reviewer-throughput measurement (records per minute,
  correction rate), a comparison of the two recommendation bundles, and the acceptance decision.
  Out of scope -- fixing findings (each finding becomes a new forward task).
- Data and artifact paths: both run bundles under `$DATA_DIR/auto-rag/<run>/`; throughput notes
  and the acceptance record under `$DATA_DIR/auto-rag/<run>/acceptance/`.
- Execution path: `make auto-rag CORPUS=<dir> SCORER_POLICY=auto`, then
  `make auto-rag CORPUS=<dir> SCORER_POLICY=human` with workbench review at each gate; both on
  the CUDA host with a real corpus the operator owns.
- Acceptance gates: the human signs the acceptance record; both bundles are complete and
  reproducible from their manifests; throughput numbers and any usability findings are captured
  as new forward tasks before this item leaves the file.
- Documentation target: `current/auto-rag.md` acceptance evidence and
  [human evaluation guide](../guides/human-tooling/human-in-the-loop-evaluation.md).

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
