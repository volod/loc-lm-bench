# Reranker Bake-off

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md). The reranker SEAM itself -- how a cross-encoder is
wired between retrieval and generation, its knobs, and the context-order policy -- is
[Reranking, context order, and query-side processing](rerank-and-query.md); this page is how the
shipped cross-encoder is CHOSEN.

## The lane (`compare-rerankers`)

`llb compare-rerankers` (`src/llb/rag/rerank_bakeoff/`; `make compare-rerankers`) answers "which
cross-encoder, and is it worth the second model in VRAM?" with evidence rather than a pin. The
design is what makes the reading a statement about RERANKERS:

- **One shared candidate pool.** The pool is retrieved ONCE per item at a fixed encoder, chunking,
  and depth (`--rerank-candidates`, default 30), and every candidate re-sorts that identical pool.
  Two rows therefore differ only by the cross-encoder -- no encoder or chunker variance leaks into
  the comparison, and the paired ledger is over the same items by construction.
- **The reranker-OFF row rides along.** `none` is the pool in its own retrieval order, scored by the
  identical metric, so "is the second model worth running at all?" is a row in the same table rather
  than a separate run.
- **Rank quality AND cost in one table.** recall@k, MRR, and mean first-hit rank (with the count of
  items that HAVE a hit beside it, because a mean rank over a shrinking denominator is not
  comparable), then mean rerank wall-clock per query, pairs/s, cold load seconds, and the VRAM the
  model holds at rest and at its scoring peak.
- **The fit gate.** `--generator-vram-mb <mb>` declares what the generator holds while serving; the
  lane then computes the headroom (device total - generator - `--vram-reserve-mb`, default 512) and
  a candidate whose resident footprint exceeds it is SKIPPED with the footprint that decided it. No
  declaration means the footprints are still measured and the gate does not run -- a measurement
  without a budget is not a verdict.
- **Both adoption bars by default.** This lane is by construction the configuration where first-hit
  rank binds -- a cross-encoder can only re-sort what it is handed -- which is exactly the scope the
  embedder lane declares for its second bar
  ([the scoped first-hit-rank bar](first-hit-rank-adoption.md#the-scoped-first-hit-rank-adoption-bar)).
  So `--adoption-bars` defaults to `recall_at_k,mrr` here, against the embedder lane's recall-only
  default. The paired intervals, the selection adjustment, and the verdict are the shared machinery
  ([paired verdicts](paired-verdicts.md)) pointed at reranker rows.
- **A floor read on the RERANK scores.** With `NOISE_FLOOR=1` each row's
  [measurement floor](retrieval-metrics.md#measurement-floor---noise-floor) perturbs the score the
  row was actually ranked on -- the cross-encoder's, not the store's -- at an amplitude scaled by
  that row's own score range, because a sigmoid head living in 0..1 and a logit head spanning tens
  do not share a scale and one absolute jitter would hand the logit model a tighter floor for free.
  `report.json` records the per-row amplitudes in `noise_floor.jitter_by_lane`.

Modules: `rerank_bakeoff/families.py` (the per-model input-convention registry: which query-side
instruction the model's own `config_sentence_transformers.json` applies, and whether it needs
`trust_remote_code`), `roster.py` (screening, over the shared
[candidate screen](#roster-screening-is-shared-with-the-embedder-lane)), `scoring.py` (the pass over
the shared pools, the row shaping, the floor pools -- all pure), `lane.py` (score, rank, assemble),
`fit.py` (the VRAM budget, the load attempt, and the recorded reason a candidate produced no row),
`readings.py` (the paired verdict and the floor), `loader.py` / `worker.py` (the heavy half), and
`report.py` (ASCII + Markdown). The report's shared sections -- paired cells, the evidence gate, the
boundary table, the verdict sentence, the not-scored table -- come from
`src/llb/rag/bakeoff_report_sections.py`, which both bake-offs render through.

Artifacts: `$DATA_DIR/compare-rerankers/<run>/{report.md,report.json}` plus the single store the
pools were retrieved from under `stores/`.

```bash
make compare-rerankers GOLDSET=<accepted.jsonl> CORPUS=<corpus-dir> SPLIT= NOISE_FLOOR=1 \
  RERANK_GENERATOR_VRAM_MB=<mb> RERANK_ALLOW_REMOTE_CODE=1
```

`RERANK_MODELS=` overrides the roster, `RERANK_BASELINE=` the incumbent every row is paired against,
`RERANK_CANDIDATES=` the pool depth, `RERANK_BATCH_SIZE=` / `RERANK_DTYPE=` the load and scoring
configuration recorded in the report. `SPLIT=` (empty) scores the whole ledger; the repo-wide
default is `final`.

### Each candidate loads in its own process

Measured on the CUDA host, and the reason `worker.py` exists: a candidate whose repository-supplied
modelling code raises a CUDA **device-side assert** poisons the CUDA context for the whole process,
and every candidate loaded after it fails with the same assert. In one process a roster of five read
as "four of these do not run on this host" when three of them run fine -- a host verdict that was an
artifact of roster order. So each candidate is loaded and scored in its own SPAWNED child behind the
same `RerankScorer` callable: the context dies with the child, the VRAM baseline is read inside the
child before its weights load (so a footprint is the candidate's own), and a child that dies
mid-pass becomes a recorded skip row rather than a hole in the table. `--in-process` is the escape
hatch, one spawn per candidate cheaper and unsafe for exactly the reason above.

### Roster screening is shared with the embedder lane

The two policies a roster needs before anything loads -- REFUSE an id with no declared convention
(scoring a model under a guessed input format is the one failure a bake-off must not commit) and
DECLINE `trust_remote_code` unless the operator opted in -- are identical for encoders and
rerankers, so they live once in `src/llb/rag/candidate_screen.py` and each lane supplies its own
registry and wording. `llb.rag.embedding_bakeoff_roster` and `llb.rag.rerank_bakeoff.roster` are
thin. A declined or unloadable candidate lands in `report.json` under `skipped[]` and in `report.md`
under "Candidates not scored", so a shorter table always says why it is shorter.

Registered reranker families (`rerank_bakeoff/families.py`), each citing the card it was read from:
`bge-reranker`, `jina-reranker-v2` (remote code), `gte-multilingual-reranker` (remote code),
`mxbai-rerank-v2`, `qwen3-reranker` (its repo config applies the retrieval-task instruction on the
query side; the report prints the prompt on the row).

Tests (no download, no GPU): `tests/llb/rag/test_rerank_bakeoff_lane.py` (shared pools, the
reranker-off row, cost columns, the fit gate, a mid-pass death, the verdict, the floor),
`test_rerank_bakeoff_roster.py`, `test_rerank_bakeoff_report.py`, `test_rerank_bakeoff_worker.py`
(the child protocol over a stub cross-encoder), `tests/llb/rag/test_compare_rerankers_cli.py`.

## What the bake-off measured (2026-08-16, CUDA host)

RTX 4060 Ti (16,380 MiB) with a UA generator ACTUALLY resident -- MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M
on Ollama at `num_ctx` 8192, holding 8,278 MiB -- so every footprint below was measured beside a
served generator rather than on an empty device. Fixed retrieval on both corpora:
`intfloat/multilingual-e5-base`, `recursive@800/120`, flat mode, pool depth 30, k=10, batch 32,
dtype `auto` (each card's own), 2000 paired resamples, seed 13.

### Corpus 1: the accepted converted-PDF ledger (40 items)

`$DATA_DIR/compare-rerankers/20260816T150456.120681Z-0cd6e2d48f61/report.md`.

| row | recall@10 | MRR | first-hit rank | items hit | ms/query | VRAM at rest (MB) | peak (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-v2-m3` (incumbent) | 0.950 | 0.930 | 1.10 | 38 | 600 | 2335 | 4075 |
| `Qwen/Qwen3-Reranker-0.6B` | 0.950 | 0.925 | 1.05 | 38 | 1060 | 1329 | 5101 |
| `mixedbread-ai/mxbai-rerank-base-v2` | 0.925 | 0.912 | 1.03 | 37 | 572 | 1175 | 4185 |
| `off (retrieval order)` | 0.925 | 0.852 | 1.38 | 37 | 0 | - | - |

Verdict: **RETAIN `BAAI/bge-reranker-v2-m3`** -- no candidate clears an adoption bar. The two
alternatives sit inside their own paired intervals against the incumbent (Qwen3 MRR
`-0.005 [-0.050, +0.048]`, mxbai recall `-0.025 [-0.075, 0.000]`), and the selection adjustment over
the 6 candidate-bar hypotheses leaves every adjusted p at 1.000. Floor: recall@10 `+/-0.000`, MRR
`+/-0.019` -- mxbai is the one row whose MRR band is non-zero (4 of 40 items fragile), which is
itself a property of its score spacing near the cut.

### Corpus 2: the committed 250-item UA fixture

`$DATA_DIR/compare-rerankers/20260816T150841.670986Z-e49d2bc4a042/report.md`, over
`samples/goldsets/ua_squad_postedited_v1/` (250 items, 250 documents).

| row | recall@10 | MRR | first-hit rank | items hit | ms/query | VRAM at rest (MB) | peak (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-v2-m3` (incumbent) | 1.000 | 0.941 | 1.16 | 250 | 534 | 2335 | 4535 |
| `Qwen/Qwen3-Reranker-0.6B` | 1.000 | 0.924 | 1.26 | 250 | 1084 | 1318 | 5188 |
| `mixedbread-ai/mxbai-rerank-base-v2` | 0.984 | 0.858 | 1.54 | 246 | 563 | 1175 | 4937 |
| `off (retrieval order)` | 0.980 | 0.847 | 1.50 | 245 | 0 | - | - |

Verdict: **RETAIN `BAAI/bge-reranker-v2-m3`** again, and at n=250 the intervals are tight enough to
say more than "not separated":

- `Qwen3-Reranker-0.6B` ties the incumbent on recall (both 1.000, 0 differing items) and sits
  `-0.017 [-0.038, +0.002]` below it on MRR -- a near-miss on the wrong side, at twice the latency.
- `mxbai-rerank-base-v2` is BELOW the incumbent on both bars with intervals that exclude zero
  (recall `-0.016 [-0.032, -0.004]`, MRR `-0.082 [-0.114, -0.052]`) and is barely above no
  reranking at all.
- The reranker-OFF row separates in the negative direction on both bars (recall
  `-0.020 [-0.040, -0.004]`, MRR `-0.094 [-0.127, -0.063]`). That is the measured justification for
  running a cross-encoder at all on this stack -- the one reading in the table whose interval
  excludes zero in a direction that decides something.

Floor: recall@10 `+/-0.006`, MRR `+/-0.016`, driven entirely by `mxbai` (45 of 250 items fragile,
against 6 for the incumbent and 0 for retrieval order) -- its scores cluster near the cut, so part
of its ranking is decided by noise rather than by relevance.

## What this establishes

- **The pinned default survives its first comparison, on both corpora.** `BAAI/bge-reranker-v2-m3`
  stays `DEFAULT_RERANKER`; it is now a measured choice rather than a default nobody had questioned,
  and it is the point-estimate leader on recall AND MRR on both item sets.
- **What the reranker buys is RANK, not recall.** Against no reranking the incumbent is worth
  +0.078 MRR / +0.025 recall@10 on the PDF ledger and +0.094 MRR / +0.020 recall@10 on the fixture,
  moving the mean first-hit rank from 1.38 to 1.10 and from 1.50 to 1.16. Only the MRR gain
  separates. That is the shape the [scoped first-hit-rank bar](first-hit-rank-adoption.md) predicts,
  and it is why this lane reads both bars by default.
- **What it costs.** ~530-600 ms per query at pool depth 30 and ~4.1-4.5 GiB of VRAM at its scoring
  peak beside an 8.3 GiB generator on a 16 GiB device. Every scored candidate fit the declared
  headroom; the decoder-based `Qwen3-Reranker-0.6B` is the cheapest at rest (1.3 GiB) and the most
  expensive at peak (5.0-5.2 GiB) while running at half the incumbent's speed -- the pair of numbers
  an operator sizing a host needs, and the reason at-rest footprint alone is not the column to
  choose on.
- **A cheaper candidate is not a cheaper option.** `mxbai-rerank-base-v2` is the smallest and among
  the fastest, and it is measurably WORSE than the incumbent on both bars on the fixture while
  carrying the widest measurement floor of any row.
- **Two of five candidates could not be scored at all, for a PACKAGING reason.** Both remote-code
  entries fail against the pinned stack: `jinaai/jina-reranker-v2-base-multilingual` raises
  `ImportError: cannot import name 'create_position_ids_from_input_ids' from
  transformers.models.xlm_roberta.modeling_xlm_roberta`, and
  `Alibaba-NLP/gte-multilingual-reranker-base` indexes its rope tables with an uninitialized
  `position_ids` buffer (`rope_cos[position_ids]`), which is an out-of-bounds gather -- a CPU load
  reports `IndexError`, CUDA reports a device-side assert. This is the SAME failure class the
  encoder roster refresh recorded for `jina-embeddings-v3` and `gte-multilingual-base`
  ([embedders](embedders.md#the-two-remote-code-candidates-do-not-run-on-the-pinned-stack)): both
  vendors target the transformers 4.x API, the repo pins 5.12.1, and the hole is packaging, not
  quality.
