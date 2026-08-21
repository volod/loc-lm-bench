# The scoped first-hit-rank adoption bar

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

The bake-off adopts on recall@k alone, so an encoder that only ranks the same evidence EARLIER --
`bge-m3` on the accepted PDF corpus, MRR +0.064 `[+0.008, +0.137]` with a recall delta spanning
zero -- is discarded by construction. Whether that is right is a downstream fact the retrieval
table cannot see: at a small `top_k`, or under a cross-encoder reranker that only re-sorts what it
is handed, first-hit rank is the binding constraint; at k=10 with a generous context budget,
ranking earlier changes nothing the answer reads. `src/llb/eval/embedder_adoption/` measures it end
to end and `decide_verdict` gains an opt-in second bar keyed to the answer.

- **The sweep** (`make compare-embedder-adoption`, `src/llb/cli/eval/embedder_adoption.py`): each
  CELL is one retrieval configuration (`top_k` x reranker); inside a cell both encoders score the
  IDENTICAL items end to end (`run-eval` each) and the candidate is paired against the baseline on
  the objective, a verbosity-robust found-rate (`contains`), token F1, recall@k, and MRR@k derived
  from the bundle's `first_hit_rank`. ONE resample draw is shared across every cell (common random
  numbers), so the cells are comparable to each other. Each (cell, encoder) pair is an ordinary
  bundle under that encoder's own `$DATA_DIR/run-eval/`, so any cell is reproducible. `decide_bar`
  reports **extend_bar** (a cell's calibrated objective test separates -> the rank gain reaches the
  answer there), **keep_bar** (the encoder ranks better but no calibrated objective test
  separates -> recall@k stays the sole bar), or **no_evidence** (the rank gain does not reproduce
  in any cell, so the sweep never tested the question). Reports are `report.md` + `comparison.json` under
  `$DATA_DIR/embedder-adoption-bar/<run>/`. The whole comparison + verdict is fake-bundle
  unit-tested (`tests/llb/eval/test_embedder_adoption.py`) -- no backend, store, or GPU.
- **The second bar** (`embedding_bakeoff/uncertainty.py`): `decide_verdict` takes a `bars`
  selection. `recall_at_k` is the default and the only UNCONDITIONAL bar; `--adoption-bars
  recall_at_k,mrr` (`EMBED_ADOPTION_BARS=`) opts into the scoped first-hit-rank (`BAR_FIRST_HIT =
  mrr`) bar. A candidate is adopted when it clears at least one enabled bar; the verdict records
  which bar(s) each separated candidate cleared. Enabling the bar demonstrably flips the accepted
  PDF bake-off from **RETAIN `e5-base`** to **ADOPT `bge-m3`** (cleared: `mrr`). The default stays
  recall@k-only: the bar EXTENDS the decision for configurations where rank binds, it never
  replaces the one reason to swap an encoder that holds everywhere.
- **The cross-model reading** (`make compare-adoption-models`,
  `src/llb/eval/embedder_adoption/cross_model.py`): whether a rank gain reaches the answer is partly
  a property of the MODEL, so `compare-adoption-models <sweep-A> <sweep-B>` reads two finished
  sweeps -- same encoder pair, cell grid, item set, and seed, each guarded so a mismatched pair is a
  hard error -- and states per cell whether the two models reach the same `answer` / `rank only` /
  `neither` reading, plus whether their headline verdicts match. It is pure and fake-report
  unit-tested (`tests/llb/eval/test_embedder_adoption_cross_model.py`); artifacts are
  `cross_model.md` + `cross_model.json`.
- **The roster reading** (`make compare-adoption-roster`,
  `src/llb/eval/embedder_adoption/roster.py`): with three or more sweeps, pairwise readings cannot
  state a trend, so this one asks whether the models that capture a cell's gain are separated from
  the rest by a property the operator knows BEFORE spending a run. Properties are DECLARED in a
  `--profiles` JSON (`params_b`, `family`), never inferred from the model id -- a guessed parameter
  count would become a wrong claim. The test is a SEPARATION, not a fit: a numeric property must
  admit a threshold with no overlap, a categorical one must have disjoint value sets AND actually
  group models (one family per model only restates the roster). Because a handful of models split
  cleanly by luck fairly often, a numeric separation is quoted with the probability it would arise
  at random (`2 / C(n, k)`). A unanimous focus cell reports `insufficient_variation` rather than a
  vacuous prediction. Verdicts: `property_predicts` / `no_property_predicts` /
  `insufficient_variation`; artifacts `roster.md` + `roster.json`; tests in
  `tests/llb/eval/test_embedder_adoption_roster.py`.
- **The screen cost study** (`make compare-adoption-screen`,
  `src/llb/eval/embedder_adoption/screen.py`): since the reranker answer must be measured per
  model, this measures what measuring costs. It re-derives each sweep's per-item deltas from the
  `run-eval` bundles the sweep names (the artifact persists aggregates only), CHECKS that they
  reproduce the sweep's own recorded reading before trusting them, then subsamples at a range of
  item counts and reports how often a screen that size reaches the same reading. `screen_supported`
  is claimed only when EVERY model survives the smaller set -- a screen that reproduces four models
  and loses the fifth is precisely the screen that reports "no gain" when there is one. Artifacts
  `screen.md` + `screen.json`; tests in `tests/llb/eval/test_embedder_adoption_screen.py`.
- **The borderline annotation** (`src/llb/eval/embedder_adoption/stability.py`): the repo-wide
  `p_positive` / `(borderline)` qualifier described under
  [how settled a paired reading is](paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag),
  specialised to this lane's THREE-state reading. `separated` / `flat` is what a single paired delta
  cuts; the adoption bar reads `answer` / `rank only` / `neither` by checking the objective delta
  first and first-hit rank second, so it computes that reading at each of the three conventional
  levels and hands them to the shared `stability_from_readings`. The persisted field, the boundary
  table, and the qualified verdict clause are then the identical ones the bake-off, fusion sweep,
  ablation, and answer lane report. Readings and verdicts are never altered. Tests in
  `tests/llb/eval/test_embedder_adoption_stability.py`.
- **Where the annotation is measured: inside the SWEEP itself.** `compare_cells` builds each cell's
  per-item delta vectors anyway, and the sweep already draws ONE resample index set shared by every
  cell and metric, so `stability_from_index_sets` annotates the very intervals the sweep publishes
  rather than re-drawing beside them. Each cell persists a `stability` block in `comparison.json`
  (`reading`, `p_positive`, `looser_reading`, `tighter_reading`, `borderline`, `side`), `report.md`
  gains a per-cell `reading` column and a `How close each cell sits to the cut` table, the terminal
  summary prints `p_positive` per cell, and `decide_bar`'s reason QUALIFIES the cell it names --
  `keep_bar` on a near-miss now says "read it as too close to call rather than as settled
  evidence" and names which neighbouring level would flip it, plus a `borderline_cells` list on the
  verdict. This is why the ONE-model recipe below is as honest as the five-model table: an operator
  never had to assemble a roster to learn their `extend_bar` rests on a knife-edge row. `decide_bar`
  and its qualifier live in `src/llb/eval/embedder_adoption/verdict.py`, split from the per-cell
  statistics so the sentence an operator acts on stays separately readable. The
  annotation is skipped (and the artifact carries none) when the sweep drew no resamples or the
  reporting confidence sits outside the two neighbouring conventions, because `p_positive` would
  not mean what it says. The roster READS the sweep's persisted value and needs no bundles at all;
  sweeps recorded before the field existed fall back to re-deriving it from the run bundles they
  name, at that sweep's own confidence, and degrade silently when those bundles are gone, so an
  archived roster still reports. The two paths are checked equal in CI over fake bundles
  (`tests/llb/eval/test_embedder_adoption_borderline.py`, which also covers the persisted fields,
  the report rendering, and both sides of the qualified verdict reason) and were checked equal on
  all 20 recorded cells (below).

CUDA host, 2026-07-25; `MamayLM-Gemma-3-12B-IT-v2.0` (Q4_K_M, Ollama), the accepted converted-PDF
goldset (40 items over final+tuning+calibration, the same 1120-chunk `recursive` 800/120 corpus the
paired re-read used), `bge-m3` minus `e5-base`, 2000 resamples, seed 13. Report under
`$DATA_DIR/embedder-adoption-bar/run-mamaylm12b/`.

| cell | config | d objective | d recall@k | d MRR@k | reading | p_positive |
| --- | --- | ---: | ---: | ---: | :-: | ---: |
| `k10` | k=10, no reranker | -0.010 `[-0.062, +0.037]` | +0.050 `[-0.050, +0.150]` | +0.064 `[+0.009, +0.137]` | rank only | 0.368 |
| `k10+rerank` | k=10, bge-reranker-v2-m3 | +0.052 `[+0.011, +0.101]` | +0.050 `[0.000, +0.125]` | +0.050 `[0.000, +0.125]` | **answer** | 0.995 |
| `k3` | k=3, no reranker | +0.034 `[+0.002, +0.073]` | +0.125 `[+0.025, +0.225]` | +0.079 `[+0.017, +0.158]` | **answer** (borderline) | 0.981 |
| `k3+rerank` | k=3, bge-reranker-v2-m3 | +0.021 `[-0.002, +0.056]` | +0.050 `[0.000, +0.125]` | +0.050 `[0.000, +0.125]` | neither (borderline) | 0.954 |

Verdict: **extend_bar** -- the answer-side gain clears zero in 2 of 4 cells, and the sweep's own
reason adds that one of the two (`k3`, p_positive 0.981) is a reading a 97.5% interval would drop.
What it establishes:

- **At the shipped default (`k10`, no reranker) the rank gain is free, exactly as predicted.**
  `bge-m3` ranks the evidence earlier (MRR +0.064 clears zero) but the answer does not move
  (objective -0.010, interval spans zero, 10/14/16 win/loss/tie). At k=10 with room in the budget,
  the model already sees the gold span whether it is ranked 1st or 3rd. This is why recall@k stays
  the DEFAULT bar and the shipped `e5-base` default is unchanged.
- **Under a reranker the rank gain reaches the answer.** `k10+rerank` gives objective +0.052
  `[+0.011, +0.101]` where recall is at ceiling (bge 1.000 vs e5 0.950) -- the cross-encoder
  re-sorts the candidate pool `bge-m3` hands it into a better first hit, and the answer improves.
  This is the cleanest pure-rank cell: the answer moves without recall separating.
- **At a small `top_k` it reaches the answer too, but partly as recall.** `k3` gives objective
  +0.034 `[+0.002, +0.073]`; recall@k ALSO separates there (+0.125), because at a tight budget a
  better ranking pulls a gold span INSIDE the k=3 cut it would otherwise miss. So the k=3 gain is
  not pure first-hit rank -- read it as "the rank advantage becomes a recall advantage when the
  budget is small", which is still a reason to prefer `bge-m3` at k=3.
- **Two of the four cells rest on the cut, and this one run says so.** `k3` clears at 0.981 against
  a 0.975 threshold (a 97.5% interval reads it `rank only`) and `k3+rerank` misses at 0.954 (a 90%
  interval reads it `answer`), while `k10+rerank` at 0.995 and `k10` at 0.368 are settled. So the
  strongest statement this single sweep supports is "the reranked k=10 cell is a settled answer
  gain, and the k=3 pair is too close to call in either direction" -- which is what the operator
  running the recipe reads off the terminal, not something they learn later from a roster.
- **Cost, for the adoption decision.** `bge-m3` embeds at ~1/3 the throughput of `e5-base` and
  builds a 1.23x index, so a cell that does not clear zero is not worth paying for. The
  configuration-by-configuration recommendation is settled by the roster below, not by this one
  model -- in particular the reranker cell does NOT generalize.

## The five-model roster

CUDA host, 2026-07-25. Four more models on the SAME corpus, cells, item set, and seed, spanning
three families and 11.8B-27B; sweeps under `$DATA_DIR/embedder-adoption-bar/run-<slug>/`, the roster
reading under `.../roster/` and the declared profiles in `.../model-profiles.json` (each model's
`parameter_size` / `family` as its own model card reports it, so both are readable before a run is
spent). The table preserves each sweep's HISTORICAL percentile reading. Its source bundles are not
present on the current host, so these rows could not be reconstituted by the calibrated audit and
must not be read as current randomization verdicts:

| model | params | family | `k10` | `k10+rerank` | `k3` | `k3+rerank` | verdict |
| --- | ---: | :-: | :-: | :-: | :-: | :-: | :-: |
| `lapa-v0.1.2-instruct` | 11.8B | gemma3 | rank only | neither | rank only | neither | keep_bar |
| `MamayLM-Gemma-3-12B` | 11.8B | gemma3 | rank only | **answer** | **answer** | neither | extend_bar |
| `qwen3:14b` | 14.8B | qwen3 | rank only | neither | **answer** | neither | extend_bar |
| `mistral-small3.1:24b` | 24B | mistral3 | rank only | neither | **answer** | **answer** | extend_bar |
| `MamayLM-Gemma-3-27B` | 27B | gemma3 | rank only | neither | rank only | **answer** | extend_bar |

Roster verdict: **no_property_predicts**. What the roster establishes:

- **The shipped default is settled: the rank gain is free there, unanimously.** All five models read
  `k10` as `rank only` -- `bge-m3` ranks the evidence earlier (MRR +0.064, identical in every sweep)
  and no recorded objective interval clears zero. This is historical percentile evidence; a new
  sweep applies the calibrated test. The recorded single-model finding rests on
  three families and a 2.3x parameter range, so `recall_at_k` staying the DEFAULT bar and `e5-base`
  staying the shipped default are not one model's quirk.
- **Neither parameter count nor family predicts the reranker cell, and the counter-examples are
  clean.** Only MamayLM-12B captures `k10+rerank` (1 of 5). `lapa-v0.1.2` has the SAME declared
  parameter count and family (11.8B, gemma3 -- both are Ukrainian fine-tunes of the same base) and
  does not capture it; within one family the LARGER MamayLM-27B does not capture what the 12B does.
  So the split is not a threshold on size and not a family label: two models an operator cannot tell
  apart from their cards land on opposite sides. `no_property_predicts` is a measured negative, not
  a shortage of models.
- **The bar itself reproduces; the cell that justifies it does not.** Four of five models return
  `extend_bar`, but via different cells -- `k3` for MamayLM-12B / qwen3 / mistral, `k3+rerank` for
  mistral / MamayLM-27B, `k10+rerank` for MamayLM-12B alone. Every cell except the shipped `k10`
  default justifies the second bar for SOME model, and only `lapa` captures nothing anywhere.
- **The operator takeaway, conditioned on the configuration rather than the model.** At the shipped
  generous-`top_k`, no-reranker default, retain `e5-base` (recall@k-only bar) -- unanimous across
  the roster. At a small `top_k`, adopt `bge-m3` with `--adoption-bars recall_at_k,mrr`: 4 of 5
  models turn its ranking into a better answer in at least one k=3 cell, though only 2 of those 4 on
  a reading a tighter convention would keep ([how settled each row
  is](paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag)). Do
  NOT assume a cross-encoder reranker will make it pay at k=10 -- that held for 1 of 5 models and is
  not predictable from the model card, so measure it with `make compare-embedder-adoption` on the
  model actually being shipped.

Run note for a ~16 GiB host: a 24B/27B generator holds ~13 GiB in Ollama, leaving too little for the
embedder and the cross-encoder, which fail with a CUDA OOM inside `retrieve`. Run those sweeps with
`CUDA_VISIBLE_DEVICES=""` so only the ENCODERS move to the CPU -- generation stays on the GPU
because Ollama is a separate process. Retrieval is model-independent, and the CPU-encoder sweeps
reproduce the GPU sweeps' recall@k / MRR@k columns exactly, which is the check that the switch
changed nothing measurable.

## What the per-model answer costs

Because the reranker reading has to be measured per model, what it COSTS is part of the
recommendation. CUDA host, 2026-07-25; study over the five recorded sweeps
(`$DATA_DIR/embedder-adoption-bar/screen/`), confirmation run under `.../screen-confirm-mamaylm12b/`.
The cost splits into two axes that behave completely differently:

- **Dropping the other cells is free, and it is the whole saving.** The reranker question lives in
  ONE cell, so `--top-ks 10 --rerankers on` scores 6 bundles instead of 24. A confirmation run of
  MamayLM-12B that way reproduced the full sweep's `k10+rerank` row BIT-IDENTICALLY on every paired
  metric (objective +0.052 `[+0.011, +0.101]`, and likewise MRR@k and recall@k) and reached the same
  `extend_bar` verdict, in **6m40s against ~18 minutes** -- 4x fewer bundles, 2.7x wall clock
  (the reranked cell costs more per bundle than the plain ones, which is why the time saving is
  smaller than the bundle saving).
- **Cutting the ITEM set is not available.** A smaller ledger can either lose or invent a
  calibrated separation, so the screen measures agreement with the full reading in both
  directions. How often each model's `k10+rerank` reading survives a subsample of N items, 120
  draws per size:

| model | full reading | n=10 | n=15 | n=20 | n=25 | n=30 | n=35 |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MamayLM-Gemma-3-12B` | **answer** | 11% | 29% | 38% | 49% | 59% | 82% |
| `lapa-v0.1.2` | neither | 96% | 90% | 87% | 82% | 78% | 76% |
| `qwen3:14b` | neither | 94% | 91% | 92% | 95% | 98% | 99% |
| `mistral-small3.1:24b` | neither | 97% | 100% | 98% | 100% | 100% | 100% |
| `MamayLM-Gemma-3-27B` | neither | 88% | 84% | 89% | 97% | 98% | 99% |

Verdict: **full_set_required**. What the table establishes:

- **The one positive reading needs the whole ledger.** MamayLM-12B is the only model whose reranked
  gain reaches the answer, and a screen reproduces that at 11% on 10 items and still only 82% on 35
  -- below the 90% target even at 7/8 of the set. Halving the items would have told the operator the
  reranker does not pay, on the one model where it does.
- **The bias is one-directional, which makes a cheap screen actively misleading here.** Every
  disagreement in the table is a `neither` where the full set says `answer`, or an `answer` where
  the full set says `neither` on a borderline row; no model ever flipped from `neither` to a
  confident `answer` it did not earn. A screen's errors are systematically "reports no gain", so
  its failures look like a clean negative result rather than like noise.
- **`lapa` is a knife-edge row, not a clean negative.** Its agreement FALLS as the subsample grows
  (96% -> 76%), and every disagreement is `answer`: its full-set objective delta is
  +0.024 `[-0.000, +0.059]`, a lower bound sitting on zero. The binary reading prints `neither`, but
  the honest statement is "too close to call" -- read that row as undecided rather than as evidence
  the reranker does not pay for `lapa`.
- **The recipe.** To decide the reranker question for a model being shipped, run the full accepted
  ledger at one cell:
  `make compare-embedder-adoption MODEL=<model> ADOPTION_TOP_KS=10 ADOPTION_RERANKERS=on
  EMBED_BASELINE_DATA_DIR=<e5-root> EMBED_CANDIDATE_DATA_DIR=<bge-root> CORPUS=<corpus>
  GOLDSET=<accepted>/goldset.jsonl SPLIT=final,tuning,calibration`. Do not reach for `ADOPTION_LIMIT`
  to make it cheaper. That single run reports its own `p_positive` and marks the cell `(borderline)`
  when a neighbouring convention would read it differently, and the `extend_bar` / `keep_bar`
  sentence names which -- so the one-cell recipe answers the same question the roster does, with no
  second command to run. New runs report `randomization_p` beside `p_positive`.

## Historical percentile stability of the adoption roster

This table records the retired `lo > 0` reading and remains useful as the audit trail explaining why
calibration was needed. It is not a current verdict table: the source bundles are absent from the
host inventory, so no randomization p can be reconstructed for these cells. New sweeps persist and
render `randomization_p` per cell; `p_positive` remains diagnostic.

| model | at 90% | `k10+rerank` (95%) | at 97.5% | p_positive | settled? |
| --- | :-: | :-: | :-: | ---: | :-: |
| `lapa-v0.1.2` | answer | neither | neither | 0.969 | **NO (below)** |
| `MamayLM-Gemma-3-12B` | answer | answer | answer | 0.995 | yes |
| `MamayLM-Gemma-3-27B` | neither | neither | neither | 0.803 | yes |
| `qwen3:14b` | neither | neither | neither | 0.767 | yes |
| `mistral-small3.1:24b` | neither | neither | neither | 0.380 | yes |

- **`lapa` is not a negative result, it is an undecided one.** Its `neither` sits at 0.969 against a
  0.975 cut and becomes `answer` at 90%, so it now prints `neither (borderline)`. The three settled
  negatives are at 0.380-0.803, nowhere near the line. Before this annotation those four rows were
  typographically identical, which is what made the roster's "1 of 5 models capture it" read as
  four clean negatives instead of three plus one too close to call. The verdict is unchanged --
  `no_property_predicts` still holds, since `borderline` is a qualifier and never an `answer`.
- **The two-sided check discriminates rather than firing on everything.** Across all 20 recorded
  rows (5 models x 4 cells) it marks 4 -- 20%. Two are `below` (`lapa` `k10+rerank` at 0.969 and
  `MamayLM-12B` `k3+rerank` at 0.954, both would clear a 90% bar) and two are `above`
  (`MamayLM-12B` `k3` at 0.981 and `qwen3` `k3` at 0.978, both dropped by a 97.5% bar). The
  remaining 16 read identically at all three conventions.
- **The k=3 claim survives in direction but not in strength.** Marking the near-miss positives
  changes how "4 of 5 models capture a k=3 gain" should be read: `mistral` (0.988 on `k3`, 1.000 on
  `k3+rerank`) and `MamayLM-27B` (0.992 on `k3+rerank`) capture it on SETTLED readings, while
  `MamayLM-12B` and `qwen3` capture it only through a row a tighter convention would drop. So the
  honest restatement is **4 of 5 capture a k=3 gain, 2 of them settled** -- still the strongest
  case for the scoped bar, and still far better supported than the reranker cell, but not the four
  independent confirmations the bare table implied.
- **The sweep's own measurement agrees with the roster's on all 20 cells.** The five recorded
  sweeps were rebuilt from the `run-eval` bundles they name and re-rendered in place (originals kept
  beside them as `comparison.pre-borderline.json` / `report.pre-borderline.md`). Every rebuilt cell
  reproduced its recorded paired intervals, per-lane means, item set, and verdict decision exactly,
  and every cell's newly persisted `stability` equalled what `row_stability` measures from those
  same bundles -- so the annotation an operator's own one-model run prints is the same number the
  roster table quotes, not a second estimate of it. The sweep reports mark the same 4 of 20 rows.
  The re-rendered `roster/` and `screen/` artifacts reproduce their recorded tables unchanged.
