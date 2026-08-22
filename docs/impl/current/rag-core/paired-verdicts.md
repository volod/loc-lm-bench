# Paired uncertainty and the adopt-or-retain verdict

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

The floor answers whether a gap is numeric noise; it cannot answer whether the SAME gap survives a
different draw of questions, and on a 40-item set a "winner" is routinely two questions.
`src/llb/rag/embedding_bakeoff/uncertainty.py` supplies the second reading, reusing
`bootstrap_index_sets` / `paired_comparison` from the fusion sweep
(`src/llb/rag/fusion_evidence/stats.py` -- they take metric vectors, not fusion rows):

- Every candidate is retrieved ONCE per item (`retrieve_pairs`), and both the published row and its
  per-item vectors (`item_vectors`: recall@k and reciprocal rank) come from that one pass, so a row
  and its interval can never disagree.
- `paired_rows` draws ONE set of resample indexes (common random numbers, seeded) and reuses it for
  every candidate and metric, so each interval is about the DIFFERENCE against the baseline
  embedder rather than two lanes' separate sampling noise. Each row carries
  `paired_vs_baseline` = `{baseline, metrics: {recall_at_k, mrr}}` with the delta interval, the
  item-level win/loss/tie ledger, and the exact sign-test p.
- The baseline is `--baseline` (`EMBED_BASELINE=`), defaulting to the shipped
  `intfloat/multilingual-e5-base`, because a swap recommendation is a statement about replacing
  THAT row. A baseline the run did not score leaves the rows bare and the verdict `undecided`
  instead of silently re-pointing the comparison. `--resamples` (`EMBED_RESAMPLES=`, default 2000)
  and `--seed` (default 13) pin the draw.
- `decide_verdict` states the recommendation as **adopt** or **retain**: a candidate is "separated"
  only when its 95% paired recall@k interval lies wholly above zero. Otherwise the incumbent is
  retained, whatever the point estimate says. `best_recall` is still reported, relabeled in
  `report.md` as the point-estimate leader -- it is no longer the recommendation.
- Artifacts: `report.md` (the table gains `recall delta vs <baseline>` / `w/l/t` / `sign p` /
  `recall reading` columns, a boundary table, and the verdict line) plus `report.json` beside it
  with every interval bound and ledger, so a later re-read recomputes from numbers instead of prose.
- `decide_verdict` and the bar helpers live in `src/llb/rag/embedding_bakeoff/verdict.py`, split
  from the paired statistics so the sentence an operator acts on stays separately readable.

Tests: `tests/llb/rag/test_embedding_bakeoff_uncertainty.py` (vector means matching the published
row, the paired ledger, seed determinism, a baseline paired against itself at exactly zero, a
one-item lead that does NOT separate, adopt/retain/undecided, the report columns, and the CLI's
`report.json`) -- all over fake stores, no FAISS, no GPU.

## Paired-power contract for comparison lanes

`src/llb/rag/fusion_evidence/power.py` is the shared a priori sensitivity seam beside the paired
bootstrap. A lane supplies an earlier per-item candidate-minus-baseline delta vector, a
predeclared minimum detectable effect (MDE), its reporting confidence, target power, and the
planned item count. Before retrieval or inference, the seam writes `power-plan.json` and prices:

- the two-sided paired normal-approximation variance floor,
  `ceil(((z_(1-alpha/2) + z_power) * sample_sd / MDE)^2)`;
- the exact-sign-test discordance floor at the reference ledger's differing-item rate; and
- one `required_n`, the larger floor, plus `binding_floor` (`variance`, `discordance`, or `both`).

The completed comparison repeats both floors with the run's own per-item SD and discordance rate.
It replaces the planned `target_reached` reading with that realized check, retains
`planned_target_reached` for audit, and reports
`resolvable_mde = (z_(1-alpha/2) + z_power) * realized_sd / sqrt(realized_n)`. This is sensitivity
at the reached item count, not post-hoc achieved power computed from the observed effect.
An undecidable result therefore states both what effect the item set could resolve and whether
realized variance or discordance made the declared target miss.

Lane selectors and artifacts:

- Context ablation selects fitting `long_context - rag` objective deltas through
  `src/llb/eval/context_ablation/power.py` and emits the same current selector, variance floor,
  discordance floor, and binding-floor fields as the other lanes.
- Embedder bake-off selects `--power-candidate` against `--baseline` and either `recall_at_k` or
  `mrr` via `--power-metric`. `report.json` now retains `paired_items`, the per-item metric ledger
  with gold item ids needed to price and audit a later run.
- Graph-vector fusion selects `--power-row` against the vector baseline and one
  `--power-metric` on the declared focus slice. Its existing `focus_items` ledger supplies the
  paired deltas. Details and its Make variables are in
  [GraphRAG](../graphrag-backend/fusion-sweep-evidence.md#the-graph-weight-sweep-lane).

Use the existing workflow targets; MDE selection remains an operator decision:

```bash
make compare-embeddings GOLDSET=<goldset-jsonl> \
  EMBED_POWER_REFERENCE=<earlier-report-json> EMBED_POWER_CANDIDATE=<candidate-model> \
  EMBED_POWER_METRIC=recall_at_k EMBED_MDE=<minimum-gain>

make compare-graph-fusion GOLDSET=<goldset-jsonl> \
  FUSION_POWER_REFERENCE=<earlier-comparison-json> FUSION_POWER_ROW=<fusion-row> \
  FUSION_POWER_METRIC=recall_at_k FUSION_MDE=<minimum-gain>
```

`tests/llb/rag/fusion_evidence/test_paired_power.py` covers item-count arithmetic, inverted MDE, and
realized-SD rechecking. `tests/llb/rag/fusion_evidence/test_paired_power_contract.py` covers both
selectors, artifact output, and the plan-before-build/retrieval order. The context-specific
regression suite remains `tests/llb/eval/test_context_ablation_power.py`. Host validation on
2026-07-26 completed `make ci`: 2,215 passed and 45 opt-in/slow tests were deselected; no heavy
model run belongs to this delivery.

The repository-wide count audit reuses this distinction between inferential evidence and resource
budgets. `src/llb/quality/acceptance_gate_registry.py` declares each retained row, trial, finalist,
and sample control; `src/llb/quality/acceptance_gates.py` discovers matching Make and Typer
defaults, rejects unexplained controls, and writes the machine inventory through
`make acceptance-gate-audit`. Trial counts in joint search, screening, knowledge-cutoff fitting,
and finetuning remain explicit because they cap optimizer work rather than establish uncertainty.
Human verification row defaults and chain promotion counts moved to finite-population precision
and relative-retention plans; their workflow details are in
[data prep](../data-prep/verification-gate.md#experiment-derived-verification-and-acceptance-gates).

## How settled a paired reading is -- `p_positive` and the borderline flag

Every adopt-or-retain call cuts the calibrated one-sided sign-flip p at the alpha corresponding to
the reported two-sided confidence convention. `src/llb/rag/fusion_evidence/stability.py` owns the
shared persisted shape:

- **`randomization_p`** is the decision-driving quantity. **`p_positive`** remains the diagnostic
  share of paired bootstrap resamples in which the candidate is ahead. **`borderline`** is raised
  when the calibrated reading would change at either
  NEIGHBOURING conventional level, looser (90%) or tighter (97.5%). The check is two-sided on
  purpose: `side: below` is a negative that would clear a looser bar (an undecided negative),
  `side: above` is a positive a tighter bar would drop (a positive resting on the convention). No
  constant is fitted to the data; all three levels are conventions the repo already reports at.
- **It rides on `paired_comparison`, so no two-state lane wires it.** Every paired delta goes
  through that one function, which now attaches a `stability` block to each `PairedComparison`.
  That reaches the embedder bake-off, the fusion sweep, the context ablation, and the
  answer-quality slices at once -- and any future lane for free.
- **The bootstrap draw is still single-pass.** Its sorted means supply the unchanged percentile
  interval and `p_positive`; the sign-flip engine uses the same configured draw count and a stable
  seed derived from the shared index sets.
- **The annotation is omitted rather than faked** when no resample was drawn (`p_positive` is a
  share OF resamples, and a zero would read as a settled negative) or when the reporting confidence
  sits outside the two conventions the flag is defined against.
- **Every verdict reason carries the shared clause.** `borderline_note` produces ONE phrasing for
  all four lanes, and each lane passes the rows its verdict was DECIDED ON -- not only the one its
  sentence quotes, because the ablation's `rag_pays_off` is reached by the long-context check
  failing first. Each lane's `report.md` also renders a `boundary_table` over those rows, and the
  bake-off / ablation tables gain a per-row `reading` cell that prints `flat (borderline)`.
- **A lane whose reading is richer than separated/flat supplies its own states.** The embedder
  adoption bar reads `answer` / `rank only` / `neither`, so it computes its reading at each of the
  three levels and hands them to `stability_from_readings`, producing the identical
  `ReadingStability` shape ([the scoped first-hit-rank bar](first-hit-rank-adoption.md#the-scoped-first-hit-rank-adoption-bar)).
- The confidence conventions, metrics, percentile intervals, and row rankings stay unchanged. The
  separation rule changed from the percentile lower bound to the calibrated p. Tests:
  `tests/llb/rag/fusion_evidence/test_paired_stability.py` (the shared annotation, the assembly, the
  rendering, the clause) and `tests/llb/rag/fusion_evidence/test_paired_stability_lanes.py` (each of
  the four lanes qualifying its own verdict).

The three remaining uncertainty shapes now state their own distance from their actual cut instead
of borrowing a paired-delta interpretation that does not fit:

- `bootstrap_ratio` returns a `BootstrapRatio` whose optional `stability` block is read from the
  SAME sorted draw as its interval. Route precision and recall therefore carry `p_positive`, the
  90% / reporting / 97.5% readings, and `borderline`, but no discordant count or exact-sign gate:
  a count ratio is not a paired delta. The sidecar-free calibration renders these rows through the
  shared `boundary_table(..., evidence_counts=False)` path.
- Query robustness is genuinely paired but directional. Its new
  `query_robustness/uncertainty.py` seam reads `improved` / `degraded` /
  `indistinguishable` at all three levels, gates either directional claim on the exact sign-test
  minimum, and records the ordinary interval / win-loss ledger / `p_positive` shape. Both
  lane-versus-clean deltas and mitigation-versus-off recoveries are computed for all items and the
  affected subset. `query_robustness/summary.py` can rebuild the aggregates from persisted case
  rows without executing a model, while `query_robustness/report.py` renders the paired table.
- The measurement floor is not a sampling CI: `clears_floor` compares the leader's fixed recall
  gap with the widest score-jitter band. Its honest continuous signal is therefore `clearance =
  delta - floor` plus `floor_multiple = delta / floor` (or null when the measured floor is zero),
  not `p_positive`. Both fields are additive in `noise_floor.margin` and appear in the shared
  ASCII/Markdown reading. `fragile_items` remains a descriptive count with no pass/fail cut; the
  margin is the floor lane's only binary reading.

Recorded-artifact re-render on 2026-07-26 rebuilt the sidecar-free calibration from its gold-item
order, both current query-robustness reports from their persisted per-case rows, and the graph
fusion floor from its stored lane bands. All pre-existing point estimates, interval bounds,
aggregate table cells, thresholds, and decisions reproduced exactly. Of 38 route precision/recall
rows, 2 are borderline; each 100-reading query report has 2 borderline rows and no
minimum-evidence relabeling. The recorded graph-fusion leader's +0.0105 recall gap against a
+/-0.0211 floor is now stated as -0.0105 clearance, or 0.50x the floor. Current implementation
coverage lives in `test_paired_stability.py`, `test_fusion_calibration.py`, `test_noise_floor.py`,
`tests/llb/eval/test_query_robustness_run.py`, and
`tests/llb/eval/test_query_robustness_variants.py`. Host validation completed `make ci` with 2,221
tests passing and 45 opt-in/slow tests deselected, plus `make lint-md`.

Re-render evidence (2026-07-25, from the recorded artifacts on disk, no new run): every recorded
paired block of the three lanes whose artifacts persist per-item values was rebuilt with the same
resamples / confidence / seed and diffed field by field -- **1222 blocks over 6 fusion sweeps, 3
answer-quality comparisons, and 9 context ablations reproduced their recorded interval, ledger, and
verdict decision exactly**, with only the additive `stability` blocks and the qualified reasons
new. 139 of those blocks (11.4%) are borderline. The bake-off's recorded artifacts persist
aggregates without per-item vectors, so they cannot be rebuilt from disk; the invariance there is
the same function, checked differentially against the pre-annotation code path over 4000 randomized
configurations spanning every recorded `(n, resamples, seed, confidence)` -- **zero delta
mismatches** -- with a compact sweep of that check kept in CI.

What the annotation found on evidence already recorded:

- **The context ablation's one `rag_pays_off` row is settled, but the verdict above it is not.** On
  `qwen3.6-35b` over the 82-item UA fixture the retrieval uplift is +0.421 `[+0.333, +0.503]` at
  `p_positive` 1.000 -- decisive. The LONG-CONTEXT delta on the same run is +0.060 `[-0.008, +0.130]`
  at `p_positive` 0.960, which a 90% interval would read as separated. Since `_judge` checks the
  long-context lane FIRST, that run's `rag_pays_off` is one convention away from
  `long_context_wins`, and the reason now says so. The other eight recorded ablations are settled.
- **Three of six recorded fusion sweeps now qualify their `inconclusive`.** The deciding gain
  metric sits on the cut in the base graph-weight sweep, the candidate-depth sweep, and the
  noise-floor re-read; the three `adopt` sweeps have borderline rows elsewhere in the grid but
  a settled winner, so their reasons stay unqualified -- the clause is scoped to the deciding row
  rather than fired on any unsettled row anywhere.
- **Two of three recorded answer-quality comparisons qualify their `retrieval_only`.** The
  coverage-versus-objective split that verdict rests on is itself a near-miss in the drafted
  multi-hop bundle and the overlap-policy re-run.

## Audit of the `lo > 0` cut itself

2026-07-26. The borderline flag says how close a reading sits to the cut. This audit asked what the
cut
COSTS -- its false-positive rate, whether the exact test in the same block could even reach the
reporting level, and what selecting a row out of a grid does to it. Method: a paired sign-flip
(randomization) null over the per-item delta vectors the recorded artifacts persist, applied
JOINTLY across a family so the real cross-row correlation survives, with the repo's own resample
index sets, seed, and nearest-rank percentile convention; 4000 flips for the rate studies and
20000 for the selection study. Read-only, no new inference. The reachability finding below is now
shipped behavior ([the minimum-evidence gate](#the-minimum-evidence-gate-on-a-paired-reading));
the per-test size finding is now enforced by
[randomization-calibrated paired readings](#randomization-calibrated-paired-readings); selection
control is now enforced by
[selection-adjusted grid verdicts](#selection-adjusted-grid-verdicts).

- **Per-test size.** On the 20 recorded adoption cells (n=40) the one-sided `lo > 0` cut fires on
  **3.0%-7.8% of null draws (mean 4.3%)** against the 2.5% its 95% two-sided interval implies, and
  the inflation tracks the DISCORDANT-item count rather than n: the cells with 7-9 non-tied items
  land at 0.062-0.078, those with 24-27 at 0.036-0.037. On the 82- and 250-item ablation lanes it is
  0.024-0.036, near nominal. On synthetic symmetric deltas it is at or below nominal at every size,
  so the inflation is driven by SKEW plus sparsity, which is exactly the regime a token-F1 delta on
  a small accepted ledger sits in.
- **Reachability.** Across 7408 paired blocks in the recorded fusion sweeps, adoption sweeps,
  context ablations, and answer-quality comparisons, 971 read `separated`, and **719 of those (74%)
  carry fewer than 6 discordant items** -- the point where the exact two-sided sign test the same
  block reports (`2 * 0.5^d`) cannot reach 0.05 under ANY arrangement of the data. Most are slice
  rows: 576 of the 712 separated fusion SLICE readings come from slices of <= 5 items, where a
  2-item slice with 2 wins prints `+1.000 [+1.000, +1.000]` beside its own `sign_test_p` of 0.5.
- **Selection.** A fusion sweep publishes 408-3048 paired cuts (17-127 rows x 4 metrics x slices)
  and then reads ONE selected `best_row`. Re-read under a Westfall-Young max-statistic sign-flip
  null over the family the row was selected from, the three recorded `adopt` sweeps' deciding row
  (`fused/global_community@0.30/d50/ioverlap`, focus slice n=35) gives raw randomization p 0.063
  (recall@k), 0.031 (span coverage), 0.032 (MRR), and **FWER-adjusted p 0.14-0.29** -- so neither
  the selection adjustment nor the unadjusted randomization test reproduces the recorded reading at
  the same 2.5% level. The `any cell clears zero` rule of the adoption sweep fires on 13.7%-16.4% of
  null draws over its 4 cells, and 44.1% over the 20-cell roster (0.82 expected false positives per
  roster). The maintained implementation and full recorded-artifact re-read are below.

The headline verdicts of the two dense-only lanes are not implicated: the ablation lanes are near
nominal at their sample sizes, and their deciding rows carry 16-181 discordant items.

## Randomization-calibrated paired readings

2026-07-28. `src/llb/rag/fusion_evidence/randomization.py` makes the audit's per-row remedy the
shared decision rule. It tests the candidate-ahead mean under sign exchangeability of the non-zero
per-item deltas. Equal-magnitude ledgers use an exact binomial tail at any item count, arbitrary
ledgers with at most 16 discordant items are enumerated exactly, and larger arbitrary ledgers use a
deterministic Monte Carlo tail with the plus-one correction. Each `PairedComparison` persists
`randomization_p`, `randomization_method`, and `randomization_samples`; the same values ride beside
`p_positive` in its `stability` block.

- `separates()` cuts `randomization_p <= (1 - confidence) / 2` and then applies the existing
  minimum-evidence gate. The percentile bootstrap remains the interval estimator: its point,
  bounds, resample count, seed convention, and the exact sign-test ledger are unchanged. Archived
  aggregate-only blocks without a calibrated p retain their historical reading until a
  vector-backed audit can rebuild them.
- `regresses()` is the mirrored reading `separates()` structurally cannot give. The calibrated test
  is ONE-SIDED by construction ("candidate ahead"), so it can never state a LOSS -- and a lane that
  buys one slice by paying for another has to be able to say so. A loss is therefore read off the
  paired interval (`delta.hi < 0`), the same fallback an uncalibrated archived block gets, and it
  carries the same minimum-evidence gate, so a loss resting on three differing items is not
  reported as one either. Its first consumer is the budget-conversion cost scan
  ([GraphRAG](../graphrag-backend/answer-quality-budget-evidence.md#the-retrieval-budget-dimension)).
- The three-state adoption reading (`answer` / `rank only` / `neither`) calibrates its objective
  and reciprocal-rank vectors separately while preserving objective-first order. Directional
  query-robustness rows test both sign-flip directions and persist the p for the observed direction.
  Fusion, answer quality, context ablation, the embedder bake-off, and adoption-bar verdicts all
  reach the shared `separates()` path.
- Reports render `rand p` beside `sign p`; boundary tables render the decision-driving
  randomization p beside diagnostic `p_positive`. The existing borderline qualifier now compares
  the calibrated reading at 90%, the reporting level, and 97.5%.
- The maintained null harness is `tests/fixtures/paired_randomization_null.json` plus
  `tests/llb/rag/fusion_evidence/test_paired_randomization.py`. It checks the implementation against
  independent brute-force enumeration and enumerates every null assignment of three committed
  sparse/skewed fixtures. Their empirical one-sided sizes are 1.5625%, 2.34375%, and 2.34375%, all
  at or below the nominal 2.5%.
- `make audit-paired-readings` reconstitutes vector-backed embedder bake-off, adoption-bar, and
  context-ablation artifacts without model calls, lists every comparison reading that changes,
  and restates every artifact verdict. The CUDA-host re-read is under
  `$DATA_DIR/paired-reading-audit/20260728T-randomization-calibrated/`: the two vector-backed
  artifacts available on this host supplied 19 paired blocks; the recorded bake-off stayed
  `retain`, the context ablation stayed `long_context_wins`, and no per-row reading changed. The
  host inventory contained no adoption-bar comparison bundle to re-read. The audit's fixture test
  covers the previously vulnerable bake-off `adopt` shape and confirms it is restated `retain`
  when its calibrated p is 0.0352.

Host validation used the RTX PRO 3000 Blackwell GPU (12,227 MiB) with the installed
MamayLM-Gemma-3-12B model available. The audit path deliberately made no inference, as its contract
is to re-read persisted vectors. Implementation coverage also includes the shared stability,
minimum-evidence, lane-verdict, adoption-borderline, query-robustness, and artifact-audit tests.

## Selection-adjusted grid verdicts

2026-07-28. Verdicts that SEARCH a grid now carry the error rate of that search. The shared
implementation is `src/llb/rag/fusion_evidence/selection.py`: it applies one aligned item-level
sign-flip draw to every hypothesis in a declared family, computes a studentized max statistic, and
returns Westfall-Young STEP-DOWN adjusted p-values. Joint signs preserve the measured cross-row
correlation. Item columns that are zero for the whole family are removed before choosing exact
versus Monte Carlo inference; up to 16 active items are enumerated, while larger families use at
least 20,000 deterministic draws with the plus-one correction. The adjusted p is cut at the same
`(1 - confidence) / 2` one-sided alpha as the per-row calibrated p.

The lane declarations are narrow and explicit:

- The fusion verdict selects from every non-baseline fused/routed row x all four metrics on its
  FOCUS slice. Its `best_row` ranking is unchanged, but a recall/all-spans gain must clear both its
  ordinary `PairedComparison` and the selected family.
- The adoption-bar verdict adjusts the objective delta over every scored cell because only
  `objective_score in ANY cell` can trigger `extend_bar`. Its rank-only evidence remains a per-row
  reading: it can retain the existing bar or say the premise was absent, but cannot extend it.
- The embedder bake-off adjusts every non-baseline candidate x ENABLED adoption bar. A candidate
  enters `separated`, `cleared`, and the adopt ranking only when its bar survives that family.

Each verdict persists an additive `selection_adjustment` block with the method, statistic,
exact/Monte-Carlo provenance, draw count, seed, item count, family size, and each hypothesis's
marginal and adjusted p. The adoption and bake-off verdicts also preserve their pre-adjustment
positive cells/candidates in `per_row_answer_cells` / `per_row_cleared`; the existing row tables
continue to print the calibrated per-row p. Lane adapters live in
`fusion_evidence/selection_family.py`, `eval/embedder_adoption/verdict.py`, and
`embedding_bakeoff/selection.py`.

CUDA-host re-read: `make audit-paired-readings
PAIRED_READING_AUDIT_OUT=$DATA_DIR/paired-reading-audit/<run>` produced
the selection-adjusted grid re-read. It reconstituted 14
vector-backed grid artifacts and 7,129 paired blocks:

| lane | grid artifacts | selected family survives | adjusted verdict |
| --- | ---: | ---: | --- |
| fusion sweep | 6 | 0 | three historical `adopt` calls become `inconclusive`; three stay `inconclusive` |
| adoption bar | 6 | 1 | five full sweeps do not survive; the one-cell confirmation stays `extend_bar` |
| embedder bake-off | 2 | 0 | both canonical corpus runs stay `retain` |

The fusion row that would decide an `exact` to `overlap` default change,
`fused/global_community@0.30/d50/ioverlap`, has a family-draw marginal recall p of 0.0628 and a
step-down adjusted p of **0.2310** in the span-identity, routing, and merge-ratio grids. It
therefore ALSO fails selection even if a larger accepted item set removed the current
minimum-evidence failure. No shipped default changes on this reading.

The adoption result shows why the family matters. MamayLM-12B's best full-grid objective cell moves
from marginal p 0.0145 to adjusted p 0.0438; Mistral's two strongest cells move from 0.0184 and
0.0045 to 0.0560. The one-cell MamayLM confirmation has no search multiplicity and stays at
adjusted p 0.0157. On the regenerated 250-item fixture bake-off, E5-large recall moves from a
family-draw marginal p of 0.0307 to adjusted p 0.0458 and remains `retain`; the accepted 40-item PDF
run is flatter still.

Seven historical bake-off JSON files persist aggregate intervals but no aligned `paired_items`, so
their cross-candidate correlation is unrecoverable rather than guessed. The audit lists each one as
legacy and uses fresh vector-backed runs of the same canonical 250-item fixture and 40-item accepted
PDF corpus. `make compare-embeddings CONFIG=<config>` now lets the config own its goldset and split
unless either is explicitly overridden on the make command line; this prevents the repository-wide
fixture and `final` defaults from silently changing a recorded family.

Coverage is in `tests/llb/rag/fusion_evidence/test_selection_adjustment.py` plus the fusion, adoption-bar,
bake-off-verdict, and paired-reading-audit suites. The shared procedure is checked against an
independent brute-force family; lane fixtures include a case where every per-row reading clears but
the family-wise verdict does not. Host validation used an NVIDIA GeForce RTX 4060 Ti (16,380 MiB,
CUDA 13.2), regenerated both bake-offs through their Make configs, and completed `make ci` with
2,354 tests passing and 45 slow/opt-in tests deselected.

## The minimum-evidence gate on a paired reading

The reachability finding above is a rule, not a caveat, and it ships as one. A paired block's
evidence is its DISCORDANT items -- the pairs where the two lanes actually differ, since ties carry
no information about direction -- and the exact two-sided sign test each block already prints beside
its interval bounds what `d` of them can ever show. Its smallest attainable p is `2 * 0.5^d` (every
pair falling the same way), so below that many items the reporting level is unreachable whatever the
interval says. `src/llb/rag/fusion_evidence/evidence_gate.py` is the one place that rule lives:

- **The bound is derived from the reporting confidence, so no constant is introduced.**
  `minimum_discordant_pairs` solves `2 * 0.5^d <= 1 - confidence`: **5 items at 90%, 6 at 95%, 7 at
  97.5%**. `apply_evidence_gate` relabels a claim resting on fewer than that
  **`insufficient_evidence`**, a state no lane may adopt on. Only a CLAIM is gated: a `flat` reading
  (or a richer lane's `neither`) says nothing was found, which a thin item set is entitled to say.
- **One separation test, used by every verdict.** `separates()` in `paired.py` is
  `randomization_p <= (1 - confidence) / 2 AND the block differs on enough items`, and it is used in
  every lane that had one: the fusion sweep's adopt, the bake-off's adoption bars, the ablation's
  long-context and uplift checks, the answer lane's gain and retrieval-only checks, the adoption
  bar's per-cell reading, the sidecar-free routing gate's coverage half (its single-span half is a
  non-inferiority check, not a separation, and is unchanged), and the long-context power
  resolution's direction. Aggregate-only archived comparisons fall back to their historical
  interval reading until their vectors can be reconstituted.
- **The gate and the borderline flag are one scale.** The bound moves WITH the level, so a row that
  differs on exactly 6 items reads `separated` at 95% and `insufficient_evidence` at 97.5% -- which
  is precisely a `borderline` row, `side: above`, and it is now marked as one automatically.
- **Every renderer prints the state.** The per-row `reading` cell of the bake-off, ablation, and
  adoption tables; the boundary block's new `d` column (the discordant count, persisted in
  `ReadingStability`); a `Minimum evidence: N of M paired readings are insufficient evidence` line
  in each lane's report; and the shared `. INSUFFICIENT EVIDENCE: ...` clause on any verdict reason
  whose deciding row the gate relabeled -- the same call shape as the borderline clause.
- The calibrated task changes the separation half; this gate still owns only the evidence-count
  entitlement. A sweep keeps the same percentile intervals and row ranking.

Tests: `tests/llb/rag/test_paired_minimum_evidence.py` -- the derived bound against the sign test's
own arithmetic, the reading at 5 / 6 / 7 discordant items, that the interval / ledger / point
estimate / sign-test p are untouched when the reading is relabeled, that the two discordant counts
(ledger and delta vector) agree, and the verdict guard driven end to end per lane.

Re-read of the recorded evidence base (2026-07-26, read-only, no new inference; harness and output
). Recorded artifacts on disk
were NOT rewritten -- they stay as produced, and this is what the gate says about them:

| lane | artifacts | paired blocks | read `separated` | relabeled |
| --- | ---: | ---: | ---: | ---: |
| fusion sweep | 6 | 7008 | 874 | 695 (79.5%) |
| answer quality | 3 | 210 | 14 | 12 (85.7%) |
| context ablation | 10 | 111 | 73 | 2 (2.7%) |
| adoption bar | 5 | 100 | 27 | 10 (37.0%) |
| embedder bake-off | 4 | 32 | 6 | 2 (33.3%) |

- **No recorded number moves.** Every block whose per-item vectors the artifacts persist (all
  derived ablation deltas, all focus-slice fusion blocks) was rebuilt with that artifact's own
  resamples / confidence / seed and diffed field by field: **0 fields differ** across delta, bounds,
  win/loss/tie ledger, and exact sign-test p. The gate touches the reading string and nothing else.
- **Eight recorded verdicts are restated.** Three fusion sweeps' `adopt` becomes `inconclusive`:
  the deciding row (`fused/global_community@0.30/d50/ioverlap`, focus slice n=35) gains +0.114
  recall@k on **4 differing items** -- the same row the selection study above put at FWER-adjusted
  p 0.14-0.29. Three answer-quality `retrieval_only` verdicts become `no_gain`: their coverage half
  rests on 4-5 differing items. Both committed-fixture bake-off runs' `adopt
  intfloat/multilingual-e5-large` becomes `retain`: +0.020 recall@10 on n=250 is **5 wins and 0
  losses**, which the recorded reading already flagged by hand as unable to reach 0.05 on the exact
  sign test. The gate makes that automatic rather than a footnote. Each of the eight is restated
  with the item count that would settle it in
  [the re-decision](#the-re-decision-what-a-withdrawn-reading-needs) below.
- **The dense-only lanes hold, as the audit predicted.** No ablation verdict changes; the only two
  relabeled ablation blocks are a side `retrieval_hit` reading on one early 4-item run, never a
  deciding uplift. No adoption-bar verdict changes either: the relabeled cells are the `k3` cell's
  `retrieval_hit` and `reciprocal_rank` readings (5 differing items) in all five roster sweeps, so
  `k3` drops out of `rank_cells` while the objective-side `extend_bar` / `keep_bar` calls stand.
- **What the fusion share means.** 79.5% of that lane's separated readings are per-SLICE rows on
  slices of a handful of items; the overall rows and the focus slice (n=35) are where the verdicts
  are read, and only the deciding row named above changes a verdict.

Live confirmation, CUDA host 2026-07-26 (`LLB_EMBED_DEVICE=cuda make compare-embeddings
CONFIG=$DATA_DIR/compare-embeddings/paired-uncertainty-fixture.yaml
GOLDSET=samples/goldsets/ua_squad_postedited_v1/goldset.jsonl SPLIT= NOISE_FLOOR=1`, report in run
the fixture bake-off):
the bake-off whose recorded verdict the re-read downgrades was re-run end to end through the real
encoders. **Every number reproduces bit-identically** -- all four candidates' recall@10, MRR, dim,
indexed count, index bytes, each paired delta bound, each win/loss/tie ledger, each exact sign-test
p, and the whole measurement floor block: 0 fields differ against the recorded `report.json`. The
verdict is the only thing that moved, `adopt` -> **`retain`**, and the reason states both qualifiers
at once: `e5-large`'s recall bar differs on 5 items (below the 6 needed at 95%), and it is
borderline because a 90% interval -- which needs only 5 -- would read it `separated`.

## The re-decision: what a withdrawn reading needs

A withdrawn reading is an OPEN question, not a measured absence of a difference, and the two must
not print alike: two of the eight verdicts the gate downgraded are recommendations an operator may
still be acting on. Every withdrawn row therefore states what would settle it, from the same
arithmetic the bound comes from. `resolving_item_count` in
`src/llb/rag/fusion_evidence/evidence_gate.py` inverts the gate: at the block's own discordance
RATE, `d` differing items out of `n` extrapolate to `minimum_discordant_pairs(confidence) * n / d`
items before the level becomes reachable at all.

- **It is a floor on the ITEM SET, not a detectable effect.** Below it no arrangement of the data
  could reach the level; above it only effects large enough to survive the interval are resolved.
  Reading it as "run this many items and the question is answered" is the one misreading to avoid
  -- that is what an a priori MDE contract prices, and the repo prices it separately.
- **It moves with the reporting convention, exactly as the bound it inverts does** (4 of 35 needs
  44 items at 90%, 53 at 95%, 62 at 97.5%), so nothing new is tuned.
- **Every renderer carries it.** The shared `. OPEN QUESTION: ...` clause rides on the
  insufficient-evidence clause in every lane's verdict reason, so no lane can withdraw a reading
  without stating its price; the boundary table gains an `n to reach` column (backed by the new
  `pairs` field in `ReadingStability`, additive and absent on artifacts recorded before it); and
  the `Minimum evidence:` line says a relabeled row is an open question rather than "no difference".
  The column prices ANY row short of the bound, including a `flat` one -- a reading that could not
  have shown a difference is not the same as one that looked and found none.

Re-decision of the recorded evidence base (2026-07-26, read-only, no new inference; harness and
). All eight
withdrawn verdicts, restated with what each one needs:

| lane | runs | recorded -> re-read | deciding row | differs | floor | what would settle it |
| --- | ---: | --- | --- | ---: | ---: | --- |
| fusion sweep | 3 | `adopt` -> `inconclusive` | `fused/global_community@0.30/d50/ioverlap` recall@k, multi-hop | 4 of 35 | 53 | a multi-hop slice of >= 53 ACCEPTED items |
| answer quality | 2 | `retrieval_only` -> `no_gain` | `fused/global_community@0.10/d10` span coverage, multi-hop | 4 of 35 | 53 | the same accepted slice |
| answer quality | 1 | `retrieval_only` -> `no_gain` | `routed/global_community@0.30/d50/ioverlap` span coverage | 5 of 35 | 42 | the same accepted slice |
| embedder bake-off | 2 | `adopt` -> `retain` | `intfloat/multilingual-e5-large` recall@10 | 5 of 250 | 300 | nothing this repo has -- undecidable at these sample sizes |

- **The three fusion sweeps and the three answer-quality comparisons ride on one item set**, the
  drafted multi-hop slice of 35. The floor says the accepted slice must reach 53 items (42 for the
  routed coverage row), and human acceptance can only SHRINK a drafted ledger -- so the re-run
  planned as `multihop-ledger-human-acceptance` cannot resolve the span-identity row at the current
  drafted size whatever the reviewer decides. Widening the drafting is the prerequisite, and that
  is stated where the work is tracked in [`plan.md`](../../plan.md).
- **The encoder question is undecidable at the sample sizes this repo has, and that is now
  recorded.** `e5-large` leads `e5-base` on 5 of 250 committed-fixture items (0 losses, 245 ties);
  at that 2% rate the level needs 300 items, and the largest committed goldset is 250
  (`ua_squad_postedited_v1`) -- the next largest is 60. The route out is not simply "more items":
  `embedder-decision-on-a-resolvable-item-set` enriches the ledger with questions the incumbent
  MISSES, which raises the discordance rate and lowers the floor proportionally (double the rate,
  halve the items). Until such a ledger is accepted, the recommendation stands on the incumbent by
  `retain`, not by evidence of equivalence.
- **Census over the whole recorded base**: 723 withdrawn readings, all priced (median floor 6
  items, min 6, max 300). 719 of them sit at or below the largest committed goldset -- most are the
  handful-of-items fusion slice rows, which need only a few more items each. The 4 that do not are
  the same `e5-large` recall bar across every re-run of the fixture bake-off, which is exactly the
  one open question this repo cannot currently close.
- **No number, threshold, interval, or adoption rule changed.** The re-decision is a statement
  appended to readings the gate had already withdrawn.

Tests: `tests/llb/rag/test_paired_open_question.py` -- the floor as the smallest item count whose
rate reaches the bound (and that one item fewer does not), the two unpriced cases (already
reachable, nothing differing), the movement across the three conventions, the shared clause, the
table column including the archived-artifact fallback, the recorded prices the table above quotes,
that no committed goldset reaches the encoder floor, and the withdrawn verdict of each affected
lane naming its own item count end to end.

Live confirmation, CUDA host 2026-07-26 (same command as above):
the fixture bake-off was re-run end to end through the real encoders once more. **Every retrieval
number reproduces bit-identically** against the pre-re-decision run -- 0 fields differ across all
four candidates' recall@10, MRR, dim, indexed count, index bytes, every paired delta bound, every
win/loss/tie ledger, every exact sign-test p, and the whole measurement-floor block. Three fields
move, none of them a measurement of retrieval: the wall-clock embed seconds (and the chunks/s
derived from them), the `pairs` count now persisted in each `stability` block, and the verdict
reason, which now ends with the open-question clause naming
`intfloat/multilingual-e5-large recall_at_k`, its 5 of 250 differing items, and the 300-item floor.
The verdict itself stays `retain`.
