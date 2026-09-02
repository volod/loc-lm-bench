# Measured Claim-Tier Precision (corpus-conflict-audit)

The semantic tier's cutoff is a RANK, not a false-positive guarantee, and the corpus alone cannot
supply the independent null a real rate would need ([conflict
detection](conflict-detection.md#known-limitation-there-is-no-independent-null)). This page holds
what the audit measures INSTEAD: the share of the returned candidate list that survives claim
adjudication, the clustered bound that earns it, the frozen probe that decides whether an
adjudicator may be quoted at all, and the optional cross-encoder ordering that changes what gets
adjudicated first. The detector itself is on [conflict detection](conflict-detection.md).

## Measured claim-tier precision

What the rank cutoff cannot say, the claim tier can: `--effort claim` adjudicates every returned
candidate row anyway, so the share of THAT list which survives adjudication is measurable at a
sample size equal to the adjudication budget rather than to the pair space. `summary.json` carries
it as `claim_precision` and `report.md` renders a **Claim-tier precision** section. It is still not
a false-positive rate over the corpus -- it is the precision of the list the operator was handed.

Four properties are what make the number publishable, and each is enforced rather than assumed:

- **Rank order.** `detect_semantic_pairs` returns candidates sorted by descending cosine (ties on
  chunk ordinals), so a prefix of the list is the TOP of the corpus's own similarity ordering.
  Both `--max-claim-pairs` and the precision curve's budgets read a prefix; before this they read
  whatever order the tree traversal produced.
- **A two-way clustered bound.** Rows that share a left or a right chunk are not independent
  evidence, so the lower bound is `two_way_proportion_bound` (`statistics/clusters.py`) --
  literally the estimator the independent-null research established, imported rather than
  reimplemented, with `tests/llb/conflicts/claim/test_claim_precision.py` asserting the audit's curve
  equals the research lane's curve on the same rows. The shared helpers live in
  `src/llb/conflicts/claim/precision.py`; `interval_stats.py` holds the Wilson interval both sides
  use.
- **A budget sweep for free.** Rank order also makes the precision curve a genuine
  candidate-budget sweep over one adjudicated list, so `budget_resolution` can name the smallest
  budget whose clustered bound clears zero without paying for a run per budget.
- **A calibration gate.** Before measuring, the audit adjudicates a COMMITTED frozen-label probe
  with the same prompt and the same endpoint, and suppresses the whole block -- with the reason,
  naming the tier that gated -- when the model does not clear it. A precision figure computed from
  a model's own verdicts is otherwise only as good as the model.

An unparsable verdict is kept as a row and counted as NOT actionable, so it biases the figure
downward; the printed precision is therefore a lower bound whenever `unparsed_rows` is non-zero.
The block is suppressed only when unparsed rows exceed `unparsed_allowance` (5% of the list, floor
1), because at a 12-row budget a single malformed completion would otherwise erase a usable
conservative measurement.

### The frozen calibration probe

`samples/corpora/conflicts_uk_v1/adjudicator_probe.json` is committed, and it carries **two tiers**
because one difficulty cannot answer two questions. A probe every model passes proves only that an
adjudicator is not badly broken; a probe that separates two working adjudicators is a different
artifact, and mixing them into one accuracy number hides which of the two a reading came from.

| tier | corpus | pairs | actionable / complementary | what it asks | gates |
| --- | --- | --- | --- | --- | --- |
| `base` | `conflicts_uk_v1/corpus` | 24 | 12 / 12 | is this adjudicator broken? | yes, at 0.60 |
| `hard` | `conflicts_uk_v1/probe_hard` | 16 | 8 / 8 | which of two working adjudicators is better? | no |

**The base tier** is drawn from the planted detector fixture: the changed deadline, the restated
sections, the byte-identical re-upload, the reformatted reissue, the absorbed note, the vague
restatement, against the unrelated archive control and cross-section pairs that state different
compatible facts. Its answers are plain on a shallow reading, which is exactly what a floor needs.

**The hard tier** has its own five-document corpus, authored so that each pair's split is arguable
on a shallow reading and determinate on a close one. Its actionable half restates one fact under a
different heading, in different units (two working weeks against ten working days), or with only
one clause of two changed under a heading the revision widened; its complementary half puts two
numeric claims about different quantities into the same sentence shape -- thirty calendar days to
submit against thirty calendar years to retain, an advance and a balance that share one
five-working-day window, permanent retention against a five-year schedule. Every pair records the
shallow reading it exists to catch in its `trap` field.

The hard tier is a separate corpus rather than more sections of the fixture on purpose: the fixture
plants one instance of every relation the DETECTOR must find, each with a known answer, and an
arguable pair is precisely what it must not contain. `probe_hard/` is never audited, only
adjudicated, so the detector's measured properties over the fixture are untouched.

Both tiers store `doc_id` + heading line, never passage text, and
`src/llb/conflicts/claim/probe.py` resolves each side to the exact corpus bytes at run time. A
fixture edit that moves the text fails the run instead of silently leaving a frozen label attached
to a passage that no longer exists. No two pairs, in either tier, present the same prompt.

Agreement is scored on the **actionable binary**, not on the exact relation: a duplicate reported
as `subsumes` sends the operator to the same decision, while a conflict reported as `complementary`
is exactly the error a precision figure would hide. `src/llb/conflicts/claim/calibration.py` scores
each tier separately and the whole probe together, and `TIER_ACCURACY_GATES` names the tiers that
decide anything -- currently only `base`, at the `MIN_ADJUDICATOR_ACCURACY_LCB` (0.60) Wilson 95%
lower bound the research lane also applies. A tier absent from that map is measured, reported, and
compared across models without gating; the measurement below is why.

A run that drops the gating tier does not silently pass: with no gating tier present the
calibration reports `no gating tier`, and the precision block stays suppressed.
`--no-calibrate-adjudicator` (Make: `NO_CALIBRATE_ADJUDICATOR=1`) skips the 40 probe calls
altogether and suppresses the block rather than printing it uncalibrated; `--probe-tiers`
(`PROBE_TIERS=`) adjudicates a subset; `--calibration-probe` points at a different probe. A probe
file in the older single-corpus shape (`corpus` + `pairs` at the top level) still loads, as one
`base` tier.

Every measured reading on these pages that quotes agreement with **24** frozen probe pairs was
taken before the hard tier existed and is a base-tier reading; the runs below are the first to
carry both tiers.

### Scoring an adjudicator on its own

Choosing BETWEEN adjudicators is the opposite shape from auditing a corpus: the probe is the whole
measurement, not a prelude to one, and paying for a store and an adjudication budget to reach it
makes the comparison expensive and corpus-dependent. `make calibrate-conflict-adjudicator` runs the
probe alone:

```bash
make calibrate-conflict-adjudicator CONFLICT_MODEL=<host-fit-model> \
  CONFLICT_TEMPERATURE=0 NULL_SEED=0
```

It writes `calibration.json` (the full payload, including every verdict) and `calibration.md` (the
tier ladder, the accuracy delta from the floor tier, and a row per pair the model got wrong) under
`$DATA_DIR/corpus-conflict-calibration/<run>/`. `PROBE_TIERS=base` costs 24 calls instead of 40 when
the separator would tell a run nothing new. The implementation is
`src/llb/cli/prep/conflict_calibration.py` with `src/llb/conflicts/report/calibration.py`;
`tests/llb/conflicts/claim/test_adjudicator_probe.py` covers both tiers' passage resolution, their
label balance, what gates, and the command's artifacts.

### Measured: five model families against both tiers

RTX 4060 Ti 16 GB CUDA host, 2026-09-02, Ollama, temperature 0, seed 0, the committed
`conflicts_uk_v1_adjudicator_probe_v2` (24 base + 16 hard pairs), one
`make calibrate-conflict-adjudicator` run per model -- no corpus and no store, so the probe is the
whole measurement. One model per family, chosen as the host-fit member of each row of the family
register.

| family | model | base agree | base LCB | hard agree | hard accuracy | hard LCB | hard recall / specificity | 40 probe calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mamaylm | MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M | 24/24 | 0.862 | **16/16** | 1.000 | 0.806 | 1.000 / 1.000 | 3 min 10 s |
| mistral | mistral-small3.1:24b | 24/24 | 0.862 | **16/16** | 1.000 | 0.806 | 1.000 / 1.000 | 8 min 12 s |
| qwen | qwen3.8:27b | 24/24 | 0.862 | 15/16 | 0.938 | 0.717 | 0.875 / 1.000 | 25 min 17 s |
| lapa | lapa-v0.1.2-instruct Q4_K_M | 24/24 | 0.862 | 12/16 | 0.750 | **0.505** | 1.000 / **0.500** | 2 min 25 s |
| gemma | gemma4:12b | 2 of 24 parsed | -- | 8 of 16 parsed | -- | -- | -- | 22 min 0 s |

**The base tier separates nothing.** Every family that returned parsable verdicts scored 24/24 on
it -- identical accuracy, identical Wilson bound, four different architectures and two different
Ukrainian lineages. The floor tells an operator that an adjudicator works, and nothing else. Raising
`MIN_ADJUDICATOR_ACCURACY_LCB` on the base tier would therefore change no verdict at any threshold
below 0.862, and reject every model above it.

**The hard tier does separate, and the split is the one it was built to find.** All four of lapa's
misses are complementary pairs it called `subsumed_by`: the report's electronic form against its
quarterly timing, the advance against the balance that share one five-working-day window, permanent
retention against a five-year schedule, and the definition of a budget request against the rule for
submitting one. Its hard-tier specificity is 0.500 against 1.000 on the base tier -- the base tier
had already scored it 12/12 on complementary pairs and called that fine. Qwen's single miss is the
opposite direction: it read the vague deferral to "the term established by the procedure" as
complementary, so its recall is 0.875 and its specificity 1.000.

**gemma4:12b is not an adjudicator on this path at all.** 30 of its 40 completions were not JSON:
it is a reasoning model, the conflict adjudicator sets no `think` control, and the reasoning
consumes the token budget before any verdict is emitted. The gate refused it for exactly that --
`22 of 24 base-tier probe pairs returned an unparsable verdict` -- which is the floor doing its job.
The 10 completions that did parse all agreed, so this is a plumbing result, not a quality one.

**The decision: the gate stays a floor, at 0.60 on the base tier.** `TIER_ACCURACY_GATES` names
`base` and nothing else, and the hard tier is measured, reported, and compared without deciding
anything. Two reasons, and the second is the one that decides it:

- Raising the EXISTING constant is a no-op. The base tier gives every working family the same
  0.862, so no threshold on it can rank them.
- Promoting the hard tier to a gate at the same 0.60 bound WOULD rank them -- it passes mamaylm,
  mistral, and qwen and rejects lapa, whose 0.505 falls below it. But at 16 pairs the pass mark is
  `>= 14/16` (13 of 16 bounds at 0.570, 14 at 0.640), so ONE relabelled pair moves a family across
  it, and this is one host, one seed, one model per family. Refusing to publish a precision figure
  for an adjudicator that clears the floor is not a decision to make from a reading that thin.

What an operator gets instead is the ranking itself: `make calibrate-conflict-adjudicator` prints
the ladder, and an audit's own report now carries a hard-tier line beside the base-tier one, so
choosing between two adjudicators no longer requires two full audits.

**What would license promoting the hard tier to a gate:** the same ordering reproduced on a second
host or a second seed, and a hard tier wide enough that the pass mark is not decided by one pair.
**What would overturn the reading above:** a relabelling of any of lapa's four misses -- each is
frozen with the shallow reading it exists to catch, and if a careful reader disagrees with one of
those labels, the separation it produces goes with it.

### The budget that buys a non-zero floor

A precision figure whose lower bound is 0.0 tells an operator nothing about how much of the list is
real, so the question the block has to answer is which candidate budget first buys a floor above
zero. That sweep is **free**: because candidates come out in rank order, the top-N list is a PREFIX
of the top-M list, so ONE adjudicated run at the widest budget measures every smaller budget on
exactly the same rows. `precision_curve` is the sweep and `budget_resolution` names the answer --
no run per budget, and no re-adjudication noise between points.

What the curve is really reading is the census beside it. `actionable_left_clusters` /
`actionable_right_clusters` count the distinct chunks the ACTIONABLE rows sit on, and that is what
decides whether the bound can clear zero at all: a two-way resampled draw returns zero whenever it
misses every chunk carrying a conflict, so the floor stays pinned at 0.0 while the conflicts are
concentrated, however high the point estimate reads.

### Measured, both quickstart corpora

CUDA host, RTX 4060 Ti, multilingual-E5 stores, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M adjudicating,
`MAX_CANDIDATE_PAIRS=100` (HR: 1 min 50 s for the probe, 10 min 54 s for 100 rows). The adjudicator
agreed with all 24 frozen probe pairs on both runs (accuracy 1.000, Wilson 95% lower bound 0.862,
recall 1.000, specificity 1.000), so the block was reported on both.

**HR** (8 docs, 2,432,676 comparable pairs, resolved cosine 0.5087, rows spanning 0.509 - 0.801,
1 unparsable verdict inside the 5-row allowance):

| budget | actionable | precision | Wilson 95% | two-way clustered LCB | left / right chunks | actionable left / right |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 4 | 0.667 | [0.300, 0.903] | 0.000 | 5 / 5 | 3 / 3 |
| 12 | 5 | 0.417 | [0.193, 0.680] | 0.000 | 9 / 9 | 4 / 3 |
| 25 | 6 | 0.240 | [0.114, 0.436] | 0.000 | 17 / 19 | 5 / 3 |
| 50 | 8 | 0.160 | [0.083, 0.285] | **0.022** | 37 / 33 | 7 / 5 |
| 100 | 13 | 0.130 | [0.078, 0.210] | **0.042** | 69 / 56 | 11 / 10 |

**goods** (5 docs, 71,736 comparable pairs, resolved cosine 0.3648, rows spanning 0.365 - 0.539,
no unparsable verdicts):

| budget | actionable | precision | Wilson 95% | two-way clustered LCB | left / right chunks | actionable left / right |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 3 | 0.500 | [0.188, 0.812] | 0.000 | 3 / 6 | 1 / 3 |
| 12 | 4 | 0.333 | [0.138, 0.609] | 0.000 | 7 / 12 | 1 / 4 |
| 25 | 4 | 0.160 | [0.064, 0.347] | 0.000 | 14 / 22 | 1 / 4 |
| 50 | 6 | 0.120 | [0.056, 0.238] | 0.000 | 28 / 34 | 3 / 6 |
| 100 | 8 | 0.080 | [0.041, 0.150] | 0.000 | 42 / 51 | 3 / 8 |

**HR resolves at budget 50, which is already `SUGGESTED_MAX_CANDIDATE_PAIRS`.** The shipped
suggestion is therefore measured rather than assumed, and no default changes. Budget 12 -- the
value the earlier evidence in this page used -- is now known to sit below the resolving budget on
both corpora.

**Precision and the floor move in OPPOSITE directions.** HR's point estimate falls 0.667 -> 0.130
while its bound rises 0.000 -> 0.042. That is not a contradiction: precision is a point estimate
over a list whose tail is mostly `complementary`, while the floor is limited by how many distinct
chunks carry conflicts. Spending more adjudication buys certainty about a smaller share, and an
operator choosing a budget is choosing between those two, not maximizing one number.

**goods never resolves, and the census says why it is not a budget problem.** All 8 of its
actionable rows share the SAME right document, and 6 of the 8 share ONE left chunk
(`pdf-6c325abf4b92.md#recursive#0003`). Eight rows is one chunk against one document, so widening
the budget adds complementary rows without adding independent evidence -- `actionable_left_clusters`
is stuck at 1 through budget 25 and reaches only 3 at budget 100. The pair-row Wilson interval at
budget 100 is `[0.041, 0.150]`, a non-zero floor it has not earned; the clustered bound refuses it.
That contrast is the whole reason the audit quotes the clustered bound.

### Optional cross-encoder claim prefilter

`CLAIM_PREFILTER=1` scores every semantic candidate with the pinned
`BAAI/bge-reranker-v2-m3` scorer. An uncapped run keeps cosine prompt order, labels the full list,
and measures the cross-ranked prefix for the same actionable pairs; a reducing `MAX_CLAIM_PAIRS`
spends that prefix. Flat or unresolved scores, non-monotone bins, and no positive rank delta all
recommend the full list. Unadjudicated rows remain provisional findings in the complete ledger.
Scores only order rows; they are never probabilities, rates, confidence values, or verdicts.

```bash
make audit-corpus-conflicts CORPUS=<corpus-dir> EFFORT=claim STORE=<store-dir> \
  CONFLICT_MODEL=<host-fit-model> CONFLICT_TEMPERATURE=0 NULL_SEED=0 \
  MAX_CANDIDATE_PAIRS=50 CLAIM_PREFILTER=1 CLAIM_PREFILTER_DEVICE=cpu
```

The implementation is in `claim/prefilter.py` and `tiers/claim_run.py`, with CLI/Make controls,
report rendering, injected-scorer tests, an exact flat fallback, and a regression keeping uncapped
prompts identical to baseline. `summary.json` records the scorer/device, both ranks, every score
and label, bins, actual adjudication order, omitted rows, saving, fallback, and recommended budget.

Measured 2026-09-01 on the RTX 4060 Ti 16 GB CUDA host with the 12B
MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M Ollama adjudicator on CUDA, the cross-encoder on CPU,
temperature 0, seed 0, the 24/24 probe, centered multilingual-E5 stores, and no gold set:

- **HR:** 8 docs, 2578 chunks, 2,432,676 comparable pairs, cosine 0.5349, and 50 candidates. The
  uncapped baseline and scored run produced identical claim findings: eight actionable pairs and
  no unparsed rows. Cross-score bin fractions `[0.083, 0.083, 0.083, 0.250, 1.000]` were monotone;
  the last actionable pair moved from cosine rank 44 to cross rank 42. The 42-call validation kept
  all eight pair identities and recorded eight provisional rows. Versus the uncapped 50-call
  baseline, it avoided eight calls; two calls are attributable to cross rank versus the equivalent
  cosine prefix of 44. Its 333.479 s claim phase plus 23.815 s scoring was 25.118 s (6.6%) below
  the baseline claim phase's 382.412 s.
- **goods:** 5 docs, 1099 chunks, 71,736 comparable pairs, cosine 0.3907, and 50 candidates. Both
  runs produced identical claim findings: five actionable pairs plus the same one unparsed row.
  Its bins `[0.000, 0.000, 0.083, 0.333, 0.000]` were non-monotone, so 22.378 s of scoring licensed
  zero saved calls and the artifact recommends all 50 rows.

The reading is corpus-local and modest: the HR order buys two incremental calls and a small net
runtime reduction; goods safely falls back. A changed corpus/store, scorer, or adjudicator can
overturn it, as can a repeated full-list pair where an actionable baseline identity is absent or
the HR bins cease to be monotone. Re-evaluate the complete list before adopting a new cap.
