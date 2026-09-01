# Corpus Hygiene: Conflict Detection (corpus-conflict-detection)

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

This lane owns DOCUMENT-level duplication and contradiction, which needs a human decision. The
chunk-level counterpart is automatic and separate: exact-duplicate chunk text inside one index is
collapsed at build time ([duplicate chunk
collapse](../rag-core/retrieval-store.md#duplicate-chunk-collapse)), which changes no corpus byte
and reports its rate in `store_meta.json`.

`llb audit-corpus-conflicts` (`make audit-corpus-conflicts CORPUS=<dir> EFFORT=<tier>`) reports
duplicated, stale, and mutually inconsistent knowledge in a corpus. It is **detection only**: no
tier edits, deletes, or reorders a corpus byte, and a CI test asserts the corpus is unchanged after
a run. Implementation lives in `src/llb/conflicts/`, Typer wiring in `src/llb/cli/prep/conflicts.py`,
Make orchestration in `make/data-prep/corpus.mk`.

## Effort tiers

Four cumulative `--effort` tiers; each settles what it can so the next has less to look at.

| tier | mechanism | needs | cost on the 8-doc / 2578-chunk HR corpus |
| --- | --- | --- | --- |
| `hash` | content sha, raw and Ukrainian-normalized | nothing | 0.26 s |
| `lexical` | word 5-gram shingles: Jaccard + containment | nothing | 0.51 s |
| `semantic` | chunk-vector pair search over a built store | a store | 1.5 s |
| `claim` | local-model adjudication of surviving pairs | a store + a model | 77 s / 11 pairs |

The `claim` tier costs one model call per surviving candidate plus 24 calls for the frozen
adjudicator-calibration probe that earns the [measured precision
block](#measured-claim-tier-precision) (about 105 s on the CUDA host; skip it with
`--no-calibrate-adjudicator`).

`hash` splits duplicates into `raw` (byte-identical) and `normalized` (identical after casefold,
whitespace, punctuation, apostrophe unification, and front-matter removal) -- the second is the
re-ingested-edition case. Content hashing is deliberately **not** `corpus_doc_fingerprints`, which
folds the governance contract into each document's hash: that is right for refresh and wrong here,
since two byte-identical documents carrying different `effective_date` values must still read as
duplicates.

Duplicate groups are transitive, so the tier reports `n-1` chained pairs for a group of `n` but
marks the group's **full pair closure** as settled. Without that split the later tiers re-derive
(and re-report) the pairs the chaining left implicit.

`--linkage` (`LINKAGE=1`) adds an opt-in lane BESIDE these two tiers rather than inside them: it
prices their duplicate and subsumption evidence as one match probability per document pair and
clusters the result into edition groups, leaving every finding, relation, and threshold exactly as
it is. It runs where the lexical tier runs, needs the `linkage` extra, and declines below 20
documents. The lane, what it measures against the current cutoffs, and its runs are documented in
[the document-edition lane](../entity-resolution.md#the-document-edition-lane); its two settings
that a reader of this page will meet -- the prior taken from the hash tier's settled pairs, and the
pseudo-count floor without which the ranking below the top class collapses -- are documented there
too.

## Relation vocabulary

Relations are assigned per **claim pair**, never per document: `duplicate`, `subsumes` /
`subsumed_by`, `contradicts`, `superseded_by`, `complementary`. That is what makes partial
supersession representable -- a revision that changes one fact while restating another produces a
`superseded_by` for the first and a `duplicate` for the second, from one document pair.

`superseded_by` is **derived, never asked for**. The adjudication prompt shows the model two
passages and no provenance, so it cannot rationalize a verdict from dates; a `contradicts` verdict
is promoted to `superseded_by` only when the governance fields order the two sides, with side `a`
always the deprecated claim. An undated contradiction stays an honest `contradicts` for a human.

Every finding carries exact character offsets on both sides. The claim tier narrows a finding to
the span the model quoted (via `ground_span`); a quote that cannot be located falls back to the
enclosing chunk and is marked `offsets_exact: false` rather than pointing at text that is not
there. On the HR evidence run 10 of 11 findings narrowed exactly.

## What the semantic tier excludes, and why

Three classes of chunk never pair, all learned from real corpora rather than anticipated:

- **Front matter** -- every ingested document's governance block shares the same keys, so an
  archiving instruction and an appeals regulation match at cosine 0.9 on their `version:` and
  `language:` lines alone.
- **Low-content chunks** (`--min-claim-tokens`, default 25 content tokens, HTML comments stripped)
  -- a converted PDF corpus is full of `<!-- source_pdf ... -->` markers, bare page numbers, and
  stub headings. On the HR corpus these were the single largest source of findings before the
  filter: the top-ranked "conflict" was one page marker against another.
- **Repeated structured metadata blocks** -- `semantic_filter.py` groups claim-sized body chunks
  by their normalized deepest Markdown heading, requires that heading at most once per document,
  then confirms each cross-document pair from corpus-derived shared-token coverage and numeric
  field density. The detector has no language- or publisher-specific vocabulary. Repeated claim
  prose under a shared heading stays comparable when it lacks that record-like structure.

The current HR store excludes 90 of 2578 chunks: 88 low-content chunks plus two publication
records. The goods store excludes 87 low-content chunks and no repeated metadata. `summary.json`
retains `excluded_chunks` and breaks it down as `excluded_front_matter_chunks`,
`excluded_low_content_chunks`, and `excluded_metadata_block_chunks`. The same filtered ordinal set
feeds both null-distribution calibration and semantic candidate generation.

## Encoder anisotropy and centering (measured)

Sentence-encoder spaces are strongly anisotropic, and it changes what a threshold means. Measured
over 2578 real multilingual-E5 chunk vectors:

| | random unrelated pair | findings at cosine 0.9 |
| --- | --- | --- |
| raw E5 space | 0.83 (p95 0.88) | 5185 |
| mean-centered | -0.02 (p95 0.26) | 0 cross-document |

A 0.9 "near-duplicate" threshold in raw E5 space sits barely above the similarity of two completely
unrelated chunks, which is why the uncentered run produced an unusable report. `--center-vectors`
(default on) removes the corpus mean direction first. It is skipped automatically below
`MIN_CENTERING_VECTORS` (50) chunks, where the "mean" is an accident of which few documents are
present rather than an estimate -- the audit logs when it does so.

Because centering rescales similarity, `--cos-threshold` is calibrated for the centered space. On
the HR corpus the useful operating point is ~0.6, not the 0.9 that suits raw-space question dedup.

## Corpus-calibrated cosine threshold (`--max-candidate-pairs`)

Even in the centered space a fixed cosine is not portable: the same row budget lands at ~0.60 on the
HR corpus and ~0.46 on the goods corpus. `src/llb/conflicts/calibration/distribution.py` (the
record), `calibration/sampling.py` (how it is measured), and `calibration/operating_point.py` (which
knob wins) derive the cutoff from the distribution of the corpus's own comparable cross-document
chunk pairs instead of asking the operator to sweep for it.

`--max-candidate-pairs N` resolves the per-pair quantile `1 - N/total_pairs`, which over an
exhaustive distribution cuts at the N-th largest similarity. A bare `--cos-quantile` is the wrong
dial to expose because it is a per-PAIR rate, so the rows it admits grow with the pair space: at
the 99.9th percentile over the goods corpus's 74,586 comparable pairs it returned 84 rows, and the
same quantile on a 100k-chunk corpus would return millions. `--cos-quantile` remains as the
low-level escape hatch.

Precedence is `--cos-threshold` > `--cos-quantile` > `--max-candidate-pairs` > the fixed default:
an operator who names a cosine has usually swept for it and is never silently overridden.
Calibration is opt-in; with no knob the fixed `DEFAULT_COSINE_THRESHOLD` still applies.

Not to be confused with the similarly named `--max-candidate-record-pairs`, which changes nothing
about detection: it sets how much of the ranked list the BUNDLE writes down for a later budget
re-read ([bundle
record](conflict-bundle-record.md#how-deep-the-prefix-reaches-and-what-the-depth-costs)).

The distribution is **enumerated exactly** whenever the comparable pair space fits
`MAX_EXHAUSTIVE_PAIRS` (5M); sampling is only the fallback above that. That is not an
optimization. Sampling puts a `1/N` floor under the estimable tail, and the HR corpus lands below
it: against 2.4M comparable pairs, a 200k sample has just one pair above cosine 0.6, so the
estimated tail rate stops moving and the threshold silently pins to the sample maximum (measured:
0.7257 instead of 0.5959). A sampled estimate records `resolvable_quantile` and warns when the
requested tail is finer than it can express. Enumerating 2.4M pairs costs ~2 s.

`summary.json` records the basis under the semantic tier: `cos_threshold`, `cos_threshold_source`,
and a `null_distribution` block with pair counts, the resolved quantile, the `selected_rank`, and
the 0.5/0.9/0.99/0.999/0.9999 tail. `report.md` renders a **Semantic threshold** section, so a
calibrated run is comparable against a swept one by absolute cosine.

**Measured, both quickstart corpora** (recursive/800/120 multilingual-E5 stores, centered space,
exhaustive distributions; the HR store is rebuilt with
`DATA_DIR=<scratch> make build-index CORPUS=<hr-md-dir>` so the goods store survives):

| budget | corpus | comparable pairs | resolved cosine | findings | vs. swept baseline |
| --- | --- | --- | --- | --- | --- |
| 12 | HR | 2,434,651 | 0.5790 | 12 | recovers **8/8** filtered swept-0.6 pairs, adds 4 |
| 50 | HR | 2,434,651 | 0.5349 | 50 | superset of the filtered swept-0.6 pairs |
| 12 | goods | 74,586 | 0.4617 | 12 | swept 0.6 found 0 |
| 50 | goods | 74,586 | 0.3948 | 50 | swept 0.6 found 0 |

What this buys: the knob **bounds output size on any corpus** while resolving a different absolute
cosine per corpus, so the old failure mode -- 5185 rows on HR at a fixed 0.9 in raw space -- cannot
recur, and an operator can size the candidate list to the claim-tier adjudication they can afford.

## Known limitation: there is no independent null

The knob above is a **rank selector, not a statistical guarantee**, and the distinction was
measured rather than assumed. It is documented here because the naming of an earlier iteration
(`--max-false-flags`, framed as a false-positive budget) was wrong, and the same mistake is easy to
make again.

The intent was to model "what do UNRELATED chunk pairs score on this corpus?" and flag pairs above
that tail. The implementation samples random comparable cross-document pairs -- but that population
*contains whatever genuine duplicates the corpus has*. It is therefore not an independent model of
"unrelated"; it is the observed distribution itself. Once the pair space is enumerated exactly, the
null and the observed population are literally the same set, and the consequences are exact:

- Empirical FDR (`expected false / observed`) is **identically 1.000** at every threshold on both
  corpora -- 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8 -- because the numerator and denominator are the
  same count. The statistic carries no information.
- A budget of `N` therefore returns **exactly `N`** pairs on every corpus and every budget
  (verified in `test_candidate_budget_selects_exactly_n_pairs_over_an_exhaustive_distribution`).
  That is a useful, predictable contract; it is just a rank cutoff, and no claim about how many of
  those `N` pairs are real.

The sampled fallback does not escape this: a random subsample of the same population has the same
contamination, only noisier.

Consequences for reading a report:

- No threshold on this corpus geometry can be justified as "statistically significant". The
  semantic tier is a recall-oriented **candidate generator** for the claim tier, and the claim
  tier is what establishes whether a candidate pair is a real conflict. In the original HR run,
  three of 11 pairs touched publication-metadata blocks and four other claim-bearing pairs were
  `complementary`. The structural filter removes the three metadata pairs; it does not relabel the
  four honest non-conflicts as metadata.
- Two swept operating points that look comparable are not. HR's useful 0.6 corresponds to a rank
  cutoff of ~12 pairs; goods' 0.6 corresponds to a rank cutoff of 0. No single budget reproduces
  both, and none can, because the budget is a rank and the two corpora have different amounts of
  real duplication at every rank.

Getting a real false-positive rate needs an independent null -- pairs known a priori to be
unrelated -- which the corpus alone cannot supply. Four generations of research have tried to build
one and all four are negative; the matrices, the measured reason each failed, and the measured
claim-tier precision that replaces the missing rate live in [independent-null
research](conflict-null-research.md), and the arithmetic that closed the search is in [closing the
independent-null question](conflict-null-closure.md).

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
  reimplemented, with `tests/llb/conflicts/test_claim_precision.py` asserting the audit's curve
  equals the research lane's curve on the same rows. The shared helpers live in
  `src/llb/conflicts/claim/precision.py`; `interval_stats.py` holds the Wilson interval both sides
  use.
- **A budget sweep for free.** Rank order also makes the precision curve a genuine
  candidate-budget sweep over one adjudicated list, so `budget_resolution` can name the smallest
  budget whose clustered bound clears zero without paying for a run per budget.
- **A calibration gate.** Before measuring, the audit adjudicates a COMMITTED frozen-label probe
  with the same prompt and the same endpoint, and suppresses the whole block -- with the reason
  printed -- when the model does not clear it. A precision figure computed from a model's own
  verdicts is otherwise only as good as the model.

An unparsable verdict is kept as a row and counted as NOT actionable, so it biases the figure
downward; the printed precision is therefore a lower bound whenever `unparsed_rows` is non-zero.
The block is suppressed only when unparsed rows exceed `unparsed_allowance` (5% of the list, floor
1), because at a 12-row budget a single malformed completion would otherwise erase a usable
conservative measurement.

### The frozen calibration probe

`samples/corpora/conflicts_uk_v1/adjudicator_probe.json` holds 24 section pairs of the planted
fixture -- 12 actionable (the changed deadline, the restated sections, the byte-identical
re-upload, the reformatted reissue, the absorbed note, the vague restatement) and 12 complementary
(the unrelated archive control, and cross-section pairs that state different compatible facts).
Every prompt is distinct: the fixture documents restate each other verbatim, so pairs that would
present byte-identical passages were replaced rather than counted twice.

The probe stores `doc_id` + heading line, never passage text, and
`src/llb/conflicts/claim/calibration.py` resolves each side to the exact corpus bytes at run time.
A fixture edit that moves the text fails the run instead of silently leaving a frozen label
attached to a passage that no longer exists.

Agreement is scored on the **actionable binary**, not on the exact relation: a duplicate reported
as `subsumes` sends the operator to the same decision, while a conflict reported as `complementary`
is exactly the error a precision figure would hide. The gate is the Wilson 95% lower bound on that
agreement at `MIN_ADJUDICATOR_ACCURACY_LCB` (0.60) -- the same bound the research lane applies.
`--no-calibrate-adjudicator` (Make: `NO_CALIBRATE_ADJUDICATOR=1`) skips the 24 probe calls and
suppresses the block rather than printing it uncalibrated; `--calibration-probe` points at a
different probe.

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

## What the counts mean

The census behind every printed count, the two decision-grouping rules and the range between them,
the `to decide` / `to review` split, the policy projection, and the `groups.json` sidecar live on
their own page: [decision groups and their counts](conflict-decision-groups.md).

## Semantic prefix tree

`src/llb/conflicts/semantic_tree/tree.py` builds a centroid tree over chunk vectors by deterministic
bisecting 2-means for angular vectors and axis-aligned median splits for projected Euclidean
vectors. The angular tree retains the exact centroid/radius triangle-inequality path used by refresh
and inspection. Select the large-corpus path with `--project-dims` (Make: `PROJECT_DIMS=32`); its
PCA and persistence implementation lives in `projection.py` and `projected_index.py`.

The blocker is exact. Store vectors are unit length, so cosine cutoff `c` is Euclidean distance
`sqrt(2 - 2c)`. PCA is an orthogonal projection and can only shrink pairwise distance. A projected
distance above that cutoff therefore proves that the full-space pair cannot match. Surviving pairs
are confirmed against the original vectors in bounded NumPy batches. Projected rows are
deliberately not L2-normalized: normalization changes pairwise distance and invalidates the
lower-bound proof. A regression test locks that behavior down.

SciPy `cKDTree.query_pairs(..., eps=0)` performs the exact radius traversal in reduced space; this
is not an approximate ANN index. The persisted `SemanticPrefixTree` supplies the same exact query
as a dependency-light fallback. Its Euclidean nodes carry axis-aligned bounds, and leaves are
checked in projected space before full-space confirmation. CI asserts that projected candidates
contain every true match and that confirmed pair identities equal the unprojected blocked scan
across several projection dimensions.

`summary.json` reports `project_dims`, `projected_backend`, `projected_candidate_pairs`,
`projected_pruned_pairs`, `projected_pruning_fraction`, and `full_space_comparisons`.

### Persisted-index reuse rule

Only the projected path persists and reuses geometry. Its one gate is
`_load_reusable_artifacts` in `projected_index.py`: the PCA projection and semantic tree load as a
pair only when the source fingerprint, the projection's fitted-source and content fingerprints,
the encoder, source and projected dimensions, centering mode, and leaf size all match. Loading the
tree also enforces its format version. The source fingerprint covers the corpus fingerprint, store
manifest, and complete chunk table, so a foreign encoder or store generation rebuilds BOTH
artifacts before it can be queried. A corrupt or incomplete pair also rebuilds; no partial artifact
is treated as reusable.

The angular full-space path has no persisted reuse decision: every regular audit invocation builds
fresh geometry from the current store vectors. An explicitly injected in-memory tree remains a
low-level API seam, not a store artifact. The former `tree_is_reusable` predicate in `refresh.py`
was removed because production never called it and its encoder/dimension-only check was weaker
than the persisted-index identity.

`tests/llb/conflicts/semantic_tree/test_projection.py` covers matching reuse, encoder and corpus
identity rejection, all-or-nothing PCA refitting, and the fresh full-space build. The focused
semantic-tree suite and the complete conflict suite pass with these cases.

## Needle ambiguity lane

With `--goldset`, the audit adds a second, independent signal: for each gold item it locates the
chunks overlapping the item's gold spans and asks whether any **other** document carries a
near-duplicate of them. A needle answerable from two places is ambiguous -- retrieval has two
defensible answers and whichever it ranks first is luck. The report gives
`non_unique_needle_fraction`. This is derived from the gold set rather than from corpus geometry,
so agreement with the tree's findings is corroboration rather than a restatement of one
measurement.

## Artifacts

`$DATA_DIR/corpus-conflicts/<run>/` holds `findings.jsonl` (one JSON object per claim pair, both
sides with exact offsets -- the machine-readable input a resolution lane consumes, one line per row
whatever the census says), `report.md`
([actionable rows first](conflict-decision-groups.md#one-actionable-set), grouped into decisions
with [the unit census beside every count](conflict-decision-groups.md#the-count-and-the-units-behind-it)),
[`groups.json`](conflict-decision-groups.md#the-groupsjson-sidecar) (the decision groups, addressed
by the same finding ids the resolution plan uses), `summary.json` (per-tier counts, timings,
parameters, `finding_census` /
`relation_census`, [`group_granularity`](conflict-decision-groups.md#how-many-decisions-the-row-count-is)
(both grouping rules and the decision range), the optional
[`policy_projection`](conflict-decision-groups.md#projecting-the-review-count-one-command-earlier)
block, the
[`stage_attribution_inputs`](conflict-decision-groups.md#recomputing-the-stage-from-a-finished-bundle)
record a later re-read of the stage attribution is recomputed from, and the `claim_precision` block
with its per-row ledger and all 24 calibration verdicts), the optional `claim_prefilter` scorer,
rank, calibration, cost, and fallback ledger, and `tree_meta.json` (tree geometry plus the embedder
fingerprint that pins reuse, since centroids are only meaningful in the space that produced them).
With projected blocking, the resolved store
generation also holds `semantic_tree/projection.json`, `semantic_tree/tree.json`, and
`semantic_tree/tree_meta.json`. The projection JSON carries its own SHA-256 fingerprint.

With `--linkage`, the run directory also holds `linkage/` -- the record-linkage seam's standard
bundle (settings, blocking counts, fitted match parameters, the trained model, the scored pairs, and
the clusters) plus `edition_summary.json`, `document_records.jsonl`, and `editions.jsonl`; the
audit's own `summary.json` carries the compact form of that reading under `edition_linkage`, and
each decision group in `groups.json` gains the `edition_groups` its documents fall into. See
[the document-edition lane](../entity-resolution.md#artifacts-2).

`make compare-conflict-granularity` writes
`$DATA_DIR/corpus-conflict-granularity/<run>/{granularity.md,granularity.json}` and
[`make recompute-conflict-stage`](conflict-decision-groups.md#recomputing-the-stage-from-a-finished-bundle)
writes `$DATA_DIR/corpus-conflict-stage/<run>/{stage.md,stage.json}` -- both are re-readings of
audit runs already on disk, so they produce no findings of their own.

## Evidence run

CUDA host, RTX 4060 Ti, real multilingual-E5 store, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M for the
claim tier.

- **HR corpus** (8 docs, 2.77 MB, 2578 chunks): no duplicate or near-duplicate documents at any
  document tier. The original 11-pair claim run at `--cos-threshold 0.6` labelled 1 `duplicate`,
  3 `subsumed_by`, and 7 `complementary` in 77 s. Three of those complementary pairs touched two
  publication-record chunks; the other four were claim-bearing non-conflicts about personnel
  authority, medical leave, and software error handling. With the structural filter, the same
  threshold returns eight claim-bearing pairs and removes exactly those three metadata pairs.
  All eight surviving pairs occur in both the 12- and 50-candidate calibrated runs. The substantive
  findings remain: three documents cover the same "mass-edit personnel cards" procedure, and a
  2008-versus-2022 military-service statute pair is ordered by specificity.
- **Goods corpus** (5 docs, 1139 chunks) with its 19-item gold set: 0 cross-document duplicates and
  `non_unique_needle_fraction` 0.0. The two independent signals agree.

The committed fixture at `samples/corpora/conflicts_uk_v1/` plants one instance of every relation
(byte-identical copy, reformatted reissue, absorbed note, changed deadline, restated section, vague
restatement, unrelated control), plus repeated publication records and a single-occurrence prose
control, so each tier and semantic exclusion reason is asserted against a known answer in CI.

Post-filter evidence measured 2026-07-20 on the RTX 4060 Ti 16 GB CUDA host over
multilingual-E5 stores, one run per semantic-metadata-filter setting. The HR swept, budget-12, and
budget-50 runs and the goods
budget-12 and budget-50 runs are the source for the measurements above. The
[claim-tier precision](#measured-claim-tier-precision) reading comes from four runs measured
2026-08-12 -- the HR and goods corpora each at candidate budgets 12 and 100 -- each recording a
per-row ledger and 24 calibration verdicts; the budget-100 pair is the source for the sweep tables
above and the
budget-12 pair reproduces their first two rows.
