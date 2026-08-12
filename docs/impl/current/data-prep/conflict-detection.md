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

Even in the centered space a fixed cosine is not portable: the same row budget lands at ~0.60 on
the HR corpus and ~0.46 on the goods corpus. `src/llb/conflicts/null_distribution.py` (the record),
`null_sampling.py` (how it is measured), and `null_calibration.py` (which knob wins) derive the
cutoff from the distribution of the corpus's own comparable cross-document chunk pairs instead of
asking the operator to sweep for it.

`--max-candidate-pairs N` resolves the per-pair quantile `1 - N/total_pairs`, which over an
exhaustive distribution cuts at the N-th largest similarity. A bare `--cos-quantile` is the wrong
dial to expose because it is a per-PAIR rate, so the rows it admits grow with the pair space: at
the 99.9th percentile over the goods corpus's 74,586 comparable pairs it returned 84 rows, and the
same quantile on a 100k-chunk corpus would return millions. `--cos-quantile` remains as the
low-level escape hatch.

Precedence is `--cos-threshold` > `--cos-quantile` > `--max-candidate-pairs` > the fixed default:
an operator who names a cosine has usually swept for it and is never silently overridden.
Calibration is opt-in; with no knob the fixed `DEFAULT_COSINE_THRESHOLD` still applies.

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
  evidence, so the lower bound is `two_way_proportion_bound` (`null_research_clusters.py`) --
  literally the estimator the independent-null research established, imported rather than
  reimplemented, with `tests/llb/conflicts/test_claim_precision.py` asserting the audit's curve
  equals the research lane's curve on the same rows. The shared helpers live in
  `src/llb/conflicts/claim_precision.py`; `interval_stats.py` holds the Wilson interval both sides
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
`src/llb/conflicts/claim_calibration.py` resolves each side to the exact corpus bytes at run time.
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

## The count and the units behind it

A finding COUNT is the first number an operator reads, and it is the number they size a review
budget against -- so on a corpus whose conflicts concentrate it is the number most able to mislead.
The clustered precision bound already refuses to certify a floor when the conflicts sit on a handful
of chunks, but that refusal lived inside the precision section while the headline count, the
relation table, and the findings list still read as N independent results.

Every count the audit prints now carries the distinct units it rests on:
`report.md`'s headline line, every row of its relation table, the findings section, the CLI summary,
and `summary.json` (`finding_census`, plus `relation_census` beside the unchanged `relations`
counts). `src/llb/conflicts/census.py` computes the census and the grouping;
`src/llb/conflicts/report_findings.py` renders the grouped findings section.

- **Unit.** The CHUNK each side of a finding rests on, falling back to its document for the `hash`
  and `lexical` tiers, which compare whole documents and carry no chunk.
- **Census.** `documents`, `document_pairs`, `chunk_units`, `groups`, `largest_group` -- rendered as
  one line beside the count: `8 findings on 4 documents / 3 document pairs / 9 chunk units, in 2
  groups (largest 6)`.
- **Group.** Findings joined **transitively** by a shared unit, the same closure the `hash` tier
  applies to duplicate groups. A chunk that conflicts with six neighbours is ONE group, and so are
  three copies of one document: a shared unit is what makes two rows the same piece of evidence.
  The report leads the findings section with a **Decision groups** table (rows, `to decide`,
  relations, shared unit, documents, top score) and prints every row below it under its group label.
  `groups` is what an operator triages; `findings` is what a resolution lane consumes.
- **Ranked by stake, named by file order.** The two are deliberately separate. A group's ID comes
  from `findings.jsonl` in file order and is the join key `groups.json` and `plan.json` share, so it
  never moves. The decision TABLE is ordered by `stake_key`
  (`src/llb/conflicts/report_findings.py`): `to decide` rows first, then rows, then top score, then
  the id. `G3` leading the table is therefore a ranking, not a renumbering, and the row table below
  it stays in the file's order -- the order a resolution lane consumes.
- **Rendering only.** `findings.jsonl` keeps one line per row, byte-identical and in the same order
  as before -- the resolution lane reads rows, and nothing here suppresses, merges, or deduplicates
  one. `tests/llb/conflicts/test_finding_census.py` asserts that, the one-group collapse, the
  transitive closure, the document-tier fallback, and the census beside each printed count;
  `test_group_ranking.py` asserts the ranking never renumbers a group.

Why the table is not ranked on score: a score is the model's confidence in ONE pair and says nothing
about how much the group holding it is worth. Measured on the goods semantic bundle (100 rows, 6
groups) a 0.002 score difference put the 14-row decision ahead of the 29-row one; on the goods
budget-100 claim bundle four groups tie at 1.000, so three of them were ordered by the claim-identity
tiebreak underneath -- a document id. Ranked by stake, the 29-row decision moves above the 14-row
one and the 3-row decision above the 2-row one, while the group holding the only actionable row
still leads the claim bundle.

### To decide and to review are two counts, never one

A decision group carries TWO counts of its work, and they are not the same number:

| count | what it counts | where it is computed | who prints it |
| --- | --- | --- | --- |
| **to decide** (`decide_rows`) | rows whose RELATION is not `complementary` (`is_actionable`) | detection, from the relation alone | `report.md` headline, decision table, precision block; `plan.json` decisions |
| **to review** (`review_rows`) | rows whose POLICY outcome is `review_required` | resolution, from `(relation, tier, governance, policy)` | `plan.json` decisions, the CLI summary, `resolution_review.jsonl` |

Neither can serve both roles, and the reason is structural rather than an implementation gap: the
review count is a property of a resolution POLICY, and the audit runs before an operator has chosen
one. That is why the audit prints only `to decide`, names `to review` in the same breath, and says
where it lives; and why `plan.json` -- the only artifact that holds both -- prints them side by side
per group. The vocabulary is defined once in `src/llb/conflicts/constants.py` (`DECIDE_LABEL`,
`REVIEW_LABEL`, `decide_count`); `FindingGroup.decide_rows` and `group_decisions` are its two
consumers.

They diverge in BOTH directions, measured on the two goods bundles:

- **Goods budget-100 claim bundle** (`.data/corpus-conflicts/20260812T-two-counts-census-goods-budget100/`):
  **1 to decide, 0 to review.** The single `subsumed_by` row is work by the relation vocabulary --
  it is what the precision block measures and what leads `findings.jsonl` -- but the conservative
  policy resolves subsumption as `keep_both` plus an annotation, so it costs a human nothing. An
  operator funding one review off the audit report would have funded a review that does not exist.
- **Goods semantic bundle** (`.data/corpus-conflicts/20260812T-two-counts-goods-semantic/`):
  **100 to decide, 100 to review**, group for group (51/51, 29/29, 14/14, 3/3, 2/2, 1/1). Every
  semantic-tier duplicate needs review under every policy because that tier has no deletion
  authority, so the two counts coincide exactly -- and this is the bundle an operator meets first,
  which is precisely why one name for both was survivable long enough to become misleading.

**Which count each ranking uses is stated where the ranking is.** The audit's decision table ranks
on `to decide` because it has nothing else; the review ledger ranks on `to review`, the count an
operator actually funds, because by then the policy has run. On the goods semantic bundle that moves
the ledger from id order (51, 14, 29, 3, 1, 2 rows) to review stake (51, 29, 14, 3, 2, 1), so the
reviewer meets the biggest open decision first; a group's records stay one contiguous block either
way. `tests/llb/conflicts/test_two_counts.py` pins the divergence on a mixed fixture (one claim-tier
duplicate the policy accepts plus one semantic-tier duplicate it escalates: 2 to decide, 1 to
review), the naming in both artifacts, and the ledger ranking.

### One actionable set

"Actionable" -- the predicate behind the **to decide** count -- has ONE definition, `is_actionable`
in `src/llb/conflicts/constants.py`: every relation but `complementary`, plus anything the
vocabulary does not recognize. `decide_count` is the same predicate over a relation census, so the
group table, the plan's `decide_rows`, and the CLI all count one set. The claim-tier
precision block counts that set (`AdjudicatedRow.actionable`) and the report's ordering promotes
that set (`finding_sort_key`), so a row the audit counts as work can never sort below a row it
counts as none. Within each of the two buckets the order is descending score, then claim identity.

That ordering writes `findings.jsonl` as well as `report.md`, and both are read top-down, so the
two artifacts lead with the same row. Measured on the goods budget-100 bundle: its single
`subsumed_by` row -- the only row an operator has to decide -- sat at position **23 of 99**, below
22 `complementary` rows the model scored higher, and now leads the file and the report. The
resolution lane is unaffected by the reordering: planning the same 99 rows in either order yields
byte-identical overlays (only `source_findings_sha256`, which pins the file, differs), the same six
decision groups with the same ids, and an apply/rollback that leaves every corpus byte untouched.
`overlay_from_plan` sorts each document's annotations and suppress-spans by their own identity to
guarantee that, so a re-sorted audit cannot republish a store generation that changes nothing.

### The `groups.json` sidecar

The grouping is machine-readable, not only rendered: `write_audit` emits `groups.json` beside
`findings.jsonl` (`src/llb/conflicts/group_artifact.py`). Each group carries `group_id`, `rows`,
`finding_ids`, `relations`, `shared_units`, `documents`, `document_pairs`, and `top_score`; the
document also carries the census and `source_findings_sha256`, which pins it to the exact rows on
disk and equals the `source_findings_sha256` the resolution plan records.

The `finding_ids` are the SAME ids `plan.json` uses (`finding_id` in `hashing.py`), so a group id
joins the audit, the plan, and the review ledger without any consumer re-deriving anything. Both
sides nonetheless compute the grouping from `findings.jsonl` ROWS through one function, and the
audit writes the rows and the sidecar from one list, so a consumer that never reads the sidecar --
including an audit run from before it existed -- derives identical groups by grouping the file in
its own order. [Conflict resolution](conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
is the first consumer.

### Measured on the goods corpus

CUDA host (RTX PRO 3000 Blackwell, 12 GiB), goods corpus (5 markdown documents, 954-chunk
`multilingual-e5-base` `recursive@800/120` store), MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M adjudicating,
`MAX_CANDIDATE_PAIRS=100`, resolved cosine 0.3604 over 55,865 exhaustive comparable pairs. The
adjudicator agreed with all 24 frozen probe pairs (Wilson 95% lower bound 0.862) in 1 min 34 s and
adjudicated the 100 rows in 9 min 23 s, with 1 unparsable verdict inside the 5-row allowance.

**99 findings on 5 documents / 6 document pairs / 79 chunk units, in 6 groups (largest 51).** The
row count over-states the independent evidence by a factor of sixteen: a triage list of 99 rows is
six decisions, on a corpus of five documents that can only supply ten document pairs and used six.
Only 1 row was actionable (`subsumed_by`), resting on one left and one right chunk, so no measured
budget clears a precision floor -- the clustered bound and the census agree, and now the headline
count agrees with both instead of reading as 99 results.

The largest group is also the reading's sharp edge: G2 chains 51 rows across three documents through
23 shared chunks, so transitive closure can UNDER-state the work as much as a raw row count
over-states it. Read the pair, not either number alone -- the row count bounds the decisions from
above, the group count from below.

Artifacts: `$DATA_DIR/corpus-conflicts/20260812T-census-goods-budget100/`. This is a different store
generation from the [budget-100 precision runs](#measured-both-quickstart-corpora) (954 chunks at
cosine 0.3604 against 1,139 at 0.3648), and it returned 1 actionable row where that run returned 8;
the candidate list at a fixed budget is a rank cutoff into the store's own similarity ordering, so
the two lists are not the same rows.

## Semantic prefix tree

`src/llb/conflicts/tree.py` builds a centroid tree over chunk vectors by deterministic bisecting
2-means for angular vectors and axis-aligned median splits for projected Euclidean vectors. The
angular tree retains the exact centroid/radius triangle-inequality path used by refresh and
inspection. Select the large-corpus path with `--project-dims` (Make: `PROJECT_DIMS=32`); its PCA
and persistence implementation lives in `projection.py` and `projected_index.py`.

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
`projected_pruned_pairs`, `projected_pruning_fraction`, and `full_space_comparisons`. A matching
projection/tree is reused. The source, encoder, centering mode, dimensions, leaf size, and
projection fingerprint control reuse, so incompatible store generations rebuild rather than
querying foreign geometry.

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
whatever the census says), `report.md` ([actionable rows first](#one-actionable-set), grouped into
decisions with [the unit census beside every count](#the-count-and-the-units-behind-it)),
[`groups.json`](#the-groupsjson-sidecar) (the decision groups, addressed by the same finding ids the
resolution plan uses), `summary.json` (per-tier counts, timings, parameters, `finding_census` /
`relation_census`, and the
`claim_precision` block with its per-row ledger and all 24 calibration verdicts), and
`tree_meta.json` (tree geometry plus the embedder fingerprint that pins reuse, since centroids are
only meaningful in the space that produced them). With projected blocking, the resolved store
generation also holds `semantic_tree/projection.json`, `semantic_tree/tree.json`, and
`semantic_tree/tree_meta.json`. The projection JSON carries its own SHA-256 fingerprint.

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

Post-filter CUDA-host evidence (RTX 4060 Ti, multilingual-E5 stores) is under
`$DATA_DIR/corpus-conflicts/20260720T-semantic-metadata-filter-*`. The HR swept, budget-12, and
budget-50 runs and the goods budget-12 and budget-50 runs are the source for the measurements
above. The [claim-tier precision](#measured-claim-tier-precision) runs are under
`$DATA_DIR/corpus-conflicts/20260812T-claim-precision-{hr,goods}-budget{12,100}/`, each carrying
its per-row ledger and all 24 calibration verdicts in `summary.json`; the budget-100 pair is the
source for the sweep tables above and the budget-12 pair reproduces their first two rows.
