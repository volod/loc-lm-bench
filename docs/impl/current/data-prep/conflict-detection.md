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
sides with exact offsets -- the machine-readable input a resolution lane consumes), `report.md`
(actionable relations first), `summary.json` (per-tier counts, timings, and parameters), and
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
above.
