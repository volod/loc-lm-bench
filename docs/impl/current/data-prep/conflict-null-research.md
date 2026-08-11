# Corpus Hygiene: Independent-Null Research (conflict-null-model-research)

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md). The detector this research is about lives in
[conflict detection](conflict-detection.md); the question here is narrower and older than any
candidate: the semantic tier ranks chunk pairs by cosine, and nothing in the corpus says what an
UNRELATED pair scores, so no cutoff can be called a false-positive rate
([why](conflict-detection.md#known-limitation-there-is-no-independent-null)).

`llb research-conflict-nulls` (`make research-conflict-nulls GENERATION=initial|next|third`) runs
one generation of candidates over a real-embedder planted fixture, an eight-document high-recall
corpus, a five-document goods corpus, and two independent Ukrainian reference banks. The harness is
deliberately separate from the audit default: a candidate cannot change user-visible threshold
behavior until it clears every gate. Every generation stays runnable, because a later claim is only
meaningful against the evidence that produced the earlier verdict.

| generation | candidates | verdict |
| --- | --- | --- |
| `initial` | cross-corpus, token/sentence permutation, held-out document, labelled calibration | negative |
| `next` | surface-matched reference, matched residual, source-cluster FDR, traced counterfactual | negative |
| `third` | operating-point feasibility, propensity-balanced control, cosine mixture identifiability, whitened and anisotropy-stripped geometries, verified control roles, claim-tier precision | negative |
| `fourth` | in-support control synthesis, cross-encoder relation scoring, group-split conformal certification ([closure](conflict-null-closure.md)) | negative; search closed |

## First generation: negative result

The first matrix asks whether any obvious construction supplies unrelated pairs.

The implementation is split by responsibility:

- `null_research_geometry.py` reconstructs the semantic tier's exact content filter and centering
  space, scores Cartesian cross-corpus controls, builds deterministic token/sentence permutations,
  and reduces held-out document pairs to their maximum chunk cosine.
- `null_research_evaluation.py` resolves null tails, reports Wilson 95% intervals, evaluates the
  planted document-pair closure, fits the labelled comparison, and measures HR/goods transfer.
  Small permutation corpora automatically receive enough shuffles for at least 20 expected tail
  observations; this prevents sample size from deciding the fixture verdict.
- `null_research_candidates.py`, `null_research_initial.py`, `null_research.py`, and
  `null_research_report.py` apply the gates, orchestrate all candidates, and write `summary.json`
  plus `report.md`. Typer wiring is in `src/llb/cli/prep/conflict_null_research.py`; deterministic
  coverage is in `tests/llb/conflicts/test_null_research.py`.
- `VectorSet.cross_similarities` scores two separately stored corpora without inventing document
  ids, while `VectorSet.centered(mean)` applies the target corpus's mean to both sides. That detail
  matters: independently centering target and reference would compare vectors transformed in two
  different nonlinear spaces.

Run the maintained workflow with explicit stores:

```bash
make research-conflict-nulls \
  FIXTURE_CORPUS=<fixture-corpus> FIXTURE_STORE=<fixture-store> \
  HR_CORPUS=<hr-corpus> HR_STORE=<hr-store> \
  GOODS_CORPUS=<goods-corpus> GOODS_STORE=<goods-store> \
  REFERENCE_CORPUS=<unrelated-uk-corpus> REFERENCE_STORE=<reference-store> \
  EMBED_DEVICE=cuda NULL_RESEARCH_OUT=<artifact-dir>
```

### CUDA evidence and verdict

The CUDA run uses multilingual-E5-base vectors and a nominal 1% null tail. The planted fixture has
8 positive and 13 negative document pairs after closing the three equivalent 2021 editions over
their relations. The rank-budget baseline selects 12 chunk pairs and scores precision 1.000,
recall 0.875, and F1 0.933 on that document-pair fixture.

| candidate | fixture P / R / F1 | HR 0.6 pairs recovered | goods rows | verdict |
| --- | --- | --- | --- | --- |
| cross-corpus | 0.381 / 1.000 / 0.552 | 5810 / 5810 | 1619 | reject: easy reference null floods |
| token permutation | 1.000 / 0.625 / 0.769 | 379 / 5810 | 0 | reject: threshold is too strict |
| sentence permutation | 1.000 / 0.125 / 0.222 | 76 / 5810 | 0 | reject: local meaning survives |
| held-out document | 1.000 / 0.125 / 0.222 | 1 / 5810 | 1 | reject: unresolved and not independent |
| labelled calibration | 1.000 / 1.000 / 1.000 | 289 / 5810 | 0 | reject as null; fixture fit does not transfer |

The cross-corpus lane is the only genuinely independent candidate in that matrix. Its Cartesian
pair-row count met the original resolved-tail heuristic, but the second-generation two-way-unit
audit below shows that this was optimistic because target and reference chunks recur across rows.
Its 1% thresholds are 0.7752 on the uncentered small fixture, 0.2106 on centered HR, and 0.2127 on
centered goods. It labels every fixture document pair positive (13 false positives) and selects
1619 goods rows against a cap of 12. The unrelated encyclopedia reference is too easy relative to
the target corpora; a precisely measured shifted null is still the wrong null.

Permutation pair-row tails meet the original heuristic (fixture 182 shuffles; HR/goods 3), but
repeating transformations of one source chunk does not create independent source units. Their
transfer failure is substantive regardless: token shuffling retains enough E5-visible vocabulary
to put the 1% threshold around 0.91-0.97; sentence shuffling raises it to about 0.99. Both suppress
most of the HR baseline. Held-out document maxima supply only 21 fixture, 28 HR, and 10 goods
observations, cannot resolve a 1% tail, and reuse the contaminated observed population. The
labelled threshold of 0.9323 perfectly separates the tiny fixture it was fit on but recovers only
289 of 5810 HR baseline pairs; it estimates neither an independent FPR nor a transferable operating
point.

The reproducible HR input is a current-host composite: the three retained source conversions and
the five retained goods conversions receive distinct document ids. It has 2189 chunks, compared
with 2578 in the older swept evidence whose runtime store is no longer present. This makes the
transfer count a conservative current-data stress test rather than a byte-for-byte replay of the
older eight-pair report. The negative decision does not hinge on that drift: cross-corpus already
loses to the fixture rank baseline and floods goods; every stricter candidate already loses to the
fixture baseline or is not an independent null.

Artifacts are under
`$DATA_DIR/corpus-conflicts/null-research/20260811T115543Z/`; the directory includes the corrected
five/eight-document input snapshots, their real E5 stores, `summary.json`, and `report.md`. The
verdict is **negative**: keep `--max-candidate-pairs` framed as a rank cutoff, keep semantic output
provisional, and pursue the domain-matched/control-mixture or claim-tier precision directions in
the forward plan before considering another default.

## Second generation: negative result

`research-conflict-nulls GENERATION=next` replaces the initial candidates with four hypotheses
motivated by their failure modes:

- `null_research_matching.py` matches two controls from each of a general and domain-oriented
  Ukrainian reference bank on log length, numeric density, heading depth, lexical entropy,
  punctuation density, and affinity to the target encoder mean. A two-fold diagonal linear
  classifier and maximum standardized mean difference test whether the matched rows are still
  identifiable by corpus; `null_research_controls.py` assembles their clustered score payload.
- `null_research_advanced.py` evaluates the raw matched scores and a local residual geometry that
  subtracts each source chunk's matched-control median. It counts unique target/reference texts as
  effective units rather than treating their Cartesian rows as independent.
- `null_research_fdr.py` searches for the least strict nonempty threshold whose expected false rows
  fit below a Wilson upper bound over source-cluster exceedances. It reports the lane as
  unidentified instead of silently selecting an empty threshold.
- `null_research_counterfactuals.py` creates exact-span capitalized-argument, quantity/date, and
  modality edits, embeds each changed passage on CUDA, and writes hashes plus edit offsets to
  `counterfactual_traces.jsonl`. These controls are explicitly ineligible as an independent null
  until a separate relation verifier proves their semantic role: a changed quantity or modality
  can be a true contradiction that the conflict detector should retain.
- `null_research_nextgen.py` shares the controls across candidates and applies the original fixture,
  HR-recovery, and goods-flood gates plus exchangeability, effective-tail, simulation-coverage, and
  semantic-eligibility gates. Deterministic coverage is in
  `tests/llb/conflicts/test_null_research.py`; the Make target exposes
  `GENERATION=next`, `DOMAIN_REFERENCE_CORPUS`, `DOMAIN_REFERENCE_STORE`, and
  `MATCHES_PER_REFERENCE`.

Run it with the original evidence stores and two independent references:

```bash
make research-conflict-nulls GENERATION=next \
  FIXTURE_CORPUS=<fixture-corpus> FIXTURE_STORE=<fixture-store> \
  HR_CORPUS=<hr-corpus> HR_STORE=<hr-store> \
  GOODS_CORPUS=<goods-corpus> GOODS_STORE=<goods-store> \
  REFERENCE_CORPUS=<general-reference> REFERENCE_STORE=<general-reference-store> \
  DOMAIN_REFERENCE_CORPUS=<domain-reference> \
  DOMAIN_REFERENCE_STORE=<domain-reference-store> \
  EMBED_DEVICE=cuda NULL_RESEARCH_OUT=<artifact-dir>
```

### CUDA inputs and dependence correction

The run reuses the multilingual-E5-base fixture, eight-document HR, five-document goods, and
general UA-SQuAD stores from the initial matrix. The second reference bank combines 15 committed
Ukrainian documents from the apostrophe-variant, chain-context, exact-term, IP-regulation,
duplicate-chunk, near-duplicate-chunk, and intra-document-repeat fixtures. Its CUDA store contains
249 recursive chunks; the content filter retains 130. The general reference contributes 294
comparable chunks from 247 documents.

Each target selects four matched rows, but those rows repeatedly use the same reference content.
After exact-text clustering, the effective source/reference minima are only 9 on the fixture, 100
on HR, and 107 on goods. At a nominal 1% tail these provide 0.09, 1.00, and 1.07 expected
independent observations, all below the predeclared minimum of 20. By contrast, the raw row counts
would claim 0.44, 81.76, and 38.16 expected observations. This demonstrates why pair-row Wilson
intervals are anti-conservative for a Cartesian null and also corrects the interpretation of the
initial matrix. The maintained initial runner now labels its old statistic
`pair_row_tail_resolved` and gates on unique source/reference units.

The deterministic 10,000-draw Wilson coverage checks are 0.915 on the fixture, 0.916 on HR, and
0.978 on goods for the raw matched lane; the gate requires at least 0.93 on every dataset. Coverage
alone does not repair low effective tail count or corpus shift.

### CUDA matrix and findings

The rank-budget-12 baseline remains precision 1.000, recall 0.875, and F1 0.933. No new candidate
clears all gates:

| candidate | fixture P / R / F1 | HR 0.6 pairs recovered | goods rows | primary failures |
| --- | --- | --- | --- | --- |
| surface-matched reference | 0.400 / 1.000 / 0.571 | 5810 / 5810 | 817 | separable, dependent, floods |
| surface-matched residual | 0.400 / 1.000 / 0.571 | 5810 / 5810 | 1005 | separable, dependent, floods more |
| source-cluster FDR | 0.000 / 0.000 / 0.000 | 0 / 5810 | 0 | no nonempty 1% FDR point |
| traced counterfactual | 1.000 / 0.125 / 0.222 | 44 / 5810 | 0 | too strict, unresolved, not a null |

Surface matching moves the 1% thresholds to 0.8507 on the fixture, 0.2683 on HR, and 0.2560 on
goods. It preserves every HR baseline pair but selects 817 goods rows against the cap of 12 and
loses badly to the fixture rank baseline. More importantly, the held-out membership AUC remains
0.678 on the fixture, 0.840 on HR, and 0.830 on goods; maximum standardized differences are 1.831,
0.752, and 1.111. Every dataset fails the predeclared AUC 0.60 and standardized-difference 0.25
limits, so matching has not established exchangeability.

Median residualization does not remove the shift. Its thresholds are 0.0479, 0.1951, and 0.1800 on
fixture, HR, and goods; it keeps the same fixture errors and expands HR/goods selections to 109,524
and 1,005. The conservative FDR lane finds no nonempty 1% operating point on any corpus. Even after
zero control-cluster exceedances, the 95% upper expected-false counts are 13.2 fixture rows, 2,184.8
HR rows, and 238.9 goods rows; the available independent clusters cannot certify the requested
tail.

The CUDA counterfactual pass writes 6,180 verified traces: 3,002 capitalized-argument swaps, 2,471
quantity/date swaps, and 707 modality flips. E5 original-to-edit cosines put the 1% thresholds at
0.9996-0.9999. That operating point recovers only 44 HR baseline pairs. This reinforces the initial
permutation finding that vocabulary-preserving transformations are too close for a useful cosine
null, while also exposing a more fundamental label problem: these edits can instantiate positive
conflict relations. Construction provenance proves what changed; it does not prove unrelatedness.
Its 1,908 unique HR source texts yield only 19.08 expected observations at the 1% tail, so even that
largest lane misses the effective-tail gate; multiple edits of one source do not add source units.

Artifacts are under `$DATA_DIR/corpus-conflicts/null-research/20260811T131139Z/`, including the
domain-reference snapshot and store, `summary.json`, `report.md`, and the full counterfactual trace
ledger. The verdict remains **negative**. No audit default changes, no FPR is exposed, and semantic
output remains a ranked input to claim adjudication. The forward task now requires a much larger
independent-unit bank, relation-verified controls, cross-fitted exchangeability, two-way clustered
inference, and a claim-tier precision fallback.

## Third generation: negative result

`research-conflict-nulls GENERATION=third` stops proposing constructions first and asks the prior
question: what would a usable operating point require, and is the null even identifiable? Six lanes
share one control bank and one rank baseline.

- `null_research_feasibility.py` converts the operator's affordable candidate list into the per-pair
  tail it implies, then into the number of INDEPENDENT control observations that tail needs. A lane
  whose bank cannot reach that number is reported infeasible before any threshold is fitted.
- `null_research_propensity.py` and `null_research_balance.py` replace nearest-neighbour matching
  with cross-fitted ridge-logistic propensity weighting over the structural covariates plus
  encoder-neighbourhood covariates (affinity to the target cloud, leading principal directions).
  Every reference chunk is used ONCE with a trimmed odds weight, so its effective sample size is a
  meaningful quantity, and the model is refitted leave-one-domain-out to score a domain it never saw.
- `null_research_mixture.py` anchors the related component with non-cosine evidence (word 5-gram
  Jaccard/containment over comparable chunks) and the null component with the weighted control bank,
  then enumerates every (null shift, related mass) mixture the corpus's independent units cannot
  distinguish from the observed distribution.
- `null_research_geometries.py` re-expresses each corpus and its controls in a whitened space and in
  a space with the three leading directions stripped, and reports per-pair baseline recovery, since a
  rescaled space's absolute cosines are not comparable to the shipped threshold.
- `null_research_roles.py` asks the host-fit Ukrainian model what each traced control edit actually
  IS, closing the second generation's open question about whether those controls are nulls at all.
- `null_research_precision.py` ranks the candidate rows exactly as the semantic tier would,
  adjudicates them, calibrates the adjudicator against the planted fixture's frozen relations, and
  reports precision with a two-way clustered lower bound.
- `null_research_clusters.py` holds both two-way resamplers; `null_research_third.py` orchestrates
  the lanes and `null_research_report_third.py` renders their sections.

Run it with the same evidence stores plus an adjudicating model:

```bash
make research-conflict-nulls GENERATION=third \
  FIXTURE_CORPUS=<fixture-corpus> FIXTURE_STORE=<fixture-store> \
  HR_CORPUS=<hr-corpus> HR_STORE=<hr-store> \
  GOODS_CORPUS=<goods-corpus> GOODS_STORE=<goods-store> \
  REFERENCE_CORPUS=<general-reference> REFERENCE_STORE=<general-reference-store> \
  DOMAIN_REFERENCE_CORPUS=<domain-reference> \
  DOMAIN_REFERENCE_STORE=<domain-reference-store> \
  CONFLICT_MODEL=<host-fit-ua-model> NULL_RESEARCH_OUT=<artifact-dir>
```

The third generation loads no encoder: it scores stored vectors and reconstructs control edits from
their traces, which leaves the GPU to the adjudicator.

### The control bank an affordable tail would need

An operator who can adjudicate 12 rows is asking for the tail `12 / comparable pairs`. Resolving a
tail needs at least 20 expected independent observations in it, and that turns the requested
operating point into a bank size. This is the measurement that decides the generation:

| dataset | comparable pairs | operating tail | independent units required | available | short by |
| --- | --- | --- | --- | --- | --- |
| fixture | 51 | 2.4e-01 | 85 | 9 | 9x |
| HR | 1,164,694 | 1.0e-05 | 1,941,157 | 198 | 9,804x |
| goods | 59,561 | 2.0e-04 | 99,269 | 182 | 545x |

Two consequences. First, the nominal 1% tail that every earlier generation reported was never an
operating point: 1% of the HR pair space is 11,647 rows, 971 times the list an operator asked for.
Second, closing the HR gap needs a deduplicated Ukrainian control bank about a thousand times larger
than the corpus being audited. No refinement of control CONSTRUCTION reaches that; only a different
question does.

### CUDA matrix and findings

Same rank-budget-12 baseline: precision 1.000, recall 0.875, F1 0.933 on the planted fixture.

| candidate | fixture P / R / F1 | HR rows (recall) | goods rows | primary failures |
| --- | --- | --- | --- | --- |
| propensity-balanced control | 0.400 / 1.000 / 0.571 | 61,849 (1.000) | 817 | not exchangeable, tail infeasible, floods |
| whitened cosine | 1.000 / 0.125 / 0.222 | 62,157 (1.000) | 720 | loses the fixture relations, floods |
| anisotropy-stripped cosine | 0.125 / 0.125 / 0.125 | 58,890 (0.982) | 2,568 | drops baseline pairs, floods worse |

**Balancing fails on overlap, not on effort.** Weighting moves the fixture's membership AUC from
0.704 to 0.544, but on the real corpora it cannot move at all: 0.99998 on HR and 0.99989 on goods
after weighting. Leave-one-domain-out is blunter still -- a model fitted on the domain bank scores
the held-out general reference at AUC 1.000 on every dataset. That is a positivity failure: the
reference banks lie outside the target corpus's covariate support, and no weight can reweight a
region with no samples in it.

**Bulk agreement is not tail agreement.** The same weighted controls are nearly exchangeable with
the observed population by SCORE: separability AUC 0.515 on HR and 0.557 on goods. Yet the weighted
1% threshold (HR 0.2509) admits 61,849 rows -- 5.3% of the pair space, a five-fold tail error from a
null whose bulk looks right. Thresholds live in the tail, and the tail is exactly where a 198-unit
bank has nothing to say.

**Two-way clustering shows how far off pair-row intervals were.** At the fitted threshold the
pair-row Wilson interval is 11.6 times narrower than the two-way clustered bootstrap interval on HR
and 11.9 times narrower on goods (4.6 on the fixture). The second generation's correction from pair
rows to unique units was directionally right and still an order of magnitude too optimistic.

**Geometry moves the shift and not the verdict.** Whitening drives the control-versus-observed
SCORE mean gap to 0.018 standardized units on HR, the closest any lane has come to a matched null,
while collapsing fixture F1 to 0.222 and still selecting 720 goods rows. Stripping three leading
directions loses 107 of the 5,810 HR baseline pairs and selects 2,568 goods rows. A geometry that
equalizes the first moment does not deliver an operating point, which is why fixture F1 alone was
never allowed to select one.

### Cosine-only identifiability

The mixture lane fits `(1 - m) * null(x - d) + m * related(x)` to the observed distribution and
keeps every `(d, m)` within the Kolmogorov-Smirnov resolution the corpus's INDEPENDENT units
support, not the resolution its pair rows would suggest.

| dataset | related anchors | equivalent mixtures | null shift | implied rows | usable |
| --- | --- | --- | --- | --- | --- |
| fixture | 2 | 101 / 1275 | +0.04 to +0.12 | 8 to 36 | no |
| HR | 1,124 | 5 / 1275 | -0.04 to -0.02 | 15,025 to 17,021 | no |
| goods | 0 | 2 / 51 | -0.04 to -0.02 | 117 to 171 | no |

On HR the accepted mixtures are narrow in shift and still put the related mass anywhere between
1.9% and 10.0% of the pair space -- 22,000 to 116,000 genuinely related chunk pairs among eight
documents, which the corpus plainly does not contain. The data cannot tell "the null sits 0.04 lower
than the control bank says" apart from "a tenth of this corpus is related", and both readings imply
lists of 15,000 rows against a budget of 12. On goods the related component cannot be anchored at
all: no chunk pair in that corpus reaches the lexical thresholds, so the mixture has nothing to
separate. Cosine-only calibration is therefore not identifiable at the resolution an operating point
needs, which is the negative half of this task's question answered affirmatively.

### Verified control roles

The second generation could not say whether its traced edits were nulls; it could only say what
changed. The verifier answers it. Ninety-three sampled edits, adjudicated against their own source
passage on the claim-tier prompt:

| edit type | verified | adjudicated as a conflict relation | 95% interval | relations returned |
| --- | --- | --- | --- | --- |
| capitalized argument swap | 35 | 35 (1.000) | [0.901, 1.000] | 31 duplicate, 3 subsumed_by, 1 subsumes |
| quantity or date swap | 34 | 34 (1.000) | [0.898, 1.000] | 25 duplicate, 8 subsumed_by, 1 subsumes |
| modality flip | 24 | 24 (1.000) | [0.862, 1.000] | 19 contradicts, 2 duplicate, 3 subsumed_by |

Not one sampled edit came back non-conflicting. The modality flips behave as the second generation
feared -- negating a modal verb manufactures a real contradiction -- and the argument and quantity
swaps are worse for a different reason: the verifier reads the edited passage as still asserting the
same claim, so those pairs are exactly the duplicates the detector exists to report. A control
family built from these edits is a bank of planted POSITIVES. The traced-counterfactual lane is
therefore closed, not deferred, and construction provenance is confirmed to prove what changed and
never that a pair is unrelated.

### Measured claim-tier precision

The lane an operator can actually use estimates precision on the rows themselves, so its sample size
is the adjudication budget rather than the pair space. Fifty ranked rows per corpus were adjudicated
by MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M, with the same prompt the claim tier ships and no provenance
shown to the model. The adjudicator agrees with the planted fixture's frozen relations on 41 of 50
rows: accuracy 0.820, Wilson 95% [0.692, 0.902], recall 0.786 and specificity 0.864 on planted
positives -- clearing the predeclared 0.60 lower-bound calibration gate.

| dataset | cosine range of the 50 rows | budget 12 precision | two-way clustered LCB | budget 50 precision | relations |
| --- | --- | --- | --- | --- | --- |
| fixture | 0.836 - 1.000 | 1.000 | 1.000 | 0.500 | 25 actionable, 25 complementary |
| HR | 0.998 - 1.000 | 1.000 | 1.000 | 1.000 | 49 duplicate, 1 subsumed_by |
| goods | 0.394 - 0.538 | 0.000 | 0.000 | 0.000 | 50 complementary |

The measurement works and its answer is corpus-specific. On HR every one of the top 50 rows survives
adjudication, and the bound survives clustering over 40 distinct left and 40 distinct right chunks.
On goods not one of the top 50 rows is a conflict, and the interval is tight enough to say so: the
Wilson upper bound at budget 50 is 0.071. The fixture behaves like a corpus whose real relations run
out -- precision 1.000 through budget 12, then 0.500 at budget 50 as the list reaches the planted
unrelated control.

That is why the predeclared gate fails: the lane requires a precision lower bound above 0.50 on BOTH
untouched corpora, and goods has no operating point at any budget because it has no conflicts to
find. The failure is not a defect of the estimator; it is the estimator reporting that one number
cannot serve two corpora. The forward consequence is in the plan: expose the measured precision and
its bound per corpus in the audit, rather than keep searching for a universal cutoff.

### Verdict

**Negative**, and more sharply than the earlier generations. No audit default changes, no
false-positive rate is exposed, and semantic output remains a ranked input to claim adjudication.
What the third generation adds is the reason to stop widening the same search: the affordable tail
on the HR corpus needs about 1.9M independent control claims and the banks supply 198; the
propensity lane fails on positivity rather than on estimator choice; cosine-only calibration is
unidentifiable at the resolution an operating point needs; and the one control family that carried
exact provenance is now proven to be planted positives. The measurable operator-facing quantity is
claim-tier precision, which this run establishes with a calibrated adjudicator and a clustered bound.

Artifacts are under `$DATA_DIR/corpus-conflicts/null-research/20260811T145837Z/`: `summary.json`
(every lane, every gate, all 150 adjudication verdicts and 93 verifier verdicts), `report.md`, and
`counterfactual_traces.jsonl` (6,180 traced edits). The run reuses the initial and second-generation
snapshots and stores, so the three generations are directly comparable.

The three directions this verdict left open -- generating controls inside the target's covariate
support, scoring relations with a cross-encoder, and certifying the tail from units instead of rows
-- were all run in a fourth generation, which closed the question. Its evidence and the decision to
stop are in [closing the independent-null question](conflict-null-closure.md).
