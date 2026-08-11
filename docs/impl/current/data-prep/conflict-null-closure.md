# Corpus Hygiene: Closing the Independent-Null Question (fourth generation)

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md). The first three candidate matrices, the harness
they share, and the verdicts they produced are in
[independent-null research](conflict-null-research.md); this page carries the fourth generation,
which was written to test the only three directions those verdicts left open and to decide whether
the search continues at all.

`llb research-conflict-nulls GENERATION=fourth` needs no reference bank: it builds its controls out
of the target corpus, so the collected banks the earlier generations argued about are not passed at
all.

```bash
make research-conflict-nulls GENERATION=fourth \
  FIXTURE_CORPUS=<fixture-corpus> FIXTURE_STORE=<fixture-store> \
  HR_CORPUS=<hr-corpus> HR_STORE=<hr-store> \
  GOODS_CORPUS=<goods-corpus> GOODS_STORE=<goods-store> \
  CONFLICT_MODEL=<host-fit-ua-model> \
  EMBED_DEVICE=cpu CROSS_ENCODER_DEVICE=cpu NULL_RESEARCH_OUT=<artifact-dir>
```

The adjudicating model holds the GPU, so the encoder and the cross-encoder run on the CPU; both
score at most a few hundred short passages, which is minutes, not hours.

## What the three closed doors leave

| third-generation finding | direction it leaves open | fourth-generation lane |
| --- | --- | --- |
| balancing fails on POSITIVITY: reference banks sit outside the target's covariate support | generate controls from the target's own structure | `synthesized_in_support_control` |
| every geometry was a linear re-expression of one bi-encoder space | read both passages together | `cross_encoder_relation` |
| pair-row intervals understate uncertainty by an order of magnitude | certify the tail from units instead of bounding it from rows | `group_split_conformal_tail` |

Implementation:

- `null_research_synthesis.py` samples source chunks per document, asks the host-fit Ukrainian model
  for a same-genre passage about a different subject, adjudicates each generated passage against its
  own source on the claim-tier prompt, and keeps only the non-conflicting ones. Survivors are
  embedded with the target's own encoder and become a control population addressed exactly like a
  corpus, so the third generation's balancing, tail, and gate machinery applies unchanged.
- `null_research_cross_encoder.py` re-scores the cosine shortlist and the frozen controls with a
  pinned multilingual cross-encoder, then reports a calibration curve against the adjudicated labels,
  relation recall, and clustered tail coverage. A fixture-F1 improvement alone cannot accept it.
- `null_research_conformal.py` compares group-split conformal tail CERTIFICATION against the shipped
  two-way row bootstrap under duplicate reuse, domain shift, and a tail finer than the bank resolves.
- `null_research_fourth.py` orchestrates the lanes over one shared bank;
  `null_research_report_fourth.py` renders their sections. Deterministic coverage with injected model
  and cross-encoder fakes is in `tests/llb/conflicts/test_null_research_fourth.py`.

## The floor no estimator choice can move

The third generation sized a control bank by how many tail observations an interval needs. The
conformal lane asks the same question with no interval and no distributional assumption at all. Sort
the control units by their worst row and take the k-th largest as the threshold: its exceedance
probability is `Beta(k, n + 1 - k)`, so the bank certifies tail `alpha` with confidence `1 - delta`
exactly when `P(Binomial(n, alpha) >= k) >= 1 - delta`. The strictest usable rank is `k = 1`, which
makes the minimum bank size closed-form:

```text
units >= log(delta) / log(1 - alpha)
```

At 95% confidence that is **59 units for a 5% tail**. The number is distribution-free: no encoder,
geometry, weighting, or estimator can lower it.

### CUDA evidence and verdict

The run adjudicates and generates with MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M on the GPU, embeds with
multilingual-E5-base and scores with `BAAI/bge-reranker-v2-m3` on the CPU, at a nominal 1% tail and
the same budget-12 candidate cap as every earlier generation. Stores were rebuilt for this run
(heading chunks with duplicates kept for the planted fixture, defaults for the two quickstart
corpora), so the pair counts differ from the third generation's while the corpora and the planted
relations do not: fixture 16 comparable chunks over 21 document pairs, HR 2,487 chunks over 2.4M
pairs, goods 1,014 chunks over 71,736 pairs.

**Generated controls are nulls, and that is new.** Forty-four generated passages, each adjudicated
against its own source on the claim-tier prompt:

| dataset | sampled | generated | verifier called conflicting | retained | yield | s per verified claim |
| --- | --- | --- | --- | --- | --- | --- |
| fixture | 15 | 15 | 0 | 15 | 1.000 | 8.2 |
| HR | 20 | 18 | 1 (`subsumes`) | 17 | 0.850 | 15.7 |
| goods | 15 | 11 | 0 | 11 | 0.733 | 17.7 |

The contrast with the third generation is the point: 93 of 93 sampled counterfactual EDITS came back
as conflict relations, while 43 of 44 GENERATED passages came back `complementary`. Writing a new
passage in the source's genre produces a null; editing the source's own claim produces a positive.
The construction problem is solved.

**Positivity is largely repaired, and the residue is sample size.** The weighted cross-fitted
membership AUC is the third generation's own diagnostic, on its own 0.60 gate:

| dataset | membership AUC, collected banks (third) | membership AUC, generated bank | score separability | held-out domain AUC |
| --- | --- | --- | --- | --- |
| HR | 0.99998 | **0.676** | 0.570 | 0.655 / 0.549 |
| goods | 0.99989 | **0.533** | 0.555 | 0.898 / 0.550 |
| fixture | 0.704 | 0.980 | 0.691 | 0.969 / 1.000 |

On the two real corpora the covariate region the collected banks never sampled is now sampled: goods
clears the 0.60 membership gate outright and HR is within reach of it, against banks that were
previously indistinguishable from the target at five decimal places. What still fails
`exchangeable` is the standardized covariate difference (0.61 to 1.63 against a 0.25 gate) and the
leave-one-domain-out transfer -- both of which are a balancing model fitted on 11 to 17 controls,
not a support failure.

**Every lane still fails the same gate, for the same reason.**

| lane | fixture P / R / F1 | HR recovered | goods rows | effective units | first failing gate |
| --- | --- | --- | --- | --- | --- |
| `synthesized_in_support_control` | 0.727 / 1.000 / 0.842 | 8 / 8 | 150 | 3 to 6 | tail unresolved, floods goods |
| `cross_encoder_relation` | 0.727 / 1.000 / 0.842 | 3 / 8 | 2 | 11 to 17 | tail unresolved, loses HR baseline |

Both beat the budget-12 rank baseline's fixture F1 (0.933 in the third generation's store, 0.842
here on a fixture whose chunking differs), and neither can price a threshold: the weighted tail
expects 0.03 to 0.06 independent observations where 20 are needed, and the pair-row interval
understates the two-way clustered one by up to 14.3x on HR.

**The cross-encoder adds real information -- to ranking, not to certification.** Its score, binned
over the 150 adjudicated shortlist rows, tracks the adjudicator where cosine does not:

| dataset | bin 1 | bin 2 | bin 3 | bin 4 | top bin | monotone | relation recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixture | 0.000 | 0.000 | 0.750 | 1.000 | 1.000 | yes | 1.000 (23/23) |
| HR | 0.083 | 0.083 | 0.083 | 0.250 | 1.000 | yes | 0.625 (5/8) |
| goods | 0.000 | 0.000 | 0.083 | 0.250 | 0.000 | no | 0.000 (0/4) |

Each cell is the share of that score bin the local adjudicator called a conflict. On the fixture and
HR the ordering is monotone and the top bin is entirely conflicts, which is what a useful pre-filter
looks like; on goods there are four conflicts in fifty rows and the curve is noise. But the
threshold the lane must use comes from the same 11 to 17 controls, and at that threshold HR keeps
only 3 of its 8 baseline pairs. A better scorer does not buy a certified rate.

**The cost of certifying, priced.** The conformal floor is roughly six times kinder than the
interval-based requirement, and still out of reach by four orders of magnitude:

| dataset | operating alpha | interval-based units | distribution-free units | verified units | short by | at the measured rate |
| --- | --- | --- | --- | --- | --- | --- |
| fixture | 1.1e-01 | 175 | 25 | 15 | 1.7x | 3 minutes |
| goods | 1.7e-04 | 119,560 | 17,907 | 11 | 1,628x | 3.7 days |
| HR | 4.9e-06 | 4,054,460 | 607,303 | 17 | 35,724x | **110 days** |

At 15.7 seconds per verified claim, certifying the HR corpus's affordable operating point costs 110
days of uninterrupted host time -- for one corpus, at one candidate budget, with a bank that must be
regenerated when the corpus changes.

**Conformal certification refuses where the row bootstrap guesses.** Simulated over 100 replications
per grid point, 8 rows per unit; both methods publish an upper bound for the tail rate on a fresh
population, and the entry is how often that bound was right:

| scenario | n=25 | n=50 | n=100 | n=200 |
| --- | --- | --- | --- | --- |
| exchangeable units -- conformal | refuses | refuses | 0.94 | 0.95 |
| exchangeable units -- row bootstrap | 0.94 | 0.91 | 0.90 | 0.89 |
| domain shift -- conformal | refuses | refuses | 0.62 | 0.56 |
| domain shift -- row bootstrap | 0.62 | 0.55 | 0.40 | 0.33 |
| tail finer than the bank (alpha 0.005) -- conformal | refuses | refuses | refuses | refuses |
| tail finer than the bank -- row bootstrap | 0.54 | 0.70 | 0.83 | 0.92 |

Three readings. Under exchangeable units conformal reaches nominal coverage exactly where the
closed form says it must and stays there, while the bootstrap's coverage drifts DOWN as units are
added -- its accuracy at n=25 is incidental, not controlled. Under domain shift neither holds, which
is the third generation's positivity finding restated as a coverage number: no estimator repairs a
bank drawn beside the population. At a tail finer than the bank resolves, the bootstrap publishes a
bound that is wrong up to half the time while conformal declines to publish one at all. That refusal
is the lane's only adoptable behavior, and it is a refusal.

**Verdict: negative, and the search is closed rather than paused.** No audit default changes, no
false-positive rate is exposed, and the semantic tier remains a ranked input to claim adjudication.
What the fourth generation establishes is that the remaining obstruction is arithmetic, not
ingenuity: control CONSTRUCTION is solved, positivity is largely repaired, the sharpest available
estimator needs 607,303 verified units for the HR operating point, and this host produces 230 an
hour. The reopening conditions -- and the directions proven dead -- are recorded in
[future research](../../future-research.md).

Artifacts are under `$DATA_DIR/corpus-conflicts/null-research/20260811T191419Z/`: `summary.json`
carries every lane, gate, the 44 generation and verification verdicts, all 150 shortlist
adjudications, the calibration bins, and the conformal grid; `report.md` renders them.
