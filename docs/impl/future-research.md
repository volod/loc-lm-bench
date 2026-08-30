# Future research

Questions this project investigated to a decision and then STOPPED, plus the conditions that would
make each worth reopening. A task lands here instead of in [plan.md](plan.md) when its answer is
known and negative: nothing remains to build, so it is not forward work, but the reasoning is worth
keeping where a later reader can find it before spending the same effort again.

Rules for this file:

- One section per closed question, naming the task id it carried in `plan.md`.
- Every section states what was measured, what closed it, and what would have to be TRUE (not merely
  attempted) for the question to be worth reopening -- a reopening condition is a number or a change
  in the world, never "try harder".
- Directions already proven dead are listed explicitly, so a reopening does not re-run them.
- Evidence lives in the delivered docs under [`current/`](current.md); this file links to it and does
  not restate the measurements.

## robotics-boundary-contract-and-upstream-pins -- whether MHS has a public conformance contract

**Question.** Does the public Model Hardware Standard research preview expose a normative schema,
versioned conformance input, and applicable license that loc-lm-bench can inspect and test without
preview credentials?

**Answer: no.** The source and contract review is recorded with the delivered
[robotics boundary](current/robotics-rag/boundary-contracts.md#pinned-upstream-boundary). The public
material supports a protocol-neutral fake for its stated semantics, but it does not license an
`MHS-compatible` claim.

**Reopening condition.** Reopen only when an authorized preview contract or public release makes a
normative schema revision and its applicable license inspectable, and supplies a named, digestable
conformance input that can run without committing credentials or preview package bytes. The
human-gated `robotics-mhs-preview-conformance` task in [plan.md](plan.md) owns that path.

**Do not re-run these.** Do not infer a schema from announcement prose, scrape unstable rendered
pages into an imitation SDK, reconstruct a private contract from transport examples, or treat MCP,
CLI, or code-API availability as evidence of semantic conformance.

## conflict-null-model-research -- a per-pair semantic false-positive rate

**Question.** The semantic conflict tier ranks chunk pairs by cosine. Nothing in the corpus says what
an UNRELATED pair scores, so no cutoff can be called a false-positive rate. Is there any construction
that supplies an independent null at this corpus scale?

**Answer: no, and the obstruction is counting, not construction.** Four generations of candidates ran
on the CUDA host over a planted fixture and two real quickstart corpora; the matrices, gates, and
artifacts are in [independent-null research](current/data-prep/conflict-null-research.md) and
[closing the independent-null question](current/data-prep/conflict-null-closure.md). The product
decision that follows is recorded in [scope boundaries](current/scope-boundaries.md).

The fourth generation is what makes this a stop rather than a pause. Controls generated from the
target corpus's own structure ARE nulls -- 43 of 44 cleared the relation verifier, against 0 of 93
for the traced edits that preceded them -- and they largely repair the positivity failure that
killed the collected banks: weighted membership AUC falls from 0.99998 to 0.676 on the HR corpus and
from 0.99989 to 0.533 on goods. With construction solved, the only thing left between the project and
a certified tail is the number of verified units, and that number is now known exactly rather than
estimated: 607,303 for the HR corpus's affordable operating point, at 230 verified claims per hour
on this host, or 110 days of uninterrupted GPU time for one corpus at one candidate budget.

**Reopening conditions.** Any ONE of these changes the arithmetic; nothing short of one does:

- **A corpus whose affordable candidate list is a coarse tail.** The requirement is
  `units >= log(0.05) / log(1 - alpha)` for `alpha = candidate_budget / comparable_pairs`. A corpus
  small enough (or a budget large enough) to put `alpha` near 5% needs 59 verified units, which is
  reachable in an afternoon -- the planted fixture already needs only 25 and has 15. The blocked case
  is the opposite one: a large corpus with an operator budget of a dozen rows, where `alpha` reaches
  5e-06 and the requirement reaches six figures.
- **A verified control claim that costs milliseconds instead of seconds.** The bank size is bounded
  by generation plus verification, and verification is the expensive half because it is a model call
  per claim. At the measured 15.7 s per verified claim, the HR corpus's 607,303 units cost 110 days;
  a verifier three orders of magnitude cheaper, or a construction whose role is provable without a
  model, makes the same bank a matter of hours. Note the yield is already high (0.73 to 1.00), so
  nothing is won by generating better candidates -- only by verifying them faster.
- **An external Ukrainian control bank that is exchangeable with the target corpus by measurement,
  not by assumption** -- held-out membership AUC at or below 0.60 on the target's own covariates. Two
  collected banks failed this by a wide margin; a bank that passes it would restore the option of
  borrowing units instead of generating them.

**Do not re-run these.** Each is closed by measurement, not by budget:

- Collected reference banks, weighted or matched. The failure is positivity: no weight can correct a
  covariate region the bank never samples. Generated in-support banks already fix this; they do not
  fix the unit count, and the unit count is what binds.
- Traced counterfactual edits (argument swap, quantity swap, modality flip) as controls. The relation
  verifier adjudicates them as conflicts; the family is planted positives.
- Further linear re-expressions of one bi-encoder space (whitening, anisotropy stripping, residual
  centering). They move the shift without producing an operating point.
- Cosine-only mixture calibration. It is unidentifiable at the resolution an operating point needs.
- Any tail estimator swap on its own. Group-split conformal certification is the sharpest of the
  three and needs about six times fewer units than the interval-based requirement -- 607,303 instead
  of 4,054,460 on HR -- which changes nothing at a bank of 17. The estimator was never the binding
  constraint.
- A stronger scorer as a route to a RATE. The cross-encoder's score orders the adjudicated conflicts
  cleanly (monotone bins, top bin 100% conflicts on fixture and HR), and still cannot price a
  threshold, because its threshold comes from the same small control bank. Its value is ranking, and
  that is forward work rather than closed research (see below).

**What carries the operator-facing number instead.** Claim-tier precision at the returned candidate
budget, with its two-way clustered lower bound and an adjudicator calibrated against frozen labels.
Exposing it inside the audit itself is live forward work (`conflict-audit-measured-precision` in
[plan.md](plan.md)), as is spending the cross-encoder's ranking on adjudication COST rather than on a
rate (`conflict-claim-tier-cross-encoder-prefilter`).
