# Human-in-the-loop evaluation -- judge calibration, schema sign-off, and data verification

A step-by-step manual for the parts of LLM evaluation that **no AI service can legitimately
do for you**, with the essential papers, manuals, and a how-to-understand explanation for each.

These are not project bookkeeping -- they are the three places where *human ground truth* is the
whole point of the measurement. The design rationale is in the [design spec](../design/spec.md),
what already exists in code is in the [current state](../impl/current.md), and the
sequenced roadmap is in the [forward plan](../impl/plan.md). This manual turns those into
runnable procedures.

## Why a human is irreducible here (read this first)

Everything an AI *could* do -- drafting schemas, drafting data, cross-checking data with a second
model -- is implemented as pipeline code elsewhere. What remains is exactly the work that would
*destroy the guarantee the tool sells* if an AI did it:

- - ****Judge calibration**** (Independent **ground-truth ratings** to validate the model judge):
- The point is to measure the LOCAL judge against HUMAN judgment. An LLM-vs-LLM calibration cannot
- establish a "defensible against humans" claim -- it just compares two models.
- - ****Schema / ontology sign-off**** (**Accountable approval** of AI-drafted schemas + facts only
- you know): Approval is an act of accountable authority; the corpus facts (real vs synthetic, do
- references exist) are knowledge only the data owner has.
- - ****Eval-data verification**** (**Human sample-verification** of AI-drafted, cross-checked
- data): Dropping the human verification would forfeit the human-ground-truth guarantee for private
- model-selection data.


The unifying idea: **AI can draft and cross-check; only a human can be the ground truth, the
sample verifier, and the accountable sign-off.** All three are human-paced and run in parallel with
the build -- but judge calibration is on the **critical path** for any *judged* result.

> **Critical-path note.** Until the rating pass clears a Spearman rho of `>= 0.6`, the model judge
> is demoted everywhere -- on the question-answering board AND on every category that uses the judge
> (borderline unsafe-content, summarization faithfulness, agentic trajectory quality, free-form
> text-analysis / conversation analysis). Objective metrics rank alone until then. So start the
> rating pass EARLY; it needs only scaffolding that already ships.

---

## Judge calibration -- validating LLM-as-judge against human ratings

### What you are doing
Independently rating model answers on the calibration split, then computing how well your human
ratings correlate with the local judge's ratings. If the correlation clears the gate, the judge is
admitted into the ranking blend; otherwise it stays a demoted diagnostic and objective correctness
ranks alone. The decision travels in the run manifest.

### How to understand it (the mental model)
- **LLM-as-judge** means using one model to score another's free-form answers (faithfulness,
  relevancy, quality) where exact-match cannot. It is powerful but *unaccountable until validated*
  -- a judge can be confidently wrong, especially in Ukrainian.
- **Calibration** = checking the judge against humans before trusting it. You produce two columns
  over the same answers: your `human_rating` and the judge's `judge_rating`, then measure agreement.
- **Spearman's rho** is the agreement statistic. It is a *rank* correlation: "do the human and the
  judge order the answers the same way?", not "are the numbers identical?". That is the right
  question because a leaderboard cares about *ranking*, and rho is robust to the two scales
  differing. It ranges -1..+1; `>= 0.6` is the trust gate.
- **Bootstrap confidence interval** answers "is rho real, or could this many items have produced it
  by luck?". It resamples your rating pairs with replacement many times, recomputes rho each time,
  and reports the spread. A high rho whose CI dips below the gate is not yet trustworthy.
- **Why human-only, not a frontier proxy.** A frontier model rating the answers is just another
  LLM-vs-LLM comparison; it cannot establish that the *local* judge matches *human* judgment, which
  is exactly what the gate certifies. This is decided and not revisitable.
- **The bias you are guarding against.** The judge is a local Gemma-family model and much of the
  candidate pool is also Gemma-family, so the judge may self-prefer Gemma answers. Calibration is
  one of four mitigations (gate + objective weight + disclosure + an optional non-Gemma cross-check
  judge). Read the bias disclosure in [`current.md`](../impl/current.md) before you rate.

### Step-by-step procedure
The statistics, the gate, the worksheet pre-fill, and the scoring are already implemented and
tested. Your residual is the human column. Everything is offline except generating the worksheet,
which needs a running judge endpoint.

1. **Stand up the judge endpoint.** On a 16 GB box a 12B judge usually cannot co-reside with a vLLM
   candidate; use GGUF/CPU offload, a smaller test judge, or another local host while generating
   the worksheet. See the [local judge guide](judge-experiments.md) and the judge tier table in
   [`current.md`](../impl/current.md).

2. **Pre-fill the worksheet** (fills `model_answer` and an UNGATED `judge_rating`; leaves
   `human_rating` blank). The judge runs ungated here on purpose -- calibration measures agreement,
   so the gate is irrelevant at this step:

   ```
   make calibration-run JUDGE_MODEL=<served-model-id> \
       JUDGE_BASE_URL=http://127.0.0.1:8000/v1
   ```

   Equivalent direct CLI:
   `llb run-eval --split calibration --worksheet ws.csv --judge-model <id> --judge-base-url ...`

3. **Rate independently with the interactive rater.** This is the irreducible work. Use the rater
   instead of hand-editing the CSV -- it walks the worksheet item by item, hides `judge_rating` by
   default, writes through after every edit (so you can stop and resume), and lets you author your
   own answer (`a`) alongside the 1-5 rating:

   ```
   make calibration-rate
   ```

   The full command reference (`a`, `note`, `j <N>`, `u`, `c`, `q`, the `START=` / `SHOW_JUDGE=` /
   `CLEAR=` options) and the rating anchors are in the
   [calibration-tooling manual](calibration-tooling.md#rater-commands). The rules that matter:
   - **Do NOT look at `judge_rating` first.** Anchoring to the judge contaminates the ground truth.
     The rater hides it by default; reveal it (`SHOW_JUDGE=1`) only for post-hoc review.
   - **Span the full score range** -- clearly good, clearly bad, and middling answers.
   - **Deliberately include fluent-but-wrong answers** -- confident, well-written, factually
     incorrect. That is the failure mode the judge is most likely to miss, so it is where
     calibration earns its keep.
   - **Rate against the reference and the retrieved context, not your prior knowledge** -- the same
     inputs the judge sees.
   - **Write down your rubric** (what a 1 vs a 5 means) before you start and apply it consistently.

4. **Score it:**

   ```
   make calibration-score RATINGS=<filled.csv>
   ```

   This computes rho + the bootstrap CI + the mechanical trust decision. `rho >= 0.6` admits the
   judge; below it the judge stays demoted (the gate working as designed -- a small local judge may
   not clear it for Ukrainian, and a 12B is borderline).

5. **(Optional, automatable -- not your work) non-Gemma cross-check judge.** A Qwen/Llama or
   frontier judge can re-score the same split to quantify the Gemma family delta; the board's
   judge-cohort guard prevents mixing cohorts in one board. Listed only so you know it is *not* part
   of the human residual.

### Done when
`make calibration-score` produces rho + CI + decision over YOUR ratings, and that decision is
recorded in the manifest. Everything else (engine, prompts, gate, worksheet) already exists and is
unit-tested.

### Learn (essential papers + manuals)
- **Judging LLM-as-a-Judge / MT-Bench** (Zheng et al. 2023) -- <https://arxiv.org/abs/2306.05685> --
  judge agreement with humans and the position/verbosity/self-preference biases. The single most
  relevant paper for *why* this calibration exists.
- **G-Eval** (Liu et al. 2023) -- <https://arxiv.org/abs/2303.16634> -- the LLM-as-judge metric
  method the project's judge implements.
- [DeepEval docs](https://docs.confident-ai.com/) -- the maintained engine and metrics.
- **Spearman's rank correlation** --
  <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>
  -- why rank (not value) correlation is the right gate.
- **The bootstrap** -- <https://en.wikipedia.org/wiki/Bootstrapping_(statistics)> (depth: Efron &
  Tibshirani, *An Introduction to the Bootstrap*) -- why the CI tells you whether rho is real on a
  small sample.
- **Inter-rater reliability** -- <https://en.wikipedia.org/wiki/Inter-rater_reliability> (Cohen's
  kappa, Krippendorff's alpha) -- so you understand "do two humans even agree?".

### In this repo
`src/llb/judge/calibration.py` (rho + CI + trust decision + worksheet I/O),
`src/llb/judge/rate.py` (the interactive rater), `src/llb/scoring/judge.py` (the gate + bias
note). The operator walkthrough for all three commands -- and the new-goldset / text-corpus-draft
cases -- is the [calibration-tooling manual](calibration-tooling.md); see also the
[data-prep guide](data-prep.md) and [judge-experiments guide](judge-experiments.md).

---

## Schema and ontology sign-off -- accountable approval

### What you are doing
Approving AI-drafted artifacts and confirming facts only you know:

1. **Done:** the AI-drafted **text-analysis scoring schema** is approved (thresholds accepted as
   proposed), recorded at the top of
   [`docs/design/text-analysis-schema.md`](../design/text-analysis-schema.md).
2. **Remaining:** approve the AI-drafted **knowledge-graph ontology schema** + the graph-retrieval
   **scope / acceptance**.
3. **Remaining:** confirm the **corpus facts** only you have -- whether text-analysis reference
   answers already EXIST or must be authored, and which corpus is real vs synthetic (the two are
   reported separately and must never be merged).

### How to understand it (the mental model)
- A **sign-off** is not a code review -- it is an *accountable acceptance* of trade-offs that then
  becomes the trust signal downstream code reads. The signed line at the top of the text-analysis
  schema doc is literally what the scoring runner checks before it lets a `verified=true`
  text-analysis item score models. The signature has mechanical force.
- A **constrained ontology schema** (for the graph backend) is the fixed vocabulary of NODE types
  (e.g. Person, Organization, Law, Event) and RELATIONSHIP types (e.g. enacted, located_in, amends)
  the graph is allowed to contain. "Constrained" = a closed, capped set, not whatever the extractor
  invents. You approve it because the schema decides what questions the graph *can* answer and
  bounds extraction noise.
- The **corpus facts are yours alone.** Whether reference answers exist changes whether
  verification is a *checking* task or an *authoring* task. Whether a corpus is real or synthetic
  changes which board it lands on. No AI can know these about *your* private data.

### Step-by-step: how to do a schema sign-off
The text-analysis schema doc contains a worked sign-off procedure; reuse it as the template for the
ontology when its draft lands. The shape is always:

1. **Read** the proposal doc + its executable form (the engine is short) and run its tests if you
   want to see the numbers move:

   ```
   make test                                            # full suite, or:
   .venv/bin/python -m pytest tests/test_text_analysis.py -q
   ```

2. **Confirm or adjust the decisions that are genuinely yours.** For the text-analysis schema those
   were: the sub-task set, the objective-vs-judged split, the credit thresholds, and the matching
   basis. For the graph ontology they will be: the node/relationship type set, the cap sizes, and
   the extraction constraints. To change a knob, edit the named constant -- the tests assert against
   constants, not hardcoded values, so they follow your change.

3. **Confirm the dependent corpus facts** (do references exist? real vs synthetic?).

4. **Record the sign-off** as a dated line at the TOP of the proposal doc, e.g.
   `Signed off 2026-06-__ by <name>; thresholds accepted as proposed (or: full-credit -> 0.9).`
   Until that line exists, the artifact is a committed proposal only and stays un-trusted for
   headline use.

### Learn (essential papers + manuals)
- Ontology engineering primer: **Ontology Development 101** (Noy & McGuinness, 2001) --
  <https://protege.stanford.edu/publications/ontology_development/ontology101.pdf> -- how to decide
  classes, relations, and granularity for a constrained schema.
- The form a graph enforces a schema in: the
  [Kuzu data-definition docs](https://docs.kuzudb.com/cypher/data-definition/) (node/rel tables).
- Documenting the decision defensibly: **Datasheets for Datasets** (Gebru et al. 2018) --
  <https://arxiv.org/abs/1803.09010> -- and **Data Statements for NLP** (Bender & Friedman 2018) --
  <https://aclanthology.org/Q18-1041/> -- the discipline of recording provenance, real-vs-synthetic,
  and intended use (exactly the corpus facts you confirm).

### In this repo
`docs/design/text-analysis-schema.md` (the template sign-off, already done), the future ontology
proposal (drafted by `src/llb/prep/ontology/`), and the manifest, which carries the trust decisions
a sign-off produces.

---

## Eval-data verification -- human sample acceptance of AI-drafted data

### What you are doing
Spot-verifying a stratified SAMPLE of AI-drafted, frontier-cross-checked gold/eval items and
accepting it, before any item from that set is allowed to score a model. This applies to the
question-answering gold set, every evaluation category, and the graph ontology data.

### How to understand it (the mental model)
- **The verified-data gate has two stages.** First, a *second frontier model* cross-checks every
  drafted item for grounding (does the answer/label actually appear at the cited span?) and
  non-circularity (the question does not contain its own answer; the planter is not the judge). That
  stage is *pipeline code*. Second -- the irreducible human stage -- you verify a *sample* and
  accept it. AI does stage one; only you can do stage two without forfeiting the human-ground-truth
  guarantee.
- **Why a sample, not everything.** Verifying every item by hand does not scale and is not the
  point: the frontier cross-check already filtered the obvious failures. Your sample is a
  *statistical acceptance check* on the residual error rate. Clean sample -> accept the set; dirty
  sample -> reject and the drafts go back.
- **Why stratified.** A random sample can miss rare-but-important cells. **Stratified** sampling
  draws from each stratum -- per sub-task kind, per difficulty band, per section, per
  real-vs-synthetic -- so an error concentrated in (say) hard trend labels cannot hide. The draft
  pipeline already tags items with these strata, so you stratify on existing tags.
- **What "verify" means per item.** Check: (1) the span is grounded -- the cited offsets really
  support the answer/label; (2) the item is non-circular and answerable; (3) the reference answer is
  correct; (4) for synthetic, that planted labels match what the doc actually says. These are the
  same checks the frontier model ran -- you are confirming it did not *systematically* err.

### Step-by-step procedure
1. **Take the drafted bundle** (a draft directory under `$DATA_DIR/prepare-goldset/<ts>/`, or a
   synthetic-corpus output). It is `verified=false`, with a self-contained `corpus/` so it validates
   offline.
2. **Run the structural validator** before reading anything:

   ```
   make validate-goldset GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus
   ```

   It already checks every span resolves to its labeled text, ids are unique, and splits disjoint.
3. **Draw a stratified sample** across the tags (kind x difficulty x section x real/synthetic). Size
   it for the error rate you will tolerate -- a few dozen across strata is typical for an acceptance
   check; document the size and the strata.
4. **Verify each sampled item** against the four checks above, reading the cited span in the corpus.
   Record pass/fail and the reason for any fail.
5. **Decide.** If the sample's error rate is within tolerance, accept; otherwise reject and send the
   drafts back to the pipeline.
6. **Flip accepted items to `verified=true` via the ledger** (do NOT hand-edit the boolean alone).
   Keep the stable IDs, flip only human-approved entries, and adopt them through the ingester so a
   reused ID can never certify changed content:

   ```
   python -m llb.prep.ingest_squad ... --verified-goldset <accepted-ledger>
   ```

   Canonical-item *replacement* (not a boolean-only flip) is what prevents a reused ID from
   certifying changed content. See the [gold-set-from-scratch guide](goldset-from-scratch.md) for
   the ledger mechanics.

### Done when
A documented stratified sample passes the four checks and the accepted items are flipped to
`verified=true` through the ledger. Only then may they enter a scored run.

### Learn (essential papers + manuals)
- **Stratified sampling** -- <https://en.wikipedia.org/wiki/Stratified_sampling> -- the why and how
  of sampling per stratum.
- **Acceptance sampling** -- <https://en.wikipedia.org/wiki/Acceptance_sampling> -- the QC framing:
  accept/reject a batch from a sample with a tolerated defect rate (exactly this gate's logic).
- Annotation quality & the human-label trap: **Are We Modeling the Task or the Annotator?**
  (Geva et al. 2019) -- <https://arxiv.org/abs/1908.07898> -- why independent, careful verification
  matters and how annotator artifacts creep in.
- Data documentation discipline: **Datasheets for Datasets** (Gebru et al. 2018) --
  <https://arxiv.org/abs/1803.09010> -- record what you sampled, how, and what you accepted.
- The project's own grounding mechanics: read `src/llb/prep/frontier.py` (`ground_span` /
  `build_drafted_items`) so you know what "grounded" already guarantees before you sample.

### In this repo
`src/llb/goldset/validate.py` (the structural gate), `src/llb/prep/frontier.py` +
`src/llb/prep/ontology/` (the drafting + grounding the human sample verifies),
`src/llb/prep/verified_ledger.py` (the adoption mechanism), and the
[gold-set-from-scratch guide](goldset-from-scratch.md).

---

## How to learn this (reading order)

If you are new to the *evaluation-methodology* side, go from "why a judge needs validating" to "how
to accept a dataset":

1. **Judging LLM-as-a-Judge / MT-Bench** (Zheng et al. 2023) -- why an unvalidated judge is
   dangerous. Motivates all of judge calibration.
2. **G-Eval** (Liu et al. 2023) -- the specific judge method you are calibrating.
3. **Spearman rho + the bootstrap** (the two primers) -- the math of the gate.
4. **Datasheets for Datasets** (Gebru et al. 2018) -- the documentation mindset behind sign-off +
   verification.
5. **Stratified + acceptance sampling** (the two primers) -- the verification procedure.
6. **Ontology Development 101** (Noy & McGuinness) -- when the graph ontology sign-off is live.

## Checklist

- [ ] **Judge calibration:** judge endpoint up; worksheet pre-filled (`calibration-run`);
      `human_rating` filled INDEPENDENTLY via `calibration-rate` (judge column hidden), spanning
      the full range incl. fluent-but-wrong answers; rho + CI + decision computed
      (`calibration-score`) and recorded in the manifest.
- [ ] **Sign-off:** text-analysis schema signed off (done); graph ontology + scope approved with a
      dated line; corpus facts confirmed (references exist?/real-vs-synthetic).
- [ ] **Verification:** drafted set validated; stratified sample drawn + documented; four per-item
      checks passed within tolerance; accepted items flipped to `verified=true`
      via the ledger, not
      by hand.
- [ ] No `verified=true` item scored a model before its sample acceptance.
- [ ] No judged headline trusted before calibration cleared rho `>= 0.6` (else objective-only, by
      design).

The settled decisions behind every item above live in the [design spec](../design/spec.md) and the
"Resolved questions" section of [`current.md`](../impl/current.md); the categories these
gates protect are in the
[evaluation-categories learning path](learning-path-evaluation-categories.md).
