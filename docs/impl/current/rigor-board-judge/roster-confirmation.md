# Roster Confirmation And The Adoption Verdict

A bounded joint search validates that the roster/runtime path works. It cannot license a
**default-model** decision, because nothing in it fixes -- in advance -- how big a quality gain
would matter, how many items could resolve one, or when searching harder stops changing the answer.
`llb joint-search-long-run` (`make joint-search-long-run`) is the confirmation that does: one
research-scale run that predeclares its effect and stopping rule, spends trials until the finalist
ranking settles, scores the FULL held-out split, screens both finalists on the public Ukrainian
tracks, and states an adopt-or-retain verdict against a declared incumbent.

It is the ordinary joint search plus six declared seams, not a second pipeline -- the host-fit
filter, the cheap tuning-split screen, successive halving, the Optuna studies, the resume markers,
and the final-split leak fence are all the ones in
[backend resolution, sweeps, and search](tuning-and-search.md#joint-model--config-search).

## What the run commits to before it measures

`src/llb/optimize/joint_search/long_run/plan.py` builds the predeclaration, and the run writes it
into `long_run.json` before a decision is stated:

- **Minimum detectable objective gain** (`--minimum-detectable-gain`): the smallest quality delta an
  operator would swap a default model for. Declared, never fitted to the data.
- **The tuning-screen size, DERIVED** from that gain and an earlier run's paired variance through
  the shared paired-power contract (`llb.rag.fusion_evidence.power`) -- the same arithmetic the
  embedder and fusion lanes price their item counts with, so a screen size here and there mean the
  same thing. Both floors apply: the normal-approximation variance floor and the discordance floor
  the exact sign test needs.
- **The stopping rule**: how many consecutive block transitions the ranking must survive, at what
  pairwise rank agreement, and the hard trial budget that ends the search if it never settles.

The reference is two ordinary scored run bundles named with `--power-reference` /
`--power-baseline`, paired on the items both scored by the shared
`llb.eval.paired_cases` reader. A pair whose per-case objective scores describe the kind of
model-to-model gap the confirmation is about to measure is what makes the derived size meaningful.

The derived size **caps the whole tuning side of the run** -- the halving screen AND every Optuna
trial's evaluation -- because that is what the declared power priced. The held-out split is scored
separately and **uncapped**; a confirmation that scored a slice of it would be the bounded
acceptance run again. A tuning split smaller than the derived size is reported as
`binding: tuning-split-exhausted` with `satisfied: false` rather than silently rounded down.

## When the search stops

`long_run/sequential.py` advances every finalist by the same trial block, then asks
`long_run/stability.py` whether the order moved. A transition counts as stable only when both hold:

- **pairwise rank agreement** at or above the declared floor (1.0 = identical order; a looser
  declaration tolerates churn in the tail);
- **an unchanged leader**, because a run whose top row keeps swapping has not settled whatever the
  tail does.

The search ends on the first block that completes the declared streak, or when the budget is spent;
`long_run.json` records which, and what the search consumed. Rankings are read from **tuning-split**
trial values only -- that is what keeps the held-out score a score rather than a stopping criterion.

Interleaving is why this needs its own finalist phase: a ranking is only meaningful between
survivors that had equal search budget, so `long_run/stage.py` replaces the stock
tune-each-finalist-to-completion loop through the `finalist_stage` seam on `run_joint_search` and
reuses everything around it.

## What the verdict is allowed to say

`long_run/uncertainty.py` re-reads every scoreboard row from the per-case `scores.jsonl` its
final-split evaluation wrote, over shared bootstrap index sets (common random numbers): a
percentile-bootstrap interval on quality AND on latency, the paired delta against the incumbent's
best row with its win/loss/tie ledger and calibrated sign-flip reading, and the quality-versus-
latency Pareto frontier.

`long_run/verdict.py` then decides on the **calibrated paired reading**, never the point gap, using
the same separation test, minimum-evidence gate, and borderline qualifier every other adopt-or-retain
lane in the repo cuts on. A candidate that merely leads on the held-out mean is exactly the
small-sample rank reversal this run exists to refuse -- so `scoreboard.{json,md}`, whose
`recommended` row IS that argmax, carries a note pointing at the verdict that supersedes it.

The declaration is then re-priced with the run's own variance (`realized_power`): the same shared
contract's second half reports the smallest delta the item set actually reached can resolve, and
whether the strongest challenger's delta reads as separated, flat, or undecidable. A quieter
reference set cannot make an underpowered run look complete.

Two things qualify the sentence without loosening it:

- the **public Ukrainian screen** (`long_run/public_tracks.py`): both finalists are screened on the
  Tier-1 public tracks before the decision, reusing `llb.screen.backends.screen_with_backend` -- the
  same endpoint launcher `screen-public` uses. The track fence is preserved (a loglikelihood
  accuracy and a generation exact-match are never cross-ranked), coverage is explicit, and a missing
  or PARTIAL screen appends a clause to the verdict rather than reading as a pass. Reports are
  cached under `$DATA_DIR/screen/<model>.screen.json`, so a re-run does not re-pay for lm-eval.
- the **quality/latency tradeoff**: the run's objectives are multi-objective, so the verdict names
  the frontier's quality leader and its fastest row, and says explicitly when they differ.

## Running it

```bash
make joint-search-long-run \
  JOINT_SEARCH_CANDIDATES=<manifest.yaml> \
  GOLDSET=<bundle>/goldset.jsonl JOINT_SEARCH_CORPUS=<bundle>/corpus \
  LONG_RUN_INCUMBENT=<candidate name> \
  LONG_RUN_POWER_REFERENCE=<scored run bundle> LONG_RUN_POWER_BASELINE=<scored run bundle> \
  LONG_RUN_MIN_GAIN=0.10 LONG_RUN_TRIAL_BUDGET=12 LONG_RUN_TRIAL_BLOCK=3 \
  LONG_RUN_RUN_ID=<id>
```

The public screen shells out to `lm_eval`, which is not a declared dependency of this project (it is
an external harness, like the backends). Install it into any environment on `PATH`
(`lm-eval[api]>=0.4.9`; the `api` extra is what the `local-completions` model needs). Without it the
board and the verdict are still written, with every finalist recorded under
`public_screen.failures` and the verdict qualified accordingly.

Artifacts land beside the ordinary joint-search ones under `$DATA_DIR/joint-search/<run>/`:
`long_run.json` (the whole record) and `long_run.md` (the operator page: verdict first, then the
predeclaration, the block trail, the held-out board, and the public screen). `scoreboard.{json,md}`
keeps its leak fence and its rows exactly as the stock search writes them; the only addition is the
note beside its `recommended` row. Re-entry with the same `LONG_RUN_RUN_ID` resumes on the same
markers the stock search uses.

CI drives the whole schedule on injected screen, block-tune, held-out, and public-screen hooks in
`tests/llb/optimize/long_run/`: `test_long_run_plan.py` (derivation and refusals),
`test_long_run_stability.py` (the stopping rule), `test_long_run_board.py` (intervals, frontier,
verdict), and `test_long_run_schedule.py` (the end-to-end run and its artifact).

## Host evidence

None yet: the lane is built and CI-green, but no confirmation run has been carried to a recorded
verdict. Executing one and writing its reading here is the open task
[`ua-roster-confirmation-run`](../../plan.md#ua-roster-confirmation-run).

What a partial run on a 12 GiB RTX PRO 3000 Blackwell laptop GPU (2026-08-27, three host-fitting
`models_uk.yaml` candidates against the committed `ua_squad_postedited_v1` gold set) establishes is
COST, not a ranking: at a declared minimum detectable gain of `+0.100` the paired-power derivation
asked for 66 tuning items and got them, one screen cell over those 66 items cost about two minutes,
and one three-trial tune block cost about 24 minutes per finalist. Read it as sizing information for
whoever runs the confirmation -- a 12-trial budget over two finalists is a two-to-three hour run on
that class of host, before the held-out split and the public screen. It licenses nothing about which
model is better; the run was stopped before any held-out score existed.
