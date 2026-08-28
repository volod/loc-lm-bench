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
trial's evaluation -- because that is what the declared power priced. On the screen that cap is a
CEILING over every round, not a starting budget: stock successive halving multiplies the case count
by `eta` each round, so a search deep enough to need a second round would otherwise spend more items
than the declaration priced. `run_joint_search(screen_case_cap=...)` is that ceiling, and only the
confirmation run passes one -- a bounded acceptance run keeps the growing budget it was designed
with. The held-out split is scored separately and **uncapped**; a confirmation that scored a slice
of it would be the bounded acceptance run again. A tuning split smaller than the derived size is
reported as `binding: tuning-split-exhausted` with `satisfied: false` rather than silently rounded
down.

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

The trail is also the one part of the record no other artifact holds: a finalist's `result.json`
carries its tuned picks and their held-out scores, but the block ranking that stopped the search
exists nowhere else, and a re-entry re-derives nothing (it never recomputes a tuning-split value).
So the trail is written to `search_trail.json` in the run directory as it is spent and read back
when every finalist resumes -- otherwise re-entering a killed run, the exact recovery the resume
markers exist for, would rewrite `long_run.json` with an empty trail and lose the stopping rule.

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
lane in the repo cuts on. It reports the LOSS half as well: a retain where nothing separated and a
retain where a challenger separated in the wrong direction are different results, and the second is
decided rather than undecided. The regressing rows come from the same `regresses` reading every
paired lane publishes beside `separates`, carry the same minimum-evidence gate, land in the
verdict's `regressed` list, and the sentence names the worst of them with its interval and its
differing-item count. A candidate that merely leads on the held-out mean is exactly the
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
  cached under `$DATA_DIR/screen/<model>.screen.json`, so a re-run does not re-pay for lm-eval --
  but only at the SAME example cap, because a smoke screen capped at a few examples per task is a
  different measurement and must never be handed to a decision that asked for the whole track (the
  cap is recorded on the report as `limit`).

  The screen runs LAST, on a card the tuning phase has been using, so three things it inherited from
  the run matter. It launches at the run's `max_model_len` rather than a model's native 128k window,
  which would OOM the KV cache on a 16 GiB card. It runs under the same VRAM-reclaim and
  thermal-cooldown isolation contract as a sweep cell. And it takes the same pre-launch VRAM
  contention guard `run-eval` takes (`guard_vllm_contention`), with `--public-evict` on by default
  so an Ollama finalist's keep-alive residency is unloaded before a vLLM finalist's engine tries to
  start on what is left -- the reclaim gate cannot cover that, since Ollama's residency is
  deliberate and deliberately excluded from the gate. Without those, a mixed-backend screen fails on
  its second finalist and the verdict is qualified for a reason that was never about the model.
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
(`lm-eval[api]>=0.4.9`; the `api` extra is what the `local-completions` model needs) -- a throwaway
`$DATA_DIR/venvs/lm-eval` keeps it off the project lock, and the run is invoked with that venv's
`bin` prepended to `PATH`. Without it the board and the verdict are still written, with every
finalist recorded under `public_screen.failures` and the verdict qualified accordingly.

`LONG_RUN_PUBLIC_EVICT` (on by default) is what lets that screen launch a vLLM finalist after an
Ollama one; set it empty to opt out on a host you share. The same two opt-ins reached `screen-public`
as `--evict` / `--wait`, so a single-model screen has the escape hatches `run-eval` always had.

Artifacts land beside the ordinary joint-search ones under `$DATA_DIR/joint-search/<run>/`:
`long_run.json` (the whole record) and `long_run.md` (the operator page: verdict first, then the
predeclaration, the block trail, the held-out board, and the public screen). `scoreboard.{json,md}`
keeps its leak fence and its rows exactly as the stock search writes them; the only addition is the
note beside its `recommended` row, and its `quality` column is the composite ranking score, not the
raw per-case objective the verdict cuts on -- the two are different numbers for the same row by
design. Re-entry with the same `LONG_RUN_RUN_ID` resumes on the same markers the stock search uses,
plus `search_trail.json` for the block trail.

CI drives the whole schedule on injected screen, block-tune, held-out, and public-screen hooks in
`tests/llb/optimize/long_run/`: `test_long_run_plan.py` (derivation and refusals),
`test_long_run_stability.py` (the stopping rule), `test_long_run_board.py` (intervals, frontier,
verdict), and `test_long_run_schedule.py` (the end-to-end run and its artifact, the trail a resumed
entry carries, the two readings a retain can rest on, the context cap the public screen launches at,
and the capped report it refuses to reuse). `tests/llb/screen/test_screen.py` covers the guard and
the eviction the screen asks for, and `tests/llb/optimize/test_joint_search.py` the screen ceiling
over a two-round halving.

## Host evidence

The default Ukrainian model was confirmed on 2026-08-28 on the 16 GiB RTX 4060 Ti CUDA host
(Ollama 0.32.15, quiet card, nothing else served). The verdict is **RETAIN
`mamaylm-v2-12b`** -- MamayLM-Gemma-3-12B-IT v2.0, served as its `Q4_K_M` GGUF through Ollama.

**What ran.** Four candidates against the committed `ua_squad_postedited_v1` gold set (250 items,
164 verified, split 82 tuning / 82 held-out, disjoint) over its committed corpus: the two
UA-specialized 12B models (MamayLM v2.0 and Lapa v0.1.2 instruct) and the two current-generation
Gemma 4 multilingual baselines (`gemma4:12b` and `gemma4:e4b`). Every candidate was pinned to its
Ollama source, for two reasons worth recording. Both are host facts, not preferences: at the
`RunConfig` default `gpu_memory_utilization=0.85` a vLLM Gemma 4 E4B engine on this card reaches
about 14.1 GiB of the 15.5 GiB torch sees and dies in CUDA-graph capture while a desktop session
holds roughly 1 GiB, and at the time of this run no joint-search command could pass the `0.80` this
tier is documented for. Both searches take it now (`--config` / `--gpu-memory-utilization` /
`--max-model-len`, [serving knobs a search
carries](tuning-and-search.md#serving-knobs-a-search-carries)), so a re-run of this confirmation
need no longer pin a Gemma candidate to Ollama for that reason -- the reading below was taken
before that, and its Ollama pinning is what it measured.
Pinning also puts every finalist on ONE public-screen track, so the two public numbers below are
comparable rather than fenced apart. The offload-only candidates (`gemma-4-26b-a4b`,
`qwen3.8-27b`, `mistral-small-3.1-24b`) were left out as far slower on this tier, so this run says
nothing about them.

**What was declared, and what bound it.** A minimum detectable objective gain of `+0.100` -- the
smallest quality delta worth swapping a shipped default for. Priced against the paired per-case
objective of two earlier 82-item final-split bundles of DIFFERENT models over this same gold set
(Lapa v0.1.2 as candidate, MamayLM v2.0 as baseline; SD `0.331`), that asks for **87 tuning items
and the split holds 82**, so the screen ran at `82/87` with `binding: tuning-split-exhausted` and
`satisfied: false`. The stopping rule was declared as two consecutive block transitions at pairwise
rank agreement `1.00` with an unchanged leader, against a hard budget of 12 trials per finalist in
blocks of 3.

**What the search did.** The 82-item tuning screen ranked `mamaylm-v2-12b` 0.636, `lapa-v0.1.2-
instruct` 0.627, `gemma-4-12b-it-w4a16` (`gemma4:12b`) 0.612, `gemma-4-e4b-it-w4a16` (`gemma4:e4b`)
0.593, and one halving round kept the two UA-specialized models. The ranking then held at agreement
`1.00` with an unchanged leader across blocks 1 and 2, so **ranking-stability stopped the search at
9 trials per finalist (18 in total) out of the 12-per-finalist budget** -- the budget was never
exhausted.

**What the held-out split says.** All 82 held-out items, 95% percentile-bootstrap intervals over
2000 shared resamples, paired against the incumbent's best row
(`mamaylm-v2-12b::best_quality_per_second`: semantic chunking at 704/171, flat retrieval,
`top_k=3`, an 8192-token context budget):

| row | objective | delta vs incumbent | W/L/T | found rate | tok/s |
| --- | --- | --- | --- | --- | --- |
| `mamaylm-v2-12b::best_quality_per_second` | `+0.597 [+0.512, +0.681]` | baseline | - | 0.659 | 14.1 |
| `mamaylm-v2-12b::best_quality` | `+0.569 [+0.484, +0.656]` | `-0.028 [-0.064, +0.006]` | 3/8/71 | 0.634 | 10.5 |
| `lapa-v0.1.2-instruct::best_quality` | `+0.507 [+0.427, +0.592]` | `-0.091 [-0.160, -0.020]` | 9/30/43 | 0.610 | 12.5 |
| `lapa-v0.1.2-instruct::best_quality_per_second` | `+0.489 [+0.407, +0.578]` | `-0.109 [-0.180, -0.039]` | 7/29/46 | 0.610 | 16.1 |

Reading: **no candidate separates in the incumbent's favour to be adopted, and Lapa separates
against itself.** Both Lapa rows sit wholly below zero -- on 39 and 36 differing items -- which clears
the minimum-evidence gate, so this is a retain on measured evidence rather than an undecided board.
Part of that gap is the verbosity confound the objective carries -- Lapa answers longer (13.9 vs
12.1 completion tokens) and pays in token precision (`0.448` vs `0.557`) at near-equal token recall
(`0.706` vs `0.724`) -- but not all of it: the contains-based found rate favours MamayLM by 4.9
points, so Lapa finds fewer answers, not merely wordier ones. Reliability was `1.000` on every row.
The `scoreboard.md` `recommended` row names the same model at `quality=0.682`; that column is the
composite ranking score and the `+0.597` above is the raw per-case objective the verdict cuts on.

**How much the run could resolve.** Re-priced on its own variance, the held-out set is QUIETER than
the reference (`SD 0.161` against `0.331`), so 82 items resolve down to `+0.050` -- half the
declared gain -- and the discordance floor asked for 45 of the 82 reached. The strongest
challenger's delta therefore reads **flat**: its interval lies wholly inside the predeclared band.
The `82/87` shortfall on the tuning screen did not cost the decision anything, because the decision
is taken on the held-out split, which was scored uncapped and had resolution to spare.

**What the public tracks say.** Both finalists screened complete on the Tier-1 generation track
(`global_piqa_prompted_ukr_cyrl`, lm-eval 0.4.12 through the Ollama endpoint): MamayLM `0.820`
exact match, Lapa `0.790`. Same track, same direction as the private board, so the public evidence
qualifies nothing away.

**Quality versus latency.** The frontier holds two rows and they disagree:
`mamaylm-v2-12b::best_quality_per_second` leads on quality at `0.818 s` per case, while
`lapa-v0.1.2-instruct::best_quality_per_second` is the fastest at `0.763 s`. An operator buying that
7% latency saving pays `-0.109` objective for it, which is why the recommendation is the quality
side of the frontier.

**Cost.** About 90 minutes end to end: four 82-item screen cells at one and a half to three minutes
each, three blocks at about 23 minutes per block across both finalists, then the uncapped held-out
scoring and the two public screens. A prior partial run on a 12 GiB RTX PRO 3000 Blackwell laptop GPU
(2026-08-27) measured about 2 minutes per 66-item screen cell and about 24 minutes per three-trial
block per finalist; that host is roughly half the throughput per item and its numbers remain the
sizing estimate for that class.

**What would overturn it.** A gold set larger than 82 held-out items, or one drawn from a different
Ukrainian domain -- this is one post-edited SQuAD-derived corpus, and the verdict is about a default
for corpus-grounded QA on that shape of data, not about Ukrainian ability in general. A newer
generation of either UA lineage, or a Lapa release that closes a `-0.109` gap. An offload-tolerant
run that admits the 24B-27B candidates this one excluded on host fit. A different objective: the
decision is on token-F1-style `objective_score`, and a lane that ranked on found rate alone would
have a narrower 4.9-point gap to cut on. And a serving change -- every row here is a `Q4_K_M` GGUF
through Ollama at an 8192-token window, so an unquantized or vLLM-served MamayLM is not this
measurement.
