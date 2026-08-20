# Local Judge And Scorer Policy

Part of the [Evaluation rigor](../rigor-board-judge.md) area of the [current implementation index](../../current.md).

## Local Judge

`src/llb/scoring/judge/endpoint.py` resolves the local OpenAI-compatible endpoint;
`src/llb/scoring/judge/deepeval_adapter.py` supplies Ukrainian DeepEval G-Eval metrics; and
`src/llb/scoring/judge/model.py` applies the calibration-rho gate. If the judge is not trusted,
objective correctness ranks alone and judge output remains diagnostic. The scorer keeps empty
candidate answers distinct from malformed or unreachable local-judge responses.

```bash
llb judge-experiment --judge-model <model> --judge-base-url <url>
llb judge-smoke --judge-model <model> --judge-base-url <url>
```

`judge-experiment` records prompts, fixed Ukrainian sanity cases, served-model metadata, and scores
under `$DATA_DIR/judge-experiment/<timestamp>/`. `judge-smoke` runs one strict-JSON grounded case
before a long judged run and exits non-zero with a reason if the local judge cannot produce a usable
score.

The local-judge choice is deliberate: corpus data should not leave the host by default. The tradeoff
is model-family bias when the judge shares lineage with candidate models. That bias is disclosed in
manifests and controlled by the calibration gate and judge-cohort guard.

## Scorer Policy Seam

`src/llb/scoring/policy/` selects the judge lane for `run-eval` via `--scorer-policy` /
`RunConfig.scorer_policy`:

| Lane | Behavior |
| --- | --- |
| `human` | Skip automated judging; objective scores rank alone; manifest records `provider=human`. |
| `local` | Existing DeepEval path against `judge_model` / `judge_base_url` (default). |
| `frontier` | Litellm frontier judge using the registered Ukrainian G-Eval step templates. |

Frontier scoring requires one upfront `--scorer-egress-consent` plus a hard cap
(`--frontier-max-usd` and/or `--frontier-max-calls`). Spend is tracked in
`$DATA_DIR/run-eval/<run>/scorer/` (`consent.json`, `ledger.jsonl`, `ledger_state.json`). Hitting
the cap aborts with `abort.json` (`resumable: true`); resume reloads the ledger so spend never
silently exceeds the cap. Each successful (or failed-but-attempted) frontier call also
checkpoints `case_index` plus `faithfulness` / `answer_relevancy` in `ledger.jsonl`; on resume
`frontier_scorer` replays those scores and issues provider calls only for unscored cases
(`src/llb/scoring/policy/ledger.py`, `frontier.py`). Headline ranking is unchanged: judges remain
diagnostic until calibration rho clears the trust threshold.

```bash
llb run-eval --scorer-policy local --judge-model <model> --judge-rho <rho>
llb run-eval --scorer-policy human
llb run-eval --scorer-policy frontier --judge-model openai/<model> \
  --scorer-egress-consent --frontier-max-usd 2.00 --judge-rho <rho>
```

Tests live under `tests/llb/scoring/test_scorer_policy*.py` (fake litellm completions; no network),
including a mid-batch abort/resume case-checkpoint test that proves the second pass issues
`N - K` new calls after `K` cases were already scored.

### Frontier Judge Agreement and Cost Report

`src/llb/scoring/frontier_agreement/` produces the evidence an operator needs to decide whether a
frontier judge may gate autonomously. It scores a filled calibration worksheet with each named
provider through the scorer-policy seam (same consent, ledger, cap, and resume guarantees as a
scored run), then reports two rank correlations per provider -- against the human `human_rating`
and against the local judge's `judge_rating` -- plus measured cost per item.

| Module | Responsibility |
| --- | --- |
| `items.py` | Worksheet rows -> judgeable records; contexts are windows of the gold source doc |
| `provider.py` | One provider's run under `resolve_scorer(lane="frontier")` |
| `agreement.py` | Spearman rho + bootstrap CI per metric; cost-per-item and cap math |
| `report.py` | `report.md` including the operator's sign-off table |
| `run.py` | Orchestration; writes the run bundle |

Design points:

- Grounding contexts come from the gold set's source spans (a `GOLD_CONTEXT_WINDOW_CHARS`-wide
  window of the source document), not from retrieval. Judge agreement is measured with retrieval
  held constant so a retrieval regression cannot masquerade as judge disagreement.
- Correlation is rank-based, so the human 1..5 scale and the judge 0..1 scale are compared without
  rescaling. The headline metric is `mean` -- the same scalar `judge_value` that a scored run
  records per case (`src/llb/scoring/judge/scorer.py`, shared with `runner_judge`).
- A recommended cap is `cost_per_item * n_items * CAP_SAFETY_FACTOR` (2.0), rounded up to the cent.
  When litellm cannot price a model, `cost_usd` is 0; the report marks the provider unpriced and
  recommends no cap rather than guessing one.
- Providers are independent: each owns its ledger directory, so one provider's budget abort or
  transport failure is recorded under `failures` while the others keep their evidence.
- `recommendation` is a machine reading of the rho gate only. `human_decision` stays `pending`
  until an operator records an accept or reject; the report says so explicitly.

```bash
make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<litellm-id>[,<id>...] \
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>
```

Artifacts land under `$DATA_DIR/frontier-judge/<run>/`: `report.md` and `agreement.json` at the
root, and per provider a `<provider-slug>/` holding `scores.jsonl` plus the standard `scorer/`
consent and ledger files. The lane refuses to run without both an explicit egress consent and a
cap. Tests are `tests/llb/scoring/test_frontier_agreement.py` (injected fake completers; no
network, no spend), covering the rho and cap math, the artifact bundle, per-provider failure
isolation, and resume-after-abort.

The provider keys, the spend approval, and the accept/reject decision per provider remain human
steps; see the `frontier-judge-authorization` task in [plan.md](../../plan.md).

## Frontier Prep Utilities

`src/llb/prep/frontier/client.py` contains GPU-free Litellm-backed utilities that emit unverified review
material:

- `prepare_goldset`: drafts question, answer, and exact source span triples from real documents;
- `prepare_synthetic_corpus`: generates synthetic documents with planted labels.

Both are injectable for tests and write provenance. A planter model must differ from the judge model
to avoid circular evaluation.
