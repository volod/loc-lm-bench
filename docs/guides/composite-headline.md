# Composite headline close-out

This guide is the operator flow for turning the category boards into one guarded composite
headline. The composite is a separate layer over the category boards; it does not weaken the rule
that security, tooling, agentic, summarization, structured output, and text-analysis each render
under their own Tier.

The headline is allowed only when every required category for a model is:

- present for the same frozen model cohort;
- scored on final inputs, not tuning or draft inputs;
- backed by a per-case objective series so a CI can be recomputed after reload;
- stamped as human verification gate-verified with a valid verification reference.

## Weights

The first composite is category suite-only. It uses the spec taxonomy proportions
for the category suite categories and renormalizes them over this subset:

| Component | Raw weight | Normalized category suite weight |
| --- | ---: | ---: |
| `text_analysis` | 20 | 0.3077 |
| `summarization` | 10 | 0.1538 |
| `structured` | 10 | 0.1538 |
| `security` | 10 | 0.1538 |
| `agentic` | 10 | 0.1538 |
| `tooling` | 5 | 0.0769 |

RAG quality, chat-period analysis, reliability, and efficiency stay outside this first category suite
headline. Reliability is still shown as a diagnostic (`avg_reliability`) so a high composite score
does not hide flaky runs.

## One-command pipeline

Use this after each category input has its human verification gate verification artifact:

```sh
make composite-headline \
  MODEL=<model-id> \
  BACKEND=<ollama|vllm|llamacpp> \
  COMPOSITE_TEXT_ANALYSIS_BUNDLE=<text-analysis-bundle> \
  COMPOSITE_VERIFICATION_REF=<bundle>/sample_manifest.json
```

For the committed sample fixtures, the Makefile defaults already point at
`samples/text_analysis_bundle_uk`, the other `samples/*_uk.json` category files, and the
category-specific sample refs under `samples/verification/composite_samples/`:

```sh
make composite-headline MODEL=<model-id> BACKEND=<backend>
```

This is a smoke/demo path over repo-authored fixtures. For real headline use, override the inputs
and refs with the actual frozen category bundles and their human verification gate artifacts.

If categories use different bundles or verification worksheets, pass one reference per category:

```sh
make composite-headline \
  MODEL=<model-id> \
  BACKEND=<backend> \
  COMPOSITE_TEXT_ANALYSIS_BUNDLE=<text-analysis-bundle> \
  COMPOSITE_TEXT_ANALYSIS_VERIFICATION_REF=<text-analysis-bundle>/sample_manifest.json \
  COMPOSITE_SUMMARIZATION_VERIFICATION_REF=<summarization-bundle>/sample_manifest.json \
  COMPOSITE_STRUCTURED_VERIFICATION_REF=<structured-bundle>/sample_manifest.json \
  COMPOSITE_SECURITY_VERIFICATION_REF=<security-bundle>/sample_manifest.json \
  COMPOSITE_AGENTIC_VERIFICATION_REF=<agentic-bundle>/sample_manifest.json \
  COMPOSITE_TOOLING_VERIFICATION_REF=<tooling-bundle>/sample_manifest.json
```

Optional knobs:

- `COMPOSITE_BASE_URL=http://localhost:<port>/v1` uses an already-running OpenAI-compatible
  endpoint.
- `COMPOSITE_REAL_CORPUS=1` records the text-analysis run as real-corpus, not synthetic.
- `JUDGE_RHO=<rho> JUDGE_MODEL=<judge-id> JUDGE_BASE_URL=<url>` enables the gated judge
  diagnostics for categories that support it.

Judge diagnostics are admitted only when `JUDGE_RHO` clears the configured gate. They stay
diagnostic: objective scores remain the headline. Empty candidate answers receive zero judge
scores, and malformed local-judge JSON zeros only the affected metric with a warning so the
composite preflight can still finish.

The target chains the required commands in order:

```sh
llb bench-text-analysis --data-verified --verification-ref <text-analysis-ref>
llb bench-summarization --data-verified --verification-ref <summarization-ref>
llb bench-structured --data-verified --verification-ref <structured-ref>
llb bench-security --data-verified --verification-ref <security-ref>
llb bench-agentic --data-verified --verification-ref <agentic-ref>
llb bench-tooling --data-verified --verification-ref <tooling-ref>
llb bench-composite
```

The final `bench-composite` call is the preflight. It prints a ranked composite only when there are
no missing-tier, unverified-data, or missing-CI blockers.

## Verification reference rules

Every `--data-verified` category run must provide `--verification-ref`. Accepted forms are:

- a reviewed `verify_sample.csv` whose rows are all decided and whose reject rate is within
  tolerance;
- a `sample_manifest.json` whose `worksheet` points to such a reviewed worksheet;
- an accepted-ledger directory or `accepted/goldset.jsonl` whose items are all `verified=true`.

Invalid references fail before model calls and before a verified manifest can be persisted. The
failure prints the reference path, artifact kind, reason, statistics, and next steps. Worksheet
statistics include `n`, decided, accepted, rejected, undecided, undecided failed checks, reject
rate, tolerance, and failing strata. Accepted-ledger statistics include item counts and sample
unverified ids.

Fix an invalid reference with the same human verification gate loop:

```sh
make verify-sample BUNDLE=<bundle> VERIFY_N=<n>
make verify-review VERIFY_WS=<bundle>/verify_sample.csv
make verify-accept BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv
```

Then rerun the category or pipeline with:

```sh
--data-verified --verification-ref <bundle>/sample_manifest.json
```

or:

```sh
--data-verified --verification-ref <bundle>/accepted
```

Do not backfill a committed seed set by editing manifests. If a committed sample file is promoted
to headline data, first package or adapt it as a category bundle, run human verification gate,
and use the resulting verification artifact.

The repository's committed sample refs are intentionally scoped to smoke/demo composite runs over
repo-authored fixtures. They are not a substitute for human verification gate on new AI-drafted,
adapted, or external data.

## Future AI-drafted or adapted bundle flow

1. Freeze the cohort: exact model ids, backend policy, category input bundle paths, judge policy,
   and prompt/data revisions.
2. For every AI-drafted or adapted category bundle, run sample -> review -> accept. Keep the
   worksheet, `sample_manifest.json`, and accepted ledger beside the bundle.
3. Run `make composite-headline` with the frozen model and the category verification refs.
4. If a category command fails the verification gate, follow the printed stats and next steps,
   update the worksheet or accepted ledger, and rerun the same pipeline command.
5. If `bench-composite` reports a blocker, rerun only the missing or stale category command, then
   rerun `llb bench-composite` without diagnostic bypass flags.
6. Publish through `llb board` only after the clean composite preflight exists.
7. Archive the decision trail: every category `manifest.json`, every category `scores.jsonl`,
   each verification artifact referenced by `verification_ref`, and the
   `bench-composite` output.

Diagnostic options:

```sh
llb bench-composite --allow-unverified
llb bench-composite --allow-missing-ci
```

Use these only to inspect data shape. Do not use diagnostic output as a headline.

## Troubleshooting

- `--data-verified requires --verification-ref`: provide a reviewed worksheet,
  `sample_manifest.json`, or accepted ledger.
- `worksheet has undecided row(s)`: run `make verify-review VERIFY_WS=<path>` until every sampled
  row has `decision=accept` or `decision=reject`.
- `reject rate ... exceeds tolerance`: inspect rejected rows, redraft or repair the source bundle,
  then rerun sample -> review -> accept.
- `ledger contains unverified item(s)`: rerun `make verify-accept` from a reviewed worksheet so the
  accepted ledger is emitted by the tool.
- `judge ... invalid JSON`: use the warning as a judge-quality diagnostic; the affected metric is
  zeroed, and objective category scoring still controls the headline.
- `category lacks per-case CI series`: rerun the category with current code so each per-case row
  carries `objective_score`.
- `missing required tier`: run that category for the same model id and verified data revision.
