# Autonomous RAG Recommendation

Use this workflow when you have a local mixed corpus and want one scored model plus RAG
configuration without manually chaining the preparation, search, prompt, and evaluation commands.

## At a glance

```text
1. prepare       make venv; ollama serve; make prep-models
2. run           make auto-rag CORPUS=<corpus-dir> SCORER_POLICY=auto
3. inspect       rag_recommendation.yaml + report.md
4. resume        repeat with AUTO_RAG_RUN_ID=<same-id> after interruption
```

The autonomous lane sends no data to a hosted service. It uses local ontology drafting and a
local judge by default. Frontier verification is available only with explicit egress consent and
a hard budget.

## Autonomous run

```bash
make auto-rag \
  CORPUS=<corpus-dir> \
  SCORER_POLICY=auto \
  AUTO_RAG_RUN_ID=<run-id>
```

The default 16 GiB-host drafter is the installed 12B Ukrainian MamayLM GGUF. Override
`AUTO_RAG_DRAFT_MODEL` for another host tier. The default candidate manifest is
`samples/configs/models_uk.yaml`; limit a bounded or host-specific search by model name:

```bash
make auto-rag \
  CORPUS=<corpus-dir> \
  AUTO_RAG_RUN_ID=<run-id> \
  AUTO_RAG_CANDIDATE_MODELS=<model-a>,<model-b> \
  AUTO_RAG_TRIALS=10 \
  AUTO_RAG_PARITY_CHECK=1
```

The full production defaults use 20 trials and all resolvable candidates. `AUTO_RAG_EVAL_LIMIT`
and `AUTO_RAG_DOC_LIMIT` are bounded evidence/debug controls, not headline settings.

## Resume and failures

Always keep the run id when retrying:

```bash
make auto-rag CORPUS=<corpus-dir> AUTO_RAG_RUN_ID=<run-id>
```

Completed stage markers are reused. A settings change under the same id is refused so a run never
silently combines different corpora, models, thresholds, or budgets. Use a new run id for changed
settings. The journal records failed attempts as history; a later successful stage marker is the
current output.

If retrieval misses its recall gate, the pipeline tries its bounded repair configurations. If all
miss, improve the corpus/gold spans or extend retrieval deliberately; the pipeline does not tune a
generator against a known-bad retriever.

## Human-assisted verification

```bash
make auto-rag \
  CORPUS=<corpus-dir> \
  SCORER_POLICY=human \
  AUTO_RAG_RUN_ID=<run-id>
```

The command pauses with exit code 3 and prints the generated worksheet path. Open it in the shared
review workbench, decide every row, then repeat the exact command. Resume applies the verification
tolerance and continues only from an accepted ledger. The separate human acceptance comparison in
`docs/impl/plan.md` remains the place for measured reviewer-throughput and recommendation-quality
judgment.

## Frontier verification

Only use frontier scoring for a corpus whose egress is authorized:

```bash
make auto-rag \
  CORPUS=<corpus-dir> \
  SCORER_POLICY=frontier \
  SCORER_EGRESS_CONSENT=1 \
  SCORER_MAX_CALLS=<call-cap> \
  AUTO_RAG_JUDGE_MODEL=<provider-model> \
  AUTO_RAG_RUN_ID=<run-id>
```

`SCORER_MAX_USD` can replace or supplement the call cap. Consent, spend, calls, and resumable
scores are recorded inside the run.

## Read the result

Start with `$DATA_DIR/auto-rag/<run>/report.md`. Then use `rag_recommendation.yaml` as the exact
deployment/evaluation handoff. Confirm these evidence blocks before adopting it:

- `verification`: accepted count, policy, and scorer identity;
- `retrieval_validation`: recall/MRR, selected baseline or repair, and attempted configurations;
- `joint_search`: final-only scoreboard and selected overrides;
- `final_split`: prompt-equipped quality, retrieval metrics, and optional parity record.

The run is intentionally local evidence, not a universal model ranking. A small corpus or bounded
case limit supports workflow validation but should not be presented as broad model confidence.
