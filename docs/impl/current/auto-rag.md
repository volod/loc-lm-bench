# Auto-RAG Orchestrator

`llb auto-rag` and `make auto-rag` turn a mixed corpus into a scored RAG recommendation. The
pipeline owns the transitions between existing production stages; ontology drafting, retrieval,
joint model/config search, prompt-system packaging, and final evaluation remain in their focused
owner modules.

## Stage graph and resume

The fixed stage order is:

```text
ingest -> draft -> verification -> retrieval -> joint_search -> prompt_system
       -> final_eval -> recommendation
```

`src/llb/auto_rag/` contains the stage machine, atomic journal, production adapters, gate logic,
retrieval repair, final evaluation, and renderers. `src/llb/cli/auto_rag.py` exposes the command;
`make/eval/workflows.mk` exposes the standard Make target.

Each run pins every score- or artifact-affecting setting in
`$DATA_DIR/auto-rag/<run>/manifest.json`. A completed stage publishes
`stages/<stage>/result.json` atomically, then appends a completion event to `journal.jsonl`.
Reusing `--run-id` validates the settings fingerprint and skips every published stage. A truncated
or missing result marker is incomplete and is rerun. The ontology stage also reuses its
per-extraction-window journal when interruption happens inside drafting.

Tests in `tests/llb/auto_rag/` inject the full eight-stage graph and interrupt after each boundary;
all completed stages are called exactly once after resume. The tests also cover settings mismatch,
autonomous/human verification, score-cache reuse, retrieval repair, and recommendation fields.

## Verification policies

`--scorer-policy auto` resolves to the existing local scorer seam unless explicit frontier egress
consent and a hard call or USD cap are present; `local` and `frontier` select those lanes directly.
The gate scores every draft item for faithfulness and answer relevance. It also checks that the
worksheet's cited span remains present in its rendered context after whitespace normalization;
exact source offsets were already validated by the ontology bundle. Accepted rows must meet the
score threshold and leave at least one verified tuning row and one verified final row.

Every decision is appended to the run-level `scorer_ledger.jsonl`. Rows carry an algorithm
revision, policy, threshold, structural result, judge scores, and decision. A resumed gate reuses
matching rows instead of repeating completed judge calls. Frontier scoring additionally retains
the policy seam's budget ledger.

`--scorer-policy human` writes the full `verify_sample.csv` and exits with code 3 while decisions
are pending. Review that worksheet through the shared workbench, then invoke the same command with
the same run id. The resumed verification stage applies the normal tolerance gate and emits the
self-contained accepted ledger before downstream scoring can start.

## Retrieval and selection behavior

Retrieval validation first builds the recursive/hybrid baseline and requires recall@10 >= 0.8 by
default. If it misses, a bounded repair search compares smaller recursive chunks, markdown
chunking, and the dense-only control. Candidates rank by recall, then MRR, then stable declaration
order. Only a passing best store is published and carried into the evidence bundle.

Joint search reuses `llb.optimize.joint_search`: candidate resolution, tuning-only successive
halving, per-finalist multi-objective Optuna, and final-only scoreboard fences are unchanged. The
prompt stage calls `prepare_prompt_system` with the ontology bundle, then pins the
knowledge-tree candidate with the least context loss, breaking ties by tree depth and budget.
The final stage rebuilds the selected retrieval configuration and evaluates it with the pinned
prompt package on the final split. `--parity-check` performs an independent second final run and
requires the objective quality to match within `1e-9`.

## Commands and artifacts

```bash
make auto-rag CORPUS=<corpus-dir> SCORER_POLICY=auto

# bounded candidate set, stable resume id, and independent final parity
make auto-rag CORPUS=<corpus-dir> SCORER_POLICY=auto \
  AUTO_RAG_RUN_ID=<run-id> \
  AUTO_RAG_CANDIDATE_MODELS=<model-a>,<model-b> \
  AUTO_RAG_PARITY_CHECK=1

# human-assisted gate; repeat after workbench review
make auto-rag CORPUS=<corpus-dir> SCORER_POLICY=human \
  AUTO_RAG_RUN_ID=<run-id>
```

The completed run root contains:

```text
$DATA_DIR/auto-rag/<run>/
|-- manifest.json
|-- journal.jsonl
|-- scorer_ledger.jsonl
|-- artifacts.json
|-- rag_recommendation.yaml
|-- report.md
'-- stages/<stage>/result.json
```

`rag_recommendation.yaml` records the exact model/backend, serving knobs, chunking, retrieval
mode, fusion, reranking, query preparation, context budget, prompt-system id, and score evidence.
`report.md` is the compact operator view; `artifacts.json` links every stage bundle.

## CUDA evidence

The bounded deterministic run `auto-rag-ua-evidence-20260719` used the two-document Ukrainian
fixture corpus on an RTX 4060 Ti 16 GiB host. MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M drafted and
locally judged the goldset; MamayLM-12B and Lapa-12B were the joint-search candidates. The run used
9 requested drafts, 4 Optuna trials, a 2-case screen/eval cap, seed 13, and parity checking.

- Drafting kept 8 items; the local gate accepted 8/8 with faithfulness `1.0` and answer relevance
  `0.7-1.0`.
- Baseline hybrid retrieval passed without repair: recall@10 `1.0`, MRR `1.0` over all 8 items.
- Joint search selected Lapa-12B with semantic chunks of 704 characters, overlap 171, flat
  retrieval, top-k 3, and an 8192-token context budget. Its pre-prompt final quality was `0.3681`.
- Prompt preparation generated 126 candidates and pinned knowledge-tree id `13574ce894a6`
  (depth 3, 256-token tree budget).
- The prompt-equipped final run scored `0.6726` over 2 final items with recall@3 and MRR both
  `1.0`. The independent parity run also scored `0.6726`, for delta `0.0`.

Evidence lives under `$DATA_DIR/auto-rag/auto-rag-ua-evidence-20260719/`. The first verification
attempt exposed a whitespace-rendering mismatch in the structural gate; the revisioned normalized
check fixed it, and the outer journal resumed at verification without repeating ingest or draft.

Validation after the implementation: `make ci` passed Ruff, mypy, and 1,545 lightweight tests
with 1 skip and 42 slow tests deselected. `make lint-md` is part of the documentation closeout.
