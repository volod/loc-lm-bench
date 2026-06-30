# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

### gpu-tier-mistral-default

- User-visible outcome: complete the committed-goldset leaderboard family defaults by adding a
  vetted Mistral variant beside the existing MamayLM, Lapa, Gemma 4, and Qwen 3.6 defaults for each
  supported 12/16/24/32 GiB CUDA tier.
- Scope boundary: update `samples/models_uk.yaml` and `samples/config-example/manifest.yaml`; reuse
  `detect-gpu-vram`, `gen-serving-config`, `list-models`, `prep-models`, `sweep`,
  `platform-matrix`, and `board`; do not add gated models without explicit license metadata and
  operator-token guidance.
- Data and artifact paths: keep generated serving scripts under
  `$DATA_DIR/llb/serving/gpu-<tier>gb/`, model-prep caches in their backend-owned stores, and
  leaderboard runs under `$DATA_DIR/run-eval/`.
- Execution path: select one tier-appropriate Mistral source and backend, prefer local/offload GGUF
  where vLLM cannot fit, add the source to the planner and serving manifests, then run the host fit
  table and a small `sweep` on the committed fixture.
- Acceptance gates: `make lint-md`, model-planner tests for the added sources, `make list-models`
  on a representative 16 GiB host, and a documented reason for every family skipped on a tier.
- Documentation target: update the README goldset quickstart, `docs/inference/config-example.md`,
  and `docs/impl/current/platform-vector-matrix.md`.

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it.

Each task entry must include:

- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
