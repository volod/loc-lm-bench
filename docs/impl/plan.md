# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

### draft-vllm-endpoint (optional, performance)

- User-visible outcome: quickstart PDF drafting can use a vLLM-served candidate (the fastest
  ranked model on 16 GB hosts serves ~2.5x more tok/s than the best Ollama candidate), instead of
  being restricted to Ollama-served models.
- Scope boundary: reuse the existing vLLM server lifecycle in `src/llb/backends/vllm.py` to start
  the target and point `prepare-goldset-draft --base-url` at it; extend drafter selection
  (`llb.quickstart.model_choice drafter`) to accept vLLM-backed candidates once serving works. The
  blocker to solve first: reasoning-model output control -- the draft endpoint disables hidden
  thinking via Ollama-native `think=false`, and vLLM needs an equivalent (for example
  `chat_template_kwargs`) or JSON output collapses for reasoning models.
- Data and artifact paths: unchanged draft bundle layout; provenance records the endpoint base URL
  and backend.
- Execution path: `make quickstart-pdf-corpus-draft` on a CUDA host with a vLLM-ranked
  recommendation JSON present.
- Acceptance gates: a one-document probe bundle passes calibration gates with a vLLM-served
  reasoning model; drafter auto-selection covered by unit tests for both backends; `make ci`
  green.
- Documentation target: `docs/impl/current/data-prep.md` (drafter selection) and
  `docs/impl/current/robustness-ontology-backends.md` (endpoint).

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
