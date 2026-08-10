# Extended Workflows

Extended workflows cover comparison axes that sit beside the main RAG leaderboard: agentic
harnesses, agent context-management policies, judge diagnostics, and prompt-system packages.

This page is the AREA INDEX: each workflow lives in its own page under
[`extended-workflows/`](extended-workflows/).

## Agentic harness and loop policy

| Page | What it answers |
| --- | --- |
| [Agentic harness comparison](extended-workflows/agentic-harness.md) | The harness seam policies transfer on, and the host evidence behind it |
| [Agent loop-policy recommendation](extended-workflows/loop-policy-recommendation.md) | Which loop policy to run, and the powered repeat-noop comparison behind the recommendation |
| [Localized repeat feedback and its transfer](extended-workflows/repeat-feedback-transfer.md) | Whether the repeat-feedback result survives another seed, family, task family, controller, and channel |

## Agent context management

The compact-versus-cap chain reads in order: what the policies are, which one wins on a long
transcript, where the crossover sits, what has been published from it, and what a constant change
to any of it invalidates.

| Page | What it answers |
| --- | --- |
| [Agent context policies](extended-workflows/agent-context-policies.md) | The policy set, its host evidence, and aggregate-safe observation trimming with compact-finish recovery |
| [Compact versus cap](extended-workflows/compact-versus-cap.md) | Active compaction against an observation cap on long and memory-dependent transcripts, including summarizer cost and cross-family transfer |
| [Cap-fitting boundary and crossover geometry](extended-workflows/crossover-geometry.md) | Where compact stops repaying its summary call: the boundary surface, the trigger axis the routing rule lives on, the crossover as a fold step, and the step-aligned summarize-input cap |
| [Published values under the shipped cap](extended-workflows/published-values.md) | Which published compact evidence a summarize-bound change can move, how a number resolves to its run, and the registered arithmetic and reading it declares |
| [Published-value implementation map](extended-workflows/published-value-implementation.md) | The module and test inventory for provenance, derivation, arithmetic, readings, and crossover restatement |
| [Policy-constant change audit](extended-workflows/policy-constant-audit.md) | Which published agentic numbers a context-policy constant change invalidates, the geometry that tests the compound guarantee, which constant pair separates the two readings, and the CI gate that pins the shipped constants |
| [Context-policy constant sweeps](extended-workflows/context-policy-constants.md) | The cap / head-share / `keep_last_n` pin-or-expose sweep, and `keep_last_n` on longer transcripts |
| [Context-policy comparison](extended-workflows/context-policy-comparison.md) | Running one agentic category across context policies and reading the comparison |

## Prompts, judging, and fine-tuning

| Page | What it answers |
| --- | --- |
| [Judge diagnostics](extended-workflows/judge-diagnostics.md) | What a zero judge score means, and the pre-run judge smoke check |
| [Prompt systems and the self-improvement loop](extended-workflows/prompt-systems.md) | Reviewable prompt-system packages, the sample prompt assets, and the local self-improvement loop |
| [Fine-tuning search and trainability](extended-workflows/finetuning-search.md) | Hyperparameter search with split discipline, budget and resume, and which compressed checkpoints are trainable |
| [Adapter registry and lifecycle](extended-workflows/adapter-registry.md) | Adapter staleness, the contamination guard, serving, garbage collection, and the committed fixtures |
