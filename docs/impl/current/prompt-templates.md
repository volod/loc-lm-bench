# Prompt Templates

LLM-facing prompt text is package data under `src/llb/prompts/templates/`. Python modules keep the
domain logic and call the shared renderer in `llb.prompts`; long prompt literals do not live in
benchmark, prep, eval, or scoring modules.

## Runtime Surface

`src/llb/prompts/engine.py` provides a small `{{ name }}` renderer with dotted lookup and fast
failure on missing variables. `src/llb/prompts/registry.py` loads the generated registry and exposes
the runtime helpers re-exported from `llb.prompts`:

- `render_text(template_id, values)` for one prompt.
- `render_chat(template_id, values, augmentation=None)` for role-tagged chat messages.
- `render_text_list(template_id, values=None)` for fixed prompt sets such as telemetry throughput.
- `render_text_map(template_id, values=None)` for keyed prompt fragments such as judge intents.

`PromptAugmentation` adds system or user prefixes/suffixes around rendered chat messages. The RAG
path uses this for prompt-system packages: package system text is prepended to the baseline system
message, and package context is rendered through the same registry before the current RAG context.

## Registry

Each prompt directory contains one or more `.txt` templates plus `*.prompt.json` descriptors. A
descriptor declares the stable template id, kind, and referenced files. The generated registry lives
at `src/llb/prompts/templates/registry.json` and records every template path with a SHA-256 digest.

Regenerate the registry after adding or editing prompt descriptors:

```bash
uv run --no-sync python -m llb.prompts.registry
```

`tests/llb/prompts/test_prompts.py` compares the checked-in registry with a fresh scan, so stale
hashes or missing descriptors fail locally.

## Current Coverage

The registry holds the prompts for:

- backend throughput telemetry;
- single-call RAG, map-reduce, and multi-hop evaluation messages;
- category runners: security, tooling, agentic, summarization, structured output, and text analysis;
- generated agentic search tasks and CrewAI harness agent text;
- frontier goldset drafting, cross-checking, synthetic corpus generation, chat corpus labeling, and
  ontology extraction/drafting;
- graph community summaries;
- Ukrainian DeepEval judge steps, result prompt, parameter labels, and judge bias note;
- prompt-system defaults, section headings, and context item formats.

This keeps product prompt review in one tree while preserving the existing public builder
functions, such as `analysis_prompt`, `draft_prompt`, `build_messages`, and `text_tool_prompt`, for
callers and tests.
