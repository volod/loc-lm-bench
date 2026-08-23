# Context-Policy Comparison

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

`bench-chain-context` ranks context-management POLICIES for ONE fixed model over a verified
chain-of-questions set. It holds the model, the chain set, retrieval, and the scoring fixed and
varies only the policy -- the row label -- exactly as the agentic harness comparison holds
everything fixed and varies the harness. It answers "which harness and context policy improves
scores, and how should system prompts be sequenced" over multi-step questions where each step's
answer depends on the prior steps.

Core locations:

- `src/llb/bench/chain_context/policy.py`: policy execution and per-step context assembly;
- `src/llb/bench/chain_context/run.py`: run orchestration and persistence;
- `src/llb/bench/chain_context/report.py`: result contract, provenance projection, digest, and
  recommendation rendering;
- `src/llb/board/chain_context.py`: one-model policy comparison board rows under
  `TIER_CHAIN_CONTEXT` (mirrors `board/harnesses.py`);
- `src/llb/prompts/templates/bench/chain_context/`: the reviewable role/instruction prompt-system
  templates and the step scaffold.

The four policies (each a fresh retrieval per step PLUS a different memory of the prior steps):

- `fresh` -- fresh retrieval per step, NO prior-step carryover (the naive baseline);
- `history` -- the accumulated full (question, answer) transcript;
- `summary` -- a running model-written summary of the prior steps (one extra model call per step);
- `roles` -- a staged role/system-prompt sequence (librarian -> analyst -> answerer) built from
  the prompt-system role templates, plus the accumulated transcript. First step is the librarian,
  last is the answerer, middle steps are analysts; a 2-step chain is librarian -> answerer.

```bash
llb bench-chain-context --chains <chains.jsonl> --corpus <corpus-dir> \
  --model <model> --backend <backend> --policies fresh,history,summary,roles
make bench-chain-context CHAIN_CONTEXT_MODEL=<model> CHAIN_CONTEXT_BACKEND=<backend>
```

Retrieval reaches the store through the injectable `Retriever` seam (`retrieve(question, k)`), and
the model through the same injectable `complete` (prompt -> raw text) every category uses, so the
exact context assembled per policy per step is unit-tested over a FAKE endpoint with no GPU
(`tests/llb/bench/chain_context/test_chain_context.py`). Each step's answer is scored objectively
against its reference answer (`scoring.correctness` token-F1); the headline is FINAL-answer
correctness per chain (does the chain end right), with per-step correctness recorded alongside, both
with bootstrap CIs. Each policy persists its OWN run bundle under
`$DATA_DIR/chain-context/<timestamp>/` tagged with the policy (mirroring the per-harness agentic
bundles); provenance records the policy, the `prompt_system_ids` (the role/instruction template
ids), and the `chain_set_digest`. Verified-data stamping (`--data-verified` + `--verification-ref`)
follows the same category-suite gate as the other benchmarks. The board loader ranks all policies
together, and `llb recommend` gains a "Context policy" section naming the best policy per model when
bundles exist.

CUDA evidence (2026-07-11, RTX 4060 Ti 16 GB): the committed 20-chain fixture (40 steps) run
through `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, all four policies in one ~11 min invocation,
reliability 1.000 for each. Final-answer correctness ranked `roles` **0.789** [0.635, 0.915] >
`history` 0.625 > `summary` 0.534 > `fresh` 0.431: the naive no-carryover baseline is worst, any
memory of the prior steps helps, and the staged librarian -> analyst -> answerer sequence wins with
its CI resolved above the rest. One run bundle per
policy). The discriminating spread is the whole reason to measure the policy rather than assume one.
