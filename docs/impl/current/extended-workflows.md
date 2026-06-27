# extended workflow Extended Workflows Current State

Covers the delivered extended-agentic / verification / prompt-system work. The 16 GB backend matrix,
power metric, and vector-store adapters live in
[`platform-vector-matrix.md`](platform-vector-matrix.md).

## agentic harness comparison Agentic harness comparison (LangGraph vs CrewAI vs loop)

The agentic benchmark (`TIER_AGENTIC`) now runs under a pluggable HARNESS, holding the task set,
`ToolWorld`, objective scoring, and gated judge fixed so the harness is the only variable.

- Seam: `Harness` Protocol `(task, complete, catalog, max_steps) -> Episode` in
  `src/llb/bench/agentic.py`; `run_episode` is the pure `loop` harness (catalog injectable, no
  behavior change). Named harnesses: `HARNESS_LOOP | HARNESS_LANGGRAPH | HARNESS_CREWAI`.
- Harness package `src/llb/bench/harness/`: `base.loop_harness`; `langgraph.build_agentic_graph`
  (mirrors `build_multi_hop_graph` -- `agent` node + `tool` node + two conditional edges, needs the
  `[eval]` extra) with pure node/router functions + `step_graph_pure` for CI equivalence;
  `crewai.make_crewai_harness` (opt-in `[crewai]` extra, injectable `crew_runner`, a fake crew
  proves the adaptation). `get_harness(name)` resolves lazily.
- `run_agentic(..., harness_name=, harness=)` records `config.harness`; board axis
  `board.data.harness_comparison(data_dir, model)` ranks ONE model across `{loop, langgraph,
  crewai}` under `TIER_AGENTIC` (each harness a board row, candidate model fixed), reusing
  `rank_board` + bootstrap CIs. CLI: `bench-agentic --harness ...` and `bench-agentic-compare
  --model ...`; Streamlit board shows the per-model harness comparison.
- Tests: `tests/test_harness.py` (loop==run_episode, langgraph nodes reproduce the loop, fake-crew
  adaptation, manifest harness tag, board comparison); a `@slow` test runs the real compiled
  LangGraph harness when `[eval]` is installed.
- Real CrewAI path: validated on `crewai==1.15.0` (pinned `[crewai] = crewai>=1.15,<2`).
  `run_real_crew` wraps the candidate `complete` as a `crewai.llms.base_llm.BaseLLM` subclass
  (`_make_candidate_llm`), builds each `BaseTool` with a pydantic `args_schema` derived from the
  ToolDef params (`_build_crew_tool`), executes tools against the shared world via the recording
  executor, and disables CrewAI tracing/telemetry for ASCII-only, no-egress runs. A scripted-ReAct
  candidate drives a deterministic-success task to the SAME Episode as the loop (success, transcript,
  `n_tool_calls=2`, `n_steps=3`). How-to + actor/model/document extension guide:
  [`docs/guides/crewai-harness.md`](../../guides/crewai-harness.md).

## judge diagnostics Non-blocking forward verification

- Harness board: the agentic harness comparison comparison ranks objectively with bootstrap CIs from FAKE endpoints (no
  GPU), harness-tagged.
- Judge diagnostic observability (`src/llb/scoring/judge_diag.py`): per-record classification of a
  zero-valued judge score -- `empty_answer` (candidate fault) vs `malformed_judge_json` /
  `judge_transport_error` (local judge fault) vs `zero_score`. `summarize_judge_diagnostics` rolls
  up `{n, n_ok, n_zero, reasons}`; `run_gated_judge` annotates `JudgeOutcome.diagnostics`; every
  category runner records it in the manifest `config.judge_diagnostics` + `judge.diagnostics`. The
  real DeepEval path threads precise reasons via `deepeval_scorer(..., diagnostics_out=)` (an
  optional `failures` map on `_measure_judge_metric` classifies the exception); a fake scorer infers
  empty/zero from `(answer, score)`. `bench-agentic` echoes the diagnostics.
- Strict-JSON judge smoke check: `judge.experiment.judge_smoke_check` runs ONE grounded-true case
  and fails (naming the reason) when the judge cannot emit a well-formed, non-zero strict-JSON score;
  CLI `judge-smoke --judge-model ... [--judge-base-url ...]` (exit 2 on failure) -- run before a long
  judged run.
- Tests: `tests/test_judge_diag.py` (classifier, summary, gated wiring, agentic manifest, smoke
  pass/zero/malformed).

## RAG prompt-system comparison Human-assisted RAG prompt-system generation

Operator-facing package `src/llb/prompt_system/` (pure + deterministic, injectable tokenizer) turning
a supplied corpus into reviewable, budget-fitted, manifest-addressable RAG prompt systems.

- `corpus.py`: read `.md`/`.txt`, split paragraphs (exact source spans), select a salient ANTHOLOGY,
  summarize per-doc METADATA, build the knowledge-graph-to-RAG MAPPING (salient term -> grounding
  passage ids). `build_corpus_package`.
- `budget.py`: `Tokenizer` seam + dependency-free `CharRatioTokenizer`; `plan_budget` reserves
  question/chunk/transcript/answer tokens; `fit_sections` trims anthology/graph/metadata to the
  remaining budget and emits a dropped-context report.
- `template.py`: editable `TemplateFields` (role / instruction / metadata density / graph-reference
  style / anthology size) -> `render_package` -> `PromptPackage` (`system_prompt` +
  `additional_prompt`). Harness compatibility: `PromptPackage.apply(prompt)` / `wrap_complete` so the
  SAME package drives the RAG path and any agentic harness without touching scoring.
- `review.py`: `PromptCandidate` + approve/revise/pin/reject + JSON persistence + `summarize_review`.
- `tuning.py`: `variant_grid` over the knobs + `generate_candidates` (deduped by prompt-system id).
- `manifest.py`: corpus/mapping/template digests + `prompt_system_id` (same corpus + fields -> same
  id) + `prompt_system_provenance`.
- `pipeline.prepare_prompt_system` writes `$DATA_DIR/prompt-system/<ts>/` by default, or a stable
  review/sample directory via `out_dir` / CLI `--out-dir` (anthology, doc_metadata,
  graph_rag_mapping, candidates, manifest with digests + tokenizer + context budget).
- Benchmark integration: `run_agentic(..., prompt_system=)` records `config.prompt_system`; board
  axis `board.data.prompt_system_comparison(data_dir, model, harness=None)` ranks one model across
  prompt-system ids under `TIER_AGENTIC`.
- Baseline RAG integration: `run-eval --prompt-system <id> [--prompt-package <run-dir|file|run/id>]`
  loads a reviewed candidate through `prompt_system.selection`, prepends the candidate system
  prompt and attached context to the normal retrieve->generate graph, records
  `prompt_system_provenance` at the top level of the run manifest, and mirrors the id into
  `config.prompt_system` for board lookup.
- Board integration: `board.data.load_rag_prompt_system_records` scans final `$DATA_DIR/run-eval/*`
  bundles and keeps the best run per `(model, prompt_system)`. `rag_prompt_system_comparison` ranks
  one model across prompt ids. The Streamlit board shows a RAG prompt-system comparison section when
  such final bundles exist.
- CLI: `prompt-system-prepare --corpus-root ... [--out-dir ...] [--role ...] [--instruction ...]`,
  `prompt-system-review --run-dir ... --action summary|approve|pin|reject [--id]`,
  `prompt-system-compare --lane rag|agentic --model ...`.
- Tests: `tests/test_prompt_system.py` (corpus spans, budget trim, template styles, digests, review
  round-trip, tuning dedupe, stable `out_dir`, prompt package selection, pipeline artifacts,
  agentic prompt-system board axis); `tests/test_runner.py` (RAG prompt injection + run manifest
  provenance); `tests/test_board.py` (final-split RAG prompt-system records and comparison).
- How-to: [`docs/guides/prompt-system-rag.md`](../../guides/prompt-system-rag.md).

### RAG prompt-system sample IP regulation sample and Gemma 4 comparison

Committed sample assets:

- `samples/goldsets/ip_regulation_uk/`: 8 verified, human-authored items over the IP regulation
  corpus, split into 4 `tuning` and 4 `final` cases. The canonical corpus file is
  `samples/corpus/ip_regulation_uk.md`; the gold-set-local `corpus/ip_regulation_uk.md` path is a
  relative symlink for tools that expect a fixture-local corpus root.
- `samples/prompt_system/ip_regulation_uk/`: default generated prompt-system package.
- `samples/prompt_system/ip_regulation_uk/tuned/`: prompt package generated with a short-answer
  instruction override; candidate `14d263ea6a40` is pinned after tuning.
- `samples/prompt_system/ip_regulation_uk/graph/`: curated GraphRAG-shape tutorial graph. A live
  `build-graph --extract-model gemma4:e4b --extract-no-think` run reached Ollama, but Gemma returned
  unparseable extraction JSON, so the committed graph is explicitly marked `curated-sample`.
- `samples/prompt_system/ip_regulation_uk/example_results.json`: exact command/result record for the
  local Gemma run.

Manual local run on `gemma4:e4b` / Ollama under `DATA_DIR=.data/prompt_system_ip_example`:

- Retrieval: `validate-retrieval --k 5` on the IP gold set gave recall@5=1.000 and MRR=1.000.
- Tuning split: baseline objective 0.709; default prompt candidates tied 0.709; tuned lean prompt
  `14d263ea6a40` improved to 0.778; richer tuned prompt `913266aa4cb3` regressed to 0.683.
- Final split: baseline objective 0.687; tuned lean prompt `14d263ea6a40` regressed to 0.578;
  default prompt `0a68e417ea71` regressed to 0.487.
- RAG prompt-system compare: `prompt-system-compare --lane rag --model gemma4:e4b` ranked
  `14d263ea6a40` above `0a68e417ea71` among final prompt-system-tagged runs, but both remained below
  the plain baseline. The operating decision for this tiny corpus is to keep baseline RAG and use
  the prompt-system lane as the audit trail.

The arXiv paper [2406.18902](https://arxiv.org/abs/2406.18902) was used only as a methodology
guardrail: prompt tuning is a selected pipeline, so tuning wins are not reported as final wins unless
the held-out final split confirms them.
