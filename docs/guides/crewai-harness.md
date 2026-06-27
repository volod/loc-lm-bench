# CrewAI agentic harness -- validate, extend, and ground in documents

The M7.1 agentic benchmark runs one model under a pluggable harness so the comparison axis is
LangGraph vs CrewAI vs the pure loop, holding the task set, `ToolWorld`, objective scoring, and the
gated judge fixed (see [`milestone-7-extended-workflows.md`](../impl/current/milestone-7-extended-workflows.md)).
CrewAI is an opt-in extra; this guide is the host-only validation how-to plus the actor / model /
document extension guide.

## 1. Install and pin

The CrewAI harness is pinned to the 1.x line and validated on `crewai==1.15.0`:

    uv pip install -e ".[crewai]"

The base install and `make ci` never import CrewAI; only this extra pulls it. If you upgrade past
1.x, re-run the validation below -- CrewAI's custom-LLM and tool APIs change across majors.

## 2. Validate the real crew on a host

The fake-crew unit tests (`tests/test_harness.py`) cover the Episode adaptation with no dependency.
This step validates the live CrewAI integration (the LLM adapter, tool execution, ReAct parsing).

Fast scripted check (no model, no GPU) -- the candidate `complete` returns CrewAI ReAct turns and the
shipped `run_real_crew` must execute the tool against the shared world and capture the final answer:

    .venv/bin/python - <<'PY'
    from llb.bench import agentic, tool_world as tw
    from llb.bench.harness.crewai import make_crewai_harness
    script = [
        'Action: write_file\nAction Input: {"path": "result.txt", "content": "84"}',
        'Thought: done\nFinal Answer: done',
    ]
    i = {"n": 0}
    def scripted(_p):
        j = i["n"]; i["n"] += 1; return script[min(j, len(script) - 1)]
    task = agentic.AgenticTask("t", "write 84", setup={},
        success=[{"kind": "file_equals", "path": "result.txt", "value": "84"}])
    ep = make_crewai_harness()(task, scripted, tw.tool_catalog(), max_steps=6)
    assert ep.success and ep.world.files["result.txt"] == "84" and ep.n_tool_calls == 1
    print("crewai harness OK:", ep.answer)
    PY

Real model over Ollama (a tool-capable instruct model), the full benchmark cell:

    .venv/bin/llb bench-agentic --harness crewai --backend ollama --model <tag> \
      --tasks samples/agentic_tasks_uk.json

Then produce the comparison board across harnesses for one model:

    .venv/bin/llb bench-agentic --harness loop      --backend ollama --model <tag>
    .venv/bin/llb bench-agentic --harness langgraph --backend ollama --model <tag>
    .venv/bin/llb bench-agentic --harness crewai    --backend ollama --model <tag>
    .venv/bin/llb bench-agentic-compare --model <tag>

Expected on 1.15.0: a deterministic-success task scores identically to the loop/langgraph harness
(same `n_tool_calls` / `n_steps` / final env-state), confirming the harness is the only variable.

## 3. How the adapter works (the approach)

`run_real_crew` (in `src/llb/bench/harness/crewai.py`) wraps the SAME inputs every harness gets and
adapts the crew's result back to the canonical `Episode`:

- **Candidate as the crew LLM.** `_make_candidate_llm` builds a `crewai.llms.base_llm.BaseLLM`
  subclass (CrewAI 1.x requires a real `BaseLLM`, not a duck-typed object) whose `call(messages,
  ...)` flattens the ReAct conversation to one prompt string and forwards it to the candidate
  `complete`. The call count rides in a closure dict (CrewAI's `BaseLLM` is pydantic) and becomes
  `n_steps`.
- **Tools over the shared world.** `crew_tool_specs` maps each `ToolWorld` tool (minus `finish`) to a
  `BaseTool` whose `args_schema` is a pydantic model generated from the ToolDef's parameter names
  (all optional strings, so a partial Action Input reaches the world rather than failing
  validation). `_run(**kwargs)` calls a recording executor that runs the tool against the SAME
  world and appends `(name, args, observation)` to the transcript -- so the Episode is faithful
  regardless of CrewAI's internal orchestration.
- **Answer + scoring.** CrewAI's `Final Answer` becomes `Episode.answer`; `check_success` re-runs the
  objective env-state / answer assertions, so the scorer + the gated judge are unchanged.
- **No egress, clean logs.** `run_real_crew` sets `CREWAI_TRACING_ENABLED=false`,
  `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true` so a benchmark run does not phone home and
  logs stay line-oriented ASCII.

Key version-sensitive points (re-check on a CrewAI upgrade): the `BaseLLM.call` keyword set, the
`BaseTool` abstract `_run` + `args_schema` contract, and the `Agent(role, goal, backstory, tools,
llm, max_iter)` / `Task(description, expected_output, agent)` / `Crew(agents, tasks).kickoff()`
fields.

## 4. Extend the actors and models

- **Swap the model.** `--model` + `--backend ollama|vllm|llamacpp` (or `--base-url` for a running
  OpenAI-compatible endpoint) picks the candidate; the crew LLM is always that candidate via
  `complete`, so no CrewAI model config is needed and the comparison stays apples-to-apples.
- **Edit the actor.** The single agent's `role` / `goal` / `backstory` live in `run_real_crew`. Tune
  them there to change how the crew reasons; keep the task set + tools fixed so the score still
  isolates the harness, not a hidden prompt change.
- **Add tools.** Extend the deterministic `ToolWorld` (a pure `(env, args) -> observation` handler in
  `src/llb/bench/tool_world.py`) and add it to `tool_catalog()`; `crew_tool_specs` picks it up
  automatically with an args schema derived from its ToolDef.
- **Multi-actor crews (future).** `run_real_crew` builds a single-agent crew on purpose (the M7.1
  comparison is one agent under different harnesses). A multi-agent crew is a separate experiment:
  add agents/tasks to the `Crew`, but record it as a distinct harness id so its scores never mix
  with the single-agent cells.

## 5. Ground the crew in documents (RAG prompt system)

Combine the harness with the M7.3 prompt-system lane so the crew answers from a supplied corpus:

    .venv/bin/llb prompt-system-prepare --corpus-root <corpus_dir>

Then wrap the candidate `complete` with a rendered `PromptPackage` before driving any harness -- the
SAME package works for loop / langgraph / crewai without touching scoring:

    from llb.prompt_system import (
        read_corpus, build_corpus_package, plan_budget, CharRatioTokenizer,
        render_package, TemplateFields, wrap_complete,
    )
    corpus = build_corpus_package(read_corpus("<corpus_dir>"))
    budget = plan_budget(8192, question_tokens=64, chunk_tokens=1024)
    pkg = render_package(corpus, TemplateFields(), budget, CharRatioTokenizer())
    grounded = wrap_complete(complete, pkg)   # pass `grounded` to run_agentic / the harness

Record the prompt-system id on the run (`run_agentic(..., prompt_system=<id>)`) so
`prompt-system-compare --model <tag>` ranks the model across prompt systems on the SAME harness.

## 6. Troubleshooting

- "needs the [crewai] extra" -> `uv pip install -e ".[crewai]"`.
- The crew never calls a tool -> the candidate did not emit CrewAI's `Action:` / `Action Input:`
  ReAct format; use a tool-capable instruct model, or confirm the scripted check above still passes
  (a format/version drift shows up there first).
- API errors after a CrewAI upgrade -> re-validate with the scripted check; adjust
  `_make_candidate_llm.call`, `_build_crew_tool` (args schema), or the `Agent`/`Crew`/`Task` fields
  to match the new version, then re-pin `[crewai]` in `pyproject.toml`.
