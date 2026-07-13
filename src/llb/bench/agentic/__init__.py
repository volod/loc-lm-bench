"""agentic workflows runner -- in-sandbox tool execution under TIER_AGENTIC.

The agentic loop is the text analysis multi-hop controller pattern extended with TOOL CALLS + an in-sandbox
execution step: each step the model emits one tool call (reusing the tooling benchmark `parse_tool_call`), the
deterministic `ToolWorld` EXECUTES it, the observation is fed back, and the loop runs until the
model calls `finish` (or answers in prose) or the step budget is exhausted. Task success is an
OBJECTIVE assertion over the final env-state and/or the final answer; completion-rate is the
headline under `TIER_AGENTIC` and trajectory length + tool-call count are recorded as efficiency.

An OPT-IN, GATED judge adds a TRAJECTORY-QUALITY signal a deterministic check cannot cover (is the
final answer grounded in what the tools actually returned, and does it address the goal?). It is
recorded ALONGSIDE completion-rate -- never folded into the headline -- and only when the judge is
configured AND trusted (calibration `judge_rho >= threshold`, the judge calibration gate gate). The judge `scorer`
is injectable, so the wiring is provable with a FAKE judge (no DeepEval / endpoint / GPU), exactly
like the category expansion summarization faithfulness signal.

LangGraph is the fixed single agent harness for the design; this loop is the pure, langgraph-free
form of that controller -> execute -> controller cycle, so it is unit-tested from a FAKE
tool-calling endpoint with no GPU. The candidate is reached through an injectable `complete`.

Submodules: `model` (shared types + constants), `success` (objective assertion checks),
`episode` (the controller cycle + episode scoring), `trajectory` (the gated quality judge),
`persist` (run persistence), and `run` (top-level `run_agentic` orchestration).
"""
