# Scope Boundaries

## Resolved questions and scope boundaries

The design spec ([`spec.md`](../../design/spec.md)) is the source of truth for decisions; this
records the settled ones that affect WHAT is and is not built, so the forward plan
([`plan.md`](../plan.md)) stays forward-only.

Resolved open questions:
- **OQ2 -- judge locality (M3.8):** a LOCAL Gemma-4 judge, tiered by GPU class (12/16/32 GB),
  chosen for no corpus egress + reproducibility; the Gemma-family self-preference bias is
  disclosed (see "Judge model (OQ2 decided) + bias disclosure" above). The only residual is the
  human calibration ratings (M3.8 in `plan.md`), not the scorer or the model choice.
- **OQ3 -- first candidate-model list (M2):** seeded in `samples/models_uk.yaml`; the vLLM repo
  ids are verified via `prep-models`.
- **OQ6 -- MAX_JOBS build helper (M2):** the canonical `max_jobs()` lives in
  `scripts/shared/common.sh` (AGENTS.md) and caps every CUDA source build.
- **OQ4 -- text-analysis + chat corpus facts (confirmed 2026-06-25):** text-analysis reference
  answers must be AUTHORED (AI-draft -> MH.5-verify, the current pipeline -- they do NOT pre-exist);
  the text-analysis corpus is BOTH real + synthetic, scored + reported SEPARATELY via the runner's
  `synthetic` flag (never merged); and a REAL chat-log corpus exists for chat-period (run via the
  real path, reported separately). So the M5.4 text-analysis + chat-period residuals must wire the
  REAL path, not only the synthetic planter.
- **OQ-egress -- cross-check egress for the real corpus (resolved 2026-06-25):** the second-frontier
  cross-check verifier is injectable (`SecondFrontierVerify`), so egress is per-corpus: the real
  CHAT-LOG corpus uses a LOCAL verifier only (no egress -- inject a local `SecondFrontierVerify`);
  the real TEXT-ANALYSIS corpus has frontier (litellm) cross-check egress APPROVED; synthetic
  bundles keep the litellm default. MH.5 remains the human gate for all of them.

Rejected pushbacks (ruled the other way; do NOT revisit -- see spec.md "Outside-voice
resolutions"): defer-Optuna-to-finalists, LangGraph-only-where-needed, drop-MLflow,
drop-thermal-gate, defer-vLLM.

Genuinely out of scope (v-next): a FULL six-framework comparison axis -- M5.3 ranks the model under
ONE fixed harness (spec Appendix D), and only a TWO-harness comparison (LangGraph vs CrewAI) is
taken forward, as Milestone 7 in `plan.md`; the remaining frameworks (LangChain / LlamaIndex /
Haystack / AutoGen) stay deferred. Also out of scope: loc-lm-bench as a public leaderboard (it
consumes lang-uk / INSAIT results as a prior, never duplicates them).

No longer deferred (now forward work in `plan.md`, not "out of scope"): the LangGraph-vs-CrewAI
harness comparison and RAG prompt-system generation/tuning (Milestone 7), plus the remaining
multi-vector-store adapters. The 16 GB backend matrix, quality-per-watt path, and generated
per-GPU serving-config workflow are current-state facts; see
[`milestone-7-platform-matrix.md`](milestone-7-platform-matrix.md).
