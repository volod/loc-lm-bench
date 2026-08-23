# Localized Repeat Feedback Comparison

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

The repeat-feedback lane keeps that powered ledger, model roster, and fixed
`6/answer/allow,noop` policy while adding only the controller notice as an experimental axis.
`samples/benchmarks/agentic_loop_feedback_localization_design.json` predeclares `current`, `uk`,
and `bilingual` variants with the same `+0.25` completion target, activation requirements, and
paired token/wall cost ceilings. `make bench-agentic-loop-repeat-feedback` runs all four cells: one
`allow/current` reference plus three `noop` feedback cells.

`src/llb/bench/agentic/loop_policy.py` owns the validated feedback variants, while
`src/llb/bench/agentic/episode.py` records whether a suppressed repeat is followed by a changed
tool call or final answer. `src/llb/bench/loop_feedback/run.py` reports that response rate
overall and per task family, pairs each localized cell directly against `noop/current`, and admits
a family-level recommendation only when activation, material separated completion, prompt-token,
and wall-time gates all pass. Each cell id and manifest includes the feedback variant;
`feedback-study-design.json` and `feedback-analysis.json` make the prospective contract and
decision independently inspectable. The general recommendation remains isolated from these
experimental `noop` cells, so a one-family result cannot alter shipped defaults.

CUDA-host evidence (2026-07-31), RTX 4060 Ti 16 GB, identical 32-task digest:

| model family | feedback | response | completion | prompt tokens | wall seconds | support |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MamayLM-Gemma 12B | `current` | 0.000 | 0.156 | 5404.0 | 35.13 | reference |
| MamayLM-Gemma 12B | `uk` | 0.000 | 0.156 | 5371.3 | 33.92 | no |
| MamayLM-Gemma 12B | `bilingual` | 0.062 | 0.219 | 5322.3 | 35.20 | no |
| Qwen3 14B | `current` | 0.562 | 0.562 | 3780.1 | 4.43 | reference |
| Qwen3 14B | `uk` | 0.281 | 0.312 | 4831.1 | 5.47 | no |
| Qwen3 14B | `bilingual` | 0.875 | 0.875 | 2906.6 | 3.78 | yes |

For MamayLM, bilingual feedback redirected only two search-family episodes and produced a
`+0.062 [0.000, 0.156]` completion delta, below the target and statistically flat. For Qwen,
bilingual feedback redirected 28 of 32 activated episodes, including five of eight calculator
tasks, and produced a separated `+0.312 [0.156, 0.469]` completion delta with ten wins and no
losses. Its paired prompt delta was `-873.5 [-1324.4, -424.4]` tokens and wall delta was
`-0.64 [-1.13, -0.16]` seconds, so both cost gates passed. Ukrainian-only feedback regressed Qwen
completion and failed both cost gates. The evidence therefore recommends bilingual feedback for
the Qwen family only; `current` remains the cross-family and shipped default.

One run bundle per cell, eight in all: MamayLM and Qwen crossed with `allow/current`,
`noop/current`, `noop/uk`, and `noop/bilingual`.

`tests/llb/bench/loop_feedback/test_agentic_loop_feedback.py` checks exact grid construction, design
validation, redirect telemetry, prospective decision gates, rendered reporting, persistence, and the
committed design/task contract. The run path was also exercised end to end on both predeclared local
models. Validation on 2026-07-31: `make ci` passed 2,463 tests with 45 opt-in/slow tests deselected,
and `make lint-md` passed.

## Seeded Cross-Family Generalization

`samples/benchmarks/agentic_loop_feedback_generalization_design.json` extends the same immutable
32-task ledger and fixed `6/answer/allow,noop` policy to four independent model families, seeds 13
and 29, and temperature 0.2. The roster retains Qwen3 14B and MamayLM-Gemma 12B and adds Aya
Expanse 8B and Mistral Small 3.1 24B. The predeclared adoption rule requires support on both seeds
for a family, at least three of four supported families, and support from an added family before a
global change is possible.

`make bench-agentic-loop-repeat-feedback-generalization` validates the complete roster, task
digest, sampling contract, and installed Ollama models before inference. Seed and temperature now
flow through both native and OpenAI-compatible Ollama completion paths. The runner persists every
three-cell family/seed bundle before aggregation; the aggregate records the exact coordinate grid,
coverage, current and bilingual activation, response rate, completion and paired cost deltas, cell
manifests, stable family routing, and the global decision. The core analysis and reporting live in
`src/llb/bench/loop_feedback/generalization.py` and
`src/llb/bench/loop_feedback/generalization_report.py`.

CUDA-host evidence (2026-07-31), RTX 4060 Ti 16 GB:

| family | seed | response | completion | completion delta | support |
| --- | ---: | ---: | ---: | ---: | --- |
| Aya Expanse 8B | 13 | 0.107 | 0.312 | -0.094 | no |
| Aya Expanse 8B | 29 | 0.037 | 0.344 | -0.219 | no |
| Mistral Small 3.1 24B | 13 | 1.000 | 0.875 | 0.000 | no |
| Mistral Small 3.1 24B | 29 | 1.000 | 0.906 | 0.000 | no |
| Qwen3 14B | 13 | 0.844 | 0.844 | +0.344 | yes |
| Qwen3 14B | 29 | 0.812 | 0.812 | +0.312 | yes |
| MamayLM-Gemma 12B | 13 | 0.062 | 0.219 | +0.062 | no |
| MamayLM-Gemma 12B | 29 | 0.062 | 0.219 | +0.062 | no |

All eight family/seed coordinates passed task coverage and both activation gates. Qwen alone
cleared the completion and paired cost gates on both seeds, so it routes to `bilingual`; Aya,
Mistral, and Gemma remain on `current`. Stable support is one of four families, below the declared
three-family threshold, and neither added family supports the variant. The global feedback default
therefore remains `current`.

The audit-complete aggregate's analysis indexes
all 24 additive cell manifests. `tests/llb/bench/loop_feedback/test_agentic_loop_feedback_generalization.py`
checks the prospective design, exact family/seed grid, coordinate metadata, activation telemetry,
stable routing, global adoption rule, reporting, and persistence.
Validation on 2026-07-31: `make ci` passed 2,469 tests with 45 opt-in/slow tests deselected.

## Family-Adapted Repeat Feedback

The family-adaptation lane tests one concise controller notice per non-Qwen family without letting
wording leak across families. Its prospective design is
`samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json`; it fixes the powered
32-task digest, seeds 13 and 29, temperature 0.2, the `6/answer/allow,noop` policy, a `+0.25`
completion target, 10% prompt-token and 20% wall-time ceilings, and a two-of-three-family adoption
threshold. The registered ASCII notices are:

- `aya_direct`: `[loop] Repeated tool call skipped. Choose a different action or give the final
  answer now.`
- `mistral_use`: `[loop] Repeated call skipped. Use the existing result: answer now, or change the
  tool arguments.`
- `gemma_choice`: `[loop] Repeated call skipped. Output one different JSON tool call or the final
  answer; do not repeat.`

`make bench-agentic-loop-repeat-feedback-family-adaptation` validates the exact Aya, Mistral, and
Gemma Ollama roster, notice text and hypotheses, sampling contract, task digest, seed grid, and
candidate isolation before inference. The prospective contract is validated in
`src/llb/bench/loop_feedback/adaptation_design.py`, the seeded runs are read in
`src/llb/bench/loop_feedback/adaptation.py`, its report and persistence layer in
`src/llb/bench/loop_feedback/adaptation_report.py`, and its CLI orchestration in
`src/llb/cli/bench/loop/feedback_adaptation.py`. Each aggregate seed row exposes
coverage, baseline and candidate activation, completion, prompt-cost, wall-cost, and combined-cost
gate decisions in addition to response and effect values. `src/llb/bench/agentic/episode.py` also
counts a final answer after a suppressed repeated call as a redirect, including when the
malformed-call policy accepts that answer as the episode result.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, identical 32-task digest. In the gates column,
`C`, `P`, and `W` mean completion, prompt-token cost, and wall-time cost respectively; `-` is the
only failed gate. Coverage and both activation checks passed in every row.

| family | seed | candidate | response | completion | completion delta | prompt delta | wall delta | gates | support |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Aya Expanse 8B | 13 | `aya_direct` | 0.643 | 0.562 | +0.156 | -227.8 | -4.91 | `-PW` | no |
| Aya Expanse 8B | 29 | `aya_direct` | 0.667 | 0.594 | +0.031 | -412.1 | -4.56 | `-PW` | no |
| Mistral Small 3.1 24B | 13 | `mistral_use` | 1.000 | 0.875 | 0.000 | +4.0 | -0.02 | `-PW` | no |
| Mistral Small 3.1 24B | 29 | `mistral_use` | 1.000 | 0.906 | 0.000 | +4.4 | -0.20 | `-PW` | no |
| MamayLM-Gemma 12B | 13 | `gemma_choice` | 0.281 | 0.375 | +0.219 | -585.3 | -1.15 | `-PW` | no |
| MamayLM-Gemma 12B | 29 | `gemma_choice` | 0.250 | 0.375 | +0.219 | -579.4 | -1.65 | `-PW` | no |

Aya's completion gains were seed-sensitive and below the material target. Mistral already
redirected every activated candidate episode but gained no completion, so its adapted wording tied
`current`. Gemma produced seven wins and no losses on each seed, a separated
`+0.219 [0.094, 0.375]` completion delta, but missed the predeclared `+0.25` mean target by one of
32 tasks. Its notice redirected all eight search tasks on both seeds, zero read and calculator
tasks, and only one mutation task on seed 13. This stable but narrow effect does not justify a
family-wide route.

No candidate cleared the completion gate on either seed. Each family therefore has zero of two
supporting seeds, every family remains routed to `current`, and the supported-family fraction is
zero of three, below the declared cross-family threshold. The audit-complete aggregate indexes all 18
source cell manifests and carries the explicit per-gate decisions.

`tests/llb/bench/loop_feedback/test_agentic_loop_feedback_adaptation.py` checks the immutable
design, exact wording, family/seed grid, candidate isolation, stable routing, aggregate gate
reporting, persistence, and an end-to-end fake run. The redirect regression is in
`tests/llb/bench/loop_policy/test_agentic_loop_policy.py`. Validation on 2026-08-01: `make ci`
passed 2,475 tests with 45 opt-in/slow tests deselected.

## Task-Family-Neutral Gemma Transfer

The transfer lane tests one task-family-neutral Gemma notice on a fresh holdout instead of tuning
against the family-adaptation ledger. Its prospective design is
`samples/benchmarks/agentic_loop_feedback_task_family_transfer_design.json`; the balanced
32-task ledger is `samples/benchmarks/agentic_loop_feedback_task_family_transfer.json`, with eight
new ASCII cases in each of read, calculator, search, and mutation. Mutation success requires both
the state change and a confirming final answer, so a successful first write cannot hide failure to
advance after suppression. The holdout digest is
`10fef23bc2b2d855f6b7395d7e94ac42013005b4967d29d1a968ada99a215465`, distinct from the powered
ledger digest recorded in the design.

The immutable `gemma_progress` notice is `[loop] The previous action already succeeded. Continue
from its result instead of repeating it.` It contains no tool family, task name, or expected value.
The validator fixes that exact registered text, its completed-state hypothesis, seeds 41 and 73,
temperature 0.2, 8192-token served context, a 25% response floor in at least three of four task
families on both seeds, a `+0.125` material paired completion target, and maximum relative cost
increases of 10% for prompt tokens and 20% for wall time. It also refuses the prior ledger digest
and any task-specific word in the controller notice before inference.

`make bench-agentic-loop-repeat-feedback-task-family-transfer` runs the fixed
`6/answer/allow,noop` comparison on the local MamayLM-Gemma 3 12B model. The prospective contract
(notice wording, fresh ledger, seeds, gates) is validated in
`src/llb/bench/loop_feedback/transfer_design.py`, shared with the controller-authority
study; the two-seed decision lives in `src/llb/bench/loop_feedback/transfer.py`; report and aggregate
persistence live in `src/llb/bench/loop_feedback/transfer_report.py`; orchestration lives
in `src/llb/cli/bench/loop/feedback_transfer.py`. Aggregate rows retain baseline
and candidate response, per-family response deltas, the full paired completion comparison, both
full cost-gate objects, and links to every source cell manifest.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M. In the gates
column, `R`, `C`, `P`, and `W` mean task-family response, completion, prompt-token cost, and
wall-time cost; `-` is a failed gate.

| seed | current response | candidate response | responsive families | completion delta | prompt delta | wall delta | gates |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 41 | 0.000 | 0.094 | mutation 0.375 | +0.000 | +27.4 [6.2, 43.3] | -0.19 [-0.84, 0.35] | `--PW` |
| 73 | 0.031 | 0.062 | mutation 0.250 | +0.000 | +37.3 [26.0, 43.4] | +2.52 [0.38, 4.87] | `--PW` |

Coverage and baseline/candidate activation passed on both seeds. Read, calculator, and search had
zero candidate redirects on both seeds; mutation alone reached its response floor, with three
redirects on seed 41 and two on seed 73. None of those redirects completed a task. Candidate and
current completion were both `0.000`, all 32 completion pairs tied on each seed, and the paired
completion interval was `[0.000, 0.000]`. Both paired cost gates passed. Thus the completed-state
hypothesis does not transfer a useful redirect across task families, and `current` remains the
recommended Gemma feedback variant.

The audit-complete aggregate indexes the six
all six source cell manifests and preserves the exact prospective design. The first
aggregate is retained; the later
aggregate adds baseline response and the complete paired gate objects from those same source
cells, with no additional inference.

`tests/llb/bench/loop_feedback/test_agentic_loop_feedback_transfer.py` checks the immutable neutral
notice and hypothesis, fresh digest, balanced ledger, exact seed grid, candidate isolation,
three-family and two-seed response rule, completion and cost decisions, report persistence, and
end-to-end fake runs with exact injected episode durations. The wall-cost regression case measures a
1.25-second candidate against a 1-second baseline and refuses it at the prospective 20% ceiling.
Validation on 2026-08-01: `make ci` passed 2,479 tests with 45 opt-in/slow tests deselected.

## Controller-Authority Gemma Transfer

The controller-authority lane tests whether an explicit controller ruling can overcome Gemma's
literal repetition on a second fresh balanced holdout. The registered `gemma_authority` notice is
`[loop] Controller ruling: suppression satisfies the requested repetition. You must now take the
next distinct action.` It is ASCII, contains no task name, expected value, tool family, or
family-specific action choice, and remains a controller observation rather than a task-specific
hint.

The prospective design is
`samples/benchmarks/agentic_loop_feedback_controller_authority_design.json`; its 32-case ledger is
`samples/benchmarks/agentic_loop_feedback_controller_authority.json`, with eight new read,
calculator, search, and mutation cases. The ledger digest is
`a2e8e0bf49c04ca27cebb9d06072e7008026f874a93a99ae5098d3b938b98f82`, distinct from both prior
ledgers fixed in the design. The contract fixes MamayLM-Gemma 3 12B, seeds 107 and 149,
temperature 0.2, an 8192-token context, the `6/answer/allow,noop` grid, a 25% response floor in at
least three families on both seeds, a `+0.125` paired completion target, minimum four discordant
pairs, and maximum relative increases of 10% for prompt tokens and 20% for wall time.

`make bench-agentic-loop-repeat-feedback-controller-authority-transfer` validates the full
contract before inference and writes a two-seed aggregate. The immutable notice lives in
`src/llb/bench/agentic/loop_policy.py`; authority validation and decision wrapping live in
`src/llb/bench/loop_feedback/authority.py`; response-versus-completion summaries live in
`src/llb/bench/loop_feedback/outcomes.py`; reporting lives in
`src/llb/bench/loop_feedback/authority_report.py`; and CLI orchestration shares
`src/llb/cli/bench/loop/feedback_neutral.py` with the earlier neutral-transfer
lane. Aggregate persistence uses the design's study kind, so authority artifacts cannot be
mislabelled as the earlier task-family-transfer study.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M. Each family cell
below is `response rate / redirected completion rate`; the latter counts completions after a
changed post-suppression action over activated tasks.

| seed | current response | candidate response | calculator | mutation | read | search | completion delta | prompt delta | wall delta | gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 107 | 0.031 | 0.344 | 0.125 / 0.000 | 0.625 / 0.000 | 0.625 / 0.625 | 0.000 / 0.000 | +0.156 | -374.1 | -1.23 | `--PW` |
| 149 | 0.094 | 0.312 | 0.000 / 0.000 | 0.625 / 0.000 | 0.500 / 0.500 | 0.125 / 0.125 | +0.156 | -365.8 | -7.44 | `--PW` |

Coverage and baseline/candidate activation passed on both seeds. Read cleared the response floor
and every read redirect completed; mutation also cleared the response floor but none of its ten
redirects produced the required confirming final answer. Calculator stayed below the floor on
both seeds, and search reached only one successful redirect on seed 149. Thus only two families
were responsive per seed, below the required three.

The candidate produced five wins, no losses, and 27 ties on each seed for a paired `+0.156`
completion delta with interval `[+0.031, +0.281]`. The standard stability reading remained
borderline and `insufficient_evidence` (`randomization p=0.03125`, sign-test `p=0.0625`), so the
completion gate also failed. Prompt-token deltas were `-374.1 [-748.3, -16.2]` and
`-365.8 [-741.4, -7.3]`; wall-time deltas were `-1.23 [-4.07, +1.40]` seconds and
`-7.44 [-9.80, -5.33]` seconds. Both paired cost gates passed on both seeds.

The authority wording therefore shows stable read completion and mutation response, but it does
not establish task-family transfer; `current` remains the recommended Gemma feedback variant. The
audit-complete aggregate links all six
source cell manifests and preserves response-versus-completion outcomes per family.

`tests/llb/bench/loop_feedback/test_agentic_loop_feedback_authority.py` checks the immutable wording,
hypothesis, fresh balanced ledger, seeds, candidate isolation, breadth and paired gates,
authority-specific study identity, persistence, and an end-to-end fake run. The shared feedback
tests check per-family redirected completion accounting. Validation on 2026-08-01: `make ci`
passed 2,483 tests with 45 opt-in/slow tests deselected, and `make lint-md` passed.

## Controller-Channel Authority

The controller-channel lane isolates transcript authority from authority wording. After an
identical repeated call is suppressed, both cells send the same task message and the same immutable
authority text in the same message position. Only the authority message role changes:
`observation` serializes to `user`, while `controller` serializes to `system`. The exact mapping is
declared for native Ollama and OpenAI-compatible chat in
`samples/benchmarks/agentic_controller_channel_authority_design.json`; typed serialization lives in
`src/llb/bench/agentic/controller_channel.py`. This keeps the task prompt and all ordinary tool
observations fixed and gives the agent loop a backend-neutral controller-message seam.

`make bench-agentic-loop-controller-channel-authority` runs the predeclared two-seed comparison.
The fresh 32-case ledger is
`samples/benchmarks/agentic_controller_channel_authority.json`, balanced over eight read,
calculator, search, and mutation cases. Its digest is
`5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1`. The contract fixes
MamayLM-Gemma 3 12B Q4_K_M, seeds 211 and 257, temperature 0.2, 512 completion tokens, an
8192-token context, six controller steps, and the same 25% response floor in at least three
families on both seeds. Adoption also requires a paired completion gain of at least 0.125, at
least four discordant completion pairs, and maximum relative increases of 10% for prompt tokens
and 20% for wall time.

Every source cell persists `prompt-snapshots.json`. Analysis pairs the first authority-bearing
snapshot by task and refuses the run unless the full message content is byte-identical while only
the final role changes. Runner, analysis, and persistence live in
`src/llb/bench/controller_authority/episodes.py`,
`src/llb/bench/controller_authority/run.py`, and
`src/llb/bench/controller_authority/report.py`. The general backend adapter now exposes
typed-message `local_chat` and `launcher_chat` callables alongside the legacy string-prompt
adapters.

CUDA-host evidence (2026-08-01), RTX 4060 Ti 16 GB, MamayLM-Gemma 3 12B Q4_K_M:

| seed | gates | observation response | controller response | observation completion | controller completion | completion delta | prompt delta | wall delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | `SA--PW` | 0.094 | 0.031 | 0.000 | 0.000 | +0.000 | +7.4 | -0.32 s |
| 257 | `SA--PW` | 0.094 | 0.062 | 0.000 | 0.000 | +0.000 | +5.8 | -2.05 s |

`S`, `A`, `R`, `C`, `P`, and `W` denote snapshot, activation, family response, completion,
prompt-cost, and wall-cost gates. Snapshot and activation coverage passed for all 32 tasks on both
seeds, and both paired cost gates passed. The controller role responded only in mutation: 0.125
on seed 211 and 0.250 on seed 257, with zero redirected completions. Calculator, read, and search
response were zero. Both placements completed 0/32 tasks on both seeds, leaving zero discordant
completion pairs and a flat paired completion reading.

Structural controller authority is therefore not supported for this Gemma model and transcript
shape; `observation` remains the recommended placement, and no shipped default changes. The
audit-complete aggregate's analysis
links the four source manifests and all 64 paired snapshot proofs. The negative result is scoped to
this model and serialization, not a claim that role never matters across model families.

`tests/llb/bench/controller_authority/test_agentic_controller_authority.py` checks exact role-only serialization,
fresh-ledger and two-seed validation, balanced family coverage, snapshot refusal, every adoption
gate, persistence, the committed contract, and an end-to-end fake run.
Validation on 2026-08-01: `make ci` passed 2,487 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

**Qwen cross-model transfer.** `make bench-agentic-loop-controller-channel-cross-model` applies
the same authority text, role mapping, message order, task shape, sampling policy, and six adoption
gates to the non-Gemma `qwen3:14b` family. Its prospective design is
`samples/benchmarks/agentic_controller_channel_cross_model_design.json`; its fresh 32-case ledger
is `samples/benchmarks/agentic_controller_channel_cross_model.json`, balanced over eight cases in
each family with digest
`177adb511124b972f748a1ef8beb21365f1bcee315c3039c11fb43e4413bcc70`. The contract fixes seeds
307 and 353, temperature 0.2, 512 completion tokens, and an 8192-token context. A distinct
`agent_loop_policy_controller_channel_authority_cross_model` study kind and immutable reference to
the Gemma study prevent the transfer row from being mislabelled or pointed back at Gemma.

CUDA-host evidence (2026-08-02), RTX 4060 Ti 16 GB, Qwen3 14B:

| seed | gates | observation response | controller response | observation completion | controller completion | completion delta | prompt delta | wall delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 307 | `SA--P-` | 0.031 | 0.000 | 0.031 | 0.000 | -0.031 | +89.3 | +2.25 s |
| 353 | `SA--P-` | 0.031 | 0.000 | 0.031 | 0.000 | -0.031 | +89.3 | +1.92 s |

Snapshot and activation gates passed for all 32 tasks on both seeds, and the prompt-cost gate
passed. The controller role redirected zero tasks in every family on both seeds, so the response
gate failed. The observation role redirected and completed one search task per seed; the controller
role completed none, producing one loss and 31 ties per seed with interval `[-0.094, 0.000]`.
Controller wall-time increases were separated above the 20% ceiling on both seeds, so the wall-cost
gate also failed. Observation remains the recommendation: structural controller authority does not
transfer to this Qwen chat template and transcript shape, and no shipped default changes.

The audit-complete aggregate links four
source cell manifests, 64 role-only snapshot proofs, and per-cell throughput from 16.21 to 22.26
tokens/s. Cross-model validation lives beside the base contract in
`src/llb/bench/controller_authority/design.py`; the CLI model preflight now queries the
configured Ollama host, and the dedicated Make target pins the committed cross-model design and
ledger.
Validation on 2026-08-02: `make ci` passed 2,489 tests with 45 opt-in/slow tests deselected, and
`make lint-md` passed.

**Template-native preamble placement.**
`make bench-agentic-loop-controller-preamble-placement` separates canonical template placement
from the earlier role-only comparison. The observation baseline serializes
`[task prompt:user, authority:user]`; the candidate serializes
`[authority:system, task prompt:user]`. The immutable authority bytes and task-prompt bytes are
identical across the pair. Both Ollama and OpenAI-compatible transforms are declared exactly in
`samples/benchmarks/agentic_controller_preamble_placement_design.json`; inference is refused if a
transform, authority byte, model-seed cell, or first authority-bearing prompt pair differs.

The prospective design reuses the 32-case balanced controller-channel ledger and its
`5d6148e0851ca749a65fd75768b388931419ae3dccf2c03be20c095c97e9ead1` digest so only placement and
model family vary. It fixes MamayLM-Gemma 3 12B Q4_K_M and `qwen3:14b`, seeds 401 and 443,
temperature 0.2, 512 completion tokens, an 8192-token context, six steps, and the existing
snapshot, activation, three-family response, paired completion, prompt-cost, and wall-cost gates.
Adoption requires all four model-seed cells to pass.

Typed source/role transforms live in `src/llb/bench/agentic/controller_channel.py`; the episode
seam is in `src/llb/bench/agentic/episode.py`. Multi-model grid validation, execution, analysis,
reporting, and CLI orchestration reuse the controller-authority modules. The result schema exposes
generic candidate-placement support and a preamble-specific decision while retaining the earlier
controller-channel fields for artifact compatibility. Every source cell persists its exact first
authority-bearing prompt snapshots.

CUDA-host evidence (2026-08-02), RTX 4060 Ti 16 GB:

| family | seed | gates | observation response | preamble response | observation completion | preamble completion | completion delta | prompt delta | wall delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma | 401 | `SA--PW` | 0.062 | 0.125 | 0.000 | 0.000 | +0.000 | -5.7 | -1.91 s |
| Gemma | 443 | `SA--P-` | 0.062 | 0.062 | 0.000 | 0.000 | +0.000 | +3.8 | +7.36 s |
| Qwen | 401 | `SA--P-` | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.0 | +2.09 s |
| Qwen | 443 | `SA--P-` | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 | +0.0 | +2.10 s |

All 128 prompt pairs passed the snapshot proof and every cell passed activation and prompt-cost
gates. Gemma preamble response was confined to mutation: 4/8 on seed 401 and 2/8 on seed 443,
with no redirected completion. Read, calculator, and search response was zero. Both placements
completed 0/32 tasks on both seeds, so every completion comparison was flat with zero discordant
pairs. Gemma wall cost passed on seed 401 and failed on seed 443.

Qwen produced no redirect or completion under either placement on either seed. Its preamble added
`+2.092 [+2.038, +2.124]` and `+2.095 [+2.042, +2.128]` seconds per paired task, above the 20%
wall-cost ceiling on both seeds; prompt-token deltas were exactly zero. The template-native
preamble therefore does not improve repeated-call recovery for either tested family and makes
Qwen materially slower. `observation` remains the recommendation, with no shipped-default change.

The audit-complete aggregate links eight
source manifests, four gate rows, and 128 paired snapshot proofs. Source-cell throughput was
4.9-5.2 tokens/s for Gemma and 20.2 tokens/s for Qwen.

`tests/llb/bench/controller_authority/test_agentic_controller_preamble.py` checks both backend
transforms, the exact two-family/two-seed design, snapshot refusal, every gate, and an end-to-end
fake run. The existing controller-channel tests protect backward compatibility. Validation on
2026-08-02: `make ci` passed 2,493 tests with 45 opt-in/slow tests deselected, and `make lint-md`
passed.

CrewAI is optional and lazy-imported. The adapter wraps the candidate completion function as a
CrewAI LLM, builds tools from the benchmark tool definitions, and disables telemetry/tracing for a
local no-egress run.

The `[crewai]` extra is a standalone install lane in `uv`: upstream CrewAI pins older Chroma,
LanceDB, and `tomli` ranges than the repo's RAG/vector/dev extras. `pyproject.toml` declares those
extra conflicts so `uv lock` stays resolvable while
`UV_PROJECT_ENVIRONMENT=<dir> uv sync --frozen --extra crewai` still installs that fork verbatim
into a dedicated environment for host validation.
