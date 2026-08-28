# Composed Agent Operating Profile

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

Every ingredient of an agent configuration is measured somewhere in this repository and, until this
lane existed, nowhere together. `llb recommend` renders host-adaptive model picks plus separate
miss-analysis, self-improvement, fine-tune-campaign, and context-policy sections; the context-order
recommendation comes out of the position probe; the retrieval knobs out of the comparison lanes; the
prompt-system id out of `prompt-system-compare`; and the adapter out of the registry. An operator
standing an agent up had to read five sections and hand-assemble, and nothing checked that the
pieces were measured on the same corpus, store, or model.

`llb recommend --agent-profile` composes them into ONE artifact.

```bash
make recommend-agent-profile          # RECOMMEND_MIN_CASES= RECOMMEND_GPU_GB= RECOMMEND_MIN_TOK_S=
llb recommend --agent-profile --no-chart
```

It writes `$DATA_DIR/agent-profile/<run_timestamp>/{agent_profile.json,profile.md}`. There is no new
evidence root: every value points back into the per-lane root that measured it.

## The ten fields and where each comes from

| Field | Lane | Depends on |
| --- | --- | --- |
| `model` | `run-eval` (the host pick) | store, adapter |
| `backend` | `run-eval` | store, adapter |
| `prompt_system_id` | prompt-system-tagged `run-eval` bundles | store, adapter |
| `adapter` | the adapter registry (or `none`) | adapter |
| `context_policy` | `agentic-context` | adapter |
| `context_order` | `context-position` | store, adapter |
| `top_k` | `run-eval` (the winning cell) | store |
| `reranker` | `compare-rerankers` (or `none`) | store |
| `context_budget` | `run-eval` (the winning cell) | -- |
| `loop_policy` | `agentic-loop-policy` | adapter |

Each field carries its value, the artifact path the value came from, that lane's own verdict and
uncertainty, and its freshness in days. The verdict is the lane's, never a re-derivation: the
reranker field reports the bake-off's adoption `decision`, the context-policy field reports the
bundle's own paired completion reading against the `full` baseline, and the loop-policy field
reports the grid's `verdict` with its paired stability block. `agent_profile.json` keeps every
reason in full; the markdown rationale clips a long one and says where the rest is.

For memory-dependent work the context-policy field additionally carries the guard-dependent routing
rule from the [cap-fitting boundary surface](crossover-geometry.md#cap-fitting-boundary-surface) as
a CONDITION on its value rather than as a competing second value.

## The four states

A composed profile invites exactly one failure -- a default dressed up as a recommendation -- so a
field is in one of four states and only the first is actionable:

- **`measured`.** The lane ran, the reading is current, and the value is a recommendation.
- **`unmeasured`.** The lane never ran on this host. There is no value at all, and never the value
  the code would have defaulted to. A profile built with no bundles is entirely `unmeasured` and is
  still a valid artifact: each row names the LANE that would close the gap.
- **`refused`.** The consistency guard found the value was measured against a different model,
  corpus, or store fingerprint than the profile anchors on, and names the axis that disagreed.
- **`demoted`.** The value stands but something it rests on moved. The value stays visible -- an
  operator still needs to know what it WAS -- and stops being a recommendation.

A `none` is a MEASURED answer, not a gap: an empty adapter registry means no adapter, and a
retained no-rerank verdict means no cross-encoder.

## The anchor and the consistency guard

The `run-eval` host pick is the only reading that fixes model, corpus, and store at once, so it is
the anchor. Every other field is checked against it on three axes -- model, corpus root, and the
retrieval fingerprint -- and refused on any disagreement. A lane that did not record an axis is not
checked on it: silence is not a disagreement, and inventing one would refuse fields that are
perfectly consistent. With no `run-eval` bundle there is no anchor, and the profile says so rather
than composing unchecked.

A refusal outranks freshness. The newest reading of a knob is not the right one if it was taken
somewhere else, so a same-day probe against a re-chunked store is refused while a month-old policy
comparison on the anchor's own model stands. That is why the probe was taught to record the
retrieval fingerprint it queried (below): a lane that cannot state where it ran cannot be checked,
and an unchecked field composes silently.

## The staleness demotion

Two drift findings demote, and both reuse machinery that already exists rather than adding a second
opinion:

- **A moved store.** The anchor run's recorded retrieval knobs are compared with the CURRENT
  `store_meta.json` through the adapter registry's own retrieval-fingerprint axis
  ([adapter registry](adapter-registry.md#staleness)), so a knob that makes an adapter stale makes
  the profile's retrieval fields stale by the same rule. Every store-dependent field is demoted with
  the changed knob named (`chunk_size 800 -> 704`).
- **A stale adapter.** The registry's own `staleness()` verdict is carried through unchanged; a
  `stale` verdict demotes every adapter-dependent field with the registry's reasons named.

Both can fire on the same field, which then collects both reasons. A field with no value is never
demoted: telling an operator that their empty `context_order` rests on a moved store explains
nothing about why it is empty.

## Replay

The measured fields are emitted as the commands that reproduce them, so a recommendation is never
hand-translated into flags:

```bash
llb run-eval --model <model> --backend <backend> --top-k <k> --context-budget <tokens> \
  --reranker <cross-encoder> --context-order <order>
llb bench-agentic --model <model> --backend <backend> --context-policy <policy> --max-steps <n>
llb bench-agentic-loop --model <model> --backend <backend> --agent-max-steps <baseline,cell> ...
```

Only `measured` fields contribute a flag; every other field is listed under `replay.omitted` with
its state. A field whose measured answer is "nothing here" (`reranker=none`, `adapter=none`)
contributes no flag, because the lane default IS off. **No measured `model` suppresses every replay
command**: a command that pins `--top-k` but not the model does not reproduce the recommended
configuration, it reproduces whatever the caller's config already said wearing one knob from this
profile -- the exact silent mixing the profile exists to prevent.

`run-eval` gained `--top-k` and `--context-budget` for this, so every retrieval-side field the
profile recommends has a flag to replay through.

## Core locations

- `src/llb/board/agent_profile/model.py`: the field roster, the four states, the dependency axes,
  and the records;
- `src/llb/board/agent_profile/artifacts.py`: lane-root readers, newest-bundle selection by run
  timestamp, and the freshness clock;
- `src/llb/board/agent_profile/sources_rag.py` and `sources_agent.py`: the per-lane readings;
- `src/llb/board/agent_profile/compose.py`: the anchor, the consistency guard, and the demotion;
- `src/llb/board/agent_profile/render.py` and `persist.py`: the payload, the rationale, the bundle;
- `src/llb/board/agent_profile/replay.py`: the replay commands;
- `src/llb/cli/recommend.py`: `llb recommend` and its `--agent-profile` flag;
- `src/llb/prompts/templates/board/agent_profile/`: the report prose;
- `tests/llb/board/agent_profile/`: fixture bundles for every lane, all four states, both drift
  axes, both consistency axes, and the replay round-trip through `RunConfig` -- no GPU.

Three supporting changes landed with it, all on `llb probe-context-position`, whose recommendation
had no machine-readable form at all:

- it writes `probe.json` beside `report.md` and `cases.jsonl`
  (`src/llb/eval/position_probe_report.py`), carrying the model, the per-position means and CIs, the
  recommendation, and a `separated`/`flat` reading of the head-versus-tail comparison -- re-deriving
  that decision by parsing a prose sentence would make the report's wording load-bearing;
- `probe.json` records the RETRIEVAL FINGERPRINT of the store the probe queried. A `context_order`
  recommendation is only about the store whose real distractors produced it, and without that field
  the consistency guard had no axis to check it on;
- it gained `--corpus-root` (`CORPUS_ROOT=`), so the probe can be pointed at the corpus the profile
  anchors on rather than only the configured default -- `run-eval` already accepted one.

## CUDA-host evidence (2026-08-28, RTX 4060 Ti 16 GB)

Composed with `make recommend-agent-profile RECOMMEND_MIN_CASES=20` over the lane bundles present on
the host. The anchor was `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` on Ollama, the host pick of the
82-case `ua_squad_postedited_v1` final-split cohort. Reading: **measured=4 demoted=4 refused=1
unmeasured=1** -- all four states exercised on real bundles, and the three non-measured ones are the
point of the lane rather than a shortfall in it.

| field | value | state | verdict | age |
| --- | --- | --- | --- | --- |
| `model` | MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M | demoted | ranked_first | 0.2d |
| `backend` | ollama | demoted | ranked_first | 0.2d |
| `prompt_system_id` | -- | unmeasured | -- | -- |
| `adapter` | none | measured | no_registered_adapter | 48.9d |
| `context_policy` | observation_cap | measured | separated | 29.1d |
| `context_order` | rank | refused | flat | 0.0d |
| `top_k` | 3 | demoted | ranked_first | 0.2d |
| `reranker` | BAAI/bge-reranker-v2-m3 | demoted | retain | 11.8d |
| `context_budget` | 8192 | measured | ranked_first | 0.2d |
| `loop_policy` | `max_steps=6 answer allow current` | measured | flat | 26.7d |

**The demotion.** The anchor's winning cell was scored against a `semantic/704/171` store, while the
host's built store now holds `recursive/800/120`. All four store-dependent fields were demoted with
all three changed knobs named, and the replay block emptied itself rather than emit a command that
would run the recommended model against a store it was never measured on. That is the failure the
lane exists to catch, caught on the first real composition -- and it is not a stale-artifact
artifact: re-running `run-eval` for the same model against the CURRENT store scored 0.545 (82 cases,
11.3 tok/s, `top_k=3`, `context_budget=8192`) against the anchor run's 0.598, so the two stores are
not interchangeable for this model. Those are two unpaired single runs, so the gap is a point
estimate, not a verdict; what it establishes is only that the demoted fields could not have been
silently reused.

**The refusal.** A full 82-item context-position probe run on the same day recommends `rank`
(head 0.546, middle 0.537, tail 0.500; head/tail CIs overlap, so the reading is `flat` -- the
default surviving an honest test rather than a resolved position preference). It is the FRESHEST
reading in the profile at 0.0d and is still refused, because `probe.json` records the store it
queried and that store is not the one the anchor was measured on. Freshness is not consistency: a
composed profile that preferred the newest reading would have shipped a `context_order` measured
against different chunking.

**The gap.** `prompt_system_id` stays `unmeasured`: no prompt-system-tagged `run-eval` bundle exists
for this model on this host. The profile reports that rather than naming the baseline prompt as a
recommendation.

Reading the four measured fields: both agentic ones are shipped defaults RETAINED on a `flat` paired
reading rather than adopted on a win; `adapter=none` is the registry's answer (it holds two adapters,
neither trained on this base model); and `context_budget` survives because it is the one field that
does not depend on the store. The freshness spread is the load-bearing part for an operator -- the
retrieval side is same-day while the agentic side is a month old, so a context-policy constant change
would invalidate the two oldest rows first.

What would overturn it: rebuilding the store at the anchor's chunking, or re-running the sweep so a
current-store cell becomes the host pick (either retires the demotion and the refusal together);
registering an adapter for this base model (the `adapter` field stops reading `none` and every
adapter-dependent field inherits its staleness); or a `bench-agentic-context` re-run whose paired
completion reading separates a policy other than `observation_cap`.
