# Model Roster

The candidate roster is a register of model FAMILIES and their GENERATIONS, not a list of tags, and
the family and license tables published in the README and the reference docs are generated from it.
This page is what was built for the `model-roster-currency` capability: where the register lives,
what it guarantees, and how a generation upgrade lands.

Serving is NOT here. Which artifact runs on which host stays with the resolver and the memory
planner ([robust backends](robustness-ontology-backends.md#memory-planner),
[platform matrix](platform-vector-matrix.md)); this page owns identity only.

## The register

[`samples/configs/models_uk.yaml`](../../../samples/configs/models_uk.yaml) carries two joined
blocks:

- `families:` -- one entry per family: `id`, published `label`, `role`
  (`ua-specialized` | `multilingual-baseline`), a one-line `focus`, an `upstream` block naming where
  a currency check reads the family (`hf_author`, `hf_prefix`, `ollama_namespace`), and
  `generations:` with each generation's `id`, `status` (`current` | `previous`), `label`, `license`,
  `license_url`, and `weights_url`.
- `models:` -- every logical model, each declaring the `family` and `generation` it carries.

`llb.backends.roster` joins the two into a `Register` of `Family` -> `Generation` -> models and
reports every way the join can be wrong. The invariants it enforces:

| Invariant | Why it exists |
| --- | --- |
| Every model names a registered family and one of its declared generations | A model no generation carries is invisible to the published tables and to any currency check |
| Exactly one generation per family is `current` | "Which Qwen do we run" must have one answer, and the previous generation is what a comparison is against |
| Every declared generation carries at least one model | A generation with no artifact is history, not roster |
| Every generation records a license and its URL | Weights travel with terms; the register is where they are stated once per generation |
| A model's `license` agrees with its generation's | Catches an entry copied from a differently licensed family |

## The published tables are generated

`llb.quality.roster_docs` renders the register into marked blocks and fails when a block drifts:

| Document | Block | Shape |
| --- | --- | --- |
| [`README.md`](../../../README.md#model-families-and-licenses) | `model-families` | one row per family: role, current generation, generations also carried, license |
| [model families reference](../../reference/model-families.md#the-roster) | `model-roster` | one row per generation: status, the models on it, weights, license |

```bash
make list-model-families        # the register in the terminal (MARKDOWN=1 for the table)
make sync-model-family-docs     # republish both blocks after editing the manifest
make lint-model-roster          # report drift instead of rewriting (runs inside make ci-checks)
```

Only the marked blocks are owned by the tool; the prose around them is written by hand. A drift is
a `make ci` failure, so the README cannot keep naming a generation the roster no longer runs.

## Qwen carries three generations

Qwen is the only family carrying more than one generation, and it is why the register exists:

- `3.8` (current) -- `qwen3.8-27b`, the dense 27.3B `qwen3_5` checkpoint (64 layers,
  `full_attention_interval 4` so 16 layers own a growing KV cache, 248320 vocab, untied embeddings,
  262144 max context). vLLM needs FP8 (~40 GiB) or bf16 (~58 GiB), so the 16 GiB dev host serves the
  `qwen3.8:27b` Ollama tag (`q4_k_m`, ~17 GB) with CPU offload. The tag is the `qwen35` architecture
  and needs Ollama >= 0.32.12.
- `3.6` (previous) -- `qwen3.6-27b` and `qwen3.6-35b-a3b-fp8`, kept so a Qwen reading is a
  generation comparison rather than a single point.
- `3` (previous) -- `qwen3-30b-a3b`, the curated `qwen3:30b` tag that serves on every CUDA tier and
  has no single-artifact vLLM equivalent.

The serving manifest follows the register: primary target ids in
[`samples/config-example/manifest.yaml`](../../../samples/config-example/manifest.yaml) are FAMILY
ids (`qwen`), and the family target carries the current generation on every tier, with the previous
generations reachable as the `qwen3.6-27b`, `qwen3.6-35b`, and `qwen3-30b` extra targets. A
generation upgrade is therefore a manifest edit, not a code edit --
`llb.inference.serving_selection.PRIMARY_TARGETS` names families, not versions.

## Upgrading a family to a new generation

1. Add the generation under its family with `status: current` and demote the outgoing one to
   `previous` (drop a generation no model carries any more).
2. Add the logical model entries that carry it, each with `family` and `generation`, then verify
   with `make list-model-families` (register) and `make list-models` (host fit).
3. Run `make sync-model-family-docs` to republish the README and reference tables.
4. Pull the artifacts (`make prep-models`) and re-measure: a generation swap invalidates readings
   taken against the generation it replaces, and those are re-run rather than carried over. For the
   throughput row that is `make measure-throughput MODELS=<model>`, which re-takes it under the
   protocol the committed rows used -- see
   [refreshing one row](backend-telemetry.md#refreshing-one-row-after-a-generation-upgrade).

## Where it lives

| Piece | Module / file |
| --- | --- |
| Family register and its invariants | `src/llb/backends/roster.py` |
| Manifest schema for `families:` and per-model `family` / `generation` | `src/llb/backends/prepare/manifest.py` |
| Published-block rendering, sync, and drift check | `src/llb/quality/roster_docs.py` |
| CLI: `list-model-families`, `sync-model-family-docs` | `src/llb/cli/models/families.py` |
| Make targets | `make/models.mk`, `lint-model-roster` inside `ci-checks` (`make/dev.mk`) |
| Tests | `tests/llb/quality/test_roster_docs.py`, `tests/llb/backends/prepare/test_model_roster.py` |

Remaining work -- reading the upstream registries to say whether a carried generation is still the
newest one -- is in [the plan](../plan.md) under `model-roster-currency`.

## Related

- [Model families, tiers, and licenses](../../reference/model-families.md) -- the long form: what
  each family answers, per-tier artifacts, serving traps, and how to add a family.
- [Inference config examples](../../inference/config-example.md) -- the generated serve and
  `run-eval` artifacts per tier, and the documented hosts.
- [Platform matrix](platform-vector-matrix.md) -- one logical model compared across backends.
