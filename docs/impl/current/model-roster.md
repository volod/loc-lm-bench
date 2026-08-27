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

## Is the roster still current?

The register states which generation is current for US; `make check-model-currency` asks the
registries those artifacts come from what is current for THEM, and reports the gap per family. It
reads only -- it never edits the roster, pulls weights, or promotes a generation, and it makes no
claim that a newer generation is better. That is a measurement the sweep takes, not a fact a
register can carry.

```bash
make check-model-currency                              # read both registries, report every family
make check-model-currency FAMILY=qwen,gemma            # one or more families
make check-model-currency CURRENCY_RECORD=<cassette>   # read upstream AND record the responses
make check-model-currency CURRENCY_REPLAY=<cassette>   # replay recorded responses, no network
make check-model-currency CURRENCY_STRICT=1            # exit non-zero unless every family is current
```

Two registries answer, and both are asked for every family:

| Registry | What is read | Namespace field |
| --- | --- | --- |
| Ollama library | `https://ollama.com/library`, the index page listing every library model | `ollama_namespace` |
| Hugging Face model API | `https://huggingface.co/api/models` filtered by author and repo prefix | `hf_author` + `hf_prefix` |

Each family gets a row -- `current`, `behind`, or `unknown` -- and under it one line per registry
naming the namespace asked, the newest generation offered, the artifact that generation was read
from, and **the time that response arrived**. A family that is already current is a printed row,
never silence: a report an operator has to read as "no news" cannot be told apart from one that did
not run.

### What a verdict means

- **`behind`** -- a registry offers a generation newer than the carried one. The row names it and
  the artifact it was read from, so the claim is checkable against the registry by hand.
- **`current`** -- every registry that answered offers nothing newer than the carried generation.
  A roster carrying a generation the registries have not indexed yet also reads `current`; the
  probe reports a gap, and there is none.
- **`unknown`** -- no registry produced a reading for that family, with the reason on the row (an
  HTTP failure, an unparseable body, a namespace nothing matched, or no declared namespace at all).
  One failing registry does NOT make the row unknown while the other answered: the failure is
  still printed under the row, but a family one registry DID answer for has a real verdict, and
  discarding it to keep the error visible would trade an answer for a warning.

### Reading a generation out of an artifact name

A registry answers with artifact names, not generations, so the family's `upstream` block declares
how to get from one to the other. The default reads the version that follows the namespace, which
is where Qwen, Gemma, and Mistral Small all put it (`qwen3.8`, `gemma-4-E4B-it`,
`mistral-small3.2`). Two traps are handled rather than guessed at:

- **A parameter count sits where a generation does.** `gemma-7b` is Gemma 1 at 7B and
  `Mistral-Small-24B-Instruct-2501` is Mistral Small 1 at 24B. A number a size unit follows is
  refused, so neither is read as generation 7 or 24.
- **Some families put the generation last.** `MamayLM-Gemma-3-27B-IT-v2.0` names the Gemma 3
  ARCHITECTURE first and the MamayLM generation last, and `lapa-12b-pt` puts a parameter count
  where `lapa-v0.1.3-instruct` puts a version. Both families declare a `generation_pattern` (one
  regex, one capture group) in their `upstream` block, which wins over the default.

`hf.co/<author>` is a Hugging Face pull-through rather than an Ollama library namespace, so the
Ollama reading for MamayLM and Lapa says exactly that and the Hugging Face reading is the
authoritative one for them.

### Recorded responses

`CURRENCY_RECORD=` writes every response verbatim -- URL, arrival time, body -- into a cassette, and
`CURRENCY_REPLAY=` replays one instead of reading upstream. That is what makes the report
reproducible: the committed cassette at `tests/fixtures/roster_currency/upstream.json` is a trimmed
capture of both live registries, so the tests parse the real HTML and JSON shapes with no network.
A URL missing from a cassette is a missing reading, never a silent live read.

### What the probe read on 2026-08-27

Run on the CUDA dev host against both live registries (`make check-model-currency`), five families:

| Family | Carried | Verdict | Newest upstream | Read from |
| --- | --- | --- | --- | --- |
| `mamaylm` | v2.0 | current | 2.0 (`MamayLM-Gemma-3-12B-IT-v2.0`) | Hugging Face |
| `lapa` | v0.1.2 | behind | 0.1.3 (`lapa-v0.1.3-instruct`) | Hugging Face |
| `gemma` | 4 | current | 4 (`gemma4`) | Ollama library |
| `qwen` | 3.8 | current | 3.8 (`qwen3.8`) | Ollama library |
| `mistral` | 3.1 | behind | 4 (`Mistral-Small-4-119B-2603`) | Hugging Face |

The reading: three families carry what upstream currently offers, and two have a newer generation
published that the roster has not adopted. It is a statement about AVAILABILITY only -- neither
`lapa v0.1.3` nor `Mistral Small 4` has been measured here, and adopting either invalidates every
reading taken against the generation it replaces. A later run overturns this table whenever either
registry publishes a new generation, which is the point of the command; the durable claim is that
the probe reproduces both verdicts, not which family was behind on one day.

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
| Upstream currency: registry adapters, cassette, verdicts, report | `src/llb/backends/currency/` |
| Manifest schema for `families:` and per-model `family` / `generation` | `src/llb/backends/prepare/manifest.py` |
| Published-block rendering, sync, and drift check | `src/llb/quality/roster_docs.py` |
| CLI: `list-model-families`, `sync-model-family-docs` | `src/llb/cli/models/families.py` |
| CLI: `check-model-currency` | `src/llb/cli/models/currency.py` |
| Make targets | `make/models.mk`, `lint-model-roster` inside `ci-checks` (`make/dev.mk`) |
| Tests | `tests/llb/quality/test_roster_docs.py`, `tests/llb/backends/prepare/test_model_roster.py`, `tests/llb/backends/currency/` |
| Recorded registry responses | `tests/fixtures/roster_currency/upstream.json` |

Remaining work for this capability is in [the plan](../plan.md) under `model-roster-currency`.

## Related

- [Model families, tiers, and licenses](../../reference/model-families.md) -- the long form: what
  each family answers, per-tier artifacts, serving traps, and how to add a family.
- [Inference config examples](../../inference/config-example.md) -- the generated serve and
  `run-eval` artifacts per tier, and the documented hosts.
- [Platform matrix](platform-vector-matrix.md) -- one logical model compared across backends.
