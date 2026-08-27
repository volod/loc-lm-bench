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

## What adopting a generation costs

The currency probe says a newer generation EXISTS. It does not say what changing the answer costs,
and that half was reconstructed by hand from the board -- which is the step that gets skipped, since
the swap itself is a manifest edit and the invalidated readings stay silently readable until someone
notices. `make report-generation-invalidation` builds the list instead: given a family and the
generation an operator proposes to adopt, it resolves every model identity recorded in the repo's
evidence back to the family generation it was measured on, and names the records the OUTGOING
generation carries.

```bash
make report-generation-invalidation FAMILY=mistral GENERATION=4      # what a Mistral Small 4 swap voids
make report-generation-invalidation FAMILY=qwen GENERATION=4 INVALIDATION_JSON=1
make report-generation-invalidation FAMILY=lapa GENERATION=v0.1.3 INVALIDATION_STRICT=1
```

It reports only, on the same terms the currency probe does: it re-runs nothing, edits neither the
roster nor the board, and takes no position on whether the target generation is better.
`INVALIDATION_STRICT=1` exits non-zero when a swap costs anything, which is for a pre-adoption gate
rather than for CI -- every real swap costs something.

### Three surfaces record a model

A measured number lives in more than one place, and a swap voids all of them at once. Each surface
is read from the repo, never from `$DATA_DIR`, so the report reproduces on a checkout that never
held the runs:

| Surface | What is read | The model field |
| --- | --- | --- |
| `committed-aggregates` | the verbatim run analyses pinned by the [provenance manifest](extended-workflows/published-values.md) | `held_fixed.model` |
| `published-values` | every value each registered published-value design states, one row per VALUE | the design's `held_fixed.model` |
| `baseline-tables` | rows of every model-column markdown table under `docs/impl/current/` | the backticked identity in a `model`, `models`, `served artifact`, or `tag` column |

Published values are listed per value rather than per design because a design publishing six
crossovers out of three runs leaves six statements to restate; collapsing them to one design row
would report a third of the cost. Baseline rows are listed once even when the table names the
logical model AND the served artifact of the same row, because that is one edit.

The GENERATED family and license tables (README, [model families](../../reference/model-families.md))
are deliberately not on this list. They are republished by `make sync-model-family-docs` from the
register, so a swap does not invalidate them -- it regenerates them, and `make lint-model-roster`
fails until it has.

### Resolving a recorded identity

Evidence records a model the way the run reached it: an aggregate holds the Ollama tag
(`mistral-small3.1:24b`), a doc table holds the logical roster name (`mistral-small-3.1-24b`) beside
the served artifact, a vLLM lane holds the Hugging Face repo id. The register is the only thing that
can join them, so every identity a model can be recorded under -- its name and every source the
resolver would serve it from -- becomes a lookup key for that model's family and generation. Case is
folded; nothing else is normalized, because a quant suffix or a namespace prefix changes WHICH
artifact is named and trimming one would resolve a row to a model that never ran.

A recorded identity the register cannot place is listed as UNRESOLVED rather than dropped. That is
the case where a swap's cost is undercounted -- a model the roster retired is invisible to the
count -- so "nothing else is affected" and "nothing else was recognized" have to be distinguishable.
For the same reason a surface that cannot be read becomes a stated reason on its own row: losing a
surface silently understates a swap in the direction that makes it look cheap.

The doc surface is the exception, and deliberately: the delivered docs publish embedder, reranker,
and judge tables whose models are not roster entries at all, so a cell that resolves to nothing
there is a table about something else, not a gap.

### What the report read on 2026-08-27

Run on the CUDA dev host (`make report-generation-invalidation`), one proposed swap per family,
against the evidence this repo carries. The command reads committed files only -- no GPU, no
network, no `$DATA_DIR` -- so any checkout at this commit reproduces the table exactly. All five
walks scanned the same 54 records across the three surfaces, with none unread:

| Proposed swap | Roster entries voided | Aggregates | Published values | Baseline rows |
| --- | --- | ---: | ---: | ---: |
| `mistral` 3.1 -> 4 | `mistral-small-3.1-24b` | 3 | 6 | 6 |
| `gemma` 4 -> 5 | the three `gemma-4-*` entries | 0 | 0 | 18 |
| `lapa` v0.1.2 -> v0.1.3 | `lapa-v0.1.2-instruct` | 0 | 0 | 5 |
| `mamaylm` v2.0 -> v3.0 | `mamaylm-v2-12b`, `mamaylm-v2-27b-fp8` | 0 | 0 | 3 |
| `qwen` 3.8 -> 4 | `qwen3.8-27b` | 0 | 0 | 2 |

The reading: cost does not track how far behind a family is. The two swaps the currency probe
currently calls for sit at opposite ends -- Lapa is five doc-row edits, while Mistral Small is the
only family whose swap reaches the committed aggregates and the published crossovers at all, because
every agentic compact study held `mistral-small3.1:24b` fixed. Adopting Mistral Small 4 therefore
means re-running three studies and restating six published crossovers, not editing a table. Gemma's
18 rows are the widest doc surface and the shallowest work -- three entries measured in many places,
with nothing publishing a value out of them -- and Qwen's two rows are the cheapest swap on the
roster despite Qwen carrying the most generations.

What overturns this table: any new run committed against a carried generation, any study that
adopts the published-value resolver, or a doc table that gains or loses a model row. The durable
claim is that the report reaches all three surfaces and resolves a tag, a repo id, and a logical
name to one entry -- not the counts on one day, which is why the command exists rather than a list.

## Upgrading a family to a new generation

1. Cost the swap first: `make report-generation-invalidation FAMILY=<id> GENERATION=<new>` lists
   every committed aggregate, published value, and baseline row step 4 will have to re-take.
2. Add the generation under its family with `status: current` and demote the outgoing one to
   `previous` (drop a generation no model carries any more).
3. Add the logical model entries that carry it, each with `family` and `generation`, then verify
   with `make list-model-families` (register) and `make list-models` (host fit).
4. Run `make sync-model-family-docs` to republish the README and reference tables.
5. Pull the artifacts (`make prep-models`) and re-measure everything step 1 listed: those readings
   are re-run rather than carried over. For the throughput row that is
   `make measure-throughput MODELS=<model>`, which re-takes it under the protocol the committed rows
   used -- see
   [refreshing one row](backend-telemetry.md#refreshing-one-row-after-a-generation-upgrade). Re-run
   the report afterwards: a row it still lists is a reading the upgrade missed.

## Where it lives

| Piece | Module / file |
| --- | --- |
| Family register and its invariants | `src/llb/backends/roster.py` |
| Upstream currency: registry adapters, cassette, verdicts, report | `src/llb/backends/currency/` |
| Generation-swap invalidation: identity index, evidence surfaces, report | `src/llb/backends/invalidation/` |
| Manifest schema for `families:` and per-model `family` / `generation` | `src/llb/backends/prepare/manifest.py` |
| Published-block rendering, sync, and drift check | `src/llb/quality/roster_docs.py` |
| CLI: `list-model-families`, `sync-model-family-docs` | `src/llb/cli/models/families.py` |
| CLI: `check-model-currency` | `src/llb/cli/models/currency.py` |
| CLI: `report-generation-invalidation` | `src/llb/cli/models/invalidation.py` |
| Make targets | `make/models.mk`, `lint-model-roster` inside `ci-checks` (`make/dev.mk`) |
| Tests | `tests/llb/quality/test_roster_docs.py`, `tests/llb/backends/prepare/test_model_roster.py`, `tests/llb/backends/currency/`, `tests/llb/backends/invalidation/` |
| Recorded registry responses | `tests/fixtures/roster_currency/upstream.json` |

Remaining work for this capability is in [the plan](../plan.md) under `model-roster-currency`.

## Related

- [Model families, tiers, and licenses](../../reference/model-families.md) -- the long form: what
  each family answers, per-tier artifacts, serving traps, and how to add a family.
- [Inference config examples](../../inference/config-example.md) -- the generated serve and
  `run-eval` artifacts per tier, and the documented hosts.
- [Platform matrix](platform-vector-matrix.md) -- one logical model compared across backends.
