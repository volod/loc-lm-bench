# Model Families, Tiers, and Licenses

The default candidate sweep compares open-weight FAMILIES, and a family carries one or more
GENERATIONS: exactly one is current, and a superseded one is kept only while a model still serves
from it, so a family result reads as a generation comparison rather than a single point. This page
is the long form of the [README roster](../../README.md#model-families-and-licenses) -- what each
family is in the sweep to answer, which artifact actually serves on which GPU tier, the traps that
cost a run, and the license that travels with the weights.

Two files are the machine-readable source of truth, and they win over any prose here:

- [`samples/configs/models_uk.yaml`](../../samples/configs/models_uk.yaml) -- the candidate roster:
  the `families:` register (generations, their status, upstream namespaces, licenses) and, per
  logical model, its `family` / `generation`, per-backend sources, quants, VRAM floors, and
  planning fields.
- [`docs/inference/config-example.md`](../inference/config-example.md) -- the generated serve and
  `run-eval` artifacts per detected tier, the documented hosts, and the vLLM knobs.

## The roster

The table below is generated from that register -- run `make sync-model-family-docs` after editing
it, and `make list-model-families` to print it in the terminal.

<!-- generated: model-roster (make sync-model-family-docs) -->

| Family | Generation | Status | Models carried | Weights | License |
| --- | --- | --- | --- | --- | --- |
| MamayLM (INSAIT) | MamayLM v2.0 (Gemma 3) | current | `mamaylm-v2-12b`, `mamaylm-v2-27b-fp8` | [upstream](https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3) | [Gemma](https://ai.google.dev/gemma/terms) |
| Lapa (lang-uk) | Lapa v0.1.2 instruct | current | `lapa-v0.1.2-instruct` | [upstream](https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct) | [Gemma](https://ai.google.dev/gemma/terms) |
| Gemma (Google) | Gemma 4 | current | `gemma-4-e4b-it-w4a16`, `gemma-4-12b-it-w4a16`, `gemma-4-26b-a4b` | [upstream](https://huggingface.co/collections/google/gemma-4) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Qwen (Alibaba) | Qwen3.8 | current | `qwen3.8-27b` | [upstream](https://huggingface.co/Qwen/Qwen3.8-27B) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Qwen (Alibaba) | Qwen3.6 | previous | `qwen3.6-27b`, `qwen3.6-35b-a3b-fp8` | [upstream](https://huggingface.co/collections/Qwen/qwen36) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Qwen (Alibaba) | Qwen3 | previous | `qwen3-30b-a3b` | [upstream](https://huggingface.co/Qwen/Qwen3-30B-A3B) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Mistral Small (Mistral AI) | Mistral Small 3.1 | current | `mistral-small-3.1-24b` | [upstream](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

<!-- end generated: model-roster -->

Comply with the listed license when serving or redistributing. The repository's own code is
[MIT](../../LICENSE); the weights are not, and neither is the benchmark data
([data licenses](data-licenses.md)).

## Ukrainian-specialized families

Both UA families are Gemma 3 derivatives, so both inherit the Gemma Terms rather than the Apache
license their Gemma 4 relative carries. That difference is the reason the roster records a license
per logical model instead of one license per vendor.

### MamayLM v2 (INSAIT)

The Ukrainian-specialized reference. Two logical models are carried: `mamaylm-v2-12b`
(MamayLM-Gemma-3-12B-IT-v2.0, bf16, ~26 GiB to serve through vLLM) and `mamaylm-v2-27b-fp8`
(MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic, ~30 GiB), which is also the local-judge candidate behind
the [judge calibration gate](../guides/human-tooling/calibration-tooling.md). On 12 and 16 GiB hosts
neither bf16/FP8 artifact fits, so the resolver picks the `Q4_K_M` GGUF and lets Ollama or llama.cpp
offload layers to system RAM.

The FP8 checkpoint is PRE-quantized: never pass `--quantization fp8` to vLLM for it.

Upstream: [MamayLM v2.0 (Gemma 3) collection][mamay-col] and the
[MamayLM project](https://models.mamay.ai/).

### Lapa v0.1.2 (lang-uk)

A second Ukrainian lineage on the same Gemma-3-12B base, tuned independently by lang-uk
([lapa-llm/lapa-v0.1.2-instruct][lapa-repo], bf16, ~26 GiB). Carrying two UA-specialized families
is what keeps a Ukrainian result from being one team's tuning choice: when MamayLM and Lapa agree on
a corpus, the finding is about Ukrainian; when they disagree, it is about the tune. Below 24 GiB the
resolver takes the same `Q4_K_M` GGUF offload path as MamayLM.

Prior art for both UA families is tracked by the [lang-uk leaderboard](https://github.com/lang-uk).

## Multilingual baselines

### Gemma 4 (Google)

The architecture the UA models derive from, which makes it the control that separates "Ukrainian
tuning helped" from "the base model was already good". Three logical models are carried:

- `gemma-4-e4b-it-w4a16` (~8B int4) -- the reference real-backend run on the 16 GiB dev host:
  measured 9.8 GiB of weights, ~64 tok/s steady, 15.7 GB peak VRAM at `gpu-memory-utilization 0.80`,
  8192 served context, ~112 s cold load.
- `gemma-4-12b-it-w4a16` (12B int4) -- 16 GiB supports 8192 context; 12 GiB only at a short context.
- `gemma-4-26b-a4b` (25.2B total, ~3.8B active MoE) -- the strongest host-runnable addition backed
  by a public lang-uk result. bf16 needs ~54 GiB and the FP8 conversion ~30 GiB, so 12/16 GiB hosts
  use the first-party Q4 GGUF with CPU offload.

Gemma 3 and 4 use sliding-window attention, so `make list-models` over-estimates KV cache at long
context -- its feasible-context numbers are conservative for this family, not optimistic.

### Qwen (Alibaba)

The MoE lane, the one family with an artifact that serves on EVERY CUDA tier, and the only family
carrying more than one generation.

- **Qwen3.8 (current).** `qwen3.8-27b` is the dense 27.3B `qwen3_5` checkpoint: 64 layers with
  `full_attention_interval 4`, so only 16 of them own a growing KV cache, 248320 vocab, untied
  embeddings, 262144 max context. vLLM takes bf16 (~58 GiB) or the first-party FP8 (~40 GiB); the
  16 GiB dev host serves the `qwen3.8:27b` Ollama tag (`q4_k_m`, ~17 GB) with CPU offload, and that
  tag is the Qwen target the generated local configs use. The tag needs Ollama >= 0.32.12 -- older
  runners do not know the `qwen35` architecture and refuse to load it.
- **Qwen3.6 (previous).** `qwen3.6-27b` (dense, FP8 at ~40 GiB) and `qwen3.6-35b-a3b-fp8` (36B
  stored / ~3B active) stay in the roster so a Qwen result reads as a generation comparison: a
  3.8-vs-3.6 gap on the same corpus is what says an upgrade paid for itself.
- **Qwen3 (previous).** `qwen3-30b-a3b` is the curated Ollama `qwen3:30b` tag whose ~18.6 GiB
  `q4_k_m` weights stay VRAM-resident on 24/32 GB and CPU-offload on 12/16 GB. vLLM has no
  equivalent single-artifact path, which is exactly why the curated tag is still carried.

Hybrid thinking is ON by default on every Qwen tag here. Turn it off for scoring (Ollama native
`think=false`); generated run configs also pin `temperature: 0.0` for reproducible runs. Qwen3.8 is
multimodal and the benchmark scores text only, so its vLLM serve script keeps the text-only
`--limit-mm-per-prompt`.

### Mistral Small 3.1 (Mistral AI)

A dense, non-Gemma, non-Qwen control at 24B
([mistralai/Mistral-Small-3.1-24B-Instruct-2503][mistral-repo]). The resolver prices each vLLM quant
and picks the highest-quality one that fits: FP8-dynamic (~24 GiB weights) on a 32 GiB card, w4a16
(~14 GiB) on a 24 GiB card, and the curated Ollama `mistral-small3.1:24b` tag below that.

Two traps: both compressed-tensors checkpoints are auto-detected, so never pass `--quantization`;
and use the CURATED Ollama tag rather than a third-party HF GGUF mirror of the same checkpoint --
the mirrors crash the Ollama 0.20 llama.cpp runner on load. Mistral Small 3.1 is multimodal and the
benchmark scores text only, so the generated serve script keeps
`--limit-mm-per-prompt '{"image": 0}'`.

## Which artifact serves on which tier

A family name is not a served model. The resolver expands each logical model's `sources` into
per-backend candidates, prices each one against the detected host, and picks the highest-quality
artifact that fits, in backend priority `vllm > ollama > llamacpp`:

```bash
make detect-gpu-vram       # total VRAM -> tier (<14 GiB -> 12, <20 -> 16, <28 -> 24, else 32)
make list-models           # per-model fit estimate and feasible context on this host
make gen-serving-config    # serve_*.sh + run_eval_*.{yaml,sh} for the detected tier
make prep-models           # pull Ollama tags, snapshot-cache HF weights once
```

The estimate is architecture-aware because `params * bpw` is wrong for partially quantized
checkpoints: `w4a16` and FP8 keep the embedding (and Gemma 3n Per-Layer Embeddings) at high
precision while quantizing only the linear layers, so the planner prices that mass separately from
`vocab_size` / `hidden_size` / `tie_word_embeddings`. GGUF k-quants quantize the embedding too, so
the premium does not apply to Ollama and llama.cpp sources. See the
[memory planner](../impl/current/robustness-ontology-backends.md#memory-planner).

The full per-tier fit table (12 / 16 / 24 / 32 GiB, per family, with the vLLM knobs) lives in
[config-example.md](../inference/config-example.md#tier-fit-summary-from-manifest), beside the
[serving traps](../inference/config-example.md#traps) and the documented host profiles.

## Licenses and gating

- Every logical model records `license` and `license_url`. The Gemma-3-derived Ukrainian families
  carry the [Gemma Terms][gemma-lic]; Gemma 4, Qwen, and Mistral artifacts carry
  [Apache 2.0][apache-lic]. Accepting a license is your obligation, not the tool's.
- None of the roster artifacts are gated. To add a gated model (a Gemma 2/3 or Llama repo, for
  example) set `gated: true` and `license_url: <hf-page>` on the entry: `make prep-models` then
  prints the acceptance link, and does so automatically when a download hits a gate. Accept the
  terms on that page, then set `HF_TOKEN` in `.env` before the download will proceed.
- Redistributing weights, merges, or adapters trained on them carries the upstream terms with it.
  Adapter provenance is recorded by the
  [adapter registry](../impl/current/extended-workflows/adapter-registry.md).

## Adding a family or a generation

A family is registered once under `families:` in
[`samples/configs/models_uk.yaml`](../../samples/configs/models_uk.yaml), and every logical model
under `models:` names the family and generation it belongs to. The register is what the published
tables are generated from, so a new family or generation reaches the README by editing it:

| Field | Why it exists |
| --- | --- |
| `id`, `label`, `role` | The family key, its published name, and whether it is a UA-specialized entry or a multilingual baseline |
| `focus` | One line saying what the family is in the sweep to answer |
| `upstream` | Where a currency check reads the family: `hf_author`, `hf_prefix`, `ollama_namespace` |
| `generations[].id`, `.status` | The generation key and whether it is `current` (exactly one per family) or `previous` |
| `generations[].label`, `.weights_url` | How the generation is published and where its weights live |
| `generations[].license`, `.license_url` | The terms that travel with that generation's weights |

Adding a generation is three steps: add the generation to its family with `status: current` and
demote the outgoing one to `previous`, add the logical model entries that carry it, then run
`make sync-model-family-docs` to republish the tables. A generation no model carries any more is
removed rather than kept, and `make ci` fails while the register and the docs disagree.

Then add the model entry itself under `models:`. The fields that matter:

| Field | Why it exists |
| --- | --- |
| `name`, `backend`, `source` | The logical model, its primary backend, and the tag or HF repo id |
| `family`, `generation` | Which registered family and generation this model carries |
| `sources:` | Per-backend (and per-quant) artifacts the resolver may substitute when the primary does not fit |
| `min_vram_gb` | Rough floor to SERVE at a sane context; used to skip or flag, never as the fit test |
| `params_b`, `quant`, `n_layers`, `kv_dim`, `max_context` | Planning fields for the weight and KV estimate |
| `vocab_size`, `hidden_size`, `tie_word_embeddings`, `hi_precision_params_b` | Embedding-aware weight pricing for partial quants |
| `license`, `license_url`, `gated` | What travels with the weights, and whether acceptance is required first |

Architecture values are also auto-read from a cached Hugging Face `config.json` when one is present,
so curated fields are an override and a fallback rather than the only source. Verify a new entry
with `make list-models` (fit) and `make list-model-families` (register) before `make prep-models`
downloads anything.

## Related

- [Inference config examples](../inference/config-example.md) -- tier detection, generated serve
  scripts, vLLM memory tuning, documented hosts.
- [Robust backends](../impl/current/robustness-ontology-backends.md) -- memory planner, preparation
  contracts, VRAM contention guard, llama.cpp and vLLM preflight.
- [Platform matrix](../impl/current/platform-vector-matrix.md) -- comparing one logical model base
  across serving backends with power and throughput telemetry.
- [Data licenses](data-licenses.md) -- the terms on the corpora and task sets, which are separate
  from the terms on the weights.

[mamay-col]: https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3
[lapa-repo]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[mistral-repo]: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
[gemma-lic]: https://ai.google.dev/gemma/terms
[apache-lic]: https://www.apache.org/licenses/LICENSE-2.0
