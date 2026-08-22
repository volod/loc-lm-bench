# Adapter Registry And Lifecycle

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

Adapters are first-class artifacts, not loose directories. `$DATA_DIR/adapters/registry.jsonl` is an
append-only event log (`register` / `merge` / `delete`) folded into the current entry set on read, so
a partial write can never lose earlier history. The entry id IS the `adapter_digest`, so it can never
be reassigned to different weights.

Modules:

- `src/llb/finetune/registry/`: `model.py` owns `AdapterEntry`; `io.py` folds the event log;
  `register.py` performs idempotent registration and lifecycle writes; `resolve.py` handles id /
  unique prefix / label / directory lookup; `staleness.py` compares benchmark fingerprints; and
  `rows.py` renders CLI/board rows;
- `src/llb/finetune/lifecycle.py`: run-bundle citation scan, supersession, and garbage collection;
- `src/llb/finetune/serving/model.py`: immutable serve and merge contracts;
- `src/llb/finetune/serving/run.py`: serve-plan construction and launcher lifecycle;
- `src/llb/finetune/serving/merge.py`: cached merges, GGUF conversion, and Ollama Modelfiles;
- `src/llb/finetune/serving/launcher.py`: backend-specific launcher construction.

```bash
llb register-adapter --adapter-dir <dir> [--goldset <g>] [--corpus <c>] [--source-run <run>]
llb list-adapters [--json]
llb serve-adapter --adapter <id> --backend vllm|ollama|llamacpp [--smoke]
llb gc-adapters [--dry-run] [--force]
llb run-eval --adapter <id> --model <base> --backend vllm
make list-adapters ; make serve-adapter ADAPTER=<id> BACKEND=<b> ; make gc-adapters GC_DRY_RUN=1
```

Entries record the base model, dataset digest, dataset item ids and split counts, the goldset and
corpus digests observed AT TRAINING TIME, the source run, and an eval summary. Self-improvement and
campaign rounds auto-register through `register_round_adapter` after the adapter's own final eval,
so the entry carries the evidence the board later cites. Registration is best-effort: an injected
trainer that writes no `adapter_manifest.json` logs a warning instead of aborting the round. A bare
`llb finetune-adapter` does not register, so `llb register-adapter` exists to adopt a hand-trained
adapter into the registry rather than leave its board row silently dropped.

## Staleness

`staleness()` compares the recorded goldset/corpus digests against the present ones
(`durability.goldset_digest` and `corpus_governance.corpus_fingerprint`, the same functions the
durable-run journal and the stale-store check use). Verdicts are `current`, `stale`, and `unknown`;
a missing digest yields `unknown` and never `current`. Detection reports, it never retrains.

A third axis covers the RAG store (adapter-staleness-retrieval-fingerprint): an adapter is
trained on retrieved CONTEXT, so re-embedding or rechunking the same corpus invalidates its
training contexts while `corpus_fingerprint` stays unchanged. Registration records a
`retrieval_fingerprint` (embedder, chunk strategy/size/overlap, retrieval mode) read from the
store's `store_meta.json` (`register_adapter --index-dir` on the CLI; `self-improve` /
`finetune-campaign` rounds record the config's index dir automatically), and `staleness()`
compares it per knob against the store's present meta -- a rebuilt store flips the entry `stale`
with the changed knob named in the reason (for example
`retrieval embedding_model changed since training (a -> b)`). An adapter registered without an
index directory reads `unknown` on the retrieval axis (reason `retrieval fingerprint unavailable`),
never `current`.

`board/runs.py` resolves every adapter-backed bundle through the registry before it can rank:

- an unregistered adapter's row is DROPPED (a tuned number nobody can trace is not comparable);
- a registered-but-stale adapter's row is stamped `<base>+adapter-<digest> [stale]`.

`recommend.load_run_summaries` reuses `load_run_records`, so both the board and `llb recommend`
inherit the rule from one seam.

## Contamination guard through the registry

`validate_adapter_for_eval` reads training provenance from the registry when the adapter is
registered, falling back to `adapter_manifest.json` only when it is not (a freshly trained adapter
registers after its first eval). The manifest beside the weights is operator-writable, so a
hand-edited one could otherwise launder a final-split adapter past the gate. The refusal message
names the intersecting ids, the offending splits, and which provenance was consulted.

## Serving

vLLM serves the LoRA directly through the existing `--enable-lora --lora-modules` wiring, sized by
`--max-lora-rank`. That flag defaults to 16, so an adapter trained at a higher rank fails
`add_lora` at engine startup (`LoRA rank 64 is greater than max_lora_rank 16`) and vLLM exits before
serving anything. Both adapter launch paths (`executor/runner.py` for `run-eval`, `serving.py` for
`serve-adapter`) therefore read the rank off the adapter they are about to serve --
`trainer.adapter_lora_rank` prefers PEFT's own `adapter_config.json` over our manifest, since it
describes the weights actually on disk -- and `backends/vllm.served_lora_rank` rounds it up to the
nearest value vLLM accepts (`1, 8, 16, 32, 64, 128, 256, 320, 512`). An adapter of unknown rank
leaves the flag off and vLLM keeps its default.

Ollama and llama.cpp serve whole model artifacts, so `serving.py` merges the adapter into its base
weights
(PEFT `merge_and_unload`), converts to GGUF via the llama.cpp checkout's `convert_hf_to_gguf.py`, and
for Ollama registers a `llb-adapter-<short-id>` tag. The merge is expensive and one-way, so it is
cached under `$DATA_DIR/adapters/merged/<short-id>/<backend>/` behind a `merge.json` and recorded as
a registry `merge` event. Both the merge and the launcher are injectable, so CI exercises all three
backends without CUDA, llama.cpp, or a running Ollama daemon. `serve-adapter` probes the endpoint
with one generation -- an empty completion FAILS the probe (a served-but-mute endpoint is not
serving) -- and then holds it in the foreground until Ctrl-C; there is no serving daemon.

Chat-template preservation is required because llama.cpp's server applies
the `tokenizer.chat_template` GGUF metadata natively, but **Ollama ignores it** when a model is
created from a bare `FROM <gguf>` Modelfile -- the tag serves raw completions and a merged
instruct model degrades to gibberish or empty chat answers. `modelfile_text` therefore reads the
merged tokenizer's `chat_template.jinja`, detects the template family by its unambiguous marker
(ChatML
`<|im_start|>`, Gemma `<start_of_turn>`, Llama 3 `<|start_header_id|>`), and writes the
equivalent Go `TEMPLATE` plus its `PARAMETER stop` tokens into the Modelfile; an unrecognized
template stays a bare FROM with a loud warning naming the fix. Family detection, the bare-FROM
fallback, and the empty-probe failure are unit-tested with fixtures.

Pristine tokenizer files are copied from the base model because a LoRA never changes the tokenizer,
while `AutoTokenizer.save_pretrained` can be lossy for GGUF conversion: it drops the sentencepiece
`tokenizer.model` (the converter's GPT-2-style fallback then asserts on vocabularies whose added
tokens sit past `config.vocab_size`) and rewrites `tokenizer_config.json` so the control-token
markings are lost: `<start_of_turn>`/`<end_of_turn>` exported as NORMAL instead of CONTROL token
types, Ollama then never matched the template's turn markers as specials, and the merged Gemma
answered every non-trivial prompt with an immediate `<end_of_turn>` (final-split objective 0.199
vs 0.410 served properly -- while the SAME safetensors answered correctly in transformers).
`copy_base_tokenizer_assets` overwrites the resaved files with the base repo's originals
(`tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`),
best-effort per file so repos without a given file (Qwen has no sentencepiece model) keep the
resaved copy that already converts fine. Unit-tested with an injected downloader.

## Garbage collection

An adapter is superseded once a newer adapter exists for the same base model, ordered by
`(created_at, log sequence)`. `created_at` has second resolution, so two fast rounds tie; the
append-log position breaks the tie exactly. Only superseded adapters are GC candidates, and GC
refuses any that a durable artifact still cites. The citation scan covers published run bundles
(`$DATA_DIR/run-eval/*/manifest.json`, matched by recorded digest or by served `adapter_path`)
AND the orchestrator journals that also link adapter directories: self-improvement
`$DATA_DIR/self-improve/*/state.json` (`rounds[].adapter_dir`) and campaign
`$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` (`entry.adapter_dir`), both resolved
through the registry's adapter-dir index the way the served-path match is. Every citation
carries its artifact kind (`run-bundle` / `self-improve-state` / `campaign-journal`) in
`GcDecision.cited_by`, the refusal reason names the citing artifact(s), and `gc_rows` exposes
the kinds in a `cited_kinds` column. `--force` overrides the citation refusal but never the
safety rule that GC only deletes directories inside `$DATA_DIR`. Deletions append a `delete`
tombstone.

## Committed fixtures

- `samples/finetune/registry/registry.jsonl`: a stale entry (with a folded `merge` event) and a
  poisoned-digest entry, both pointing at adapter dirs outside `$DATA_DIR`;
- `samples/finetune/gc-journals/`: a data-dir-shaped fixture whose campaign journal cites the
  committed stale adapter, proving a journal-only citation blocks an unforced GC;
- `samples/finetune/stale-adapter/`: recorded digests that no longer match
  `samples/goldsets/ip_regulation_uk/`;
- `samples/finetune/laundered-adapter/`: an `adapter_manifest.json` that CLAIMS a clean tuning-only
  training set while the registry records the `final`-split ids it was really trained on;
- `samples/finetune/poisoned-adapter/`: the simpler case where the manifest itself declares the
  protected split, refused even when unregistered.

`tests/llb/finetune/test_adapter_registry.py` covers registry round-trip and idempotence, the
staleness flip when the goldset digest changes, the `unknown` verdict, guard resolution through
the registry, serving smoke over a fake launcher for all three backends, merge-event recording and
merge caching, GC citation refusal plus `--force` (run-bundle, self-improve-state, and
campaign-journal citations, including the committed journal fixture), the same-second supersession
tie, the outside-`$DATA_DIR` safety rule, and board drop/stamp behavior.

Merge-serving CUDA evidence (2026-07-10, RTX 4060 Ti 16 GB, adapter-merge-serving-cuda-evidence;
the first time the real merge lane ran end to end):

- Adapter: `ea848f7e160e` (`Qwen/Qwen2.5-0.5B-Instruct`, one `self-improve` round over the
  `ua_squad_postedited_v1` tuning split, registered; campaign
  `.data/self-improve/merge-evidence-qwen05b/`).
- Both GGUF backends merged and answered the smoke probe: PEFT merge + `convert_hf_to_gguf.py`
  (f16) + launch + probe in **~15 s wall-clock per backend** for the 0.5B model, GGUF size
  **949 MB** (vs ~1 GB safetensors); converter accepted the Qwen2 architecture without complaint.
- Three-way final-split objective (n=82, same goldset/store/seed):
  base (vLLM) **0.2880** [0.204, 0.370]; vLLM LoRA row **0.3272** [0.239, 0.422]; merged tag on
  ollama **0.3119** [0.218, 0.402] -- inside the LoRA row's CI and above the base point estimate,
  so the merged artifact answers as the ADAPTER, not the base model. One run bundle per row: base,
  LoRA, and merged (the merged row uses the fixed template).
- The Ollama Modelfile carries the explicit chat template described above, while the smoke probe
  rejects an empty completion. The `finetune` extra includes both the converter's `gguf` import and
  the trainer's `bitsandbytes` dependency so failures occur during dependency validation.

Second cohort model, `google/gemma-3-1b-it` (2026-07-10, same host; adapter `db80e8440b7d` from
one `self-improve` round trained with the effective-batch search's best config, campaign
`.data/self-improve/merge-evidence-gemma-3-1b/`):

- Merge cost: ~24 s (Ollama) / ~18 s (llama.cpp) wall-clock per backend, 1.9 GB f16 GGUF. The
  converter uses the base repository's pristine tokenizer files to preserve the sentencepiece
  vocabulary and control-token types.
- Three-way final-split objective (n=82): base (vLLM) **0.3872** [0.299, 0.480]; vLLM LoRA row
  **0.4103** [0.326, 0.498]; merged tag on ollama **0.3427** [0.260, 0.428] -- inside the LoRA
  row's CI, so the merge passes the fidelity gate, with the honest caveat that the point
  estimate sits 0.068 below the LoRA row (unresolved at n=82, and partly a cross-backend
  comparison: the merged row is f16-GGUF-on-ollama while both reference rows are
  safetensors-on-vLLM). One run bundle per row: base, LoRA, and merged.

CUDA evidence on the 12 GB RTX PRO 3000 host:

- Command shape: `LLB_EMBED_DEVICE=cpu llb finetune-campaign --config
  .data/quickstart-leaderboard/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml --models
  Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct --corpus
  samples/goldsets/ua_squad_postedited_v1/corpus --rounds 1 --limit 1 --out-dir
  .data/finetune-campaign/task19-evidence-qwen-small-12gb`.
- Campaign report:
  `.data/finetune-campaign/task19-evidence-qwen-small-12gb/report.md`.
- Recommend summary:
  `.data/recommend/task19-summary.md`.
- Shared dataset digest: `5b99939c91b02500eda6fe3aa7cb27c46012928929f93def380a245b4a6711b0`.
- `Qwen/Qwen2.5-0.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.7800` s, adapted peak VRAM `11862` MiB.
- `Qwen/Qwen2.5-1.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.4219` s, adapted peak VRAM `11690` MiB.
- `llb recommend --gpu-gb 12 --no-chart` rendered the fine-tune campaign section and selected the
  0.5B base model for this smoke cohort because all one-case objectives were tied at zero and the
  base model was faster than its adapter-backed row.
- `google/gemma-4-12B-it-qat-w4a16-ct` served on the same host at `max_model_len=1024`
  (`41.8` to `42.9` tok/s, peak VRAM about `11523` MiB), but PEFT LoRA injection could not train
  the compressed-tensors QAT checkpoint because its compressed linear modules do not expose the
  normal `weight` attribute.
