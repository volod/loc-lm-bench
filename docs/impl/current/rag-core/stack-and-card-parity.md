# The Scoring Stack And The Card-Parity Gate

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

Two things decide whether a bake-off candidate is allowed to produce a row at all, and neither is
about its quality:

- **can it run HERE** -- four roster candidates ship their forward pass as repository code written
  against the transformers 4.x API while the repo pins 5.x for the shipped path;
- **does it reproduce its OWN model card** -- loading is not evidence a model can be ranked, and
  `Alibaba-NLP/gte-multilingual-base` is the case that proves it: on the pinned stack it loads,
  encodes without raising, and returns similarities its card does not publish.

This page owns both gates and the declared load precision that makes two passes comparable. The
lanes they guard are the [embedder bake-off](embedders.md#the-bake-off-lane) and the
[reranker bake-off](reranker-bakeoff.md).

## The card-parity gate

`src/llb/rag/card_parity.py` is the shared verdict: the comparison arithmetic, the three states a
row can be in, and the skip entry a failure produces. Each lane supplies its own reference table --
`src/llb/rag/encoder_cards.py` and `src/llb/rag/rerank_bakeoff/cards.py` -- and its own probe.

Every reference is the model card's OWN example, run through the query/passage convention this repo
registered for that family ([embedder conventions](embedders.md#the-convention-registry),
[reranker conventions](reranker-bakeoff.md)). That is deliberate: it makes one check cover both
halves of a row's readability. A candidate whose weights load wrong and a candidate scored under a
prefix its card never documents fail identically here, because either way the number in the table
cannot be read.

Cards publish their reference in whichever space their own snippet prints, so a reference declares
the transform that carries the published numbers into the space the loaded model returns -- never
the other way round:

| card | published as | how it is compared |
| --- | --- | --- |
| `Alibaba-NLP/gte-multilingual-base` | cosine similarities | as printed |
| `intfloat/multilingual-e5-large-instruct` | similarities x 100 | divided by the declared `scale` |
| `Qwen/Qwen3-Embedding-0.6B` | cosine similarities | as printed |
| `Alibaba-NLP/gte-multilingual-reranker-base` | raw classifier logits | squashed by the same sigmoid `CrossEncoder` applies |
| `jinaai/jina-reranker-v2-base-multilingual` | sigmoid probabilities | as printed |

**A card with no published numbers still gets a gate.** `jinaai/jina-embeddings-v3` publishes a
runnable snippet and no values, so its reference is that snippet: the card's own
`encode(texts, task=t, prompt_name=t)` call, which lets the MODEL apply the prompt its repository
declares, against this registry's hand-applied copy of the same prompt. What that isolates is the
format rather than the weights -- a registry entry that has drifted from the repo's own prompt
understates the encoder exactly the way a wrong prefix does, and no published number could catch it.
The two agree to `0.0000`.

Three states, and the report prints which one each row is in, because they are not
interchangeable:

- `reproduced` -- the row is evidence;
- `mismatch` / `probe_failed` -- the candidate is REFUSED. On the encoder side that happens before
  a store is built, so the expensive half never runs; on the reranker side before the scoring pass.
  It lands in `skipped[]` with the diagnosis, so a shorter table still says why;
- `no_reference_declared` -- nobody checked. Several cards publish nothing checkable, and "nobody
  checked" must never read as "it reproduces".

Both `report.md` files carry a **Model-card parity** table with the status, the reference mode, the
worst absolute delta, the tolerance, and the card URL. `report.json` carries the same record on
each scored row under `card_parity` and on each refused entry under `skipped[]`.

Tests (no download, no GPU): `tests/llb/rag/test_card_parity.py` (the verdict, the transforms, the
scale, a shape mismatch), `test_encoder_cards.py` and `test_rerank_cards.py` (the gates over
injected encoders/scorers, including each registry's self-consistency), and
`test_bakeoff_card_gate.py` (the lane statement: a mismatching candidate is never built and never
scored, and a cleared row carries the verdict that let it in).

## Declared load precision

`src/llb/rag/encoder_precision.py`. Warm chunks/s is read as a MODEL property, and on a mixed roster
it is not one: the published checkpoints differ in precision, so a half-precision upload outruns a
float32 one at identical parameter count and dimension. `--encoder-dtype` (Make: `EMBED_DTYPE=`)
loads EVERY candidate at one declared precision; the default `auto` keeps each checkpoint's own,
which is what reproduces every recorded reading.

The knob is exported process-wide as `LLB_EMBED_DTYPE`, like `LLB_EMBED_DEVICE` and the
`trust_remote_code` opt-in, because the store build, the lazy reload behind `retrieve()`, the
card-parity probe, and the throughput profiler each construct their own `Embedder` and all four
must agree. `PUBLISHED_CHECKPOINT_DTYPE` records what each roster repository uploaded, so every
encoder row prints the precision it was MEASURED at plus the card's, as `float32 (card float16)`
when the two differ. The reranker lane's existing `RERANK_DTYPE=` is now recorded in its report
header for the same reason.

`auto` is not the same thing on both stacks, which is itself a reason to declare one: transformers
5.x honors the checkpoint's `torch_dtype`, transformers 4.x materializes float32 regardless. A
gte-multilingual-base row scored in the legacy pass therefore reads `float32 (card float16)` on an
`auto` run.

## The legacy transformers pass

`src/llb/rag/model_stack.py` declares the contract; each family's convention record carries
`requires_transformers_major`, and the shared [roster screen](embedders.md#roster-screening) turns
that into a third check beside the unregistered-id refusal and the `trust_remote_code` decline. A
candidate whose repository code targets a major this interpreter is not is SKIPPED with the pin it
needs and the target that provides it -- never failed, never silently dropped.

The pass itself is a second virtualenv, not a second repo-wide pin:

```bash
make venv-encoders-legacy          # $DATA_DIR/venvs/encoders-legacy, transformers 4.x
make compare-embeddings-legacy CONFIG=<run.yaml> NOISE_FLOOR=1 EMBED_ENCODER_THROUGHPUT=1
make compare-rerankers-legacy GOLDSET=<accepted.jsonl> CORPUS=<corpus-dir> SPLIT= NOISE_FLOOR=1
```

`scripts/setup_encoders_legacy_venv.sh` builds it with `uv sync --extra encoders-legacy` against the
SHIPPED `uv.lock`, exactly as `scripts/setup_venv.sh` builds `.venv`. The
[`[encoders-legacy]` extra](../../../pyproject.toml) is declared as conflicting with `[rag]` under
`[tool.uv] conflicts`, which is what lets one lock hold both resolutions without the shipped path
being resolved down to transformers 4.x to satisfy both. (A bare `uv pip install` here resolves
outside the lock and was observed to drop `click`, failing the CLI at import rather than at
install -- hence the sync.)

Both legacy targets are the SAME recipe as their pinned siblings with `BAKEOFF_PY` pointed at that
interpreter, rather than a `$(MAKE)` re-entry: `SPLIT` is exported, so in a sub-make its origin
becomes `environment` and the config-owns-its-split rule would silently flip. `LEGACY_MODELS=`
overrides the roster; the default names the incumbent BESIDE the legacy candidates, because a pass
that scored only the unrunnable rows would have no baseline to pair them against.

What the two environments differ in, declared rather than discovered: transformers 4.57.6 against
5.12.1, and torch 2.12.1 against 2.13.0 (the lock's alternative resolution). Nothing in `src/`
imports the legacy pass, no third-party modelling code is vendored, and the repo-wide pin is
unchanged.

## What the four unscorable rows measure

CUDA host (RTX 4060 Ti, 16,380 MiB, `LLB_EMBED_DEVICE=cuda`), committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` (250 items, 311 chunks), `recursive` 800/120, flat mode,
k=10, 2000 resamples, seed 13, `NOISE_FLOOR=1`. Every table below reports a `+/-0.000` measurement
floor with 0 fragile items on the encoder rows, so every delta is a SAMPLING statement.

### The two encoders

Report: `$DATA_DIR/compare-embeddings/cardgate-legacy/report.md`.

| model | dtype | card parity | recall@10 | MRR | recall delta vs `e5-base` | w/l/t | warm c/s | peak VRAM (MB) |
| --- | --- | --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `jinaai/jina-embeddings-v3` | bfloat16 | reproduced (0.0000) | 0.988 | 0.829 | +0.008 [-0.016, +0.032] | 5/3/242 | 122.3 | 6644 |
| `intfloat/multilingual-e5-base` | float32 | (no reference) | 0.980 | 0.847 | baseline | 0/0/250 | 256.7 | 3213 |
| `Alibaba-NLP/gte-multilingual-base` | float32 (card float16) | reproduced (0.0000) | 0.956 | 0.787 | -0.024 [-0.052, +0.004] | 3/9/238 | 183.5 | 4400 |

**Verdict: RETAIN `intfloat/multilingual-e5-base`.** Neither newly scorable encoder clears the
recall@k adoption bar, and neither is close to the roster's leaders:

- `gte-multilingual-base` reproduces its card EXACTLY once its repository code has the transformers
  it targets (`[[0.3017, 0.7504, 0.3203]]`, worst delta 0.0000) and then ranks BELOW the incumbent
  on both bars -- 9 lost items against 3 won, and the worst MRR of any retrieval-tuned row on this
  fixture. The hole in the roster was packaging; the row it was hiding is not a contender.
- `jina-embeddings-v3` is the only one of the two above the baseline (+0.008 recall), but its
  interval spans zero on 8 differing items, its MRR is 0.018 BELOW the incumbent, and it holds
  6,644 MB peak -- the highest of any encoder measured on this host, 2.1x the incumbent.
- **Both cost more VRAM than every pinned-stack candidate except the 1024-dim pair**, which is the
  operational reason neither changes the recommendation even before the paired reading.

### The two rerankers

Report: `$DATA_DIR/compare-rerankers/cardgate-legacy/report.md`. Same fixture and retrieval
configuration, pool depth 30, batch 32, dtype `auto`, on an otherwise idle device (the recorded
2026-08-16 reranker run held a resident generator, so its VRAM column is not comparable with this
one; its quality columns are).

| row | card parity | recall@10 | MRR | first-hit rank | ms/query | VRAM at rest (MB) | peak (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-v2-m3` (incumbent) | (no reference) | 1.000 | 0.941 | 1.16 | 546 | 2329 | 4649 |
| `jinaai/jina-reranker-v2-base-multilingual` | reproduced (0.0030) | 0.996 | 0.904 | 1.35 | 73 | 729 | 1773 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | reproduced (0.0002) | 0.984 | 0.886 | 1.42 | 80 | 809 | 1721 |
| `off (retrieval order)` | - | 0.980 | 0.847 | 1.50 | 0 | - | - |

**Verdict: RETAIN `BAAI/bge-reranker-v2-m3`.** Both newly scorable rerankers are below the incumbent
on both bars, and the gte row's recall interval excludes zero downward
(`-0.016 [-0.032, -0.004]`); both MRR deltas do too (jina `-0.037 [-0.064, -0.013]`, gte
`-0.055 [-0.083, -0.027]`). But the cost columns are the finding worth keeping:

- **Both are ~7x cheaper per query than the incumbent** (73 and 80 ms against 546) and hold roughly
  a third of its resident VRAM. `jina-reranker-v2` buys +0.057 MRR over no reranking at all for 73
  ms, where the incumbent buys +0.094 for 546 ms.
- So the reranker roster now has a real latency/quality frontier rather than one usable point. The
  shipped default is unchanged -- quality is the primary axis -- but an operator on a latency budget
  has a measured second option instead of an unscorable row.
- `jina-reranker-v2` carries the widest measurement floor of the three (20 of 250 items fragile, MRR
  band `+/-0.001`), against 6 for the incumbent and 4 for gte.

## What a declared float32 does to the throughput column

Same fixture and host, the seven pinned-stack candidates, `EMBED_DTYPE=float32` against the `auto`
default. Reports: `$DATA_DIR/compare-embeddings/cardgate-fp32/report.md` and
`.../cardgate-auto/report.md`.

| model | card dtype | warm c/s at `auto` | warm c/s at float32 |
| --- | --- | ---: | ---: |
| `intfloat/multilingual-e5-small` | float32 | 733.6 | 715.2 |
| `lang-uk/ukr-paraphrase...` | float32 | 440.9 | 429.1 |
| `intfloat/multilingual-e5-large-instruct` | **float16** | 272.5 | **79.3** |
| `intfloat/multilingual-e5-base` | float32 | 261.6 | 255.7 |
| `BAAI/bge-m3` | float32 | 80.9 | 79.1 |
| `intfloat/multilingual-e5-large` | float32 | 79.8 | 78.2 |
| `Qwen/Qwen3-Embedding-0.6B` | **bfloat16** | 51.5 | **22.4** |

- **The recorded 3.4x lead of `e5-large-instruct` over `e5-large` was its dtype, and nothing else.**
  At a declared float32 the two are 79.3 against 78.2 warm chunks/s -- a 1.4% difference at
  identical parameter count and dimension. The throughput column is only a model comparison when
  the precision is declared; at `auto` it is a comparison of what publishers uploaded.
- **The float32 rows are unmoved** (every one within 3% of its `auto` rate), which is what says the
  two runs differ in the one variable they were meant to.
- **Retrieval quality is nearly, but not entirely, precision-invariant.** Six of the seven rows
  reproduce their recall@10 and MRR exactly at float32. `Qwen3-Embedding-0.6B` does not: 0.976 ->
  0.968 recall and 0.814 -> 0.813 MRR, two items changing rank order, moving it from
  `-0.004 [-0.024, +0.016]` to `-0.012 [-0.036, +0.012]` against the baseline. Its bfloat16 upload
  is doing measurable work on the ranking, so a Qwen3 row's precision belongs beside its numbers.
- **Precision is not free in VRAM.** `e5-large-instruct` moves from 2,906 to 3,981 MB peak and
  `Qwen3-Embedding-0.6B` from 2,981 to 4,167 MB. Declaring float32 is the right call for reading
  the throughput column; it is not automatically the right call for deploying.

## Reproduction

The pinned-stack pass at `auto` (`$DATA_DIR/compare-embeddings/cardgate-auto/report.md`) reproduces
[the 2026-08-16 roster refresh](embedders.md#the-refreshed-candidate-roster-2026-08-16) on the
fixture EXACTLY -- every recall@10, MRR, paired bound, and win/loss/tie ledger for all seven scored
rows, the `RETAIN` verdict, and the zero measurement floor -- and its warm chunks/s land within 1%
of the recorded values (733.6 / 440.9 / 272.5 / 261.6 / 80.9 / 79.8 / 51.5). `e5-base` also
reproduces its recall@10 0.980 / MRR 0.847 inside the legacy pass, on transformers 4.57.6 and torch
2.12.1, which is what makes the two passes' rows comparable at all. On the reranker side the
incumbent and the reranker-off row reproduce their recorded 2026-08-16 fixture numbers exactly
(1.000 / 0.941 / 1.16 and 0.980 / 0.847 / 1.50).
