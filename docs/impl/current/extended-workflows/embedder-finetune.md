# Corpus-Adapted Embedder Fine-Tuning

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

## The problem this lane owns

Which general encoder to pin is settled by measurement, not assumption -- that is the
[embedder bake-off](../rag-core/embedders.md#the-bake-off-lane). But every candidate on that roster
was trained on general web text, so the ceiling the bake-off ranks against is a general-corpus
ceiling: on a corpus whose vocabulary the encoder never saw, no roster entry recovers the recall
that domain terms cost. This lane trains the incumbent encoder ON the operator's corpus and hands
the result back to the same bake-off, so the uplift is a measured final-split number and never a
claim about fine-tuning in general.

## Command and artifacts

```bash
llb finetune-embedder --goldset <goldset> --corpus-root <corpus-dir> \
  [--base-model <encoder>] [--negatives 4] [--max-positives 2] \
  [--epochs <n>] [--batch-size <n>] [--mini-batch-size <n>] [--learning-rate <lr>] \
  [--seed 13] [--trainer auto|fake]
make finetune-embedder GOLDSET=<goldset> CORPUS=<corpus-dir> \
  [FT_EMBED_BASE=<encoder>] [FT_EMBED_EPOCHS=] [FT_EMBED_BATCH=] [FT_EMBED_MINI_BATCH=] \
  [FT_EMBED_LR=] [TRAINER=fake]
```

`FT_EMBED_BASE` defaults to `EMBED_BASELINE`, the shipped `RunConfig.embedding_model` -- the
incumbent a tuned row has to beat. One run writes
`$DATA_DIR/finetune-embedder/<model-slug>/<timestamp>/`:

| path | what it holds |
| --- | --- |
| `pairs/pairs.jsonl` | one training row per (item, positive chunk): question, positive text, its hard negatives, and the chunk ids of each |
| `pairs/pairs_manifest.json` | what was trained on: item ids, split counts, corpus fingerprint, chunking, requested vs used negative width, rows that needed the fill, dataset digest |
| `model/` | the tuned sentence-transformers model directory, loadable by `Embedder` like any other id |
| `model/embedder_manifest.json` | which encoder this is: base model, convention family, dataset digest, item ids, split counts, seed, trainer, hyperparameters, loss curve, and the `tuned_digest` |

`--trainer fake` runs the whole lane except the weights: it validates the row set, writes both
manifests and a marker, and needs no GPU or download. That is the CI path and the control-plane
smoke check.

## What becomes a positive, and what becomes a negative

`src/llb/finetune/embedder/pairs.py` derives both from the gold labels rather than asking for new
ones:

- **Positives** are the chunks whose character range overlaps one of the item's source spans --
  `chunk_hits_any`, the same predicate `evaluate_retrieval` scores with
  ([retrieval metrics](../rag-core/retrieval-metrics.md)). Training and scoring therefore agree on
  what a hit is. Candidates are ranked by how many gold characters each chunk carries, so
  `--max-positives` truncates to the BEST chunks, not the first ones: a span split across a chunk
  boundary leaves one chunk holding a sentence and the next holding three words.
- **Negatives** (`src/llb/finetune/embedder/negatives.py`) are BM25 rows for the same question that
  carry none of its evidence -- the confusions a general encoder actually makes on a domain corpus,
  drawn from the same lexical index the [hybrid lane](../rag-core/hybrid-retrieval.md) uses. Random
  negatives would teach almost nothing; a multilingual encoder already separates a question from an
  unrelated paragraph.
- **Two rejections are load-bearing.** A chunk overlapping the item's own spans is never a
  negative, however far down the corpus it sits; and a chunk repeating the positive's TEXT
  elsewhere hits no labelled span, so the overlap predicate would call it a negative -- training
  against it would teach the encoder to separate a passage from itself.
- **The rows are rectangular.** A row short of lexical matches is filled deterministically from the
  rest of the corpus. If even the fill cannot reach the requested count on EVERY row, the width
  drops to what the thinnest row can supply rather than leaving a ragged set no batch can hold; the
  manifest records `requested_negatives`, the `negatives_per_pair` actually used, and
  `rows_needing_filled_negatives`, so a thin corpus reads as thin rather than as hard.
- **A gold item with no chunk over its spans is skipped and named** (`items_without_positive`), and
  a gold set where that is true of every item fails the export rather than training on nothing.

The chunking is the RUN CONFIG's (`strategy` / `chunk_size` / `chunk_overlap`), not a training-only
choice: the positives have to be the chunks a query will actually meet in the store, or an uplift
measured here would not be an uplift there.

## The split guard

Only verified TUNING items are read, and that is re-checked rather than trusted.
`assert_tuning_only` (`src/llb/finetune/guard.py`) refuses a pair set whose manifest declares a
non-tuning split, and -- because a manifest is operator-writable -- cross-checks its item ids
against the calibration and final ids in the gold set itself. It runs at the trainer's entry point,
before a single weight moves, since a leaked id cannot be un-trained once the run starts. The same
function guards the [LoRA hyperparameter search](finetuning-search.md#split-discipline); the lane
name in the refusal is the only thing that differs.

## The training objective

`src/llb/finetune/embedder/trainer.py` trains with multiple-negatives ranking (InfoNCE) through
`SentenceTransformerTrainer`: within a batch, a question must score its own gold chunk above its
own hard negatives AND above every other question's passages. That is the objective the
retrieval-tuned encoders on the roster were themselves trained with, so a fine-tune under it moves
the encoder along the axis its weights already encode.

Two details are not cosmetic:

- **The convention travels with the weights.** A tuned E5 is still queried with `"query: "` and
  indexed with `"passage: "`, so the training rows carry those prefixes -- taken from the BASE
  model's registered convention ([the convention registry](../rag-core/embedders.md#the-convention-registry)).
  Training bare text and querying prefixed text would tune the encoder for inputs it never meets
  again.
- **Batch size is part of the objective; the mini-batch is not.** In-batch negatives come from the
  batch, so `--batch-size 4` is a weaker objective than `--batch-size 16`, not merely a slower run.
  The default loss is therefore the CACHED one (GradCache): it reaches the same gradients in
  `--mini-batch-size` chunks, so peak VRAM stops scaling with the batch and a 12 GiB card can train
  at the batch the objective needs. The uncached `multiple-negatives-ranking` stays selectable; on
  this host it runs out of memory at the default batch, which is what the cached default exists to
  avoid. Defaults: `learning_rate=2e-5`, `num_train_epochs=3`, `per_device_train_batch_size=16`,
  `mini_batch_size=4`, `warmup_ratio=0.1`, `max_length=512`; the manifest records whatever ran.

Dependency contract: `make install-extras EXTRAS=rag,finetune` -- `[rag]` supplies
sentence-transformers, `[finetune]` supplies `datasets` and `accelerate`. All three are imported
lazily and checked up front, because `SentenceTransformerTrainer` is a `transformers.Trainer` and
raises for a missing `accelerate` only after the arguments are built, which is late enough to look
like a bug in this lane. Every unit test runs on the `fake` trainer, so the lightweight `make ci`
suite needs none of it.

## Measuring the result

The tuned directory is a bake-off candidate like any other id:

```bash
make compare-embeddings SPLIT=final GOLDSET=<goldset> \
  MODELS="<base-encoder>,<tuned-dir>" EMBED_BASELINE="<base-encoder>" NOISE_FLOOR=1
```

The verdict is the bake-off's own PAIRED one, not the point-estimate order: the tuned row has to
clear zero against the base encoder on the held-out final split
([paired verdicts](../rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
A tuned encoder that leads by two questions and whose interval spans zero is a retain, and the run
says so. How the bake-off resolves a directory into a candidate -- and what stops a tuned store
from being queried by anything else -- is
[the tuned-encoder section of the embedder page](../rag-core/embedders.md#a-locally-fine-tuned-encoder-as-a-candidate).

## What it bought on the quickstart corpus (2026-08-27)

First run of the lane end to end, on the RTX PRO 3000 Blackwell 12 GB CUDA host. Base encoder
`intfloat/multilingual-e5-base` (the shipped `RunConfig.embedding_model`), corpus and gold set
`samples/goldsets/ua_squad_postedited_v1` -- 250 Ukrainian post-edited SQuAD documents chunked
`recursive` 800/120 into 311 chunks, 82 verified tuning items and 82 verified final items.
Training exported 83 rows (82 items; one item had a second gold chunk) with 4 hard negatives each,
and ran 3 epochs at batch 16 / mini-batch 4 under the cached multiple-negatives loss -- 18
optimizer steps, 16 seconds, training loss 2.68 -> 1.10. Scoring: `compare-embeddings` at k=10,
2000 paired bootstrap resamples, seed 13, both encoders over the same corpus and chunking.

| split | encoder | recall@10 | MRR |
| --- | --- | ---: | ---: |
| tuning (trained on) | `multilingual-e5-base` | 0.976 | 0.877 |
| tuning (trained on) | tuned | 1.000 | 0.943 |
| final (held out) | `multilingual-e5-base` | 0.976 | 0.838 |
| final (held out) | tuned | 0.963 | 0.828 |

Paired against the base encoder, tuned minus base:

| split | metric | delta | 95% CI | win/loss/tie | randomization p | reading |
| --- | --- | ---: | --- | ---: | ---: | --- |
| tuning | recall@10 | +0.024 | `[+0.000, +0.061]` | 2/0/80 | 0.2500 | flat |
| tuning | MRR | +0.066 | `[+0.022, +0.117]` | 14/2/66 | 0.0031 | separated |
| final | recall@10 | -0.012 | `[-0.037, +0.000]` | 0/1/81 | 1.0000 | flat |
| final | MRR | -0.011 | `[-0.049, +0.026]` | 7/8/67 | 0.6967 | flat |

**The verdict is RETAIN `intfloat/multilingual-e5-base`, and the shipped default does not move.**
What the two splits say together:

- **The lane trains.** On the questions it saw, the tuned encoder's first-hit rank is separated
  from the base encoder's -- +0.066 MRR on 16 differing items, 14 of them wins, at p=0.0031 -- and
  it retrieves every tuning item at k=10. Loss fell by more than half over 18 steps. A broken
  convention, a mis-derived positive, or a negative set full of true positives would show here as
  nothing happening, and something happened.
- **None of it transfers.** On the held-out final split both metrics are flat and MRR's wins and
  losses are the same size (7 against 8). The encoder learned these 82 questions, not this corpus's
  vocabulary. On 83 rows from a 311-chunk corpus that is the expected shape of the result, and it
  is why the adoption gate is the held-out paired verdict rather than the training-split gain a
  reader would otherwise be shown.
- **There was almost nothing to win.** The base encoder already answers 80 of 82 final items at
  k=10 (0.976). This corpus is post-edited Ukrainian Wikipedia -- general-domain text of exactly
  the kind multilingual E5 was trained on -- so the premise the lane exists for, domain vocabulary
  a general encoder misses, is not what this corpus tests. The result is evidence about this
  corpus, not about corpus adaptation.
- **Cost is not the reason to retain.** Both encoders are the same architecture: 768 dimensions,
  identical 1.29 MB index over the same 311 chunks. The build-time difference between the two runs
  is cache and ordering, not a model property.

What would overturn this: an operator corpus with real domain vocabulary (statute, clinical, or
firm-internal Ukrainian) and a base recall@10 well below ceiling, where transfer has room to show;
or the same corpus with a training set an order of magnitude larger. Either is a new run of the
same two commands, and the gate it has to clear is unchanged.

## Modules and tests

| module | what it owns |
| --- | --- |
| `src/llb/finetune/embedder/pairs.py` | the export: positives, the drafted rows, the shared width, and `pairs_manifest.json` |
| `src/llb/finetune/embedder/negatives.py` | the lexical hard-negative pool and the deterministic fill |
| `src/llb/finetune/embedder/manifest.py` | both manifests, the pair digest, and the tuned identity digest |
| `src/llb/finetune/embedder/trainer.py` | the split guard call, the convention-prefixed rows, the sentence-transformers path, and the CI fake |
| `src/llb/finetune/embedder/run.py` | one run: export, train, and the run directory layout |
| `src/llb/cli/finetune/embedder.py` | the `finetune-embedder` command |

Tests: `tests/llb/finetune/embedder/test_embedder_pairs.py` (split boundary, derived positives,
rejected negatives, the width rule, digest stability, both refusals) and
`tests/llb/finetune/embedder/test_embedder_trainer.py` (the manifests, the tuned identity, both
guard refusals, the convention prefixes, the ragged-row refusal). Both run in the lightweight
`make ci` suite on a fixture corpus with no GPU and no download.
