# Embedder Conventions And Bake-off

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## The convention registry

Per-family query/passage conventions live in `src/llb/rag/encoders/families.py`: a retrieval-tuned
encoder scored with the wrong instruction silently loses recall, so `Embedder`
(`src/llb/rag/encoders/embedder.py`) applies each model FAMILY's declared convention. Resolution is an
ORDERED registry, not a chain of substring tests -- `embedding_family` walks a match table whose
first row whose every substring appears in the lowercased id wins, `resolve_convention` returns the
`EmbeddingConvention` record, and `apply_query_convention` / `apply_passage_convention` are pure +
unit-tested (`tests/llb/rag/encoders/test_embedding_families.py`). Every entry names the model card its
prefixes were read from, so a row's format is auditable without reading module history.

| family | models | query side | passage side | notes |
| --- | --- | --- | --- | --- |
| `e5` | `intfloat/multilingual-e5-{small,base,large}` | `"query: "` | `"passage: "` | |
| `e5-instruct` | `intfloat/multilingual-e5-large-instruct` | `"Instruct: <task>\nQuery: "` | none | card: "no need to add instruction for retrieval documents" |
| `bge-m3` | `BAAI/bge-m3` | none | none | FlagEmbedding retrieval default |
| `bge` | other BGE retrieval lines (`bge-large-en-v1.5`) | query-only instruction | none | |
| `gte-multilingual` | `Alibaba-NLP/gte-multilingual-base` | none | none | needs `trust_remote_code` |
| `jina-v3` | `jinaai/jina-embeddings-v3` | `"Represent the query for retrieving evidence documents: "` plus `task="retrieval.query"` | `"Represent the document for retrieval: "` plus `task="retrieval.passage"` | needs `trust_remote_code`; `task=` selects the LoRA adapter, which the prompt text alone does not |
| `qwen3-embedding` | `Qwen/Qwen3-Embedding-0.6B` | `"Instruct: <task>\nQuery:"` (NO trailing space) | none | prompt taken verbatim from the repo's `config_sentence_transformers.json` |
| `plain` | paraphrase/STS (`lang-uk/ukr-paraphrase-multilingual-mpnet-base`, LaBSE, `all-mpnet`) | none | none | symmetric |
| `unknown` | anything unregistered | none | none | see below -- NOT a convention |

`<task>` is the shared `RETRIEVAL_TASK` constant, "Given a web search query, retrieve relevant
passages that answer the query", which both instruct-style cards use verbatim.

**An unregistered id resolves to `unknown`, not to `plain`.** That distinction is the point of the
table. `plain` -- symmetric, no instruction -- is a DOCUMENTED property of the paraphrase/STS line,
not a safe default for an encoder whose card nobody read, and the old fall-through made those two
cases indistinguishable: `multilingual-e5-large-instruct` landed in plain `e5` and would have been
scored with `query:` / `passage:` prefixes its card never documents, while any unknown id was
encoded with no instruction at all. `Embedder` now WARNS on an `unknown` family, and
`compare-embeddings` refuses such a candidate outright (below).

Some conventions are more than a prefix, so an `EmbeddingConvention` also carries per-side
`SentenceTransformer.encode()` kwargs (jina-v3's task adapter) and a `trust_remote_code` flag.

## The `trust_remote_code` opt-in

Two current-generation candidates ship their forward pass as repository code. Executing downloaded
code is an operator decision, so it is opt-in: `LLB_TRUST_REMOTE_CODE=1` (or
`Embedder(..., trust_remote_code=True)`, which wins over the env knob). Without it `Embedder._load`
refuses with a message naming both the knob and the card to review. `compare-embeddings
--allow-remote-code` (Make: `EMBED_ALLOW_REMOTE_CODE=1`) exports the knob process-wide -- like
`LLB_EMBED_DEVICE`, because the store build, the lazy reload behind `retrieve()`, and the throughput
profiler each construct their own `Embedder` and all three must agree.

## Roster screening

`screen_candidates` (`src/llb/rag/embedding_bakeoff/roster.py`) runs before any store is built and
splits the roster two ways:

- an id with **no registered convention** RAISES `UnregisteredCandidateError`; the CLI exits 2
  before building anything. Scoring an encoder under a guessed format understates it rather than
  ranking it, which is the one failure a bake-off must not commit;
- a **`trust_remote_code`** candidate without the opt-in is SKIPPED, and the reason lands in
  `report.json` as a `skipped[]` entry and in `report.md` under "Roster entries not scored" -- so a
  declined row reads as declined, never as beaten. The rest of the roster still ranks.

A third check joins those two once the caller declares which stack it is on: a candidate whose
repository code targets a different transformers major is SKIPPED with the pin it needs and the
target that provides it ([the legacy transformers
pass](stack-and-card-parity.md#the-legacy-transformers-pass)).

Every scored row now also carries the `family` it was scored under, the precision it was measured
at, its [card-parity verdict](stack-and-card-parity.md#the-card-parity-gate), and, when repo code ran,
`trust_remote_code: true`; `report.md` prints both plus a peak-VRAM column fed from the
`--encoder-throughput` decomposition.

## The bake-off lane

`llb compare-embeddings` (`src/llb/rag/embedding_bakeoff/run.py`; `make compare-embeddings`) answers
"which embedder for Ukrainian?" with evidence, not assumption. It builds one store per candidate
over the SAME corpus + chunking (each under its own family convention), scores recall@k / MRR by the
model-independent source-span metric (reusing `evaluate_retrieval`), and reports embed throughput,
index size, dimension, and device -- ending in the [adopt-or-retain verdict](paired-verdicts.md),
which the operator applies via `build-index --embedding-model <winner>` +
`RunConfig.embedding_model`. Artifacts: `$DATA_DIR/compare-embeddings/<timestamp>/report.md` and
`report.json` plus one saved store per candidate under `stores/<model-slug>/`.
`DEFAULT_LOCAL_CANDIDATES` (`src/llb/rag/embedding_bakeoff/models.py`) is nine ids in three bands:
the incumbents (`intfloat/multilingual-e5-base` -- the current default and the paired baseline --
plus `-small`, `-large`, and `BAAI/bge-m3`), the current multilingual retrieval generation
(`intfloat/multilingual-e5-large-instruct`, `Alibaba-NLP/gte-multilingual-base`,
`jinaai/jina-embeddings-v3`, `Qwen/Qwen3-Embedding-0.6B`), and the paraphrase/STS
`lang-uk/ukr-paraphrase-multilingual-mpnet-base` whose objective differs. The store builder is an
injectable seam, so
scoring, ranking, the consent gate, and report shaping are fake-store unit-tested
(`tests/llb/rag/test_embedding_bakeoff.py`) with no GPU/FAISS/network. The lane is five modules:
`embedding_bakeoff/models.py` (the item/store seams and the row + report shapes every consumer
reads), `embedding_bakeoff/scoring.py` (the retrieval pass, the report row, and the ranking over one
built store), `embedding_bakeoff/run.py` (drive the roster, the gated API row, and the report),
`embedding_bakeoff/uncertainty.py` (the paired intervals and [the verdict](paired-verdicts.md)), and
`embedding_bakeoff/report.py` (ASCII + Markdown rendering).

`NOISE_FLOOR=1` (`--noise-floor`) adds the [measurement floor](retrieval-metrics.md#measurement-floor---noise-floor)
per candidate to both the ASCII table and `report.md`, ending in the one sentence the
recommendation needs: how far the winner leads the runner-up and whether that lead clears the
floor. Without it `report.md` says so explicitly instead of leaving the reader to assume a lead is
real. This lane is where the floor matters most -- several candidates on one corpus routinely differ
by a single item.

## The recommendation re-read against the floor

CUDA host, 2026-07-24; report under
`$DATA_DIR/compare-embeddings/floor-reread/compare-embeddings/<run>/report.md`. The recorded
recommendation was measured on a verified 44-item quickstart-PDF accepted goldset that is no longer
on disk, so the re-read uses the human-ACCEPTED converted-PDF goldset the repo still has (40 items,
one document, 1120 chunks at `recursive` 800/120, flat mode, k=10) -- a different item set, stated
where the numbers are.

| model | recall@10 | MRR | dim | chunks/s | size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 0.975 | 0.917 | 1024 | 5.5 | 5.89 |
| `intfloat/multilingual-e5-large` | 0.925 | 0.871 | 1024 | 5.5 | 5.89 |
| `intfloat/multilingual-e5-base` | 0.925 | 0.852 | 768 | 17.7 | 4.79 |
| `lang-uk/ukr-paraphrase-multilingual-mpnet-base` | 0.475 | 0.241 | 768 | 33.2 | 4.79 |

Floor: recall@10 `+/-0.000`, MRR `+/-0.000`, 0 of 40 items fragile in every lane.

What the re-read establishes:

- **The floor is not what limits this recommendation.** Every candidate's band is exactly zero:
  no item's rank-10 / rank-11 scores sit within `1e-6` in any of the four stores.
- **The recorded ranking does not reproduce on this item set.** `BAAI/bge-m3` leads by 0.050
  recall@10 -- two items, and it clears the zero floor -- where the recorded run had it 0.023
  BELOW `e5-base`. `e5-base` and `e5-large` tie exactly here (0.925), so the recorded separation
  between those two is not reproduced either. The `lang-uk` paraphrase model collapses on both
  runs, which is the one part of the recorded reading that holds.
- **A zero floor does not make a 2-item lead a ranking.** 0.050 on n=40 is two questions, and the
  floor answers only the numeric-noise question; SAMPLING is the binding constraint on this lane,
  which the paired lane below measures.
- **The default is unchanged.** `RunConfig.embedding_model` stays `intfloat/multilingual-e5-base`:
  the row that would replace it is one item-set's 2-question lead bought at 3.2x the embed cost
  (5.5 vs 17.7 chunks/s) and 1.23x the index.

## The recommendation re-read with paired uncertainty

CUDA host (`LLB_EMBED_DEVICE=cuda`), 2026-07-24; the four default local candidates at k=10,
`recursive` 800/120, flat mode, 2000 resamples, seed 13, `NOISE_FLOOR=1`. Reports (`report.md` +
`report.json`) under `$DATA_DIR/compare-embeddings/paired-uncertainty-pdf/compare-embeddings/<run>/`
and `.../paired-uncertainty-fixture/compare-embeddings/<run>/`; the two run configs are
`$DATA_DIR/compare-embeddings/paired-uncertainty.yaml` and `...-fixture.yaml`. Both corpora report
a `+/-0.000` floor with 0 fragile items, so every number below is a SAMPLING statement.

Accepted converted-PDF goldset (40 items, 1120 chunks) -- deltas against `e5-base`:

| model | recall@10 | MRR | recall delta | w/l/t | sign p | chunks/s |
| --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `BAAI/bge-m3` | 0.975 | 0.917 | +0.050 [-0.050, +0.150] | 3/1/36 | 0.625 | 48.1 |
| `intfloat/multilingual-e5-large` | 0.925 | 0.871 | 0.000 [-0.075, +0.075] | 1/1/38 | 1.000 | 47.8 |
| `intfloat/multilingual-e5-base` | 0.925 | 0.852 | 0.000 [0.000, 0.000] | 0/0/40 | 1.000 | 75.9 |
| `lang-uk/ukr-paraphrase...` | 0.475 | 0.241 | -0.450 [-0.600, -0.275] | 1/19/20 | 0.000 | 128.2 |

Committed UA fixture `samples/goldsets/ua_squad_postedited_v1/` (250 items, 311 chunks):

| model | recall@10 | MRR | recall delta | w/l/t | sign p | chunks/s |
| --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `intfloat/multilingual-e5-large` | 1.000 | 0.879 | +0.020 [+0.004, +0.040] | 5/0/245 | 0.062 | 34.5 |
| `BAAI/bge-m3` | 0.992 | 0.849 | +0.012 [0.000, +0.028] | 3/0/247 | 0.250 | 33.3 |
| `intfloat/multilingual-e5-base` | 0.980 | 0.847 | 0.000 [0.000, 0.000] | 0/0/250 | 1.000 | 32.0 |
| `lang-uk/ukr-paraphrase...` | 0.856 | 0.600 | -0.124 [-0.164, -0.084] | 0/31/219 | 0.000 | 54.0 |

The 2026-07-28 rerun on the 12,227 MiB RTX PRO 3000 Blackwell reproduced every
host-independent field above exactly: the four recall/MRR values, all paired bounds and ledgers,
the zero measurement floor with no fragile items, and the `retain` verdict with the 300-item open
question for e5-large. Every row records `device=cuda`. Throughput on this 80 W laptop GPU was
25.6 chunks/s for e5-base, 6.8 for e5-large, 5.9 for BGE-M3, and 6.1 for the paraphrase model in
the one-pass store build; those rates mixed cold load with encoding, which the warm decomposition
below separates. Artifact:
`$DATA_DIR/compare-embeddings/20260728T110500Z-blackwell12/{report.md,report.json}`.

## Blackwell encoder throughput decomposition

`make compare-embeddings ... EMBED_ENCODER_THROUGHPUT=1 EMBED_ENCODER_COMPARE_CPU=1` on the same
12,227 MiB RTX PRO 3000 Blackwell host (power limit held at 80 W) over the 311-chunk committed UA
fixture. Each candidate records cold load, first-pass (compile+encode), and adaptive warm encodes
until IQR/median <= 0.05 or a pass/time cap. Additive fields land on the bake-off bundle; the host
summary is `$DATA_DIR/encoder-throughput/20260729T124909.208587Z-cb457ea736f4/`.

CUDA warm rates (311 texts; median over warm passes that cleared 0.05 relative precision):

| model | load_s | first_s | compile_est_s | warm_chunks/s | one_pass_chunks/s | peak_vram_MB | mean_power_W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lang-uk/ukr-paraphrase...` | 5.72 | 0.81 | 0.00 | 342.0 | 47.6 | 9015 | 72.2 |
| `intfloat/multilingual-e5-base` | 5.61 | 1.45 | 0.00 | 208.5 | 44.1 | 2371 | 80.7 |
| `BAAI/bge-m3` | 5.94 | 5.04 | 0.05 | 62.4 | 28.3 | 6839 | 79.9 |
| `intfloat/multilingual-e5-large` | 5.82 | 5.08 | 0.06 | 62.0 | 28.5 | 4597 | 77.0 |

What the decomposition establishes:

- **Cold load dominates the one-pass number.** Every model spends ~5.6-6.0 s loading weights; the
  first-pass compile estimate is near zero on this torch/CUDA stack. A one-pass bake-off that
  folds load into `embed_seconds` therefore understates steady encode by 3-7x and is not a model
  throughput recommendation.
- **The architecture-dependent spread survives warm measurement.** e5-base remains ~3.4x e5-large
  on warm chunks/s (208 vs 62). The 2026-07-28 one-pass spread was not an artifact of load alone.
- **e5-large vs BGE-M3 are tied when warm.** One-pass CUDA order puts e5-large ahead of BGE-M3 by
  a hair; warm order flips them (62.4 vs 62.0). Prefer warm chunks/s, and do not rank those two on
  throughput. Headline CUDA ordering does NOT survive (`ordering_survives=false`); CPU ordering
  does.
- **CPU twin confirms the same shape at ~10x lower rate.** paraphrase 31.9 / e5-base 20.2 /
  BGE-M3 5.6 / e5-large 5.5 warm chunks/s. Use `LLB_EMBED_DEVICE=cpu` on this host when the GPU
  is reserved for a served generator.
- **Quality verdict unchanged.** The paired bake-off still RETAINs `e5-base`; throughput is a
  cost column beside that call, not a reason to adopt the paraphrase model (its recall still
  separates negatively).

Reusable knobs: `--encoder-throughput`, `--encoder-precision`, `--encoder-min-warm`,
`--encoder-max-warm`, `--encoder-max-warm-seconds`, `--encoder-compare-cpu` (Make:
`EMBED_ENCODER_THROUGHPUT=1`, `EMBED_ENCODER_COMPARE_CPU=1`, ...). Only the literal
`EMBED_ENCODER_COMPARE_CPU=1` enables the CPU twin (`=0` is off). Host summaries name
`faster_than_baseline` when the bake-off baseline is set. `Embedder.release()` plus
`torch.cuda.empty_cache()` runs between candidates so peak VRAM is per-encoder, not stacked.
CI covers the aggregation with an injected clock and fake encoders
(`tests/llb/rag/encoders/test_encoder_throughput.py`).

## The refreshed candidate roster (2026-08-16)

RTX 4060 Ti (16,380 MiB) CUDA host, `LLB_EMBED_DEVICE=cuda`, both scored corpora, k=10,
`recursive` 800/120, flat mode, 2000 resamples, seed 13, `NOISE_FLOOR=1`,
`EMBED_ENCODER_THROUGHPUT=1`. Run configs unchanged
(`.data/compare-embeddings/paired-uncertainty{,-fixture}.yaml`); reports under
`$DATA_DIR/compare-embeddings/paired-uncertainty-pdf/compare-embeddings/20260816T120805.613959Z-d843778462a0/`
and `.../paired-uncertainty-fixture/compare-embeddings/20260816T120246.110009Z-1d83e6004ec4/`, with
the throughput host summaries beside them under each corpus's `encoder-throughput/<run>/`.

Both corpora report a `+/-0.000` measurement floor with 0 fragile items, so every delta below is a
SAMPLING statement. Seven of the nine roster ids were scored; the two `trust_remote_code` rows were
declined by default and are read separately below.

Committed UA fixture `samples/goldsets/ua_squad_postedited_v1/` (250 items, 311 chunks) -- deltas
against `e5-base`:

| model | family | recall@10 | MRR | recall delta | w/l/t | warm c/s | peak VRAM (MB) |
| --- | --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `intfloat/multilingual-e5-large` | e5 | 1.000 | 0.879 | +0.020 [+0.004, +0.040] | 5/0/245 | 79.9 | 4001 |
| `intfloat/multilingual-e5-small` | e5 | 0.996 | 0.836 | +0.016 [+0.004, +0.032] | 4/0/246 | 738.9 | 2691 |
| `intfloat/multilingual-e5-large-instruct` | e5-instruct | 0.992 | 0.850 | +0.012 [0.000, +0.028] | 3/0/247 | 272.6 | 5077 |
| `BAAI/bge-m3` | bge-m3 | 0.992 | 0.849 | +0.012 [0.000, +0.028] | 3/0/247 | 80.7 | 4001 |
| `intfloat/multilingual-e5-base` | e5 | 0.980 | 0.847 | baseline | 0/0/250 | 261.2 | 2959 |
| `Qwen/Qwen3-Embedding-0.6B` | qwen3-embedding | 0.976 | 0.814 | -0.004 [-0.024, +0.016] | 3/4/243 | 51.6 | 4049 |
| `lang-uk/ukr-paraphrase...` | plain | 0.856 | 0.600 | -0.124 [-0.164, -0.084] | 0/31/219 | 440.1 | 2905 |

Accepted converted-PDF goldset (40 items, 1120 chunks) -- deltas against `e5-base`:

| model | family | recall@10 | MRR | recall delta | w/l/t | warm c/s | peak VRAM (MB) |
| --- | --- | ---: | ---: | ---: | :-: | ---: | ---: |
| `BAAI/bge-m3` | bge-m3 | 0.975 | 0.917 | +0.050 [-0.050, +0.150] | 3/1/36 | 65.4 | 3982 |
| `intfloat/multilingual-e5-large` | e5 | 0.925 | 0.871 | 0.000 [-0.075, +0.075] | 1/1/38 | 64.8 | 3957 |
| `intfloat/multilingual-e5-base` | e5 | 0.925 | 0.852 | baseline | 0/0/40 | 211.2 | 2915 |
| `Qwen/Qwen3-Embedding-0.6B` | qwen3-embedding | 0.925 | 0.832 | 0.000 [-0.100, +0.100] | 2/2/36 | 50.1 | 4032 |
| `intfloat/multilingual-e5-large-instruct` | e5-instruct | 0.900 | 0.850 | -0.025 [-0.075, 0.000] | 0/1/39 | 221.9 | 5058 |
| `intfloat/multilingual-e5-small` | e5 | 0.900 | 0.770 | -0.025 [-0.075, 0.000] | 0/1/39 | 586.0 | 2647 |
| `lang-uk/ukr-paraphrase...` | plain | 0.475 | 0.241 | -0.450 [-0.600, -0.275] | 1/19/20 | 437.7 | 2898 |

**Verdict: RETAIN `intfloat/multilingual-e5-base` on both corpora.** No candidate clears the
recall@k adoption bar on either. Adding the current generation did NOT resolve the standing
undecidability -- it reproduced it with more rows:

- **Neither new encoder separates upward anywhere.** `e5-large-instruct` ties `bge-m3` exactly on
  the fixture (+0.012 `[0.000, +0.028]`, 3 wins / 0 losses -- an interval touching zero on 3
  differing items, below the 6 an exact sign test needs) and is the second-WORST retrieval-tuned row
  on the PDF corpus (-0.025, one lost item). `Qwen3-Embedding-0.6B` ties the baseline on the PDF
  corpus (0.925, 2 wins / 2 losses) and sits BELOW it on the fixture (-0.004, 3 wins / 4 losses) --
  the only retrieval-tuned candidate with a negative point estimate there.
- **The incumbent rows reproduce bit-identically.** Every recorded recall/MRR value, paired bound,
  and win/loss/tie ledger from [the 2026-07-24 paired
  re-read](#the-recommendation-re-read-with-paired-uncertainty) reappears unchanged for `e5-base`,
  `e5-large`, `bge-m3`, and the `lang-uk` paraphrase model on both corpora, so the new rows are
  measured against an unmoved ruler. (`e5-small`'s recorded 1.000 / 0.819 was taken on the 82-item
  `final` split, a different item set; on the full 250 it is 0.996 / 0.836.)
- **The corpora still disagree, and now on four candidates rather than two.** The PDF corpus's
  point-estimate leader is `bge-m3`, the fixture's is `e5-large`; the two new candidates rank 4th
  and 6th on one corpus and 3rd and 6th on the other. A single-corpus bake-off remains unreadable as
  a general Ukrainian encoder ranking.
- **The measurement floor is still not the binding constraint.** Zero band, zero fragile items, on
  both corpora, across all seven candidates. Sampling is.

### Read the throughput column with the checkpoint dtype

Warm chunks/s is NOT comparable across these rows as an architecture property, because the shipped
checkpoints differ in precision -- measured with `SentenceTransformer(...)` defaults on this host:

| model | params | dtype | max_seq |
| --- | ---: | --- | ---: |
| `intfloat/multilingual-e5-small` | 118M | float32 | 512 |
| `intfloat/multilingual-e5-base` | 278M | float32 | 512 |
| `intfloat/multilingual-e5-large` | 560M | float32 | 512 |
| `intfloat/multilingual-e5-large-instruct` | 560M | **float16** | 512 |
| `BAAI/bge-m3` | 568M | float32 | 8192 |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | **bfloat16** | 32768 |
| `lang-uk/ukr-paraphrase...` | 278M | float32 | **128** |

So `e5-large-instruct` running 3.4x `e5-large` (272.6 vs 79.9 warm c/s) at the same parameter count
and dimension is a PRECISION difference in the published weights, not a faster model, and its peak
VRAM is nonetheless the roster's highest (5077 MB). Precision is now a DECLARED knob rather than an
inherited one (`--encoder-dtype` / `EMBED_DTYPE=`), and the re-read at a declared float32 settles
the caveat: the two rows land at 79.3 against 78.2 warm chunks/s, a 1.4% difference
([what a declared float32 does to the throughput
column](stack-and-card-parity.md#what-a-declared-float32-does-to-the-throughput-column)).
Every encoder row states the precision it was measured at, and the card's when the two differ.

`Qwen3-Embedding-0.6B` is the slowest row on
both corpora (~50 c/s) despite being a 0.6B model: its 32,768-token window dominates. And the
`lang-uk` paraphrase model's 128-token window is the mechanical reason its collapse is worst on the
PDF corpus, whose 800-character chunks are truncated hard. Every candidate's peak VRAM stayed under
5.1 GiB with `Embedder.release()` between candidates, so all seven fit the 16 GiB host beside a
served generator.

### The two remote-code candidates are scored in the legacy transformers pass

`Alibaba-NLP/gte-multilingual-base` and `jinaai/jina-embeddings-v3` cannot be scored on this repo's
pinned `transformers 5.12.1`: both remote-code repositories target the transformers 4.x API, so
jina raises `AttributeError: 'XLMRobertaLoRA' object has no attribute 'all_tied_weights_keys'` at
load, and gte loads but returns embeddings that do not reproduce its own card, because 5.x
materializes its non-persistent `position_ids` buffer as uninitialized memory and `rope_cos[...]`
gathers out of bounds. That is a PACKAGING fact, so the screen routes both rows to the
[legacy pass](stack-and-card-parity.md#the-legacy-transformers-pass) rather than failing the run,
and both are scored there against a reproducing `e5-base` baseline -- gte at `-0.024` recall and
jina at `+0.008` with an interval spanning zero, neither changing the recommendation
([the four unscorable rows](stack-and-card-parity.md#what-the-four-unscorable-rows-measure)).

Every scored candidate now clears a [card-parity gate](stack-and-card-parity.md#the-card-parity-gate)
before a store is built for it, which is what makes a row readable: `multilingual-e5-large-instruct`
reproduces `[[0.9193, 0.6758], [0.7038, 0.9213]]` to 0.0001 and `Qwen3-Embedding-0.6B` reproduces
`[[0.7646, 0.1414], [0.1355, 0.6000]]` to 0.0036, both on the pinned stack; `gte-multilingual-base`
reproduces `[[0.3017, 0.7504, 0.3203]]` exactly once its repository code has the transformers it
targets. A CUDA device-side assert poisons the process's CUDA context, so a candidate that fails
the gate is refused BEFORE its store is built rather than scored on numbers that do not reproduce.

## Blackwell sub-base encoder roster (e5-small)

`DEFAULT_LOCAL_CANDIDATES` now includes `intfloat/multilingual-e5-small` (same `e5` query/passage
convention as base/large). Full five-candidate bake-off + warm decomposition on the committed UA
fixture (n=82 final; 311 chunks; RTX PRO 3000 / 12 GiB / ~80 W):
`$DATA_DIR/compare-embeddings/20260729T131520.054732Z-1d36908e745c/` and
`$DATA_DIR/encoder-throughput/20260729T131520.054732Z-1d36908e745c/`. Corrected per-candidate
VRAM after the release fix (base vs small only):
`$DATA_DIR/encoder-throughput/20260729T133400.407347Z-c79df0776706/`.

| model | recall@10 | mrr | dim | warm CUDA c/s | peak_vram_MB (corrected) | d_recall vs base |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `intfloat/multilingual-e5-small` | 1.000 | 0.819 | 384 | ~638-642 | 1587 | +0.024 [0.000, 0.061] (2/0/80) |
| `intfloat/multilingual-e5-base` | 0.976 | 0.838 | 768 | ~209 | 1851 | baseline |
| `intfloat/multilingual-e5-large` | 1.000 | 0.864 | 1024 | ~62 | (see full run) | +0.024 [0.000, 0.061] (2/0/80) |
| `BAAI/bge-m3` | 0.988 | 0.844 | 1024 | ~62 | (see full run) | +0.012 [0.000, 0.037] |
| `lang-uk/ukr-paraphrase...` | 0.878 | 0.627 | 768 | ~334 | (see full run) | -0.098 [-0.171, -0.037] |

What this establishes for a 12 GiB host that must keep embeddings on CUDA beside a served
generator:

- **Cheap CUDA alternative is named.** e5-small is ~3.05x e5-base on warm chunks/s and uses less
  peak VRAM (~1.6 GiB vs ~1.9 GiB). The host summary lists it under `faster_than_baseline`.
- **Quality is flat on this item set, not an adopt.** Point estimate favors e5-small (+2 recall
  wins, 0 losses), but the paired CI includes zero and the verdict remains
  **RETAIN `intfloat/multilingual-e5-base`**. Do not change `RunConfig.embedding_model` until a
  resolvable item set clears an adoption bar
  ([paired uncertainty](#the-recommendation-re-read-with-paired-uncertainty)).
- **Paraphrase is faster but not a substitute.** It is 1.6x base on warm CUDA yet loses recall
  (-0.098); throughput alone must not promote it.
- **e5-small ties e5-large on recall@10 here** (both 1.000) at ~10x the warm rate and a third the
  dimension -- useful when the operator prioritizes index/encode cost over MRR (e5-large still
  leads MRR 0.864 vs 0.819).

Recorded verdicts: **RETAIN `intfloat/multilingual-e5-base`** on the accepted PDF goldset, **ADOPT
`intfloat/multilingual-e5-large`** on the committed fixture -- which the shipped minimum-evidence
gate now reads as **RETAIN** as well, because that adopt rests on 5 differing items ([the
gate](paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)). That withdrawn adopt is an
OPEN question, and the item set it would take to close it is recorded: 300 items at the observed 2%
discordance rate, which no committed goldset reaches, so the encoder choice is **undecidable at the
sample sizes this repo has** ([the
re-decision](paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs)). What the run
establishes:

- **The `bge-m3` lead the floor re-read surfaced is an item set, not a ranking.** +0.050 on 40
  items is 3 wins against 1 loss with 36 questions tied; the paired interval spans zero
  (`[-0.050, +0.150]`) and the exact sign test is p=0.625. The floor said the delta is not numeric
  noise and the paired lane says it is not evidence of a better encoder either.
- **The two corpora disagree, so the ranking is corpus-specific.** The PDF corpus's separated
  candidate is none; the fixture's is `e5-large` (+0.020, 5 wins, 0 losses), where `bge-m3` --
  the PDF leader -- does not separate (`[0.000, +0.028]`). A single-corpus bake-off cannot be read
  as a general Ukrainian embedder ranking.
- **The shipped default is unchanged.** `RunConfig.embedding_model` stays
  `intfloat/multilingual-e5-base`. The one ADOPT is on a committed toy fixture whose baseline is
  already at 0.980 recall (5 questions of headroom), its sign test is p=0.062 -- 5 discordant
  pairs cannot reach 0.05 on an exact two-sided sign test whatever their direction, which is
  exactly the rule the gate now applies for every lane -- and the same
  candidate is flat on the accepted operator-corpus ledger while embedding 1.6x slower there
  (47.8 vs 75.9 chunks/s) for a 1.23x index.
- **First-hit rank is where `bge-m3` does separate on the PDF corpus.** Its MRR delta is
  +0.064 `[+0.008, +0.137]` (5 wins / 1 loss) while its recall delta does not clear zero -- it
  ranks the same evidence earlier without finding more of it. The DEFAULT verdict bar is recall@k
  alone, so this does not adopt anything by itself; whether that rank gain is worth adopting is
  measured end to end in [the scoped first-hit-rank adoption
  bar](first-hit-rank-adoption.md#the-scoped-first-hit-rank-adoption-bar) below.
- **The lane can resolve a real gap at these sample sizes.** The `lang-uk` paraphrase row separates
  in the NEGATIVE direction on both corpora (-0.450 and -0.124, p=0.000), so a wide interval on the
  leaders is headroom exhaustion, not an inert statistic.

Multi-objective tune (`llb tune --objectives ...`) may sample that same shortlist as a categorical
knob; the tuner `StoreRegistry` (`src/llb/optimize/store_registry.py`) rebuilds when the embedder
or chunking fingerprint changes, prewarms the shortlist for the base chunking shape before the
Optuna loop, fans out once per new chunking fingerprint, and may reload from
`$DATA_DIR/optuna/<study>/stores/`. It never reuses a store built under a different embedder. See
[evaluation rigor](../rigor-board-judge/tuning-and-search.md#multi-objective-rag-tuner).

## Context budget

`RunConfig.context_budget` is an optional token budget that couples `top_k`, `chunk_size`, and
(for vLLM) `max_model_len`. When set, `fits_context` prunes configs whose estimated retrieved
prompt exceeds the budget, and multi-objective search samples the budget from
`{2048, 4096, 8192, 16384}` then sets `max_model_len` to that value on vLLM backends. Single-objective
`llb tune` leaves the budget unset unless the operator pins it in the run config.

Store/query embedder fingerprint: `store_meta.json` records the `embedding_model` a store was built
with, and `_load_store` refuses a run whose `config.embedding_model` differs
(`store_embedder_mismatch` in `src/llb/rag/vector_store/store.py`), because a store is embedded and
queried by one encoder -- a mismatch would silently score the wrong model. A non-default-embedder
store runs normally with the embedder recorded in the manifest fingerprint.

Opt-in API row (open corpora only): `--api-model cohere/embed-multilingual-v3.0`
(`src/llb/rag/encoders/api.py`) embeds the corpus through a hosted API -- full egress, so it is
bake-off EVIDENCE ONLY (never usable as `RunConfig.embedding_model` for a scored run), refused
unless `--data-classification open`, gated on an interactive consent prompt naming the corpus, and
capped by `--max-usd` (`record_embed_cost` aborts when the running cost crosses the cap). Cohere's
`input_type` (`search_query` / `search_document`) maps onto the query/passage seam. litellm is
lazily imported and the embed callable is injectable, so the consent gate + budget arithmetic are
unit-tested with a fake client, no network in CI. The drafting-side pinned-E5 seams (ontology dedup,
semantic scoring, retrieval-uniqueness annotation) are deliberately NOT switched by this task.

Durable evidence, full corpus (2026-07-10, embedding-bakeoff-full-corpus on the CUDA host,
`LLB_EMBED_DEVICE=cuda`, outside quick CI): the four local candidates over the verified 44-item
quickstart-PDF accepted goldset (5 PDF documents, ~1.2 MB markdown, 1139 chunks at 800/120 --
the recall spread is finally NON-saturated at both cutoffs):

| model | recall@10 | MRR@10 | recall@20 | MRR@20 | dim | chunks/s (GPU) | index MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `intfloat/multilingual-e5-base` | **0.955** | 0.740 | 0.977 | 0.742 | 768 | 69 | 4.99 |
| `intfloat/multilingual-e5-large` | 0.932 | **0.795** | 0.977 | 0.798 | 1024 | 38 | 6.10 |
| `BAAI/bge-m3` | 0.932 | 0.753 | 0.955 | 0.755 | 1024 | 38 | 6.10 |
| `lang-uk/ukr-paraphrase-multilingual-mpnet-base` | 0.455 | 0.307 | 0.500 | 0.311 | 768 | 122 | 4.99 |

Winner for the 16 GB host: `intfloat/multilingual-e5-base` (the current default) -- it holds the
highest recall@10 (the gate metric; the score an operator's answers are capped by), embeds ~1.8x
faster than the 1024-dim pair, and builds the smallest index. e5-large trades a small recall@10
loss (-0.023) for the best early ranking (MRR 0.795 vs 0.740) and ties e5-base at recall@20 --
pick it only when a downstream reranker or a small `top_k` makes first-hit rank the binding
constraint. bge-m3 trails e5-large on both axes at the same cost, and the paraphrase/STS
`lang-uk` model collapses to 0.455/0.500 recall on a real corpus (the tiny-fixture 1.000 was
saturation, not quality) -- the "paraphrase objective loses to retrieval-tuned encoders"
hypothesis is supported by this corpus. Embed VRAM peaked ~4 GB (sequential model loads),
so every candidate fits the 16 GB host with a co-resident judge stopped; steady-state GPU
throughput at 1139 chunks is no longer cold-load-dominated. Reports:
`$DATA_DIR/compare-embeddings/20260710T044652*/report.md` (k=20) and
`.../20260710T044914*/report.md` (k=10).
