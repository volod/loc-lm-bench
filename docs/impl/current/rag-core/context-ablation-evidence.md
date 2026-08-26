# Context-ablation evidence

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md). The lane itself -- what each context lane
sends, how the window binds a skip, and what the artifact reports -- is
[Context ablation](context-ablation.md).

Each derived delta carries `p_positive` and a `(borderline)` flag, and the verdict names both the
rows it was decided on -- the retrieval uplift AND the long-context delta, because `_judge` checks
the long-context lane first
([how settled a paired reading is](paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag)).
The original `qwen3.6-35b` final-only row was the one exception: its `rag_pays_off` rested on a
settled uplift (`p_positive` 1.000) but a long-context delta at `p_positive` 0.960 that a 90%
interval read as separated. The power-resolved run below removes that exception; every recorded
context-ablation verdict is now settled at the neighbouring 90%, 95%, and 97.5% conventions.

## The lanes reproduce; the closed-book lane is not the noisier one (2026-08-25)

Three independent invocations of `make compare-context-strategies` on the RTX 4060 Ti 16 GB CUDA
host with Ollama, MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M, the committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` (82 verified `final` items, 250-document corpus, 311
chunks at 800/120, `intfloat/multilingual-e5-base`, `top_k=5`, `retrieved_document_top_n=1`,
`max_tokens=512`, temperature 0, served window 4096 against a declared 131072). All four lanes each
time: once at `--repeats 4`, then twice at `--repeats 3`, the second and third invocations each
started after `ollama stop` had unloaded the model -- ten four-lane passes in all, scored to price
how much of a lane's number is the decode rather than the model.

| lane | quoted (first pass) | settled (later passes) | band | items moved | answers moved | repeat groups |
| --- | ---: | ---: | ---: | ---: | ---: | :-: |
| `closed_book` | 0.1502 | 0.1509 | +/-0.0004 | 4/82 | 12/82 | 1+2 |
| `rag` | 0.5089 | 0.5152 | +/-0.0032 | 3/82 | 4/82 | 1+2 |
| `retrieved_document` | 0.4975 | 0.4975 | +/-0.0000 | 0/82 | 0/82 | 3 |
| `long_context` | 0.6330 | 0.6330 | +/-0.0000 | 0/82 | 0/82 | 3 |

**The premise this task was written on does not hold on this host: the ungrounded lane is the
QUIETER measurement.** Closed-book's mean objective moves +/-0.0004 across identical repeats,
eight times TIGHTER than the `rag` lane's +/-0.0032, and the two whole-document lanes never moved a
single answer. The expectation was the opposite -- a flatter next-token distribution under an
ungrounded prompt should flip more tokens -- and the two lanes that do move are simply the two with
short prompts, closed-book and chunked RAG at ~1.2k tokens, while the two lanes carrying whole
documents are immovable. The contamination rate does move, which is the part of the original
concern that survives: 10/82 (12.20%) on the quoted pass against 11/82 (13.41%) on the settled
ones, a band of +/-0.61 percentage points, so that rate is a one-decimal number with a real
half-point of slack under it.

**Every derived delta clears its decoding floor by a wide margin,** so no verdict on this page rests
on the decode: `retrieval_uplift` +0.3588 against a +/-0.0035 floor (101x), `long_context_delta`
+0.1241 against +/-0.0032 (39x), and even the smallest row, `retrieved_document_delta` -0.0114,
clears its +/-0.0032 floor 3.6 times over. The 2026-08-23 reject therefore stands on a delta that
is 3.6x the noise it could have been made of, not on a coin flip.

**The lanes are reproducible, and byte-exactly so.** Pass for pass, all three invocations agree on
every one of the 82 answers in every lane, and their lane means agree to six decimals -- the second
and third each in a fresh process, against a model Ollama had unloaded and reloaded. That is the
whole of the "make the lanes reproducible" remedy already satisfied, and it makes the drift visible
here something narrower than run-to-run noise: **within one invocation the first pass differs from
every later pass, and the later passes are identical to each other.** The difference tracks what a
lane inherits from the backend when it starts -- the first pass follows the process's own one-token
warm-up request, every later pass follows a full pass of whole-document prompts -- and since that
request sequence is identical between invocations, so is the outcome. `repeat groups` is the column
that says this: `1+2` is one odd pass and then a settled pair, and it is a completely different
finding from the same band spread over three unrelated outcomes.

The metric is steadier than the text it is computed from. Twelve closed-book answers change between
the first and later passes but only four scores move, because most of the rewrites are already-wrong
answers being restated -- one item swaps a wrong `вірмени` for a wrong `араби`, another replaces a
plausible-sounding invention with `Не знаю.` -- and a token-F1 of 0.0 does not care which wrong
answer it scored.

**What this licenses.** Quote a closed-book number with its band and read a first-pass-versus-later
difference of this size as a warm-up transient before reading it as a model or corpus change. The
quoted first pass reproduces the [2026-08-23
artifact](#the-shippable-document-lane-does-not-pay-reject-2026-08-23) exactly -- `rag` 0.5089,
`retrieved_document` 0.4975, `long_context` 0.6330 -- so those rows are re-derivable two days later
to the last recorded digit. It does NOT license reading the July rows the same way: `rag` 0.501 and
`long_context` 0.643 belong to a different host and roster state, and a month-scale gap of that size
is a change in what was serving, not the +/-0.003 measured here.

**The remedy the measurement supports is the first one, and the second is unavailable rather than
untried.** `RunConfig.temperature` is 0.0, so decoding is already greedy: there is no sampling RNG
left for a seed to pin, and `OllamaLauncher` accepts a `seed` that cannot change a token at
temperature 0. What remains is to quote the numbers with the band, which the report now does
automatically for every lane, every derived delta, and the contamination line in its header.

What would overturn it: a non-zero temperature, which reintroduces the sampler and would widen these
bands by orders of magnitude; a backend that batches concurrent requests, where the eval path is
serial today; a different Ollama or llama.cpp build, driver, or model artifact, which is exactly the
axis the July and August rows differ on; a corpus of longer or more variable documents, whose batch
shapes would change within a lane rather than between passes; or a model whose closed-book answers
run past the 280-character answer preview the answer-divergence count is read off, which would make
that count an undercount rather than the exact one it is here.

## The served window is 32x smaller than the declared one on this host (2026-08-24)

A four-lane run on the RTX 4060 Ti 16 GB CUDA host with Ollama, MamayLM-Gemma-3-12B-IT v2.0 GGUF
Q4_K_M, the committed UA fixture `samples/goldsets/ua_squad_postedited_v1/` (8 verified `final`
items, `top_k=5`, `max_tokens=512`), scored to check what the document lanes were measuring their
skips against. Both document-lane manifests recorded
`{"declared_max_model_len": 131072, "served_max_model_len": 4096, "budget_source": "served"}`; both
`closed_book` and `rag` recorded `null`.

**The declared window was 32x the served one.** At `max_tokens=512` that is a document budget of
9,216 usable characters, not the 390,144 the declared 131072 implies -- a 42x drop in what the
lanes will accept. No item was skipped in either lane, before or after: the fixture's largest
corpus document is 1,671 characters and its median is 615, three orders of magnitude inside both
budgets, which is why this corpus never exposed the gap and why the numbers are unchanged. The
finding is the gap itself: on any corpus whose documents run past ~9k characters, this host would
have handed Ollama a document it truncated at 4096 tokens and reported the answer as a fully
delivered long-context result.

This run is a binding check, not a quality reading: 8 items is far below every evidence floor this
page's verdicts are held to, so its lane means are deliberately not recorded here.

What would overturn it: an Ollama build whose `/api/ps` reports the GGUF window rather than the
served `num_ctx`, or a host with `OLLAMA_CONTEXT_LENGTH` raised, in which case `budget_source` reads
`declared` and the two windows agree. The skip behaviour itself is pinned deterministically in both
binding directions by `tests/llb/eval/context_ablation/test_context_ablation_window.py` rather than
by this run, which could not produce a skip on a corpus this small.

## The shippable document lane does not pay: reject (2026-08-23)

Four runs on the RTX 4060 Ti 16 GB CUDA host with Ollama, the committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` (82 verified `final` items, 250-document corpus, 311
chunks at 800/120, `intfloat/multilingual-e5-base`, `top_k=5`), scoring all four lanes on the SAME
item set: the two roster models whose oracle gap priced this work (MamayLM-Gemma-3-12B-IT v2.0 GGUF
Q4_K_M and Lapa v0.1.2-instruct GGUF Q4_K_M), each at both ends of the document-selection rule --
`retrieved_document_top_n=1` (the top-ranked chunk's document alone) and `=5` (every distinct
document in the retrieved top-5, which holds the retrieved SET fixed and changes only the unit).

| model | top_n | rag | retrieved_document | long_context | `retrieved_document - rag` | w/l/t | `long_context - retrieved_document` | adoption |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | ---: | --- |
| MamayLM 12B | 1 | 0.509 | 0.498 | 0.633 | -0.011 [-0.077, +0.052] | 14/13/55 | +0.135 [+0.076, +0.207] | inconclusive |
| MamayLM 12B | 5 | 0.515 | 0.498 | 0.621 | -0.017 [-0.034, -0.004] | 0/6/76 | +0.122 [+0.060, +0.186] | **reject** |
| Lapa 0.1.2 | 1 | 0.485 | 0.446 | 0.556 | -0.039 [-0.099, +0.018] | 11/13/58 | +0.110 [+0.056, +0.172] | inconclusive |
| Lapa 0.1.2 | 5 | 0.485 | 0.487 | 0.556 | +0.002 [-0.027, +0.035] | 4/3/75 | +0.069 [+0.016, +0.132] | inconclusive |

**The verdict is do not adopt, on both models and at both settings.** Not one of the four
`retrieved_document - rag` intervals lies above zero; the one interval that separates at all
separates BELOW it. Retrieval still pays decisively in every run (+0.359 to +0.393 over
closed-book, 49-58 item wins against 3-4 losses), and `long_context` still wins the ablation
(+0.071 to +0.124) -- but `oracle_document_gap` is separated above zero in all four runs, which is
the finding: on this corpus the whole long-context gain was the gold LABEL, not the document size.
Widening the unit of retrieval recovers none of it.

The two settings fail for different and instructive reasons, and the lane's own document-level
recall@k is what separates them:

- **`top_n=1` narrows retrieval as well as widening the unit.** One document is document-level
  recall@1 = 0.768 against the `rag` lane's chunk-level recall@5 = 0.951, identical on both models
  because retrieval is pinned. The answer follows the coverage: the found-rate (`contains`) falls
  0.646 -> 0.561 on MamayLM (1 win / 8 losses / 73 ties) and 0.610 -> 0.524 on Lapa (0/7/75). It is
  not a document-versus-chunk reading at all -- it is depth 1 versus depth 5.
- **`top_n=5` is the clean test, and it is flat to slightly negative.** Document-level recall is
  0.9512, exactly the `rag` lane's, so the retrieved set is held fixed and only the unit changes.
  On MamayLM the found-rate is then IDENTICAL (0.6463 both, 0 wins / 0 losses / 82 ties) and so is
  answer-side span coverage (0.8293 both); the entire -0.017 is token precision, 0.470 -> 0.452, on
  answers that got longer (mean 16.7 -> 17.5 completion tokens). All six "losses" are correct
  answers stated more fully -- `1169 році.` becoming `У 1169 році.`. The reject is real at the cut
  but it prices VERBOSITY, not knowledge, and it rests on exactly 6 discordant items, the minimum
  the sign test needs at 95%. On Lapa the same setting is +0.002 (4/3/75) with the found-rate
  slightly up (0.610 -> 0.622) -- indistinguishable from `rag` in either direction.

Because `separates` is one-sided by construction ("candidate ahead"), the derived table prints the
rejected MamayLM row's reading as `flat` while the adoption verdict says `reject`: the loss is read
off the mirrored interval gate, and the interval `[-0.034, -0.004]` in the same row is what the
verdict quotes.

**Operationally,** `top_n=1` is a cost lane rather than a quality lane: the prompt shrinks from
1202 to 328 tokens on MamayLM (872 -> 235 on Lapa) and throughput nearly doubles (10.0 -> 17.8
tok/s; 10.7 -> 19.3), which buys roughly 8-9 points of found-rate away. `top_n=5` costs a little of
both (1202 -> 1310 prompt tokens, 10.0 -> 9.2 tok/s) and buys nothing. No item was skipped in any
of the four runs: SQuAD-derived documents are ~1.5k characters, so the budget path never fired and
this reading says nothing about how the lane behaves on a corpus of long documents.

Contamination was 10-11/82 (12-13%) in every run, unchanged from the earlier cohort, so the uplift
is measured against the same non-zero baseline as before.

Reproducibility, measured, and one correction to the earlier claim: Lapa's `closed_book`, `rag`,
and `long_context` lanes were IDENTICAL across its two back-to-back runs (0.092 / 0.485 / 0.556,
same intervals), so the lane machinery adds nothing. Against the 2026-07-24 MamayLM artifact,
however, the grounded lanes did NOT reproduce exactly this time: retrieval is byte-identical
(recall@5 0.951, same ranked chunks), but 5 of 82 `rag` answers and 7 of 82 `long_context` answers
differ in phrasing, moving `rag` 0.5005 -> 0.5089 and `long_context` 0.6428 -> 0.6330. That is the
same GGUF decode nondeterminism previously recorded only for `closed_book`, now visible in the
grounded lanes across a month of host and roster change; it is far inside every interval above and
changes no verdict, but the earlier "the `rag` and `long_context` lanes reproduce exactly across
runs" claim holds only within a host state, not across one. Within one, it holds to the last digit:
every lane row in this table is reproduced two days later, pass for pass, by three independent
invocations
([2026-08-25](#the-lanes-reproduce-the-closed-book-lane-is-not-the-noisier-one-2026-08-25)).

Adding the fourth lane changes nothing the other three report. Recomputing the comparison from the
same four persisted run bundles with and without `retrieved_document` gives byte-identical
`retrieval_uplift` and `long_context_delta` entries, byte-identical per-lane reports for
`closed_book` / `rag` / `long_context`, and the same ablation verdict and reason string -- the
paired bootstrap index sets depend only on the item count and the seed, and each fitting population
is scoped to its own pair.

Artifacts: `$DATA_DIR/context-ablation/<run>/` for the four runs above, each with its own
`report.md` and `comparison.json`, plus one ordinary `run-eval` bundle per (lane, run).

## Power-resolved Qwen3.6 long-context verdict (2026-07-25)

The target was declared from the earlier 82-item `final` artifact BEFORE new inference:
minimum detectable delta +0.060 objective, 80% power, two-sided alpha 0.05. Its per-item paired SD
was 0.3078, pricing the run at 207 items. Pooling all three verified splits of the same committed
fixture supplied 250 items (`final,tuning,calibration`), above target; this is a diagnostic
ablation, not a leaderboard or tuning result.

| model | n | closed_book | rag | long_context | retrieval uplift | long-context delta | p_positive | resolution | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `batiai/qwen3.6-35b:iq3` | 250 | 0.121 | 0.534 | 0.593 | +0.413 [+0.363, +0.461] | +0.058 [+0.025, +0.096] | 0.9995 | separated | `long_context_wins` |

The long-context row has 51 wins / 23 losses / 176 ties (two-sided sign p=0.0015), no context
skips, and is separated at all three reported confidence conventions. The extra power therefore
changes the earlier final-only `rag_pays_off` reading to a settled `long_context_wins`: for this
model and corpus, chunked retrieval loses a small but real amount to whole-document context.
Retrieval itself still pays decisively over closed-book (+0.413, `p_positive` 1.000), and 33/250
closed-book answers match the reference (13.2%).

The operator boundary remains important: `long_context` is oracle-grounded on each item's gold
document, so this result supports sending the whole document AFTER a source is known; it does not
remove the need to retrieve or route to that source.
[`retrieved_document`](context-ablation.md#the-shippable-sibling-retrieved_document) is the
lane that closes that distance, and its own evidence below says how much of this gap it reaches.

The `final` split inside the pooled run independently reproduces the earlier grounded rows exactly
(`rag` 0.554, `long_context` 0.615), so the changed verdict comes from added items rather than a
changed lane. The `power-resolution` run bundle is not retained on either GPU host; the counts,
intervals, and ledgers above are the record.

## MamayLM 12B rerun on 12 GiB Blackwell (2026-07-28)

The full 82-item `final` comparison used the fitting Ukrainian MamayLM Gemma 3 12B Q4_K_M model on
the RTX PRO 3000 Blackwell and the same 311-chunk e5-base store. Closed-book scored 0.155, RAG
0.510, and oracle whole-document context 0.624. Retrieval uplift was +0.356
`[+0.272, +0.438]` with 49/4/29 wins/losses/ties; long-context minus RAG was +0.114
`[+0.051, +0.180]`. Both readings are separated at the neighbouring confidence conventions, no
item was skipped, 11/82 closed-book answers matched, and the verdict remained
`long_context_wins`. A new powered run was unnecessary for this host/model pair because the
82-item long-context reading is not borderline. Reading: the verdict survives a change of GPU
generation, driver, and quantized runtime -- the 12 GiB rerun lands +0.114 [+0.051, +0.180] against
the 16 GiB host's +0.142 [+0.083, +0.206], overlapping intervals on the same 82 items -- so
`long_context_wins` is a property of this model and corpus rather than of one box. What would
overturn it: a corpus of documents long enough that whole-document stuffing stops fitting the
served window, which the SQuAD-derived ~1.5k-character documents here never test. Lookup key:
`context-ablation` run `blackwell12-mamaylm12b`.

Durable evidence (2026-07-22, CUDA host, Ollama, committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` -- 82 verified `final` items, 250-document corpus,
311 chunks at 800/120, `top_k=5`, `DATA_DIR=.data/context-ablation-host`):

| model | closed_book | rag | long_context | retrieval uplift | long-context delta | closed-book matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.160 | 0.501 | 0.643 | +0.340 [+0.262, +0.423] | +0.142 [+0.083, +0.206] | 10/82 (12.2%) |
| Lapa v0.1.2-instruct GGUF Q4_K_M | 0.100 | 0.496 | 0.576 | +0.396 [+0.314, +0.484] | +0.080 [+0.036, +0.133] | 12/82 (14.6%) |

Both models return `long_context_wins`, and both agree on the shape of the result:

- Retrieval pays for itself, decisively. The uplift interval is far clear of zero for both models
  (sign-test p<0.001, 50/6/26 and 59/3/20 item wins/losses/ties). RAG is not decoration on this
  corpus.
- Whole-document stuffing still beats chunked retrieval, by a smaller but separable margin. That
  is expected here and is NOT an argument to ship long context: SQuAD-derived documents are ~1.5k
  characters, the lane is oracle-grounded on the item's own gold document, and `rag` retrieval was
  already near-ceiling (`recall@5=0.951`). The measured gap is what the retrieval layer still
  loses to chunk boundaries when the right document is known for free.
- Roughly one item in eight is answered correctly with no context at all -- parametric knowledge
  or contamination of a public post-edited SQuAD set. Any uplift on this fixture is therefore
  measured against a baseline that is not zero.

Skip path, measured (same model and item set, `context_budget: 1250` to force overflow):
28/82 items skipped, and the two populations diverge exactly as designed -- all-items
`long_context_delta` reads `-0.085 [-0.188, +0.018]` (the 28 skips score zero) while
`long_context_delta_fitting` over the remaining 54 reads `+0.165 [+0.091, +0.250]`. The verdict
reads the fitting delta, and the report carries both.

Reproducibility, measured: the `rag` lane's bundle is byte-identical to a plain `run-eval` of the
same configuration (all 82 items: same answers, same per-case scores), which is the check that the
lane machinery adds nothing to the leaderboard path. The `rag` and `long_context` lanes reproduce
exactly across runs WITHIN one host state -- across a month they drift too, by 5-7 of 82 answers
([2026-08-23](#the-shippable-document-lane-does-not-pay-reject-2026-08-23)); the `closed_book` lane
does not reproduce even back to back -- 11/82 answers differed between two
identical invocations (lane mean 0.160 vs 0.153), because an ungrounded prompt leaves a much
flatter next-token distribution for GGUF kernel nondeterminism to flip. The drift is well inside
the uplift interval half-width (~0.08) and changed no verdict, but a closed-book number is a
noisier measurement than a grounded one and should be quoted with that in mind. That last
generalization did NOT survive being measured: on the August host state the closed-book band is
eight times TIGHTER than the `rag` lane's and all three invocations reproduce byte-exactly
([2026-08-25](#the-lanes-reproduce-the-closed-book-lane-is-not-the-noisier-one-2026-08-25)), so the
July observation prices that host state rather than the lane.

None of the three 2026-07-22 run bundles (MamayLM, Lapa, and the budget-constrained skip run) is
retained on either GPU host; the tables, intervals, and ledgers above are the record. The 12 GiB
rerun of the MamayLM row, which IS held on the Blackwell host, is the section above.

## Roster-wide ablation cohort (2026-07-24)

The same lane, host, index fingerprint, and item set extended to the Gemma 4, MamayLM v2.0, and
Qwen3.6 rosters. `rag` recall@5 is 0.951 for every row (retrieval is pinned), so all differences
are answer-side. Throughput is the `rag` lane's measured tokens/s.

| model | closed_book | rag | long_context | retrieval uplift | long-context delta | closed-book matches | rag tok/s | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `batiai/qwen3.6-35b:iq3` | 0.133 | 0.554 | 0.615 | +0.421 [+0.333, +0.503] | +0.060 [-0.008, +0.130] | 10/82 (12.2%) | 6.1 | `rag_pays_off` |
| MamayLM-Gemma-3-27B-IT v2.0 GGUF Q4_K_M | 0.193 | 0.546 | 0.609 | +0.353 [+0.269, +0.436] | +0.063 [+0.014, +0.124] | 12/82 (14.6%) | 3.3 | `long_context_wins` |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.153 | 0.501 | 0.643 | +0.348 [+0.268, +0.429] | +0.142 [+0.083, +0.206] | 10/82 (12.2%) | 11.8 | `long_context_wins` |
| `gemma4:e4b` | 0.062 | 0.365 | 0.470 | +0.303 [+0.242, +0.364] | +0.105 [+0.056, +0.163] | 5/82 (6.1%) | 31.8 | `long_context_wins` |
| `gemma4:26b` | 0.097 | 0.288 | 0.410 | +0.190 [+0.138, +0.240] | +0.122 [+0.081, +0.169] | 11/82 (13.4%) | 12.1 | `long_context_wins` |

None of the five 2026-07-24 cohort run bundles (gemma4:e4b, gemma4:26b, MamayLM-12B,
Qwen3.6-35B-A3B, MamayLM-27B) is retained on either GPU host; the table above is the record.

What the wider cohort adds beyond the two-model result:

- **The original final-only Qwen3.6-35B-A3B row is the cohort's only `rag_pays_off`.** Its
  `long_context_delta` is +0.060 [-0.008, +0.130] (sign p=0.210), the one 82-item interval that
  straddles zero, and it posts the largest retrieval uplift measured (+0.421). The powered
  250-item run above resolves that near-miss as `long_context_wins`, so the final-only row is kept
  here as the reference observation that priced the larger run, not as the current operator
  verdict.
- **A tie at the top, at very different cost.** Qwen3.6-35B and MamayLM-27B are statistically
  indistinguishable on `rag` (0.554 vs 0.546) and on the context-position probe (paired
  +0.006 [-0.048, +0.059]), but Qwen serves from VRAM at 13 GB with ~3B active parameters while
  the 27B's 18 GB artifact runs at 23%/77% CPU offload -- 6.1 vs 3.3 tok/s on this lane, and
  18.5 vs 6.5 on closed-book. Quality-first ranking calls this a tie; the tiebreak is throughput.
- **Closed-book tracks Ukrainian specialization, not size.** MamayLM-27B leads the cohort at
  0.193 and the Gemma 4 rows sit at 0.062-0.097, so the contamination/parametric baseline a given
  uplift is measured against is model-specific and must be quoted with the uplift.
- The `long_context` lane skipped nothing for any model, so no fitting-population split applies.

Reproducibility, measured: MamayLM-12B reproduced its 2026-07-22 grounded lanes exactly
(`rag` 0.501, `long_context` 0.643, `long_context_delta` +0.142 [+0.083, +0.206]) while its
closed-book lane again landed at 0.153 against the original 0.160 -- an independent confirmation
of the closed-book nondeterminism documented above, on a re-run 2 days later.
