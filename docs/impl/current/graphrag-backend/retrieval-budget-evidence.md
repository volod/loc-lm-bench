# Retrieval Budget And Per-Hop Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

## The per-hop probe lane

`llb probe-multihop-hops` (`make probe-multihop-hops`) answers the question the fusion sweep
cannot: when a two-hop item misses `all-spans@k`, WHY does it miss? The sweep varies ranking knobs
and reports whether both hops arrive; when that number does not move for any knob, the remaining
explanations are not about ranking, and they lead to opposite fixes.

The lane ranks every labeled span TWICE and reads the coverage curve against those ranks:

- **by the item's own question**, in one deep pass (`--probe-depth`, default 200), giving each hop
  a rank or "never reached";
- **by the span's own text**, as the retrievability CONTROL. The span text is a verbatim slice of
  the chunk that has to be retrieved, so it is the most favorable query that hop can ever be
  given: when the question cannot reach a hop the span text reaches at rank 1, the gap is the
  query, not the index.

Each item is then classified by its WORST hop, at the smallest compared budget (the operating
budget):

| diagnosis | what it reads | the fix it points at |
| --- | --- | --- |
| `covered` | the retrieval AT the operating budget carried every labeled span | none |
| `budget` | every hop is reachable by the question, below the cut | a larger k, or a second pass |
| `query` | a hop the question never reaches, that its own text reaches at the operating budget | query decomposition |
| `unreachable` | a hop no query form reaches at the operating budget | chunking or the index, not ranking |

The counted diagnoses name one explanation per slice (`budget`, `query`, `unreachable`, or
`mixed` on a tie). The coverage curve itself is retrieved once per budget -- at that budget, not
by cutting one deep pass -- because a hybrid dense/lexical lane re-fuses its candidate pool per
requested depth, so the top 10 of a depth-200 pass is not the top 10 of a k=10 retrieval. The
report says which of the two each table is.

**`covered` is the served outcome, not a deep-pass rank.** The same re-fusion that separates the
histogram from the curve also separates them item by item, in both directions: a hop can sit
inside the deep pass's top 10 and outside a k=10 retrieval, or the reverse. `diagnose_item` in
`src/llb/rag/multihop_probe/diagnose.py` therefore takes the item's measured `all-spans@k` at the
operating budget and lets that alone decide `covered`, so the diagnosis ledger and the coverage
curve can never disagree about which items were served; the other three diagnoses stay rank-based,
because they answer "what would fix the miss", which is a question about depth rather than about
the cut. Reading `covered` off the deep ranks instead had counted one factoid item covered that a
k=10 retrieval missed (the 95-item ledger is 45/34/9/7, not 46/33/9/7) and, on the paired lane,
had swapped two multi-hop items between `covered` and `budget` against what k=10 actually
returned. Three fake-store cases in `tests/llb/rag/test_multihop_hop_probe.py` pin both directions
of the divergence and the ledger/curve agreement.

```bash
make probe-multihop-hops CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> SPLIT= \
  HOP_PROBE_BACKEND=faiss HOP_PROBE_BUDGETS=10,25,50 HOP_PROBE_DEPTH=200
make probe-multihop-hops CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> SPLIT= \
  HOP_PROBE_BACKEND=faiss HOP_PROBE_BUDGETS=10,25,50 HOP_PROBE_DEPTH=200 \
  QUERY_PREP=decompose QUERY_PREP_MODEL=<local-model> QUERY_PREP_BACKEND=ollama
llb probe-multihop-hops --config <cfg> --budgets 10,25,50 --probe-depth 200 \
  --retrieval-backend faiss --focus-slice multi-hop --out-dir <dir>
```

`HOP_PROBE_BACKEND` / `HOP_PROBE_STRATEGY` probe a different lane than the config names, so the
vector baseline and a fused row are measured through one command. `QUERY_PREP` adds a paired raw
and prepared lane. Only focus-slice items are sent to the model, one generated plan is reused at
every compared budget and the deep pass, and span-text controls remain raw. The paired report
counts conversion and regression by the RAW diagnosis cohort, which prevents an item from changing
the population whose cost it contributes to.

Artifacts are `probe.json`, `report.md`, and `run_config.json`; full generated decomposition text
and subqueries stay in JSON while the Markdown ledger records their count. The code is
`src/llb/rag/multihop_probe/`: `probe.py` retrieves, `aggregate.py` builds slice reports,
`prepared.py` drives the reusable query plan, `diagnose.py` classifies, `conversion.py` pairs the
cohorts, and the two report modules render ASCII Markdown. The CLI shares model-endpoint resolution
with `validate-retrieval` through `src/llb/cli/rag/query_prep_endpoint.py`. Fake stores and a fake
generator cover the lane in `tests/llb/rag/test_multihop_query_prep_probe.py` and
`tests/llb/rag/test_multihop_query_prep_cli.py`. Both raw and paired commands now fail before
loading a store when the requested focus slice is empty, rather than reporting zero failures as
full coverage.

## Is the both-hops ceiling a budget or a query problem?

CUDA-host evidence is under `$DATA_DIR/graph-vector-fusion-multihop/20260816T-hop-probe/` (the
probe) and `.../20260816T-k-sweep-k10`, `-k25`, `-k50` (the sweep at three budgets). All four runs
score the same 95 drafted goods items (35 multi-hop), the same rebuilt store as the
[measurement-floor re-read](fusion-sweep-evidence.md#the-sweep-re-read-against-its-measurement-floor)
(1099 indexed chunks), seed 13, and the sweep grid of the
[span-identity run](span-and-depth-evidence.md#span-identity-evidence)
(`GRAPH_FUSION_CANDIDATES=k,50 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap`).

The reproduction check passed before the comparison was read: at k=10 the vector row scores
multi-hop recall 0.686 and `all-spans@10` 0.057, and `fused/global_community@0.30/d50/ioverlap`
scores 0.800 and 0.086 -- the recorded values, to three decimals. The probe's own vector curve
(0.057 / 0.200 / 0.229) equals the sweep's vector row at the same three budgets exactly, which is
the cross-check that the probe measures the shipped retrieval path and not a re-implementation of
it.

**Result: the ceiling is a property of k=10, not of the corpus and not of the ranking.** The whole
row set moves with the budget, while at a fixed budget the whole row set is pinned in a narrow
band:

| multi-hop `all-spans@k` | k=10 | k=25 | k=50 |
| --- | ---: | ---: | ---: |
| vector | 0.057 | 0.200 | 0.229 |
| `fused/global_community@0.30/d50/ioverlap` | 0.086 | 0.171 | 0.343 |
| `fused/global_community@0.70/d50/ioverlap` | 0.057 | 0.257 | 0.657 |
| `graph/global_community` | 0.057 | 0.257 | 0.629 |
| best row of the sweep | 0.114 | 0.257 | 0.657 |
| every row, min to max | 0.029 - 0.114 (55 rows) | 0.114 - 0.257 (55 rows) | 0.229 - 0.657 (31 rows) |

At k=10 no row of the grid exceeds 0.114 (4 of 35 questions) and the spread across 55 rows is
0.086 -- three questions between the best and the worst ranking policy in the lane. Raising k
alone, with no ranking change at all, moves the vector row by more than that spread (0.057 ->
0.229, 2 to 8 questions), and the best row of the k=50 grid reaches 0.657 (23 of 35). The k=50 row
set's WORST member (0.229) beats the k=10 set's best (0.114) by a factor of two.

The per-hop probe says the same thing item by item, on the vector lane. Of the 33 multi-hop items
that miss a hop at k=10:

| diagnosis | items |
| --- | ---: |
| `budget` -- the question reaches every hop, below the cut | 19 |
| `query` -- only the span's own text reaches the hop at k=10 | 8 |
| `unreachable` -- no query form reaches the hop at k=10 | 6 |

**The measurement supports the BUDGET explanation, with a real query-side residue.** A majority of
the failures (19 of 33) are hops the question already ranks -- just not in the top 10 -- though
only some of them sit at a depth an operator would serve, which the histogram below prices. The
per-hop hit rate (share of the 70 labeled spans the item's own question ranks within k) rises
0.386 -> 0.500 -> 0.600 across the three budgets, while the same spans queried by their own text
are found at 0.800 -> 0.914 -> 0.957. That second column is the ceiling an ideal query would hit,
and at k=10 it is twice the question's own rate: decomposition has room, but it is the second
effect, not the first.

Where the budget would have to go is measured too, from the deep pass:

| smallest cutoff carrying EVERY hop | items |
| --- | ---: |
| <= 10 | 2 |
| <= 25 | 5 |
| <= 50 | 2 |
| <= 200 | 12 |
| never reached (depth 200) | 14 |

So a budget fix has a long tail: k=50 buys 9 of 35 items, and 12 more need a pool between 50 and
200 (their limiting ranks run 59, 61, 63, 65, 82, 82, 89, 100, 138, 148, 160, 179). Fourteen items
are not solvable by depth alone on this lane -- and those are exactly the 8 `query` plus 6
`unreachable` items. (The histogram counts ranks in the one deep pass while the curve retrieves at
each budget; on this hybrid lane the two disagree by a single item at k=50, 9 against 8.)

Two structural facts the item ledger makes visible, both consequences of how the slice was
drafted:

- **The multi-hop items of one document share a first hop.** Drafting walked relation paths from a
  bridge entity, so items within a document group carry the same first span, and that span's
  own-text rank repeats across the group (2 in one group, 3 in another, 20 in a third, 53 in a
  fourth). One badly-placed shared span therefore fails a whole group at once, which is why 5 of
  the 6 `unreachable` items come from a single document: four of them share a first hop at
  own-text rank 20 and the fifth sits at 53, past the operating budget even under the ideal query.
- **`recall@k` never saw any of this.** At k=10 the same slice reports recall 0.686 -- the flat
  metric is satisfied by one hop of two.

## Query decomposition conversion evidence

CUDA-host evidence from 2026-08-19 is under
`$DATA_DIR/graph-vector-fusion-multihop/20260819T-query-decomposition-conversion/`. It answers the
question against the ORIGINAL cohorts, so conversion is attributable to the same 8 query-diagnosed
and 19 budget-diagnosed items the [budget reading](#is-the-both-hops-ceiling-a-budget-or-a-query-problem)
above diagnosed.

The 95-item ledger it runs on is the shared draft bundle at
`$DATA_DIR/graph-vector-fusion-multihop/goods-draft/`: `goldset.jsonl` carries the 95 items and
`needle_items.jsonl` is its question-type sidecar (40 factoid, 35 multi-hop, 14 procedural,
4 numeric, 2 comparative). The matched store is the one the measurement-floor re-read built,
`.../20260724T-noise-floor/llb/rag/` -- recursive/hybrid, E5-base, 1,139 chunks before duplicate
collapse and 1,099 indexed after it. Neither needed recovery or recreation; the bundle is not
per-run, so it sits beside the runs rather than inside one.

**The raw identity gate passed before the paired result was read.** `raw-identity/report.md`
reproduces `20260816T-hop-probe/report.md` byte for byte, including the multi-hop diagnosis split
of 2 covered, 19 budget, 8 query, 6 unreachable and the curve 0.057 / 0.200 / 0.229. `raw/` is the
same probe re-read after `covered` was anchored to the served retrieval: the multi-hop slice is
unchanged, and only the whole-set and factoid ledgers move (45/34/9/7 and 31/8/1/0).

The paired run uses the existing `decompose` step, MamayLM-Gemma-3-12B-IT-v2.0 GGUF Q4_K_M on
Ollama at a 4,096-token context, k=10/25/50, and depth 200. It generates one bounded plan
(3 or 4 subqueries) for each of the 35 multi-hop items and reuses that plan for all four retrieval
calls; span-text controls stay raw. The lane is deterministic at temperature 0: re-running it
after the diagnosis fix reproduced every generated plan, subquery count, and rank, and moved only
the two diagnosis labels the fix was for.

**Result: generic decomposition converts NONE of the 8 original query-diagnosed items at the
operating budget, and costs the 19 original budget-diagnosed items nothing there.**

| original raw cohort | n | all-spans@10 before -> after | span coverage before -> after | coverage +/=/- | deep reachability +/- |
| --- | ---: | :-: | :-: | :-: | :-: |
| covered | 2 | 2 -> 2 | 1.000 -> 1.000 | 0 / 2 / 0 | +0 / -0 |
| budget | 19 | 0 -> 0 | 0.316 -> 0.342 | 1 / 18 / 0 | +0 / -0 |
| query | 8 | 0 -> 0 | 0.250 -> 0.312 | 1 / 7 / 0 | +2 / -0 |
| unreachable | 6 | 0 -> 0 | 0.500 -> 0.500 | 0 / 6 / 0 | +0 / -0 |

Two of the eight query items (`pdf-4c313c0e6619.md-mhop-10`, `pdf-6d8c2128b330.md-mhop-54`) become
`budget` -- decomposition does reach a hop the raw question never reached at depth 200 -- but both
land past k=10, so neither converts. Nothing regresses at the operating budget in any cohort: no
item loses all-spans@10 and no item loses covered-span share.

| multi-hop slice (n=35) | k=10 | k=25 | k=50 |
| --- | ---: | ---: | ---: |
| raw `all-spans@k` | 0.057 | 0.200 | 0.229 |
| `decompose` `all-spans@k` | 0.057 | 0.171 | 0.286 |
| raw span coverage | 0.371 | 0.514 | 0.543 |
| `decompose` span coverage | 0.400 | 0.486 | 0.586 |

The curve says where the effect actually lands: the gain is deeper than the budget an operator
serves. At k=50 decomposition adds two items and 0.043 of span coverage; at k=25 it LOSES one item
(0.200 -> 0.171); at k=10 it is a tie on all-spans and worth 0.029 of coverage share. That is a
negative result for the diagnosed population and it does not license enabling the step by default.
No prompt, retrieval strategy, or shipped `query_prep` default changed. `run_config.yaml` is the
probed lane, `raw-identity/` the byte-identical gate, `raw/` the re-anchored raw reading,
`decompose/` the paired JSON, Markdown, and endpoint fingerprint, and
`decompose-deep-rank-diagnosis/` the same paired run read under the pre-fix rule, retained because
it is the evidence for the diagnosis anchor described above.

### What the k=50 coverage is actually made of

The k=50 jump is real but it is not all the same substance. `covering_record_size.py`, archived
beside the probe run, measures the SIZE of the record that delivers each covered hop on the
multi-hop slice at k=50:

| lane | covered hops (of 70) | covering record, median chars | p10 | max |
| --- | ---: | ---: | ---: | ---: |
| vector (`faiss`) | 38 | 774 | 731 | 794 |
| `graph/global_community` | 56 | 86 | 36 | 205 |
| `graph/local_khop` | 43 | 80 | 32 | 205 |

The graph lane's coverage is delivered by entity mentions, an order of magnitude shorter than a
chunk. On the source-span metric a mention that overlaps the gold span is a hit -- correctly, the
metric is defined on offsets -- but 86 characters is a citation, not the context an answer is
written from. The high-weight fused rows that reach 0.657 at k=50 inherit that shape. Read the
k=50 row set as "the evidence IS reachable at this depth", not as a retrieval configuration ready
to serve.

## What this licenses, and what it does not

- **The diagnosis is recorded: budget first, query second.** An operator whose multi-hop coverage
  is stuck should raise the retrieval budget (or run a second retrieval pass) before touching a
  ranking knob, because on this corpus no ranking knob in the lane is worth more than three
  questions and the budget is worth eight on the vector lane alone.
- **No default moves here.** `top_k` stays 10 and no ranking policy changes. The evidence is
  measured on the DRAFTED multi-hop ledger (see the boundary in
  [span-identity evidence](span-and-depth-evidence.md#span-identity-evidence)), and the standing
  rule is that a drafted slice does not move a shipped default. The k=50 sweep's own verdict row
  reads `adopt` for `fused/global_community@0.70/d50/ioverlap` (+0.429 [+0.257, +0.600] multi-hop
  `all-spans`, adjusted p 0.0006) but carries BORDERLINE and INSUFFICIENT-EVIDENCE qualifiers on
  its recall reading (4 differing items of 35, 53 needed), so it is not a settled separation
  either.
- **A larger k is not free, and this lane did not price it.** Retrieval-side it looks free -- the
  vector row's overall recall@k rises 0.705 -> 0.853 -> 0.874 and its overall `all-spans@k` 0.474
  -> 0.621 -> 0.642 with no slice regressing -- but five times the chunks is five times the
  context, and what that costs an answer is an answer-side question this lane does not touch.
- **Decomposition now has a measured size.** Eight items are `query`-diagnosed: their missing hop
  is not in the pool at any depth under the question, and its own text finds it at rank 10 or
  better. That is the population a decomposition step would have to convert, and it is the
  measurement any such step should be read against.
