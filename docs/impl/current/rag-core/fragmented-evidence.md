# Two Levers Against Fragmented Evidence (`size` And Stitching)

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

When [evidence intactness](retrieval-metrics.md#evidence-intactness-span_char_coveragek--span_intactk)
sits far under character coverage, the evidence is being FOUND and delivered in pieces rather
than missed. This page is the measured reading of the two levers against that on the converted
Ukrainian goods PDF corpus: the index-side one (`size`) and the assembly-side one
([contiguous-chunk stitching](context-assembly.md#contiguous-chunk-stitching-fragmented-evidence-delivery-lever)).

CUDA host (RTX 4060 Ti 16 GB), measured 2026-08-23, one run:
`make compare-retrieval CONFIG=<goods.yaml> CHUNK_SIZES=200,400,800 STITCH=1`, pinned
`intfloat/multilingual-e5-base`, the converted Ukrainian goods PDF corpus and its 95-item drafted
ledger, k=10, 2000 paired resamples, 95% confidence, seed 13, baseline `recursive#size200` (the
size in production). No generator is involved -- every reading here is retrieval-side. The three
flat FAISS stores index 3,515 / 1,972 / 1,023 chunks at `size` 200 / 400 / 800 after exact duplicate
collapse (4,848 / 2,263 / 1,063 before it), all at `overlap` 30. The ledger is DRAFTED: no reviewer
accepted these items, so every reading here is diagnostic. Report, config, stores, and per-item
vectors under `$DATA_DIR/table-aware-chunking/<run>/`.

The question is the one [the intactness
re-read](chunking.md#the-intactness-re-read-of-the-same-three-chunkers) left open: on the
`procedural` slice 0.706 of the gold-span characters arrive but only 0.357 of the spans arrive
whole, and no chunker moves either number. Two levers can: raise the `size` cap (a different index
and a different retrieval), or stitch contiguous retrieved chunks into one block at assembly time ([contiguous-chunk
stitching](context-assembly.md#contiguous-chunk-stitching-fragmented-evidence-delivery-lever)),
which retrieves nothing new. The incumbent row reproduces its recorded numbers EXACTLY -- 0.695
recall@10, 0.465 MRR, 0.575 cover@10, 0.516 intact@10, and 0.786 / 0.706 / 0.357 on the procedural
slice -- so this is measured against the recorded state, not a re-tuned one.

| lane | recall@10 | MRR | cover@10 | intact@10 | chars@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recursive#size200` (baseline) | 0.695 | **0.465** | 0.575 | 0.516 | **1619** |
| `recursive#size200+stitch` | 0.695 | 0.473 | 0.575 | 0.526 | 1616 |
| `recursive#size400` | **0.705** | 0.437 | **0.593** | **0.584** | 3328 |
| `recursive#size400+stitch` | **0.705** | 0.442 | **0.593** | **0.584** | 3327 |
| `recursive#size800` | 0.642 | 0.368 | 0.553 | 0.553 | 6659 |
| `recursive#size800+stitch` | 0.642 | 0.369 | 0.553 | 0.553 | 6654 |

Paired `intact@10` deltas against `recursive#size200`, aggregate (n=95) and on the `procedural`
slice the levers were aimed at (n=14):

| lane | aggregate intact delta | w/l/t | procedural intact delta | w/l/t | reading |
| --- | --- | :-: | --- | :-: | --- |
| `size200+stitch` | +0.011 [+0.000, +0.032] | 1/0/94 | +0.071 [+0.000, +0.214] | 1/0/13 | flat |
| `size400` | +0.068 [-0.021, +0.153] | 17/8/70 | +0.357 [+0.000, +0.643] | 6/1/7 | flat |
| `size800` | +0.037 [-0.068, +0.137] | 16/13/66 | +0.357 [+0.000, +0.714] | 7/2/5 | flat |

Verdict: **NEITHER lever raises `span_intact@k` on the procedural slice by an interval clear of
zero, and no default moves.** `size` stays 200, stitching stays a comparison lane, and the
fragmentation the intactness pair diagnosed remains a diagnosis with two measured, priced, and
rejected candidate cures.

What each lever actually did, because the two null readings are null for OPPOSITE reasons:

- **`size` has the effect and misses the reading by ONE item.** On the procedural slice `size400`
  DOUBLES whole-span delivery, 0.357 -> 0.714, on 7 discordant items split 6 wins to 1 loss -- the
  largest movement anything in this repo has produced on that slice, and enough discordant items to
  license a claim at 95% ([the minimum-evidence
  gate](paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading) needs 6). It is still
  `flat`, and the margin is exactly the one loss: the exact randomization p is 0.0625 against the
  0.025 a separation needs, where a clean 7/0 would have given 0.0156 and separated. This is a slice
  that CAN be decided and was not. `size800` moves the same point estimate on a worse ledger (7/2 of
  9 discordant, p=0.0899). And neither is free: `size400` serves 2.06x the context (1619 -> 3328
  characters per query; `size800` 4.11x), `size800` costs first-hit rank (MRR -0.097 [-0.195,
  -0.001], the one `regressed` reading in the run), and a wider cap redistributes across question
  types rather than lifting them -- `numeric` loses a quarter of its intactness at both wider caps
  (0.750 -> 0.500, one item of four) while `multi-hop` gains recall (0.657 -> 0.743 at `size400`).
- **Stitching is nearly free and has almost nothing to reflow.** It costs no retrieval, no index,
  and NEGATIVE context (-3.2 characters per query at `size200`, because a merge counts a chunk
  overlap once instead of twice), and its invariance held on every lane: recall@10 and cover@10
  reproduce their base lane to six decimals, exactly as the construction requires. But the census
  says why it cannot deliver: 9.87 blocks per query out of 10 retrieved chunks, i.e. **0.13 merges
  per question**. A top-10 on this corpus rarely holds two chunks that were adjacent in the source
  -- dense retrieval spreads its hits across documents and sections, and duplicate collapse has
  already removed 1333 of the 4848 `size200` chunks -- so the lever fires on 1 item of 95 (1 of 14
  procedural) and converts that one. Its slice reading rests on a single discordant item, five short
  of what any claim at this confidence needs, so "flat" here means "not looked at", not "looked and
  found nothing". There is no tuning of stitching that changes this: the input it needs is not in
  the context.

The operator reading, in one line: on this corpus fragmented procedural evidence is a `size`
phenomenon and not an assembly one, the `size` cure costs roughly double the served context, and the
slice was big enough to decide the trade and came out one item short of deciding it.

What would overturn it: a `procedural` slice with a few more items that the two lanes differ on --
`size400` is one flipped item away from separating, so this is the cheapest open reading in the
chunking area. For stitching, a retrieval configuration whose top-k actually HOLDS neighbours: a
much larger `top_k`, a parent-child or oracle-doc-scoped lane, or a corpus whose gold spans sit
inside long single-document runs. A re-read should keep the `chars@k` column beside the intactness
one either way: the whole distinction between these two levers is visible only there.
