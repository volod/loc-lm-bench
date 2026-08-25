# Answer-Side Coverage Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

The lane itself -- what it scores, how it pairs, and what its four verdicts mean -- is
[answer-quality evidence](answer-quality-evidence.md#answer-quality-evidence). The metric itself --
what a gold span requires an answer to carry, and why -- is [answer-side gold-span
coverage](../rag-core/scoring.md#answer-side-gold-span-coverage-answer-side-span-coverage-metric).
This page owns what the two together MEASURED: whether the multi-hop answers of the recorded
comparison state the facts their gold spans carry, which the hop-blind objective could never say.

## Measured result: the answers do not state the evidence the fused lane adds

Scored 2026-08-23 on the RTX 4060 Ti 16 GB CUDA host, one comparison per model. Every earlier
reading on the answer-quality page rests on `objective_score`, which is reference-answer token F1
over the whole answer: a two-hop answer stating one fact in the reference's own words earns what a
terse answer stating both earns, so "the coverage gain does not reach the answer" was an inference
from a metric that cannot see a hop. This run re-scores the [model-invariant
comparison](answer-quality-evidence.md#measured-result-the-verdict-is-model-invariant-the-cost-slice-is-not)
unchanged -- the same three lanes (`vector`, `fused/global_community@0.10/d10`, and the overlap row
`fused/global_community@0.30/d50/ioverlap`), the same 95-item drafted goods ledger, the same
duplicate-collapsed 1,099-chunk store, k=10, all three splits pooled, 2,000 bootstrap resamples,
seed 13, a `num_ctx` of 8,192, and both Ukrainian-specialized families
(`MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`, `lapa-v0.1.2-instruct-GGUF:Q4_K_M`) -- with the
answer-side columns now measured beside the objective.

**Nothing else moved, which is what makes the two columns readable as an addition.** Every
retrieval figure of the recorded run reproduces exactly on both models (multi-hop recall 0.686 /
0.771 / 0.800, all-spans 0.057 / 0.086 / 0.086, span coverage 0.371 / 0.429 / 0.443), as does every
objective delta the earlier result was stated on: the exact row costs MamayLM -0.002 [-0.059,
+0.072] on multi-hop and -0.045 [-0.099, -0.001] on factoid, the overlap row costs Lapa -0.081
[-0.157, -0.017] on multi-hop, and so on for the rest. Both models answered all 285 of their cases
`ok`, and the largest prompt sent was 3,357 tokens (MamayLM) and 2,509 (Lapa) against the 8,192
served, so nothing was truncated.

Multi-hop slice (n=35), lane minus vector, 95% paired bootstrap CI, with the answer-side pair
beneath the two columns the verdict is cut from:

| lane | metric | MamayLM 12B | Lapa 12B |
| --- | --- | ---: | ---: |
| exact `@0.10/d10` | objective | -0.002 [-0.059, +0.072] 7/10/18 | -0.024 [-0.074, +0.016] 10/7/18 |
| exact `@0.10/d10` | span coverage | +0.057 [+0.014, +0.114] 4/0/31 | +0.057 [+0.014, +0.114] 4/0/31 |
| exact `@0.10/d10` | answer span coverage | +0.014 [-0.057, +0.100] 3/3/29 | -0.014 [-0.043, +0.000] 0/1/34 |
| exact `@0.10/d10` | answer all-spans | +0.029 [-0.057, +0.114] 2/1/32 | -0.029 [-0.086, +0.000] 0/1/34 |
| overlap `@0.30/d50` | objective | -0.010 [-0.087, +0.070] 11/9/15 | **-0.081 [-0.157, -0.017] 7/14/14** |
| overlap `@0.30/d50` | span coverage | +0.071 [+0.014, +0.129] 5/0/30 | +0.071 [+0.014, +0.129] 5/0/30 |
| overlap `@0.30/d50` | answer span coverage | +0.000 [-0.129, +0.129] 5/6/24 | -0.043 [-0.100, +0.000] 0/3/32 |
| overlap `@0.30/d50` | answer all-spans | +0.029 [-0.114, +0.171] 4/3/28 | -0.086 [-0.200, +0.000] 0/3/32 |

Verdict: **`no_gain` on both lanes for both models, unchanged.** The four outcomes are still cut
from the objective and from retrieval coverage, so the new columns decided nothing; they enter each
verdict as its `ANSWER SIDE:` clause -- `answer-side span coverage +0.014 [-0.057, +0.100], which
does not separate` on MamayLM's exact row, `-0.014 [-0.043, +0.000]` on Lapa's. What the run
settles is what those clauses now say instead of leaving unsaid:

- **The retrieval-only reading is now measured on the answer side, not inferred.** No candidate
  lane's answer-side coverage separates for either model, and not one of the four is positive with
  an interval clear of zero. The earlier conclusion ("paying for a graph build buys multi-hop
  RETRIEVAL on this corpus, not multi-hop ANSWERS") no longer rests on a metric that a half answer
  can earn: the answers themselves state no more of the gold spans than the vector lane's do.
- **Lapa's overlap row carries more evidence and states LESS of it.** Its span coverage is +0.071
  (5 wins, 0 losses) while its answer-side coverage is -0.043 on 0 wins against 3 losses, and its
  all-spans gate -0.086 on the same 3 items. That is the same direction as its objective loss
  (-0.081, past the minimum-evidence gate) and it is the sharpest instance of the pattern on this
  page: the extra graph vote re-ranks the chunk the model was answering from, and the model answers
  with fewer of the gold facts than before. Neither answer-side reading clears the gate on its own
  (3 differing items), so it corroborates the objective loss rather than adding a second claim.
- **The cross-model level confound the objective carries does not apply here, and the two metrics
  agree.** `objective_score` levels are not comparable across models -- a terser model scores
  better on token F1 for the same facts, and these two models' overall objectives (0.516 and 0.510)
  are a tie only in that sense. The answer-side reading is a RECALL-side one, so brevity can only
  lower it, and Lapa is the terser model here (median 22 completion tokens against MamayLM's 30).
  It nonetheless carries far more of the multi-hop evidence: vector-lane answer span coverage 0.771
  against 0.429, all-spans 0.571 against 0.257, on byte-identical context. Two metrics with
  OPPOSITE verbosity biases therefore agree that Lapa is the better multi-hop answerer on this
  corpus -- which is the claim the earlier page could only make with a caveat about answer length.

Boundaries, beyond the drafted ledger that bounds every result on this page:

- **The reading is recall-side and is not bounded by retrieval coverage.** On the focus slice, the
  answer states more of the gold spans than the top-k carried in 29 of 105 lane readings for
  MamayLM and 69 of 105 for Lapa -- a model can state a hop the context missed, from what it
  already knows or from the half it was given. The pair (retrieval coverage, answer coverage) is
  two measurements, not one derived from the other, and a verbose model that dumps its context
  scores well on the answer side by construction; `token_precision` in the same row is what prices
  that.
- **This ledger has degenerate spans, and they do not drive the numbers.** Of its 130 labeled
  spans, 57 fall back to the undiscounted requirement and 35 end up requiring a single content term
  (`3.На підрозділі рахується майно.` requires only `майно`), where "states the fact" and "mentions
  the subject" cannot be told apart; 3 are unjudgeable and are excluded from their item's
  denominator. Refusing every single-term span instead moves the multi-hop vector level from 0.414
  to 0.371 on MamayLM and 0.757 to 0.771 on Lapa and leaves the lane ordering unchanged, so the
  readings above are not an artifact of those spans. (That check recomputes from the persisted
  280-character answer preview rather than the full answer, which is why its levels sit within a
  point or two of the artifact's.) A widened ACCEPTED ledger would settle it properly, as it would
  every other reading on this page.
- **What would overturn it.** Any of: an accepted ledger with more than 35 multi-hop items (the
  coverage half of every verdict here is already withdrawn by the minimum-evidence gate at 4-5
  differing items), a lane whose answer-side coverage rises with an interval clear of zero while
  its objective stays flat -- which is the case the metric was built to make visible and which this
  corpus did not produce -- or a generator whose answers are long enough for the recall-side bias
  to matter, which neither of these two is.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  MODEL=<roster-model> SPLIT=final,tuning,calibration \
  ANSWER_QUALITY_LANES=vector,<best-exact-row>,<best-overlap-row> INCLUDE_DRAFTED=1 \
  ANSWER_QUALITY_OUT_DIR=<run>/<model-slug>/answer-quality
```
