# Sidecar-Free Routing Calibration

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md). The routed lane this calibrates, and the
answer-quality comparisons it is read beside, are on the
[answer-quality evidence page](answer-quality-evidence.md#answer-quality-evidence).

## Sidecar-free heuristic calibration

Measured 2026-07-22 on the RTX 4060 Ti 16 GB CUDA host. The run used the
same drafted goods ledger and matched stores, `global_community@0.30/d50/ioverlap`, k=10,
multilingual E5 on the RTX 4060 Ti, 2,000 bootstrap resamples, and seed 13. Question-type labels
were hidden from every routing decision; the evaluation truth was only whether an item carried
more than one gold span.

`make calibrate-fusion-routing` swept word thresholds 10/12/14/16/18/20 crossed with linked-entity
thresholds 0/1/2 on the 31-item tuning split. It froze `w12/e0` before constructing the final-split
retrieval caches, then evaluated that one policy on the untouched 31-item final split.

| split | tp/fp/tn/fn | precision | recall | multi-span coverage delta | single-span recall delta |
| --- | :-: | ---: | ---: | ---: | ---: |
| tuning | 9/7/14/1 | 0.562 [0.333, 0.800] | 0.900 [0.667, 1.000] | +0.050 [0.000, +0.150] | +0.048 [0.000, +0.143] |
| final | 8/6/14/3 | 0.571 [0.308, 0.833] | 0.727 [0.444, 1.000] | +0.091 [0.000, +0.227] | +0.000 [-0.150, +0.150] |

Route precision and recall now carry the lower-bound qualifier produced inside `bootstrap_ratio`:
the same draw supplies `p_positive`, readings at 90% / 95% / 97.5%, and `borderline`, without a
paired sign-test gate because these are count ratios rather than paired deltas. `report.md` renders
all tuning rows plus frozen final in **Route-quality threshold stability**. Re-rendering the
recorded gold-item order reproduced all 38 ratio point estimates and bounds exactly; 2 rows are
borderline, both on tuning `w16/e0` (precision and recall, `p_positive` 0.954): they clear a 90%
lower-bound cut but not 95%. Those descriptive rows did not select the frozen policy and do not
change the verdict.

Verdict: **no recommendation**. The frozen policy's tuning coverage interval touches zero, so it
does not pass the predeclared positive-gain gate. Final points in the same positive direction but
does not repair a failed tuning gate, and its single-span interval includes regression. The
production fallback therefore stays at 16 words plus 2 linked entities.

The standard sweep path independently reproduced the frozen policy with
`FUSION_HIDE_ROUTING_SIDECAR=1` on each split. The routed row gained multi-hop recall +0.100
[0.000, +0.300] on tuning and +0.182 [0.000, +0.455] on final, while all-spans@10 was unchanged;
both sweep verdicts were `inconclusive`. Their reports are the calibration artifact's
`tuning-compare/` and `final-compare/` children. The remaining limitation is statistical power,
not an invitation to select on final; a larger accepted-ledger repeat is forward work in
[`plan.md`](../../plan.md) (`fusion-routing-calibration-power`).
