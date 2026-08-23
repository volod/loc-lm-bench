# Re-Rendering A Recorded Answer-Quality Comparison

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

## Why a finished comparison can be re-read

An answer-quality comparison costs hours of generation -- one `run-eval` bundle per (lane, split),
and a budget sweep multiplies that by every budget
([budget evidence](answer-quality-budget-evidence.md#the-retrieval-budget-dimension)). Once
rendered, the artifact was locked to the report format of the day it ran: a new column, a new
section, or a corrected reading could not reach a recorded run without paying for every generation
again.

It never had to be. The comparison is PURE over the per-case rows, every lane's run bundles are
recorded in its own `comparison.json`, and the bundles are ordinary `run-eval` output that stays on
disk. `--from-bundles <comparison.json>` resolves each lane's recorded bundles, recomputes the
comparison over them, and renders the artifact under the CURRENT report -- with no model call, no
store, and no GPU.

```bash
make compare-answer-quality ANSWER_QUALITY_BUNDLES=<recorded-run>/answer-quality/comparison.json \
  ANSWER_QUALITY_OUT_DIR=<new-run>/answer-quality
llb compare-answer-quality --from-bundles <recorded-run>/answer-quality/comparison.json
```

The recorded artifact is never written to: the re-render lands in a NEW directory (default
`$DATA_DIR/graph-vector-fusion-multihop/<timestamp>/answer-quality/`), because overwriting the
artifact the generations produced would destroy the thing the re-render is checked against. Its
metadata is the recorded metadata plus two appended keys, `rerendered_from` and `rerendered_at`, so
stripping those two is exactly how a re-render is compared against its source. Every other option
is ignored in this mode -- the lanes, splits, model, gold set, and bootstrap settings are the
recorded ones -- and naming `--from-comparison` (which STARTS a comparison from a retrieval sweep)
alongside it is a usage error rather than a silent precedence rule.

A lane is read back with `recorded_lane_rows` (`src/llb/eval/paired_cases.py`), the same seam
`make audit-paired-readings` reconstitutes a recorded lane through
([paired verdicts](../rag-core/paired-verdicts.md#randomization-calibrated-paired-readings)), so
the two cannot drift into two notions of what a recorded lane is. What the answer-quality path adds
on top is its own multi-span coverage columns, recomputed from each bundle's `retrieval.jsonl` at
that LANE's own retrieval budget -- a budget sweep's cells differ in exactly that, so reading every
cell at the base config's `top_k` would erase the thing it measured.

## What a re-render refuses

A bundle set that drifted would re-render into a DIFFERENT comparison wearing the recorded one's
provenance, so two refusals fire before anything is written. Both list every disagreement rather
than stopping at the first, and both say the same thing: re-run the comparison instead of
re-rendering it.

**The bundles must still describe their lanes** (`bundle_match.py`). Each bundle's own
`manifest.json` records the full `RunConfig` it ran under, and that config is checked against the
lane LABEL (which parses back into retrieval knobs, including a `#k<budget>` cell and a `+headers`
twin) and against the comparison metadata: the run name that ties a bundle to its lane, the
retrieval backend / strategy / weight / fusion knobs, the retrieval budget, the model, the backend,
the gold set, the grounding, and one bundle per recorded split in the recorded order.

Two rules keep that check honest about the ARCHIVE it reads rather than refusing everything old:

- **Only the knobs the lane rides on.** A vector or graph lane carries the fusion knobs as dead
  config; holding it to them measures nothing.
- **A field the manifest never recorded is not a mismatch by itself.** That run predates the knob,
  so it is consistent with the knob's DEFAULT and with nothing else -- which still refuses a label
  asking for a non-default value, such as an `/ioverlap` lane pointing at a bundle older than span
  identity. Grounding is the asymmetric case: `run-eval` stamps `item_grounding` on a DRAFTED
  bundle and leaves a verified one unstamped, so a verified comparison expects the field to be
  ABSENT, which is what refuses a drafted bundle standing in for a verified lane.

**The rebuilt comparison must still cover what was recorded** (`rerender.py`). After the recompute
and before the write, the item set, the lanes, and every question-type slice must match the
recorded artifact, and no metric column the artifact recorded may have gone missing. That is what
catches a deleted retrieval sidecar (the coverage columns would silently vanish) and a gold set
whose question-type sidecar no longer labels the items (the slices would collapse).

Gaining a column the recorded run never had is the opposite of drift -- it is the report
improvement reaching the old run, which is the entire point -- so it passes.

## Modules and coverage

| Module | Role |
| --- | --- |
| `src/llb/eval/paired_cases.py` | `recorded_lane_rows`: the shared seam a recorded lane is read back through |
| `src/llb/eval/answer_quality/bundle_match.py` | Does a bundle's manifest still describe the lane its label claims? |
| `src/llb/eval/answer_quality/bundles.py` | Recorded lanes plus coverage columns at each lane's own budget |
| `src/llb/eval/answer_quality/rerender.py` | The recompute, the shape refusal, and the re-rendered artifact |

`tests/llb/eval/answer_quality/test_answer_quality_rerender.py` covers the round trip over fixture
bundles (a full `run-eval` shape: scores, retrieval sidecar, manifest), the byte-identity gate, a
gained column, and seven refusals -- drifted retrieval config, a lane repointed at another lane's
bundle, a different model, a drafted bundle under a verified lane, a missing bundle, a lost
retrieval sidecar, and a gold set that no longer slices the items.

## Measured result: the current report reaches four recorded runs unchanged in substance

Run 2026-08-23 on the RTX 4060 Ti 16 GB CUDA host over every answer-quality comparison the host
held -- nine recorded artifacts, each 95 drafted goods items pooled across the three splits, 2,000
bootstrap resamples, seed 13. No model was loaded and no GPU was used; the whole re-read is file
work over the recorded `run-eval` bundles.

Five of the nine re-rendered **byte-identically** to the artifact their generations produced, once
the two appended `rerendered_*` metadata keys are stripped: the two-model fused and routed
comparisons from 2026-08-22 and the table-header comparison from the same day. Those are the runs
made under the current report, and reproducing them to the byte is what says the re-render is a
re-reading rather than a recomputation with different inputs.

The other four gained exactly what the report has gained since they ran:

- **The three comparisons from 2026-07-22** (fused, the overlap span identity, and question
  routing) gained the `context_chars` column, which did not exist when they ran, and their headline
  verdict moved from `retrieval_only` to `no_gain`. That is not a new measurement -- it is the
  minimum-evidence gate, which the evidence page already records by hand for these three runs
  ([the withdrawn coverage
  reading](answer-quality-evidence.md#measured-result-the-multi-hop-coverage-gain-does-not-reach-the-answer)):
  the coverage half of each `retrieval_only` rests on 4-5 differing items of 35, fewer than the 6
  an exact sign test needs at 95%. The re-render derives that downgrade from the recorded rows
  instead of a reader deriving it, and prints the same 53-item and 42-item floors. The new context
  column also reads against the run: the fused lane served 6,277 characters per item against the
  vector lane's 6,884, so on this corpus it carried more of the multi-hop evidence in LESS context,
  not more.
- **The budget sweep from 2026-08-16** gained the `not_ok` count, and with it a fact its own
  artifact could not show: `vector` at k=50 left 1 of 95 cases unanswered. A case the model never
  answered scores zero exactly like a wrong answer, so that lane's numbers carry one missing answer
  rather than one more bad one. Its verdict is unmoved -- `retrieval_only` on the multi-hop slice,
  budget conversion `stalled` -- so the reading of that run stands, now with the caveat attached.

What would overturn this: any change to what the comparison COMPUTES (a metric, the bootstrap, a
verdict rule) makes a re-render disagree with its source by design, and the byte-identity check is
then a check of that change rather than of the re-render. The check is also only as good as the
archive: it says nothing about a bundle whose `scores.jsonl` was edited in place, since the manifest
records the config a run was launched with, not a hash of the rows it produced.
