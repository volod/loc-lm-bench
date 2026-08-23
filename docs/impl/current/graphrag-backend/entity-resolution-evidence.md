# Entity Resolution Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

Does entity fragmentation cost the graph lane recall, and would merging the fragments buy it back?
The mechanics -- the record table, the blocking keys, the overlay, and what applying one does --
are in [entity resolution](../entity-resolution.md#the-graph-node-lane). This page is the reading.

## Is fragmentation costing the graph lane recall? (2026-08-21)

**No, and merging harder costs it.** CUDA host, evidence under
`$DATA_DIR/graph-entity-resolution-host/graph-entity-resolution/`. Two runs, both over the
five-document Ukrainian PDF corpus and the 61-item multi-hop gold set of
the widened multi-hop draft, whose graph builds to 423
nodes and 242 edges. Each run fits the node linkage once (31,312 scored pairs across the three
blocking rules, seed 7, one DuckDB thread), re-clusters that one fit at every candidate cut, and
reruns both graph strategies over the identical items at k=10 with the same paired bootstrap seed.
The built FAISS lane rides along as the reference row. Each run takes about 25 seconds including
the E5 embedding of all 423 node texts.

Pre-overlay: `graph/local_khop` recall@10 0.475 / MRR 0.149, `graph/global_community` 0.607 /
0.368, `faiss` 0.639 / 0.364.

| Cut | Clusters | Nodes merged | Largest | local_khop d recall / d MRR | global_community d recall / d MRR |
| --- | --- | --- | --- | --- | --- |
| 0.99 | 1 | 1 | 2 | +0.000 / +0.000 | +0.000 / +0.000 |
| 0.9 | 2 | 3 | 3 | +0.000 / -0.000 | +0.000 / +0.000 |
| 0.75 | 2 | 3 | 3 | +0.000 / -0.000 | +0.000 / +0.000 |
| 0.6 | 5 | 7 | 3 | +0.000 / -0.001 | +0.000 / -0.011 |
| 0.5 | 22 | 27 | 4 | -0.016 / -0.004 | -0.016 / -0.003 |
| 0.4 | 24 | 29 | 4 | -0.016 / -0.005 | -0.016 / -0.003 |
| 0.3 | 40 | 61 | 6 | -0.016 / -0.011 | -0.098 / -0.037 |
| 0.2 | 52 | 82 | 8 | -0.082 / -0.025 | -0.131 / -0.050 |
| 0.1 | 52 | 113 | 11 | -0.230 / -0.051 | -0.246 / -0.164 |

The verdict on both runs and both strategies is `negative-result`: no cut separated an overlay lane
from the lane it would replace, so no overlay is adopted and none was applied to a store. What
makes the reading strong rather than merely null is its SHAPE -- the curve is monotone. Every cut
loose enough to merge a meaningful number of nodes loses recall, and it loses more the looser it
gets, to -0.246 at the model's near-coin-flip.

Four things this says about the corpus:

- **The model finds real fragmentation, and there is very little of it.** At the seam's own default
  cut of 0.9 the whole 423-node graph yields three merges:
  `Накладна З-3` / `Накладна (вимога)` / `Накладна`, and `Забезпеченість особового складу` /
  `Донесення Забезпеченість особового складу`. Both are correct. They are also 3 nodes out of 423.
- **The cut where merges become plentiful is the cut where they become wrong.** The first false
  merge appears at 0.6 -- `Додаток 27` with `Додаток 57`, two different numbered annexes whose
  names differ in one digit. Below that the false rate climbs fast, which is exactly the negative
  slope in the table.
- **The graph's own seed linker already absorbs the inflectional case.** It keys on a node's name
  PLUS its aliases and on a leading Ukrainian stem, and `_GraphBuilder` merges an alias onto a node
  whose name matches -- so the second node's name is usually already the first node's alias. The
  fragments that survive that are mostly not reachable by any lexical merge either.
- **The corpus is a system manual, not a corpus of named entities.** 234 of 423 nodes are `MISC`,
  and much of what looks like fragmentation (`вкладки` / `вкладка`, `розділи` / `розділом`) is
  common-noun inflection, where merging two nodes joins two unrelated passages.

Reproduce either run with:

```bash
DATA_DIR=<a data root> make build-graph BUNDLE=<draft-bundle>
DATA_DIR=<a data root> make build-index CORPUS=<draft-bundle>/corpus
DATA_DIR=<a data root> make resolve-graph-entities GOLDSET=<draft-bundle>/goldset.jsonl \
  RESOLVE_WITH_VECTOR=1 CORPUS=<draft-bundle>/corpus
DATA_DIR=<a data root> make resolve-graph-entities GOLDSET=<draft-bundle>/goldset.jsonl \
  RESOLVE_THRESHOLDS=0.5,0.4,0.3,0.2,0.1 RESOLVE_WITH_VECTOR=1 CORPUS=<draft-bundle>/corpus
```

## What the reading does NOT settle

It prices ONE corpus. A corpus of named entities -- people, organisations, places, in running prose
rather than a numbered manual -- is where fragmentation should cost the most, and this corpus is
the opposite of that. The lane exists to take the reading again there; the result to compare
against is the shape above, not the specific deltas.

It also prices the merge as the resolution seam can propose it. A fragment that agrees with its
entity on nothing but the type and the document -- an epithet, a pronoun-like reference -- is
coreference, which the [boundary](../entity-resolution.md#boundary) puts outside this pass. If a
later reading shows that class of fragment is what costs the graph lane, the answer is a different
capability, not a looser threshold on this one.
