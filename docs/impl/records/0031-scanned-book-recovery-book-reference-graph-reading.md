# Book reference graph reading

## Work contract

```text
#### book-reference-graph-reading

A technical handbook's hardest questions join a sentence to the table, equation, or figure it cites.
The vector lane retrieves the sentence without the table: a numeric grid embeds poorly and is its
own projected document. The book already prints the edge between them, and the projection's
reference sidecar records it. Build a graph from those printed references and prepare the paired reading
against the vector lane over the same projection
([method survey](../design/graph-construction-methods.md#the-printed-reference-graph)).

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- deterministic graph, provenance, and paired comparison fixtures.
- Dependencies: the printed references each projected document records
  ([corpus projection](current/book-recovery/corpus-projection.md#printed-references));
  the graph store, its strategies, and the fusion seam
  ([modules and CLI](current/graphrag-backend/modules-and-cli.md));
  the minimum-evidence gate on a paired reading
  ([paired verdicts](current/rag-core/paired-verdicts.md)). The derived-source contract is
  [graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology).
- User-visible outcome: `llb build-graph` over a projected book builds a graph where
  every node mention and edge names a span the book prints. A replayable comparison measures
  vector, graph, and fused retrieval on the same verified items.
- Scope boundary: in scope -- section, figure, table, equation, page, and index-term nodes and the
  `contains`, `cites`, and `indexes` relations as a closed set beside the entity vocabulary, never
  normalized into it; index terms linked to the question by the existing lexical linker; no edge
  built from a citation the sidecar records as unresolved; `local_khop` over the graph, since a
  cited block is one `cites` hop away, alone and fused with the vector lane through the existing
  fusion seam; `ppr` rows beside it when `graph-seeded-propagation-ranking` has left the plan by
  the time of the run, with their absence stated otherwise; the reading on the verified book items
  under the paired book retrieval settings
  ([comparison](current/book-recovery/corpus-projection.md#paired-retrieval-reading)), split by
  whether the spans include a table, equation, or figure block. Out of scope -- any edge the book
  does not print, blending the structural graph and an extracted graph into one store, retuning
  retrieval for the book, and answer scoring.
- Data and artifact paths: a book-reference builder beside `src/llb/graph/build.py`; the graph
  store, the index, and the comparison under `$DATA_DIR`.
- Execution path: the synthetic book inside `make ci`, where a sentence citing a table
  reaches the table block in one `cites` hop. Exercise the same inputs through
  `make build-graph`, `make build-index`, and `make compare-graph-fusion` with `NOISE_FLOOR=1`.
- Acceptance gates: `make ci` green; every mention and edge resolves to its printed evidence
  offsets; two builds are byte-identical; an unresolved citation supplies no edge. Pin the paired
  report, minimum detectable effect, intervals, and technical-evidence slices, including a null
  result. Run the real-book comparison only with the release gold set.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md` beside the book's
  vector reading, and the structural vocabulary in
  [graph ontology schema](../design/graph-ontology-schema.md) outside the signed entity vocabulary.
```

## Implementation and decisions

`make build-graph BOOK_CORPUS=<pinned-projection-dir>` publishes a separate graph generation
using the existing graph-store format. The builder reads the PDF and reference sidecars for
one source PDF, checks targets against exact block offsets, and preserves the six structural
node types and three relation types. A separately recognized label and technical body can
belong to the same node, so the graph reaches the equation or table evidence in one hop.
Index page offsets identify the exact token or range, including continuation lines.

`make compare-graph-fusion BOOK_CORPUS=<pinned-projection-dir>` uses the prebuilt FAISS index
and the structural graph for the same verified gold. It keeps the index and fusion settings
fixed in the shared run configuration. It reports page and span metrics, complete-span
recall, paired deltas and sensitivity, and both evidence slices. The report binds input
digests and stores item-level ranks and offsets for replay.

## Acceptance evidence

Accepted.

On 2026-11-28, CPU only on the Intel Core Ultra 7 265HX host:

- `make build-graph` ran twice over the 11-page synthetic book and wrote byte-identical
  graph files with 20 nodes and 22 edges. The fixture confirmed exact offsets for every
  mention and edge and the unresolved figure cue supplying no edge. A query linked only
  to the citing section reached the grid at rank 12 in one hop. With zero hops it missed.
- `make build-index` used a two-dimensional fixture encoder and CPU FAISS with fixed
  40-character chunks and zero overlap. All 24 indexed chunks had page citations.
- `make compare-graph-fusion NOISE_FLOOR=1 FUSION_MDE=0.25 RAG_K=8` compared 12 synthetic
  verified items with one hop, graph weight 0.75, exact offsets, 256 resamples, seed 13,
  and 95% confidence. Table complete-span recall rose from 0.00 to 1.00 for graph and fused,
  a +1.00 delta with interval [+1.00, +1.00] and 6 wins. The six text controls tied at 1.00
  in every lane, with a 0.00 delta and interval [0.00, 0.00]. The overall delta was +0.50
  with interval [+0.167, +0.750] and 6 wins and 6 ties. The estimated overall sample size
  for the 0.25 MDE was 35 items. `ppr` was unavailable because its implementation task
  remained open. See the [synthetic graph fixture](../current/book-recovery/corpus-projection.md#synthetic-graph-fixture)
  for the replay settings and conditions that would overturn the reading. The repeated
  Make comparison produced the same bytes.
- The tests cover deterministic builds and replay, offset and vocabulary checks, the
  detached equation label, lexical index term linking, and index continuation. They pin
  unresolved-cue exclusion and refusal of a malformed target, stale graph inputs, and
  unverified gold.

`make format`, `make lint-md`, and `git diff --check` passed. The first full `make ci`
run caught a standalone folio being read as an index continuation in the long-book
publication fixture. Continuations now require an entry with an explicit page list.
The publication fixture and synthetic graph replay passed after the fix. The full
`make ci` rerun with loopback access passed every gate and 5,338 tests, with 7 skipped
and 50 deselected.

## Handoff

`make plan-status` went from 65 tasks (40 agent and 25 human) to 64 (39 agent and 25 human).
Only scanned-book recovery changed. Its capability remains planned. The library extraction
and closing audit lose this task's prerequisite, while each retains its other prerequisites.
`book-goldset-verification` owns the release-book reading once the projected gold is accepted.
