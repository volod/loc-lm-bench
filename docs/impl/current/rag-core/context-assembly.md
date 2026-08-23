# Prompt-Side Context Assembly

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

Retrieval decides WHICH chunks reach the model. Context assembly is the step after it: what those
chunks look like once they are laid into the prompt. The two are kept apart on purpose -- a stored
chunk is a verbatim corpus slice with exact offsets, because that is what the source-span metrics
score, so anything that changes the TEXT the model reads has to happen on the prompt copies and
nowhere else. A step that respects that boundary cannot move recall@k or MRR by construction, which
is what makes its answer-quality delta readable as an answer-quality delta.

## Table-Header Restoration (table-header-context-restoration)

Shipped, OFF by default: a retrieved chunk that carries the `table` chunker's recorded header-row
offsets and does not already show that header gets the header row prepended IN THE PROMPT.

The problem it addresses is created by chunking a table at all. `table_spans` packs whole rows up to
`size` and records every block's header-row SOURCE offsets in `metadata.table_header_span`, never
copying the header text into a later block ([table-aware chunking](chunking.md#table-aware-chunking)).
A middle row block therefore reaches the model as rows of bare values whose column names sit in a
different chunk -- precisely the shape a numeric or comparative question cannot be answered from.
Until this step, nothing read those offsets.

Modules:

- `src/llb/eval/table_headers.py` -- the rule and its accounting. `restore_headers(chunks,
  doc_text)` returns `HeaderRestoration(chunks, restored, added_chars)`: prompt COPIES of the
  retrieved chunks, how many got a header, and what that added in characters. It returns the input
  list itself when nothing fired, so a no-op is free. `restored_chunk` is the per-chunk rule and
  fires only when all four conditions hold -- a well-formed `[start, end]` span is recorded, the
  resolved document reproduces that chunk's own text at that chunk's own offsets, the span resolves
  to non-blank text, and the chunk does not already carry the header (by span containment, or by
  text for a header repeated further down). `corpus_header_restorer(corpus_root)` is the resolver a
  run uses: header text read from the corpus the offsets refer to, each document read at most once,
  a missing document disabling restoration for its chunks rather than guessing.
- `src/llb/eval/graph.py` -- the retrieve node applies the restorer to the PROMPT copies only.
  `retrieved` keeps the stored records, so the retrieval sidecar, the source-span metrics, and the
  persisted run bundle see exactly what they saw with the step off; `prompt_chunks` carries the
  restored copies when they differ, and the answer-side signals (groundedness, `[i]` citations) are
  scored against those, because that is the text the model was actually asked to ground in.
- `src/llb/executor/cases.py` -- `table_headers_restored` / `table_header_chars` land on every case
  row that RETRIEVED (0 / 0 when the step is off), so an off lane and an on lane carry the same
  measured column and stay comparable. Both keys are journaled
  (`src/llb/executor/durability_journal.py`), which is what keeps a resumed bundle identical to an
  uninterrupted one.
- `src/llb/eval/answer_quality/table_headers.py` -- the comparison dimension. The step is not a
  retrieval knob, so no retrieval sweep can diagnose it; the only reading that exists is the same
  retrieval row scored twice, off and on, over one item set. It rides in the lane LABEL
  (`vector+headers`), like the retrieval budget rides in `vector#k50`, so a cell stays a plain lane
  everywhere downstream and round-trips back into the config that produced it.

Knob: `RunConfig.restore_table_headers` (bool, default False), hence in the manifest and the sweep
cell fingerprint. It needs a store built with `strategy: table` to fire at all, and reads header
text from `corpus_root`.

The restoration adds the recorded header ROW and nothing else -- not the GFM delimiter line under
it, which the chunker does not record and which a row block would not be a well-formed table with
anyway. What the step restores is the column NAMES, not a reconstructed table.

Commands:

```bash
make build-index CONFIG=<run.yaml> CHUNK_STRATEGY=table
make run-eval CONFIG=<run.yaml> RESTORE_TABLE_HEADERS=1
make compare-answer-quality CONFIG=<run.yaml> ANSWER_QUALITY_LANES=vector \
  ANSWER_QUALITY_RESTORE_HEADERS=1 SPLIT=tuning,calibration,final FUSION_FOCUS_SLICE=numeric
```

`ANSWER_QUALITY_RESTORE_HEADERS=1` twins every named lane with its `+headers` copy; the twin
differs from its base in `restore_table_headers` and the run name and in nothing else, which is the
whole basis of the reading. The report gains a `header chars` column -- the price of the step in
characters, 0.0 on the off lane -- beside the `prompt tokens` column that carries the same price in
the model's own units.

Boundary of the step, stated so nobody has to rediscover it: it applies where a lane RETRIEVES from
the store -- the single-call eval graph and everything built on it. A diagnostic lane that supplies
its own context (`closed_book`, `long_context`) restores nothing and records no accounting column,
which is correct: those lanes have no chunk boundary to repair. The abstention probe and the
position probe build their prompts on their own path and are likewise unaffected; they measure
abstention and ordering, not table readability.

Tests: `tests/llb/eval/test_table_header_context.py` -- the rule (recorded span absent, the row
block that gets its header, the block that already carries it, a header repeated further down, a
drifted document, a blank span, malformed span shapes), the accounting (only restored chunks
counted, a no-op returning the input list, an unreadable document, the per-document cache keyed on
the document rather than the first chunk seen for it), the same over real `table`-strategy chunks,
and the retrieve node recording zero accounting with the step off and restoring the prompt only with
it on. `tests/llb/eval/answer_quality/test_answer_quality_lanes.py` -- the `+headers` label
round-trip, the suffix riding a fused row and a budget together, the refusal of a suffix-only label,
the twin lane config differing in exactly one field, and the `header chars` column reaching the
report.

### Measured result: the header reaches the prompt and does not reach the answer

Scored 2026-08-22 on the RTX 4060 Ti 16 GB CUDA host. The store was built with
`make build-index CHUNK_STRATEGY=table` over the converted Ukrainian goods PDF corpus -- the
table-heavy one, 30% of its characters markdown table rows ([table-aware
chunking](chunking.md#table-aware-chunking)) -- at `size` 200 / overlap 30 with the pinned
`intfloat/multilingual-e5-base`, a flat FAISS store of 3,505 indexed chunks. The 95-item drafted
goods ledger was then scored end to end across all three splits pooled (31 tuning + 33 calibration
+ 31 final) by `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` over Ollama at k=10, 2,000 bootstrap
resamples, seed 13, 95% paired CI, under two lanes -- `vector` and `vector+headers` -- that differ
in `restore_table_headers` and in nothing else. The ledger is DRAFTED: no reviewer accepted these
items, so the objective is diagnostic and the artifact records `grounding: drafted`.

The step fired: 56 of the 95 cases had at least one retrieved chunk given its header back, 145
chunks in total.

Retrieval is identical, as the design requires. Per split, both lanes report recall@10 0.645 /
0.758 / 0.677 and MRR 0.513 / 0.479 / 0.401 -- the same six numbers, not merely overlapping ones --
and pooled over the 95 items both lanes report recall@10 0.695, all-spans@10 0.516, span coverage
0.605, and 1,618.8 context characters. That is the construction working: the two lanes retrieved
the same chunks and the metrics read the same stored records.

What the step cost and what it bought, `vector+headers` minus `vector`, pooled (n=95):

| metric | delta | interval | w/l/t | rand p | reading |
| --- | ---: | ---: | :-: | ---: | --- |
| header chars | **+36.1** | **[+26.2, +48.0]** | 56/0/39 | 0.000 | separated |
| prompt tokens | **+20.0** | **[+15.0, +25.6]** | 56/0/39 | 0.000 | separated |
| objective | +0.009 | [-0.011, +0.033] | 9/7/79 | 0.213 | flat |

Per question-type slice, objective mean off -> on and its paired delta:

| slice | n | off | on | objective delta | w/l/t | rand p | header chars |
| --- | ---: | ---: | ---: | ---: | :-: | ---: | ---: |
| numeric | 4 | 0.281 | 0.281 | +0.000 [+0.000, +0.000] | 0/0/4 | 1.000 | +20.8 |
| comparative | 2 | 0.414 | 0.712 | +0.298 [+0.000, +0.596] | 1/0/1 | 0.500 | +20.0 |
| factoid | 40 | 0.453 | 0.461 | +0.008 [-0.027, +0.049] | 3/3/34 | 0.359 | +42.6 |
| multi-hop | 35 | 0.399 | 0.393 | -0.005 [-0.029, +0.013] | 2/3/30 | 0.656 | +37.1 |
| procedural | 14 | 0.489 | 0.500 | +0.011 [-0.006, +0.035] | 3/1/10 | 0.250 | +22.1 |

Verdict: **REJECT for adoption; the step stays OFF by default and remains selectable per run.**

The reading, in operator terms. The cost side is real and measured: ~20 prompt tokens on each of
the 56 cases the step touches, 56 wins to 0 losses, an interval nowhere near zero -- this is the
one row in the comparison that separates. The benefit side is not there. Pooled, the objective
moves +0.009 on an interval straddling zero with 9 wins against 7 losses out of 16 items that
differ at all; the calibrated sign-flip probability is 0.213, five times the 0.025 the reading
would need. On the two slices the step was predicted to help, it did not: `numeric` is EXACTLY
unchanged (the step fired on 2 of its 4 cases and neither answer moved -- zero discordant items,
so the slice looked and found nothing rather than being merely underpowered), and `comparative`
turned one of its two items from 0.414 to 0.712, which at n=2 and one differing item is an anecdote,
not a reading. Both slices are far under the six differing items an exact two-sided sign test needs
to reach 95% ([the minimum-evidence gate](paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)),
so neither could have licensed adoption at this item count whatever it had shown.

Why the effect is small here, stated plainly: the headers on this corpus are SHORT. +36 characters
per case is roughly one narrow header row of two or three columns, and the values in these tables
are largely self-describing (a price, a unit, a date) -- a 12B model answers them about as well
unlabeled. The mechanism the step implements is sound and its cost is negligible; what this corpus
does not supply is a table wide enough for the column name to be the thing the answer was missing.

What would overturn it: a corpus of WIDE tables (many columns, headers whose text is the only way
to tell two numeric columns apart) with a numeric or comparative slice of at least six items that
the two lanes differ on, or a second generator -- this reading is conditioned on MamayLM 12B, and
whether delivered structure becomes a better answer is a property of the model as much as of the
prompt. A future re-read should also carry the same `header chars` / `prompt tokens` pair, because
a corpus with wide headers will pay far more than 20 tokens per case and the trade has to be read
against that price, not this one.

## Contiguous-Chunk Stitching (fragmented-evidence-delivery-lever)

Shipped as a MEASUREMENT lane, not a run knob: `compare-retrieval --stitch` twins every compared
row with a copy whose top-k has its contiguous same-document chunks merged into one block.

The problem it addresses is fragmentation rather than missing structure. On the goods corpus the
`procedural` slice retrieves 0.706 of its gold-span characters but only 0.357 of those spans arrive
inside ONE chunk ([the intactness
re-read](chunking.md#the-intactness-re-read-of-the-same-three-chunkers)), so on a whole question
type the model reassembles a procedure from pieces that were adjacent in the source document. Two
levers can convert those pieces back into whole spans: raise the `size` cap, which rebuilds the
index and changes what is retrieved, or reflow what was ALREADY retrieved, which does not.
Stitching is the second one. What the two are worth against each other is measured in [two levers
against fragmented evidence](fragmented-evidence.md).

Modules:

- `src/llb/rag/stitching.py` -- the rule and the wrapper. `stitch_contiguous(chunks)` merges two
  retrieved chunks only when they share a document and their character ranges TOUCH or OVERLAP. A
  gap is never bridged (that would serve text nobody retrieved), nothing is reordered (a block sits
  at its best-ranked part's position and inherits that part's identity and score), and no text is
  invented -- the merged text is the parts' own text with an overlap counted once, so the block is
  exactly the source slice its merged offsets name. It returns the input list itself when nothing
  merged, so a no-op is free; otherwise every block is a copy, `rank` is renumbered 1..n because the
  pre-stitch rank names a position the list no longer has, and a merged block records its parts in
  `metadata.stitched_from`. `StitchingRetriever` wraps any store on the `.retrieve` seam (the split
  dense/lexical path of a hybrid store included) and censuses blocks, merges, and served characters
  per query.
- `src/llb/rag/comparison/rows.py` -- `add_stitch_rows` builds the `<row>+stitch` twins and
  `stitch_report` records, per twin, what it merged AND whether it reproduced its base lane's
  `recall@k` and `span_char_coverage@k` exactly. That invariance is the reading's own precondition,
  so it lands in the artifact as two booleans rather than being asserted in prose.
- `src/llb/cli/rag/compare_retrieval_lanes.py` -- `--stitch` layers the twins over whatever rows the
  comparison built, AFTER any `--reranker` twins, because stitching reflows what a lane finally
  delivers rather than its pre-rerank pool. `verdict_lanes` keeps every stitched row out of the
  eligible set.

Two merges are REFUSED, both to keep a block exactly reversible onto the source: a chunk whose text
length disagrees with its own offsets (a governance overlay may rewrite chunk text), and a chunk
that collapsed byte-identical copies ([duplicate chunk
collapse](retrieval-store.md#duplicate-chunk-collapse)), whose recorded occurrences describe that
text at other places a merged block appears at none of. A refused chunk is still served as its own
block, so a refusal costs intactness, never evidence.

Why this step is read on retrieval metrics while header restoration is not: restoration changes the
TEXT the model reads, which no source-span metric can see, whereas stitching changes only how many
BLOCKS the same characters arrive in -- and `span_intact@k` asks exactly whether ONE block carries a
span whole. What follows is a boundary, not a convenience: `recall@k` and `cover@k` MUST reproduce
the base lane, and a lane that moved them did not reflow evidence, it changed it.

`mrr` is NOT readable on a stitched row and the report says so on its own line. Merging shortens the
returned list, so the first hit can only move to an earlier position; the compression is an artifact
of counting positions, not a ranking gain. For the same reason a stitched twin is a REPORTED lever
and never an adoption candidate: it ties its base lane on every metric a verdict is decided on, so
`compare-retrieval` excludes it from the eligible lanes rather than letting it win a tie-break on a
compressed `mrr`.

Commands:

```bash
make compare-retrieval CONFIG=<run.yaml> STITCH=1
make compare-retrieval CONFIG=<run.yaml> CHUNK_SIZES=200,400,800 STITCH=1
```

Boundary of the step, stated so nobody has to rediscover it: it exists on the comparison path only.
No `RunConfig` field turns it on, nothing in the eval graph stitches, and no default moved -- what
ships is the measurement that would justify shipping it. Promoting it to a run knob is future work
that needs an answer-side reading, exactly as header restoration did.

Tests: `tests/llb/rag/test_stitching.py` -- adjacent chunks merged with exact offsets, an overlap
served once, a contained chunk adding nothing, a gap never bridged, two documents left alone, the
two refusals, the block taking its best-ranked part's position and identity, a cut span turning
intact while recall/coverage/served characters do not move, and the retriever's census, delegation,
and split-query path. `tests/llb/rag/comparison/test_compare_retrieval_core.py` and
`test_compare_retrieval_cli.py` -- the twin scored beside its base through the real comparison, the
invariance flags in both directions, the rendering, and the twin's exclusion from the verdict lanes.
