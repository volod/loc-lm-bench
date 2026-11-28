# Book retrieval reading

## Work contract

```text
#### book-retrieval-reading

Prepare the paired retrieval comparison before operator review so an accepted book item can
exercise both corpora through one replayable command.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- replay over a planted corpus and synthetic gold.
- Dependencies: [RAG core](current/rag-core.md) and
  [paired verdicts](current/rag-core/paired-verdicts.md).
- User-visible outcome: a report of recall at k and MRR against source spans and a paired comparison
  against the existing PDF lane, joined on the originating PDF page.
- Scope boundary: retain the baseline OCR language and the same index and retrieval settings
  on both corpora. Report separately the cases whose baseline page text contains the answer span.
  Keep retrieval settings fixed and leave answer scoring to its own evaluation.
- Data and artifact paths: fixture corpora and items under `tests/`; corpus and index runs under
  `$DATA_DIR`.
- Execution path: reuse `make pdf-to-markdown`, `make build-index`, and the retrieval evaluation
  targets over synthetic inputs.
- Acceptance gates: `make ci` green; refuse stale or unverified gold; pin the span and page
  readings, the item count and minimum detectable effect, the baseline-text split, and the
  inconclusive and negative verdict paths. Real book evidence follows reference and gold review.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md`.
```

## Implementation and decisions

`make compare-book-retrieval` builds both corpora with the shared `build-index` configuration.
`src/llb/rag/book_comparison/inputs.py` checks the conversion manifest, scan identity, page
citations, and verified gold spans before the index builder runs. It maps labeled text by
physical PDF page with whitespace normalization. `metrics.py` reuses source-span overlap,
paired intervals, and the shared minimum-evidence and sensitivity calculations. `run.py`
records the input digests, index settings, ranks, and verdicts. The shared exact sign test
keeps its binomial denominator as an integer so large ledgers can score past 1,023 differing
pairs.

## Acceptance evidence

Accepted.

On 2026-11-28 on a CPU-only run on the Intel Core Ultra 7 265HX host:

- `make pdf-to-markdown` admitted the generated 13-page PDF through PyMuPDF4LLM.
  `make validate-goldset` accepted 12 synthetic items grounded in the projection.
- Two `make compare-book-retrieval` runs gave byte-identical reports using CPU FAISS, page
  chunking at 256 characters and zero overlap, the planted locator encoder, k=1, 95% confidence,
  2,000 resamples, seed 13, and an MDE of 0.25.
- The projection's 13 indexed chunks and the baseline's 14 both matched their respective
  standalone `make build-index` artifacts byte for byte.
- The [planted retrieval fixture](../current/book-recovery/corpus-projection.md#planted-retrieval-fixture)
  records the span and page deltas, intervals, and win/loss/tie ledgers. Overall span recall
  and MRR tied at 0.50 with a delta interval of [-0.50, 0.50]. Overall page recall and MRR
  regressed by 0.50 with an interval of [-0.75, -0.25]. The six items with surviving baseline
  evidence regressed on both readings and the six with missing baseline evidence improved
  on span hits while page hits tied. For overall span recall, the observed variance requires
  137 items to resolve the declared 0.25 effect. A changed rank, split, report, or store on
  the same inputs would overturn the fixture reading.
- `tests/llb/rag/test_book_comparison.py` exercises replay, stale and unverified gold refusal,
  unresolved spans, mismatched scans, cross-page labels, numeric substitutions, and the
  minimum-evidence gate.

`make format`, `make lint-md`, and `git diff --check` passed. The sandboxed `make ci`
attempt cleared the static gates and was stopped after a loopback bind check confirmed that
the download-guard tests could not open sockets. The full rerun with loopback access passed
all gates and 5,334 tests, with 7 skipped and 50 deselected.

## Handoff

`make plan-status` went from 66 tasks (41 agent and 25 human) to 65 (40 agent and 25 human).
Only scanned-book recovery changed. The reference-graph task and closing audit drop the
retrieval task dependency. The capability remains planned. The real-book comparison belongs
to `book-goldset-verification` after reviewer acceptance of the reference and gold.
