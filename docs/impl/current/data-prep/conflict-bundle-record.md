# Corpus Hygiene: What a Finished Audit Bundle Can Answer Alone

An audit run writes `summary.json` and `findings.jsonl` and then the store it read moves on: the
next `make build-index` collapses a different duplicate, chunks a document differently, or is one
ingest ahead of the corpus the run saw. Every question asked of a finished run afterwards therefore
has two possible answers -- the one the run would have given, and the one today's store gives -- and
they look identical from the outside.

This page is the verdict per question: which ones the bundle answers from its own record, which ones
it refuses, and why the record stops where it does. The refusals are the point as much as the
answers: "not recomputable" is correct, and a reading recomputed against a rebuilt store is not.

## The questions and their verdicts

| question | answered from the bundle | what it costs to record |
| --- | --- | --- |
| Which STAGE lost an orderable document pair? | yes | one entry per document (its ordering fields) plus the per-document chunk accounting |
| WHY is a document not comparable, and at what `--min-claim-tokens` does it come back? | yes | three counters and one integer per document |
| What would a SMALLER CANDIDATE BUDGET have returned? | yes | the ranked candidate list collapsed to document pairs, capped by the run's own budget |
| Which CHUNK would a lost pair have matched on? | **no** | a chunk ordinal and a similarity per pair -- the store, re-recorded |
| What would a different `--min-claim-tokens` have excluded, CORPUS-WIDE? | **no** | a token count per chunk plus a re-run of the metadata grouping |

The rule the table is drawn on is a size bound, not a preference: a recorded input must be linear in
DOCUMENTS or bounded by a run parameter. The first three are; the last two are per CHUNK and grow
with the corpus, so recording them would be keeping a second copy of the store inside the bundle.
Both refusals have the same honest fix -- rebuild the store and re-run the audit.

The verdicts are computed per bundle by `bundle_readings` (`src/llb/conflicts/bundle_readings.py`)
and printed by `make recompute-conflict-stage` as a table, so an operator reads them instead of
discovering a gap when an answer turns out wrong. A question that is unavailable says WHY in the
operator's own terms: a bundle written before the record says it predates it, and a run below the
semantic tier says it read no store and names `--effort` rather than a re-run.

## The record

All of it rides under `stage_attribution_inputs` in `summary.json`
(`src/llb/conflicts/bundle_record.py` builds it, `AuditResult.stage_inputs` carries it,
`RunInputs` is what the semantic pass hands back). `schema_version` is the migration seam and is
**2**; the additions are additive, so a schema-1 bundle still replays its stage and answers the two
newer questions with a refusal.

| key | what it carries | module |
| --- | --- | --- |
| `documents` | every corpus document in corpus order, with the `effective_date` / `version` it was audited under | `bundle_record.py` |
| `chunks` | `stored` / `comparable` / `copies` per document | `document_chunks.py` |
| `exclusions` | `front_matter` / `low_content` / `metadata_block` per document, plus `recovery_floor` and the run's `min_claim_tokens` | `document_exclusions.py` |
| `candidates` | the ranked candidate list collapsed to `[rank, left doc, right doc, cosine]`, with `total_pairs` and `covered_to_rank` | `candidate_record.py` |

Why `documents` and `chunks` cannot be re-derived instead of recorded -- corpus order is data, and a
rebuilt store answers about itself -- is in
[recomputing the stage](conflict-decision-groups.md#recomputing-the-stage-from-a-finished-bundle).

`chunks`, `exclusions`, and `candidates` are ABSENT below the semantic tier, never empty: a run that
read no store built no accounting, no exclusion pass, and no ranking, and an empty one would say the
opposite (a store that held nothing).

### Why a document is not comparable, and the floor that returns it

The stage attribution could already say a pair stopped at the CLAIM-TOKEN FLOOR, but the sentence it
printed there was a disjunction -- "front matter, below `--min-claim-tokens`, or a repeated metadata
block" -- because the run kept one total and threw the three reasons away. Two of those three are
not moved by the knob the stage names: front matter is excluded BEFORE the floor is consulted, and a
repeated metadata block is excluded for being ABOVE it. So "lower `--min-claim-tokens`" was advice
that could not work on two thirds of the cases it was printed for.

`ContentSelection` (`semantic_filter.py`) now keeps the exclusion ORDINALS rather than three counts,
and `DocumentExclusions.of` folds them per document in the same pass. The reading gains two things:

- the reason names the counts: `` `archive-policy.md`: 1 front matter, 2 below `--min-claim-tokens` ``;
- the knob names a VALUE: `recovery_floor` is the largest claim-token count among a document's
  low-content chunks, which is the highest floor at which that document has a candidate chunk again.
  A pair needs every filtered side back, so the printed floor is the LOWEST of its sides' floors.

A document with no low-content chunk has no `recovery_floor` entry, and that absence is the reading:
the knob then says so outright rather than naming a value -- "no `--min-claim-tokens` value returns
this pair, because nothing the floor governs is what excluded it".

**Measured, on the committed 7-document fixture** (CUDA host, real e5-base 19-chunk heading store,
no model call). Run at a floor nothing clears, then at the floor the record names, then one above it:

```text
make build-index CORPUS=samples/corpora/conflicts_uk_v1/corpus CHUNK_STRATEGY=heading CHUNK_SIZE=600
make audit-corpus-conflicts CORPUS=samples/corpora/conflicts_uk_v1/corpus EFFORT=semantic \
  STORE=<that-store> COS_THRESHOLD=0.9 MIN_CLAIM_TOKENS=200
[conflicts] ... Earliest stage an orderable document pair was lost at: the CLAIM-TOKEN FLOOR, shown
  by `archive-policy.md` + `deadline-note.md` (every chunk of `archive-policy.md` and
  `deadline-note.md` in the store is excluded from comparison (`archive-policy.md`: 1 front matter,
  2 below `--min-claim-tokens`; `deadline-note.md`: 1 front matter, 2 below `--min-claim-tokens`)).
  One knob: lower `--min-claim-tokens` to 30 or below (this run used 200), or re-chunk so the claim
  lands in a longer chunk.
```

| run | `MIN_CLAIM_TOKENS` | stage named | `deadline-note.md` comparable chunks |
| --- | --- | --- | --- |
| `20260815T-bundle-record-fixture-floor` | 200 | `claim_token_floor` | 0 |
| `20260815T-bundle-record-fixture-floor-31` | 31 | `claim_token_floor` | 0 |
| `20260815T-bundle-record-fixture-floor-30` | 30 | `candidates` | 1 |

The recorded floor is EXACT, not a hint: 30 returns the pair and 31 does not, which is what the
per-document floors say (`archive-policy.md` 51, `deadline-note.md` 30, and the pair takes the
lower). `tests/llb/conflicts/test_bundle_record.py` pins that acceptance the same way -- it re-runs
the audit at the floor the record named and asserts the document became comparable, rather than
comparing the record with itself.

**What the floor is NOT.** It is a per-document necessary condition, never a corpus-wide sweep.
Repeated-metadata detection runs OVER the candidate set the floor produces, so a lower floor can add
chunks that then re-group other chunks out of comparison; an exact "what would floor N have
excluded" needs a token count per chunk and a re-run of the grouping, which is the refused
`floor_sweep` reading above.

### What a smaller candidate budget would have returned

The claim tier's budget is a RANK CUTOFF into the store's own cosine ordering:
`detect_semantic_pairs` returns its pairs sorted by descending similarity, `--max-claim-pairs` keeps
a prefix, and `--max-candidate-pairs` picks the cosine threshold that makes the list about that
long. A smaller budget therefore returns a PREFIX of the same ranking, and a prefix is answerable
with no store, no vectors, and no re-adjudication.

What it needs is the ranking, and the ranking is the one thing `findings.jsonl` loses. Every
candidate pair does reach that file -- pairs past `--max-claim-pairs` land there as provisional
`duplicate` rows -- but the rows are written in report order (actionable first, then by the
ADJUDICATOR's score, which for a claim-tier row is not the cosine that ranked it). The bundle holds
the candidate SET and not the candidate ORDER.

`CandidateRecord` records the order, collapsed to document pairs with the rank each one first
appears at. Collapsed, because every reading built on it is a document-pair reading and the first
rank a pair appears at is exactly what decides whether a budget returns it. Capped, because the
uncollapsed list is quadratic in CHUNKS and the collapsed one is quadratic in DOCUMENTS: the cap is
the run's own `--max-candidate-pairs` when it set one and `DEFAULT_CANDIDATE_RECORD_PAIRS` (200)
otherwise, and `covered_to_rank` says how deep the record reaches. A budget past it is REFUSED, not
answered from a truncated list.

```bash
make recompute-conflict-stage STAGE_RUNS="<audit-run-dir> ..." STAGE_BUDGET=2
# -> $DATA_DIR/corpus-conflict-stage/<run>/{stage.md,stage.json}
```

The reading reports TWO things, because the attribution alone under-reports the budget: it names one
pair, and on a corpus whose corpus-first lost pair is lost at every budget the NAME never moves
however many pairs the budget takes away. So the count comes with it -- how many document pairs
would have returned, against how many the run returned.

**Measured, over the bundles on this host**
(`.data/corpus-conflict-stage/20260815T-bundle-record-archive/`, CUDA host, no model call):

| bundle | run's document pairs | at budget 2 | attribution moves |
| --- | --- | --- | --- |
| `20260815T-bundle-record-fixture-semantic` (7 docs, 19 chunks) | 8 | 5 | no |
| `20260815T-bundle-record-squad-semantic-cos060` (250 docs, 311 chunks) | 36 | 2 | no |
| `20260815T-bundle-record-squad-semantic-cos080` (250 docs) | 1 | 1 | no |
| `20260815T-bundle-record-fixture-hash` (below the semantic tier) | -- | refused | -- |

At its own budget the record reproduces the run exactly: the recorded prefix at `total_pairs` equals
the document pairs in `findings.jsonl`, checked directly on the fixture bundle (8 of 8) and pinned in
CI. **The attribution moved on none of the measured bundles**, and the reason is the corpus rather
than the reading: the fixture's corpus-first lost orderable pair is `archive-policy.md` +
`deadline-note.md`, which no budget ever returned, so it is the named pair at every cutoff. That is
precisely why the pair COUNT is reported beside the name -- on the fixture the budget silently costs
three of eight document pairs while the sentence stays word for word identical.

## The size the record actually costs

| bundle | documents | store chunks | record bytes | of which `candidates` |
| --- | --- | --- | --- | --- |
| `20260815T-bundle-record-fixture-hash` | 7 | -- (no store) | 637 | absent |
| `20260815T-bundle-record-fixture-semantic` | 7 | 19 | 2,044 | 429 |
| `20260815T-bundle-record-squad-semantic-cos080` | 250 | 311 | 25,266 | 125 |
| `20260815T-bundle-record-squad-semantic-cos060` | 250 | 311 | 27,566 | 2,425 |

The largest bundle on this host records 27.6 KiB over 250 documents -- about 110 bytes per document
-- against a 74 KiB `summary.json`, and the per-document maps (`documents` 9.5 KiB, `chunks`
14.5 KiB, `exclusions` 1.1 KiB) are what dominate it, exactly as the bound predicts. No corpus here
has a candidate list long enough for the cap to bite, so the cap and its refusal are pinned in CI
instead (`test_the_candidate_record_stops_at_its_cap_and_says_how_far_it_reaches`), alongside a
growth test that doubles a corpus past the cap and asserts the record less than doubles.

## Where it lives

| what | where |
| --- | --- |
| the record a run writes | `src/llb/conflicts/bundle_record.py` (`RunInputs`, `stage_attribution_inputs`, `recorded_inputs`, `readable_record`) |
| re-reading a bundle with it | `src/llb/conflicts/stage_replay.py` (`replay_attribution`, `replay_entry`, the budget prefix) |
| per-document exclusions | `src/llb/conflicts/document_exclusions.py`, folded in `semantic_run.py` from `ContentSelection` |
| the ranked candidate list | `src/llb/conflicts/candidate_record.py` |
| the per-question verdicts | `src/llb/conflicts/bundle_readings.py` |
| rendering | `src/llb/conflicts/report_stage_replay.py`, `src/llb/cli/prep/conflict_stage.py` |
| tests | `tests/llb/conflicts/test_bundle_record.py`, `tests/llb/conflicts/test_stage_replay.py` |
