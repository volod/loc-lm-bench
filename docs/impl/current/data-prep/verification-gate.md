# Verification Gate And Judge Calibration

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

## Verification Gate

The verification path has a mechanical half and a human half.

```bash
make cross-check-goldset BUNDLE=<draft> CROSS_CHECK_MODEL=<model>
make verify-sample BUNDLE=<draft>
make verify-review VERIFY_WS=<worksheet>
make verify-accept VERIFY_WS=<worksheet> BUNDLE=<draft>
```

`src/llb/prep/goldset/cross_check.py` checks grounding and non-circularity before calling an injectable
second verifier for support and answerability. A pass means the item is reviewable, not verified.

`src/llb/goldset/verify_sampling/` handles stratification, reviewer context, confidence ordering,
row construction, and worksheet persistence. `verify_base.py`, `verify_acceptance.py`, and
`verify_refcheck.py` own the schema/I/O, acceptance/ledger, and reference-checking seams;
`src/llb/goldset/verify/cli.py` orchestrates the commands. `verify_session/` owns the interactive
terminal loop.
The review session keeps command parsing, navigation, row edits, clear confirmation, and
persistence in small helpers so the loop reads as worksheet orchestration.
The accepted ledger writes copied corpus files plus canonical `verified=true` rows; chain samples
write `accepted/chains.jsonl`, while flat goldset samples write `accepted/goldset.jsonl`.
`prepare-goldset-draft` can also write the first worksheet in the same run with
`--derive-verification-sample`; the make wrapper exposes this as `DRAFT_VERIFY_DERIVE=1`.
`--verification-sample-size <n>` / `DRAFT_VERIFY_N=<n>` remains an explicit override.

### Experiment-derived verification and acceptance gates

`verify-sample` no longer inherits a universal row count. With no `VERIFY_N`, it prices a
conservative reject-rate proportion estimate from `VERIFY_CONFIDENCE=0.95`,
`VERIFY_PRECISION=0.10`, expected reject rate 0.50, and the actual finite population. The result is
also floored at the number of non-empty strata and capped at the population. For example, the
default target is 17 rows for a 20-item population and 66 for 200 items. `VERIFY_N=<n>` still wins
as an operator override.

Every `sample_manifest.json` carries an `acceptance_gate` object with the method, assumptions,
derived target, override, selected target, and whether an override reaches the derived target.
Single- and multi-reviewer sampling, additive merges, draft-time worksheet creation, and the PDF
quick-start all use the same helper in `verify_sampling/planning.py`.

Chain promotion uses the sibling relative gate in `goldset/chain_acceptance.py`: by default it
requires half of the reviewed sample to survive into the accepted chain ledger, rather than a
corpus-independent count. `fixture_manifest.json` persists the same derivation fields.

`make acceptance-gate-audit` writes `inventory.json` and `report.md` under
`$DATA_DIR/acceptance-gate-audit/<run>/`. The inventory classifies retained trial counts as
resource budgets, finalist minima as structural safety controls, and migrated row/count defaults
as inferential gates. The retired `ua-model-roster-long-run` name is recorded as absent; its live
joint-search successor controls are explicitly resource/structural. `make ci` and
`scripts/code_quality.sh` run the check-only form, which fails on an unexplained absolute trial,
sample, or acceptance control.

The 2026-07-28 inventory at
`$DATA_DIR/acceptance-gate-audit/20260728T135530Z/inventory.json` passed with 27 declarations, 20
live absolute-control discoveries, and no findings. Validation completed with `make lint-md` and
`make ci`; the latter passed 2,332 tests with 43 slow or opt-in tests deselected.

The rationale is anti-anchoring and auditability: automated cross-check context can be shown to a
reviewer, but it is hidden by default; the accepted ledger is a new reviewed artifact rather than an
in-place mutation of the draft.

### Reviewer throughput tooling

The review CLI carries the throughput features from the `verify-cli-throughput` task, complete
with the measured human review-pass evidence recorded at the end of this section:

- **Confidence-ordered queue** -- `make verify-review VERIFY_WS=<ws> VERIFY_ORDER=confidence`
  reviews least-confident items first: each cross-check verdict flag contributes +1/-1 and a
  needle `retrieval_rank` contributes `1/rank` (`row_confidence`/`confidence_order` in
  `verify_sampling/confidence.py`). Only the session queue is reordered; the CSV row order never
  changes.
- **On-card evidence** -- worksheet rows (new optional columns `retrieval_rank`,
  `page_citation`) carry the item's needle retrieval rank (joined from `needle_items.jsonl` /
  `item_provenance.jsonl`) and a `<source.pdf> p.N[-M]` citation resolved through the PDF lane's
  `*.citations.json` sidecars, so the reviewer can check the original page without leaving the
  terminal.
- **Ambiguous-evidence guard** -- an optional `span_occurrences` column carries how many times an
  item's primary gold span text appears verbatim across the whole corpus (`span_occurrences.py`).
  An item whose span repeats is ambiguous by construction: the answer text exists in several
  places, the retrieval metric credits any of them, and the reviewer could not otherwise tell that
  the span they are accepting is not unique. The count comes from the draft-time
  `span_occurrences.jsonl` sidecar when present, else a direct corpus scan of the sampled items;
  the review card adds an `== ambiguous evidence: this span text appears in N places ...` line so
  the reviewer decides whether the question is uniquely answerable. The guard fires above one
  occurrence (`OCCURRENCE_FLAG_THRESHOLD`) and only annotates -- it never rejects an item or
  changes the retrieval metric. The column and sidecar are BOTH absent when every sampled span is
  unique, so an all-unique bundle keeps its worksheet byte-for-byte.
- **Accept-with-edit re-grounding** -- the `e` command captures an edited reference answer and
  re-grounds it IMMEDIATELY against the bundle corpus (resolved via `sample_manifest.json`); an
  edit that is not a verbatim corpus span is refused on the spot, an accept over a stale edit is
  blocked, and `emit_accepted_ledger` re-checks authoritatively (raises) so a hand-edited CSV can
  never certify an un-grounded answer. Accepted edits flow into the ledger with the primary span
  replaced by the re-grounded offsets.
- **Additive sample enlargement** -- `make verify-sample BUNDLE=<draft> VERIFY_N=<n>
  VERIFY_MERGE=1` (`merge_sample_worksheet`) enlarges an existing worksheet to ~`n` rows by
  appending only item ids not already present: decided rows are preserved byte-for-byte, never
  re-drawn or re-shown, and re-running the merge is idempotent.
- **Session throughput stats** -- each decision prints a pace line (decided count, items/hour,
  ETA for the remaining rows); the end-of-session summary repeats it and every sitting appends a
  record to `verify_session_stats.json` beside the worksheet (the durable items-per-hour
  evidence the throughput task cites).
- **Coded rejection reasons** -- `x` infers a code from the first failed check
  (`ungrounded`/`circular`/`wrong_reference`/`label_mismatch`), `x <code>` sets one explicitly
  (also `bad_question`, `other`); `make verify-accept` exports the aggregate to
  `rejection_reasons.json` beside the accepted ledger, and the drafting pipeline reads it back
  (draft-feedback-rejection-reasons, below) to tighten its prompts on a re-draft.

All of it is unit-tested with injected input/output/clock in `tests/llb/goldset/test_goldset_verify.py`
(golden-path session tests included). The loader supplies the shared verification columns and
preserves profile-specific columns used by translation and adjudication, so one review engine can
serve each current worksheet profile without separate parsers.

The review card and controls are unified with the external-RAG human review session
(`llb.scoring.external_rag_session`, the origin interface): a `=====` banner, `== field:` labels,
a blank line before `== question:` delimiting consecutive cards, indented multi-line evidence
blocks, a two-line grouped prompt hint, and the shared `o` (note) / `w` (edit answer) keys. The
decision keys intentionally differ (`y`/`x` here) because `a`/`r`/`p` mark the verification
checks. The card layout and key aliases are documented behavior, not test surface -- session
tests cover decisions, re-grounding, merge, and stats, not print formatting.

Measured throughput evidence (2026-07-10, quickstart PDF corpus draft, 69 drafted items): a
single-sitting human pass with `VERIFY_ORDER=confidence` decided all 46 sampled rows in 9.9
minutes -- **279.5 items/h** -- recorded in `verify_session_stats.json` beside the worksheet.
`verify-accept` passed at 44 accepted / 2 rejected (reject rate 0.043 vs tolerance 0.05); both
rejects concentrated in the corpus's one long-manual document, one of them a TOC-mined
page-number question whose row also carried no `retrieval_rank` (the signal the confidence queue
sorts on). Two advisory per-stratum FAIL warnings were small-sample artifacts: at tolerance 0.05
a stratum needs >= 20 decided rows to absorb a single reject, and the flagged cells held 7 and 5.
A bare `x` reject with no failed checks exports `code: other`; marking the failing check first (or
using `x <code>`) keeps `rejection_reasons.json` actionable.

`draw_stratified_sample` allocates through `stratum_quotas`: a floor of one per non-empty stratum
(largest strata first when `n` cannot cover them all) plus a deterministic largest-remainder
top-up, each stratum
capped at its own size. An explicit `VERIFY_N=<n>` therefore draws exactly
`min(n, population)` rows at every seed, while the default draws the derived target. Both paths
stay seeded-reproducible. The sibling `sample_manifest.json` records the final per-stratum
allocation and gate plan. The allocation invariants are covered in the goldset sampling tests.

### Rejection feedback into re-drafting

Rejection feedback can directly guide a new draft:

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> \
  DRAFT_REJECTION_FEEDBACK=<bundle>/accepted/rejection_reasons.json
```

`llb prepare-goldset-draft --rejection-feedback <file>` maps each dominant reject code to a
deterministic Ukrainian draft-prompt hint (`src/llb/prep/ontology/extraction/feedback.py`; the mapping
covers exactly the closed reject-code set, ordered by rejection count, and each hint carries the
first rejected item's note as an example -- e.g. a `circular`-heavy summary adds an explicit
non-circularity instruction). The combined hint block is appended to the ontology-constraint
line of every draft prompt; an empty summary is a no-op. `provenance.json` gains an
`applied_feedback` block (source path, sha256 digest, applied hint codes + counts), the setting
is pinned in the journal meta so `--resume` replays it, and
`settings.rejection_feedback` names the file. Unit tests:
`tests/llb/prep/ontology/test_draft_feedback.py` (per-code mapping, dominant ordering, no-op
summary, prompt + provenance round-trip over a fake endpoint).

### Multi-annotator gate and adjudication

The verification gate supports more than one annotator plus configurable acceptance arithmetic
(`src/llb/goldset/verify_multi/` + policy extensions in `verify_acceptance.py`; tests in
`tests/llb/goldset/test_verify_adjudication.py`):

```bash
make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_ANNOTATORS=<k>
make verify-review VERIFY_WS=<bundle>/verify_sample.r1.csv   # each reviewer, own sheet
make verify-adjudicate BUNDLE=<draft>
make verify-review VERIFY_WS=<bundle>/adjudication.csv
make verify-accept VERIFY_WS=<bundle>/verify_sample.csv BUNDLE=<draft> \
  VERIFY_ACCEPT_POLICY=<global|per-stratum|weighted>
```

- **Multi-reviewer sampling** -- `VERIFY_ANNOTATORS=<k>` draws ONE stratified sample and writes
  it as `k` identical per-reviewer worksheets (`verify_sample.r<i>.csv`), each row stamped with
  a `reviewer_id` column, left blank for single-reviewer worksheets.
  The manifest records the reviewer worksheets and intentionally omits the single-`worksheet`
  key, so a multi-reviewer bundle can only stamp `--data-verified` through its accepted ledger.
- **Agreement report** -- `verify-adjudicate` writes `agreement.json` beside the worksheets:
  observed agreement plus Cohen's kappa (2 reviewers) or Fleiss' kappa (3+) over the jointly
  decided rows, per-reviewer decided/accept/reject counts, and the disagreement item ids. A
  unanimous accept whose accept-with-edit answers differ counts as a disagreement (the edit
  changes what the ledger would certify). Metric arithmetic is isolated in
  `verify_multi/agreement_metrics.py`; `AgreementReportBuilder` in `agreement_report.py` indexes
  the worksheets once and builds the report sections.
- **Adjudication pass** -- disagreements are drawn into `adjudication.csv` (exactly those rows),
  human columns blank for an independent decision, prior verdicts carried forward in a read-only
  `prior_decisions` column (`r1=reject:bad_question;r2=accept`) shown on the review card.
  Rebuilding preserves adjudicator decisions already made. The ordinary `verify-review` session
  reviews it unchanged.
- **Consensus acceptance** -- `verify-accept` on a multi-reviewer bundle scores the consensus:
  unanimous decisions stand, adjudicated decisions override disagreements, and anything else
  (a reviewer still undecided, an unadjudicated disagreement) stays undecided and blocks
  acceptance. `ConsensusBuilder` in `verify_multi/consensus.py` owns adjudication preference,
  unanimity checks, and clearing blocked human fields. The accepted ledger and
  adoption-through-ledger invariant are unchanged.
- **Acceptance policies** -- `--policy` (make: `VERIFY_ACCEPT_POLICY=`) selects the arithmetic:
  `global` (the original single-tolerance rule, still the default), `per-stratum` (EVERY stratum
  within its own tolerance; `VERIFY_STRATUM_TOLERANCES="<stratum>=<tol> ..."` overrides cells),
  and `weighted` (confidence-weighted reject rate where a decided row weighs
  `1 + max(row_confidence, 0)` -- a reject on a row the automated signals rated confident counts
  more, because it means those signals mispredict).

Agreement statistics are unit-tested against hand-computed kappa fixtures; the adjudication
draw, each acceptance policy, and the reused-id adoption invariant are covered by synthetic
reviewed fixtures.

## Judge Calibration

Judge calibration is a separate human-rating problem. The code measures whether a local judge
tracks human ratings on the calibration split. The trust gate is Spearman rho `>= 0.6`; below that,
the judge remains diagnostic.

Modules:

- `src/llb/judge/calibration.py`: worksheet IO, Spearman rho, bootstrap CI, trust decision;
- `src/llb/judge/rate/`: command parsing, worksheet state, presentation, and the interactive rater;
- `src/llb/scoring/judge/model.py`: runtime trust gate and judge outcome policy;
- `src/llb/scoring/judge/scorer.py`: score normalization and empty-answer handling.

```bash
make calibration-run
make calibration-rate
make calibration-score
make run-eval JUDGE_RHO=<rho> JUDGE_MODEL=<model> JUDGE_BASE_URL=<url>
```

`calibration-run` pre-fills model answers and optional ungated judge ratings.
`calibration-rate` hides judge ratings by default so the human rating is independent.
It stores only human-owned worksheet columns, supports resume/review navigation, and exits without
editing when the start-fresh clear prompt is not confirmed. The rating session uses the same
parser/navigation/edit-helper shape as verification review.
`calibration-score` computes rho and confidence interval from the filled worksheet.

Tracked calibration worksheets live in `calibration/`. Generated worksheets for temporary corpora
live under `$DATA_DIR/llb/calibration/` unless deliberately promoted.
