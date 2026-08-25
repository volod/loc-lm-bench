# Corpus Conflict Resolution (corpus-conflict-resolution)

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

`llb resolve-corpus-conflicts` and the `make resolve-corpus-conflicts` alias turn an audit
`findings.jsonl` into `plan.json`, `conflict_overlay.json`, `resolution_review.jsonl`, and
`effect.md`. The implementation is split across `src/llb/conflicts/resolution/policy.py`,
`resolution/io.py`, `resolution/review.py` (the reviewer's ledger and returned decisions),
`overlay.py`, `grouping/artifact.py`, and `resolution/effect.py`; Typer wiring lives in
`src/llb/cli/prep/conflict_resolution.py`.

The policy is deliberately narrower than the detector:

- hash, lexical, and claim-adjudicated duplicates may use `drop_duplicate`;
- `prefer-newer` may suppress an older `superseded_by` claim only when the recorded governance
  pair orders the editions;
- contradictions, unknown relations, conservative supersession, and every semantic-tier
  duplicate candidate become `escalate` records;
- complementary and subsumption findings remain `keep_both` annotations.

The semantic guard is important: the semantic tier is a recall-oriented candidate generator, not
deletion authority. Its rank-selected goods candidates coexist with the claim-level finding that
the corpus has no confirmed cross-document duplicate. Automatic suppression at that tier would
convert similarity rank into destructive policy.

## Decision groups in the plan and the review ledger

A plan that lists one escalation per ROW asks for the review the audit report was changed to stop
asking for: six rows quoting one stale chunk are one call. `plan.json` (schema 3) therefore carries
both units.

- **`items`** is unchanged -- one per finding row, and still the only thing the overlay and its
  rollback are built from. Each item now names its `group_id`.
- **`decisions`** is one entry per [decision
  group](conflict-decision-groups.md#the-count-and-the-units-behind-it): `rows`, `finding_ids`,
  `relations`, `documents`, `shared_units`, the `actions` its members resolved to, `decide_rows`,
  `review_rows`, and a `status`. `action` is the action every member agreed on and is **null** when
  they did not, so a mixed group reads as mixed. A decision never authorizes what no member row
  already authorized -- it is a view over the per-row policy, not a second policy.
- **Both counts of the work, in the one artifact that holds both.** `decide_rows` is the audit's
  relation-based count restated here; `review_rows` is this policy's count of rows still needing a
  human. They differ in both directions, so the plan is where they are reconciled rather than
  compared across two terminals -- see [to decide and to
  review](conflict-decision-groups.md#to-decide-and-to-review-are-two-counts-never-one). Schema 3 is
  exactly that addition; schema 2 added `decisions` beside `items`.
- **`resolution_review.jsonl`** keeps one record per open row -- a drop applies to one span, so the
  row stays the unit a reviewer decides on -- but every record carries `group_id`, `group_rows`,
  `group_decide_rows`, and `group_review_rows`. The records are ordered by **to review** first, then
  by group id, so a group still reads as one contiguous block while the block that costs the most
  human time leads the file. This is the one artifact ranked on the count an operator funds; the
  audit report cannot be, because it has no policy. The review TUI titles each record
  `(G1, 1 of 51 rows sharing this decision, 12 to review)`, dropping the second clause when every
  row in the group is open.
- **A whole-group decision** is a ledger row carrying `group_id` and `resolution_decision` with no
  `finding_id`; it settles every member. It may only be `keep_both`. A group-wide `drop_a` /
  `drop_b` is REFUSED with the group named, because members of one group share a unit but not a
  document pair, so "drop a" means a different act on each of them and would suppress spans no
  reviewer looked at.

The planner derives the grouping from the rows it reads, so an audit bundle written before
`groups.json` existed still plans by decision; `tests/llb/conflicts/test_decision_groups.py` asserts
the derived groups equal the sidecar's.

Measured 2026-08-12 on the RTX PRO 3000 Blackwell 12 GiB CUDA host over the 5-document goods
corpus, semantic tier at `MAX_CANDIDATE_PAIRS=100`, cosine threshold 0.360 calibrated as the 0.9982
quantile of a 55,865-pair exhaustive null, 954 chunks of which 898 were comparable:
**100 rows in 6 decision groups, largest 51**
-- all 100 escalate, as the semantic tier has no deletion authority, so a reviewer previously faced
100 undifferentiated records. Six group-wide `keep_both` rows settle all of them
(`action_counts: {keep_both: 100}`, every decision `accepted`, no suppression in the overlay), which
is the review the corpus actually requires. Reading: grouping turns a 100-record queue into a
6-decision one -- a ~94% cut in what a reviewer must open -- and it costs no authority here because
all 100 findings are `duplicate` relations over 79 chunk units and 6 document pairs, so one unit
really is one decision. What would overturn it: a corpus whose findings span several relations,
where one group can hold rows a reviewer would settle differently; the group-wide `drop_a`/`drop_b`
refusal exists because that case is not hypothetical. Lookup key: `corpus-conflicts` run
`groups-goods-semantic`.

The CLI summary names both counts on every run, so the number an operator carries out of the
terminal is labelled:

```text
[resolve-conflicts] 100 rows in 6 decision groups (largest 51 rows)
[resolve-conflicts] to decide (relation): 100 rows
[resolve-conflicts] to review (policy conservative): 100 rows in 6 decision groups
```

The same two lines on the budget-100 claim bundle read `to decide (relation): 1 row` and `to review
(policy conservative): none` -- the divergence in the other direction, with the audit's one
actionable row costing zero human decisions.

`review_rows` here is the MEASURED count. The audit can be asked to project the same number one
command earlier under a named policy (`--project-policy`), and the projection is required to equal
this plan's `review_rows` group for group -- see [projecting the review
count](conflict-decision-groups.md#projecting-the-review-count-one-command-earlier).

## Overlay and rollback contract

Applying a plan validates every document, offset, and exact quote against the current corpus, then
atomically installs `.llb/conflict_overlay.json` below the corpus root. A stale audit is rejected
before any directive is installed. Source `.md` and `.txt` bytes are never edited.

The overlay is a function of the finding SET, not of the order the rows were read in: each
document's annotations and suppress-spans are sorted by their own identity, so re-reading an audit
whose rows were merely re-sorted produces the same bytes and cannot republish a store generation
that changes nothing (the fingerprint folds the directive into the document). Only
`source_findings_sha256` differs, which is what it is for.

`chunk_corpus` consumes the control file. Whole-document duplicate directives omit that document;
claim-level directives omit chunks overlapping the accepted span; keep/escalate records add
`conflict_resolutions` metadata. `corpus_doc_fingerprints` folds each document's directive into
only that document's fingerprint. The existing `refresh_vector_store` path therefore publishes a
normal immutable generation and can reuse vectors when an overlay changes annotations only.
Removing the control file and running with `ROLLBACK=1` publishes the inverse generation and
restores the previous ranking.

The resolver can refresh and measure in one invocation:

```bash
make resolve-corpus-conflicts FINDINGS=<findings-jsonl> CORPUS=<corpus-dir> \
  POLICY=conservative APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>
```

Pass `BEFORE_RUN=<run-dir>` and `AFTER_RUN=<run-dir>` to add their manifest objective scores to
`effect.md`. Retrieval measurements persist in `effect.json`, so a later report update retains the
same recall/MRR comparison. Roll back with:

```bash
make resolve-corpus-conflicts ROLLBACK=1 CORPUS=<corpus-dir> \
  STORE=<store-dir> GOLDSET=<goldset-jsonl>
```

## CUDA-host resolution evidence

The goods quickstart evidence bundle is
the goods resolution run of 2026-07-20. It used the 1,139-chunk hybrid
multilingual-E5 store, the 19-item flat retrieval set, and the 20 human-accepted chain set with
MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M on the RTX 4060 Ti. The fixed history-policy objective run had
40 scored steps and reliability 1.000 before and after.

All 12 semantic candidates escalated to review; none received suppression authority. Applying the
annotation-only overlay reused all 1,139 vectors. Recall@10 stayed 0.8421, MRR stayed 0.5342, and
the verified final-chain objective stayed 0.6163 (all deltas 0.0000). One-command rollback and
re-apply each published a fresh generation and reproduced the same metrics exactly. The report
verdict is `REVERT` because 12 review decisions remain; unchanged metrics do not override an open
human gate. The failed endpoint probe under `before/` is excluded; the reliable baseline and
post-overlay manifests are under `before-valid/` and `after-valid/`.

Tests in `tests/llb/conflicts/test_resolution*.py` cover policy, semantic escalation, stale-source
rejection, artifact/CLI output, per-document fingerprint changes, source-byte preservation,
review-ledger decisions, all-keep ranking identity, rollback identity, objective loading, and
verdict gating. The current `make ci` passes with 1,687 tests, one skipped, and 42 slow tests
deselected.

## Large-corpus blocking evidence

On 2026-07-20, the 32-dimensional exact blocker was compared with `VectorSet.pairs_above` on both
real multilingual-E5 quickstart stores at cosine 0.9. Findings were byte-identical:

- HR: 2,578 chunks and 3,321,753 possible pairs; 4,642 reached full-space confirmation, so the
  projection pruned 99.8603%. The reused projected search took 0.150 s versus 0.073 s for the
  all-pairs matrix scan. The small corpus remains below the crossover where tree setup pays.
- Goods: 1,139 chunks and 648,091 possible pairs; 1,461 reached confirmation, so the projection
  pruned 99.7746%. Search took 0.116 s versus 0.007 s for the small all-pairs scan.

The required large run used 50,000 deterministic synthetic unit-vector chunks (64 source
dimensions, 32 projected dimensions) on the RTX 4060 Ti CUDA host. It covered 1,249,975,000 pairs,
sent 51 to confirmation, pruned 99.999996%, and returned the same zero matches as the actual
all-pairs baseline. The cold path (PCA fit, persisted-tree build, exact query, confirmation) took
9.306 s; the reusable search path took 7.571 s; the full blocked matrix baseline took 13.988 s.
This evidence includes construction cost rather than reporting query time alone.

Run the delivered path with:

```bash
make audit-corpus-conflicts CORPUS=<corpus-dir> STORE=<store-dir> \
  EFFORT=semantic PROJECT_DIMS=32
```
