# Corpus Conflict Resolution (corpus-conflict-resolution)

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

`llb resolve-corpus-conflicts` and the `make resolve-corpus-conflicts` alias turn an audit
`findings.jsonl` into `plan.json`, `conflict_overlay.json`, `resolution_review.jsonl`, and
`effect.md`. The implementation is split across `src/llb/conflicts/resolution_policy.py`,
`resolution_io.py`, `overlay.py`, and `resolution_effect.py`; Typer wiring lives in
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

## Overlay and rollback contract

Applying a plan validates every document, offset, and exact quote against the current corpus, then
atomically installs `.llb/conflict_overlay.json` below the corpus root. A stale audit is rejected
before any directive is installed. Source `.md` and `.txt` bytes are never edited.

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

The goods quickstart evidence bundle is under
`$DATA_DIR/corpus-conflicts/20260720T-resolution-goods/`. It used the 1,139-chunk hybrid
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
