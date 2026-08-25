# Product Decisions

This page records current decisions that affect implementation shape. Keep rationale here when it
helps future agents avoid re-litigating a settled tradeoff.

## Local Judge

The benchmark uses a local OpenAI-compatible judge by default. The reason is no corpus egress and
reproducibility. The tradeoff is family bias when the judge shares architecture, tokenizer, or
pretraining lineage with candidate models.

Mitigations:

- the judge enters ranking only when calibration rho clears the trust threshold;
- objective correctness remains available and ranks alone when the judge is demoted;
- manifests disclose the judge model and bias note;
- boards reject incompatible judge cohorts.

## Data Egress

Default corpus processing is local. Frontier or Litellm calls are opt-in tools, not the default
path for private material.

Current policy:

- real chat-log corpora use local drafting or verification only;
- real text-analysis corpora may use frontier cross-check when the operator explicitly approves it;
- synthetic bundles may use the configured Litellm path;
- every drafted bundle still needs human verification before headline scoring;
- frontier *scoring* (the `scorer_policy=frontier` lane on `run-eval`) is a separate opt-in: one
  upfront egress consent plus a hard per-run USD and/or call budget enforced by the scorer cost
  ledger under `$DATA_DIR/<method>/<run>/scorer/`. Over-cap aborts are resumable and never silent.
  See [evaluation rigor](rigor-board-judge/judging.md#scorer-policy-seam).

## Closed Graph Ontology

GraphRAG uses the closed node vocabulary in `docs/design/graph-ontology-schema.md`. The closed set
keeps graph queries, node typing, and relation caps stable across model runs. Model-invented types
are normalized to the canonical vocabulary or `MISC`.

## Backend Scope

The serving backends are Ollama, vLLM, and llama.cpp. All three must stay behind the
OpenAI-compatible launcher seam. New backend-specific behavior belongs in launcher, resolver,
planner, telemetry, or preflight modules; it should not leak into scoring logic.

CUDA source builds use `scripts/shared/common.sh:llb_max_jobs()` for parallelism. Ordinary dependencies
use `uv` caches. `$DATA_DIR/wheels/` is only for intentional local-source wheel outputs with ABI
and git revision encoded in the directory name.

## Evaluation Tiers

Tier mixing is out of scope for a single board. Public screens, private RAG results, and each
category suite have separate metric semantics. Use side-by-side sections or explicit handoff
commands rather than one blended leaderboard.

## Corpus-Conflict Confidence

The semantic tier reports a **ranked candidate list, not a set of statistically significant
findings**, and the audit must not be presented as though it did. Threshold calibration measures
the corpus's own comparable cross-document pair similarities, a population that contains the very
duplicates it is meant to detect; enumerated exactly, the null and the observed set are identical,
so empirical FDR is 1.000 at every threshold and a budget of N returns exactly N pairs. Detail and
measurements: [data prep](data-prep/conflict-detection.md#known-limitation-there-is-no-independent-null).

The CUDA independent-null matrix tests cross-corpus controls, token and sentence permutation,
held-out document maxima, and labelled fixture calibration. None clears the fixture, HR recovery,
goods flood, independence, and resolved-tail gates together; the detailed matrix and artifact are
in [null research](data-prep/conflict-null-research.md#first-generation-negative-result).

The second-generation CUDA matrix adds surface/encoder-matched multi-reference controls, local
residual similarity, source-cluster empirical FDR, and traced argument/quantity/modality edits. It
also corrects tail accounting from repeated pair rows to unique source/reference units. None clears
exchangeability, effective-tail, fixture, HR, and goods gates together; counterfactual edits are not
eligible nulls without relation verification. The evidence and artifact are in [null
research](data-prep/conflict-null-research.md#second-generation-negative-result).

The third-generation CUDA matrix settles why, rather than adding candidates. The candidate list an
operator can afford implies a per-pair tail whose certification would need an independent control
bank about a thousand times larger than the corpus being audited; propensity balancing fails on
positivity, not on estimator choice; cosine-only mixture calibration is unidentifiable at that
resolution; whitening and anisotropy stripping move the shift without producing an operating point;
and the traced counterfactual controls are proven planted positives, not nulls. Measured claim-tier
precision, by contrast, works and is corpus-specific: it is 1.000 with a clustered lower bound of
1.000 on the HR list and 0.000 on the goods list, which is the same statement as "no single
threshold serves both corpora", now with a number. Evidence: [null
research](data-prep/conflict-null-research.md#third-generation-negative-result).

The fourth-generation CUDA matrix removes the last excuse for reopening it. Control claims generated
from the target corpus's own structure are nulls -- 43 of 44 cleared the relation verifier, against
0 of 93 for the traced edits -- and they largely repair positivity: weighted membership AUC falls
from 0.99998 to 0.676 on HR and from 0.99989 to 0.533 on goods. With construction solved, what
remains is arithmetic. A group-split conformal threshold, the sharpest estimator available and
distribution-free, certifies tail `alpha` at 95% confidence only from `log(0.05) / log(1 - alpha)`
independent units: 607,303 for the HR corpus's affordable operating point, which this host produces
at 230 verified claims per hour -- 110 days for one corpus at one candidate budget. Evidence:
[closing the independent-null question](data-prep/conflict-null-closure.md).

The product decision therefore stands, and one search is closed rather than paused: pursuing a
per-pair semantic false-positive rate at this corpus scale is not the direction. What would change
that answer, and which directions are proven dead, are recorded in [future
research](../future-research.md). Confidence in a
corpus conflict comes from the **claim tier's adjudication**, not from a cosine or a threshold, so
no autonomous gate should branch on the semantic tier's provisional `duplicate` verdict alone. No
report, doc, or CLI string may describe a semantic-tier cutoff as a false-positive rate,
significance level, or confidence -- name it a candidate budget or a rank cutoff. A precision figure
may be published only with its clustered bound and only from an adjudicator calibrated against
frozen labels.

## Context-Ablation Lanes Stay Diagnostic

`closed_book` and `long_context` (`RunConfig.context_strategy`) are measurement lanes, never
default retrieval policies and never leaderboard rows; `rag` remains the ranked lane. This is a
decision, not an omission, and it survives the measured result that `long_context` beat `rag` on
both scored roster models
([RAG core](rag-core/context-ablation-evidence.md)):

- `long_context` is **oracle-grounded**. It reads the item's own gold `doc_id`s, so it knows the
  answer's document for free. That is a legitimate ceiling to measure a retrieval layer against
  and an illegitimate thing to ship -- a real query arrives without a gold label.
- Its gap therefore sizes what chunking still loses, not what an operator would gain. Reading
  "+0.142 objective" as "stuff whole documents instead of retrieving" would be adopting the
  oracle, not the lane -- and the `retrieved_document` lane below has now MEASURED that: holding
  the retrieved set fixed and sending documents instead of chunks recovers none of the gap on
  either model.
- `closed_book` scores what the model already knows, which is a contamination and
  parametric-knowledge signal for the corpus, not a system configuration anyone would run.

The one number the ablation is entitled to change is interpretation of a leaderboard row: an
uplift that does not clear zero says the RAG stack is not earning its cost on that corpus, and a
high closed-book match rate says the item set is measuring memory as much as retrieval.

The per-question-type slices inherit that boundary and add one of their own. A slice reading says
where retrieval did or did not pay for itself on a KIND of question, decided on that slice's items
alone, so it is read as a pointer to the next measurement rather than as a statement about the
corpus -- the pooled reading stays the ablation's conclusion. `retrieved_document`'s adopt-or-reject
call is not taken per slice at all: it is the one decision here an operator acts on, and a dozen
items of one question type is not the evidence for it.

`retrieved_document` is the deliberate exception, and the reason the boundary above is a boundary
rather than a refusal to act. It sends whole documents like `long_context` does, but it picks them
from the RANKING -- top-N distinct documents off the retrieved chunk list -- so nothing in its path
knows the gold label and its score is reachable by an operator who changes a config value. It is
therefore a shippable configuration and carries an adopt-or-reject verdict of its own
([RAG core](rag-core/context-ablation.md#the-shippable-sibling-retrieved_document)), while `rag`
stays the leaderboard row unless a measured run on the operator's own corpus says otherwise. The
split it makes possible is the point: whatever `retrieved_document` does NOT capture of the
`long_context` gap was the oracle, and no configuration will ever recover it.

On the committed UA fixture that split is now measured, and the answer is do not adopt: at equal
retrieval depth the lane is -0.017 [-0.034, -0.004] objective on MamayLM 12B (a verbosity cost --
same found-rate, same span coverage) and +0.002 [-0.027, +0.035] on Lapa, while
`oracle_document_gap` stays separated above zero in all four runs
([RAG core](rag-core/context-ablation-evidence.md#the-shippable-document-lane-does-not-pay-reject-2026-08-23)).
The lane stays in the product as the measurement that keeps the boundary honest -- an operator on
another corpus runs it and gets their own verdict -- not as a recommended configuration.

## Agentic Framework Scope

The maintained agentic harness axis is `loop`, `langgraph`, and `crewai`. Additional frameworks
should not be added just to broaden a comparison table. Add a harness only when it changes a
meaningful operational question and can share the same task set, world, scoring, and judge gates.

## Security Guardrails Scope

loc-lm-bench measures model security behavior (`bench-security`, corpus-derived probes); it is a
benchmark, not a production RAG service. Runtime guardrails -- prompt-injection filtering of
retrieved content, output PII/secret filters, and identity-backed authorization -- are out of
scope and belong to the application embedding a recommended model. The benchmark-side governance
layer is limited to plain metadata tags, ACL-scoped retrieval, deletion propagation, stale-store
refusal, and immutable store-directory rollback; see [data prep](data-prep.md) and
[RAG core](rag-core.md). This resolves the corresponding items of the Ukrainian-RAG minimum
production checklist deliberately rather than silently.

## Public Leaderboard Scope

loc-lm-bench is a local/private benchmark. It can consume public Ukrainian benchmark results as
context, but it does not try to become a public hosted leaderboard.
