# Design

The current design lives in [spec.md](spec.md). It explains why the benchmark validates data,
retrieval, and hardware fit before ranking models, and how those constraints shape the architecture.

## Contents

- [Purpose](spec.md#purpose) -- the local decision problem and why public rankings do not transfer
  directly.
- [Design intuition](spec.md#design-intuition) -- the trust chain from corpus evidence to a model
  recommendation.
- [Scope](spec.md#scope) -- supported workloads and explicit non-goals.
- [Architecture](spec.md#architecture) -- ownership across data prep, retrieval, execution, scoring,
  persistence, and analysis.
- [Data and ground truth](spec.md#data-and-ground-truth) -- source-span labels and human gates.
- [Retrieval before generation](spec.md#retrieval-before-generation) -- isolating evidence delivery
  from answer generation.
- [Backend and hardware boundary](spec.md#backend-and-hardware-boundary) -- backend-neutral
  evaluation with host-specific serving plans.
- [Scoring policy](spec.md#scoring-policy) -- objective metrics, calibrated judge diagnostics, and
  resource measurements.
- [Optimization without leakage](spec.md#optimization-without-leakage) -- tuning/final separation.
- [Persistence and reproducibility](spec.md#persistence-and-reproducibility) -- canonical run
  bundles and strict board admission.
- [Success criteria](spec.md#success-criteria) -- what a defensible selection workflow must prove.

The ontology graph schema is defined separately in
[graph-ontology-schema.md](graph-ontology-schema.md).

Current implementation detail is indexed in [../impl/current.md](../impl/current.md). Forward work
is tracked in [../impl/plan.md](../impl/plan.md).
