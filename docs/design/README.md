# Design

The current design lives in [spec.md](spec.md). It explains why the benchmark validates data,
retrieval, and hardware fit before ranking models, and how those constraints shape the architecture.

The spec is a LIVING document. A capability discovered while implementing is added to it through
[Extending this specification](spec.md#extending-this-specification), not refused for being outside
a previously written scope. The [capability registry](spec.md#capability-registry) is the index of
everything the product does, what evaluates it, and where it is implemented; its row order is the
implementation line that [../impl/plan.md](../impl/plan.md) follows. `make lint-spec-plan` fails the
build when the two disagree.

## Contents

- [Purpose](spec.md#purpose) -- the local decision problem and why public rankings do not transfer
  directly.
- [Design intuition](spec.md#design-intuition) -- the trust chain from corpus evidence to a model
  recommendation, which is also the build order.
- [Scope](spec.md#scope) -- supported workloads and explicit non-goals.
- [Benchmark tiers](spec.md#benchmark-tiers) -- why tier metrics stay separate.
- [Architecture](spec.md#architecture) -- ownership across data prep, retrieval, execution, scoring,
  persistence, and analysis.
- [Data and ground truth](spec.md#data-and-ground-truth) -- source-span labels and human gates.
- [Corpus provenance and acquisition boundary](spec.md#corpus-provenance-and-acquisition-boundary)
  -- what an acquired document must carry, why the seam is a projection rather than a format, and
  what this project refuses to do on a producer's behalf.
- [Corpus conflict and governance](spec.md#corpus-conflict-and-governance) -- contradiction and
  supersession auditing, and the confidence contract that bounds what it may claim.
- [Retrieval before generation](spec.md#retrieval-before-generation) -- isolating evidence delivery
  from answer generation.
- [Graph retrieval and ontology](spec.md#graph-retrieval-and-ontology) -- the graph lane and the
  human-signed axiom set above it.
- [Backend and hardware boundary](spec.md#backend-and-hardware-boundary) -- backend-neutral
  evaluation with host-specific serving plans.
- [Scoring policy](spec.md#scoring-policy) -- objective metrics, the answer contract, ablation
  lanes, and [judge admission](spec.md#judge-admission).
- [Optimization without leakage](spec.md#optimization-without-leakage) -- tuning/final separation.
- [Agentic and context-policy workloads](spec.md#agentic-and-context-policy-workloads) -- the
  harness axis and the published-number provenance rule.
- [Autonomous orchestration](spec.md#autonomous-orchestration) -- the unattended corpus-to-
  recommendation path and its bounds.
- [Operator review tooling](spec.md#operator-review-tooling) -- one workbench behind every human
  gate, and review cost as a measurement.
- [Persistence and reproducibility](spec.md#persistence-and-reproducibility) -- canonical run
  bundles, strict board admission, and the
  [reproducible environment](spec.md#reproducible-environment).
- [Data egress boundary](spec.md#data-egress-boundary) -- what may leave the host, and what the
  benchmark deliberately does not guard.
- [Capability registry](spec.md#capability-registry) -- every capability, its evaluation, its
  implementation, and the implementation line.
- [Extending this specification](spec.md#extending-this-specification) -- the six-step lifecycle for
  a capability discovered while building.
- [Specification and plan integrity](spec.md#specification-and-plan-integrity) -- the invariants
  `make lint-spec-plan` enforces, and what is deliberately not tracked.
- [Success criteria](spec.md#success-criteria) -- what a defensible selection workflow must prove.

The ontology graph schema is defined separately in
[graph-ontology-schema.md](graph-ontology-schema.md). Two data contracts with parties outside this
repository are also separate documents: the
[acquired-corpus projection](acquired-corpus-projection.md) an upstream acquisition service renders
into, and the [external-service draft contract](external-draft-contract.md) for artifacts drafted by
hand with a chat provider.

Current implementation detail is indexed in [../impl/current.md](../impl/current.md). Forward work
is tracked in [../impl/plan.md](../impl/plan.md).
