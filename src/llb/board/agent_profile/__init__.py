"""One composed agent operating profile: model + backend, prompt system, adapter, context policy,
context order, retrieval knobs, and loop policy in a single artifact.

Every ingredient is already measured somewhere in this repo and nowhere together. Hand-assembling
them from five report sections invites the one failure a composed profile must not have: a value
that was never measured, or two values measured against different corpora, reading exactly like
values that were. So each field carries its evidence path, its lane's own verdict and uncertainty,
and its freshness; a field whose lane never ran is `unmeasured`; a field measured against a
different model, corpus, or store is refused; and a field resting on a stale adapter or a moved
store fingerprint is demoted with the changed knob named.

Submodules (import from the specific one you need -- there is no re-export surface):
`model` (field roster, states, records), `artifacts` (lane-root readers), `sources_rag` and
`sources_agent` (per-lane readings), `compose` (anchor, consistency guard, staleness demotion),
`render` (payload + rationale), and `replay` (the flags that reproduce the configuration).
"""
