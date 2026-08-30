# Robotics RAG

Robotics RAG owns the boundary between offline episode evidence, live device state, model-authored
action proposals, and physical side effects. The protocol-neutral emulator capability is shipped:
its delivered seams are the pinned contracts, offline HFlow evidence bridge, deterministic action
gate and device emulator, and held-out paired operation benchmark. No current code operates real
hardware or claims Model Hardware Standard compatibility.

This page is the AREA INDEX. Each subject lives in its own page under [`robotics-rag/`](robotics-rag/).

| Page | What it answers |
| --- | --- |
| [Boundary contracts and upstream pins](robotics-rag/boundary-contracts.md) | Which records cross the robotics seam, which HFlow and MHS surfaces are pinned, how the offline fake is replayed, and why the current label is protocol-neutral |
| [HFlow robotics evidence bridge](robotics-rag/evidence-bridge.md) | How curated text reaches the existing corpus, which MCAP and generation facts survive, how draft/quarantine admission works, and how the pinned upstream exercise is replayed |
| [Action gate and device emulator](robotics-rag/action-gate-and-emulator.md) | How typed proposals are checked against fresh trusted state, policy, approval, locks, hard limits, faults, and no-retry semantics before one protocol-neutral adapter invocation |
| [Robotics RAG operation benchmark](robotics-rag/benchmark.md) | How the frozen emulator ledger compares retrieval on and off, which safety and adoption gates hold, and why the current model retains the no-retrieval baseline |
