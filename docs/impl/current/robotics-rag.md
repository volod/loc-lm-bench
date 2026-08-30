# Robotics RAG

Robotics RAG owns the boundary between offline episode evidence, live device state, model-authored
action proposals, and physical side effects. The capability remains in development; its delivered
seams are the pinned protocol-neutral contracts, the offline HFlow evidence bridge, and the
deterministic action gate and device emulator. No current code operates real hardware or claims
Model Hardware Standard compatibility.

This page is the AREA INDEX. Each subject lives in its own page under [`robotics-rag/`](robotics-rag/).

| Page | What it answers |
| --- | --- |
| [Boundary contracts and upstream pins](robotics-rag/boundary-contracts.md) | Which records cross the robotics seam, which HFlow and MHS surfaces are pinned, how the offline fake is replayed, and why the current label is protocol-neutral |
| [HFlow robotics evidence bridge](robotics-rag/evidence-bridge.md) | How curated text reaches the existing corpus, which MCAP and generation facts survive, how draft/quarantine admission works, and how the pinned upstream exercise is replayed |
| [Action gate and device emulator](robotics-rag/action-gate-and-emulator.md) | How typed proposals are checked against fresh trusted state, policy, approval, locks, hard limits, faults, and no-retry semantics before one protocol-neutral adapter invocation |
