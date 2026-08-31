# Robotics RAG

Robotics RAG owns the boundary between offline episode evidence, live device state, model-authored
action proposals, and physical side effects. Its shipped, protocol-neutral path is deliberately
layered: pin the external contracts, admit evidence into the normal corpus, evaluate proposals
through the deterministic action gate and emulator, then compare retrieval on and off over one
held-out operation ledger. No current code operates real hardware or claims Model Hardware Standard
compatibility.

This page is the AREA INDEX. Each subject lives in its own page under [`robotics-rag/`](robotics-rag/).

| Stage | Page | What it answers |
| --- | --- | --- |
| 1. Pin | [Boundary contracts and upstream pins](robotics-rag/boundary-contracts.md) | Which records cross the robotics seam, which HFlow and MHS surfaces are pinned, how the offline fake is replayed, and why the current label is protocol-neutral |
| 2. Admit | [HFlow robotics evidence bridge](robotics-rag/evidence-bridge.md) | How curated text reaches the existing corpus, which MCAP and generation facts survive, how draft/quarantine admission works, and how the pinned upstream exercise is replayed |
| 3. Contain | [Action gate and device emulator](robotics-rag/action-gate-and-emulator.md) | How typed proposals are checked against fresh trusted state, policy, approval, locks, hard limits, faults, and no-retry semantics before one protocol-neutral adapter invocation |
| 4. Compare | [Robotics RAG operation benchmark](robotics-rag/benchmark.md) | How the frozen emulator ledger compares retrieval on and off, which safety and adoption gates hold, and why the current model retains the no-retrieval baseline |

The next boundary is explicit rather than implied: MHS preview conformance and a supervised device
canary remain in the
[forward plan](../plan.md#robotics-rag-and-hardware-operation----robotics-rag-operation).
