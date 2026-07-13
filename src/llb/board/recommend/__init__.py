"""Operator recommendation summary from final-split run bundles.

Turns the ranked leaderboard into a few plain-language picks an operator actually needs after a
sweep: the best RAG accuracy, the most efficient model for THIS host (quality per watt), the fastest,
and the model we recommend running here -- the highest-accuracy candidate that is feasible,
Pareto-optimal, and fits the GPU tier's VRAM budget with headroom. Selection is pure and testable;
host detection and chart rendering live behind injectable seams / a guarded matplotlib import.

Submodules (import from the specific one you need -- there is no re-export surface):
`model` (constants + dataclasses + formatting primitives), `build` (bundle loading + cohort
selection + `build_recommendation`), `render` (the summary markdown + payload + config-detail
table), and `sections` (the miss / self-improvement / fine-tune / context sections).
"""
