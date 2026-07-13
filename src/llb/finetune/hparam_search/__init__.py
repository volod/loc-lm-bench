"""Budgeted per-model LoRA hyperparameter search that never leaves the tuning split.

The split discipline of `optimize/tuner.py` is extended one level down: that tuner searches RAG and
serving knobs on the tuning split while `final` stays held out. Here the search space is the LoRA
configuration itself, and the held-out set is carved from *inside* the tuning split -- a seeded dev
sub-slice that no trial ever trains on. Calibration and final never enter a trial at all, and a
guard refuses a dataset that so much as names one of their item ids.

The Optuna conventions are the tuner's: a seeded `TPESampler`, a persistent SQLite study so a killed
search resumes, and pruned rather than crashed trials on a measured OOM. The trainer and the
objective are injectable, so CI runs a complete study over the fake trainer with a synthetic
objective and no CUDA.

Submodules (import from the specific one you need -- there is no re-export surface): `model`
(vocabulary + dataclasses), `dev_slice` (held-out carve), `space` (LoRA search space + guards +
footprint estimate), `objective` (per-trial fine-tune + score), `search` (study orchestration), and
`manifest_io` (read the recorded best back).
"""
