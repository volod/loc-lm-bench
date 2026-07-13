"""Explain a finalized run's wrong answers (miss analysis).

After any run or sweep, `llb analyze-misses` classifies every miss of one run bundle into
exactly one class -- retrieval miss (gold span absent from the retrieved context), generation
miss (evidence present, answer wrong), refusal, format/scoring artifact, or judge disagreement
-- clusters the misses by document, topic, and question type, and emits ranked, evidence-backed
recommendations (raise or lower `top_k`, change chunking, add prompt-system dictionary terms,
try the named alternative model). Every recommendation line names its numeric evidence.

Classification is span-overlap based: it reads the additive per-case `retrieval.jsonl` record
the runner persists beside `scores.jsonl` (falling back to the scored `retrieval_hit` for
legacy bundles). Everything here is pure and file-driven -- no endpoint, GPU, or store -- so the
whole classifier is unit-testable over a synthetic scored bundle. The bounded probe mode that
re-runs the miss subset at alternative retrieval depths lives in `miss_probe.py`; run bundles
are never mutated.

Submodules (import from the specific one you need -- there is no re-export surface):
`model` (vocabulary + dataclasses), `load` (bundle reading), `classify` (classification +
clustering + `analyze_run`), `recommendations` (ranked advice), and `report` (Markdown + JSON
artifacts).
"""
