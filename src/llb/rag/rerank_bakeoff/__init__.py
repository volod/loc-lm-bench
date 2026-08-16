"""Cross-encoder reranker bake-off: rank rerankers on one accepted ledger at a fixed encoder.

The package is deliberately import-light at the top level: submodules are imported directly by the
lane, the CLI, and the tests (`families`, `models`, `roster`, `lane`, `loader`, `report`).
"""
