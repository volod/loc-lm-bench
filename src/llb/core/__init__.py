"""Foundational, cross-cutting modules shared across the whole llb vertical.

These carry no domain logic of their own -- run configuration (`config`), typed cross-boundary
records (`contracts`), environment-variable names (`env`), project path resolution (`paths`),
safe filesystem helpers (`fsutil`), and the shared CLI runtime (`runtime`). Everything else in
the package builds on top of them, so they live together here to keep the package root clean.
"""
