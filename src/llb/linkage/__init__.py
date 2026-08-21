"""Probabilistic record linkage: one seam for every identity decision the project makes.

Answers "do these two records denote the same thing" and nothing else. Match probabilities are
never a contradiction verdict, and a proposed cluster never rewrites the table it was read from --
see the confidence contract in `docs/design/spec.md#entity-resolution-and-record-linkage`.
"""
