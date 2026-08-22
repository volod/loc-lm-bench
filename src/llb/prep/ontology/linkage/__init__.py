"""Gold-item duplicate suppression read as a record-linkage decision.

The shipped drop policy is a cosine constant over the question alone. This package expresses the
same decision as a comparison specification over the fields a gold item actually carries, fits it
through `llb.linkage`, and scores the fitted model BESIDE the constant -- it never changes which
items a drafting run drops.
"""
