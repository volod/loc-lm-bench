"""Quality metrics for a scanned-book recovery.

``score_reference`` compares a reference transcription with a ``book-ocr`` run, with the existing
PDF lane's citation sidecar, or with both. The report keeps every metric, and it does not fold
the dangerous-substitution counts into the character error rate.
"""

from llb.prep.book.metrics.score import score_reference

__all__ = ["score_reference"]
