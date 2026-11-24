"""Pilot comparison of book-recovery pipelines.

Importing this package does not import a recognition library. Scoring reads saved
ledgers. The roster runner is the only piece that starts an engine, and it starts
one at a time.
"""

__all__ = ["LaneInput", "compare_submissions"]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from llb.prep.book.compare.assemble import LaneInput, compare_submissions

    return {"LaneInput": LaneInput, "compare_submissions": compare_submissions}[name]
