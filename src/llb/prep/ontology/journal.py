"""Per-document, per-window extraction journal for resumable ontology drafting.

The extraction stage (stage 2) is the expensive, multi-hour part of a full-corpus draft: one
model call per extraction window. This journal appends one line per COMPLETED window so an
interrupted run can resume the extraction stage instead of re-spending those calls. Everything
after extraction (induce -> sample -> draft -> refine -> emit) is deterministic given the merged
extractions and the seed, so replaying it produces the same bundle.

Keying is `(doc_id, window_index)`: `split_document` is deterministic, so window `i` of a document
is always the same text as long as the extraction settings are unchanged -- which the meta sidecar
(`extraction_journal.meta.json`) pins and `--resume` reads back. A journaled `DocExtraction` is
already grounded against the FULL original text, so replaying it needs no re-grounding.

A window that raised inside the adapter (swallowed transport error -> empty extraction) IS
journaled: that window is done-as-empty, matching the non-resumed run which would have written the
same empty result. A hard interruption (process kill / KeyboardInterrupt) propagates and simply
leaves the window un-journaled, so resume re-runs it.
"""

import json
import logging
import threading
from pathlib import Path

from llb.prep.ontology.models import DocExtraction

_LOG = logging.getLogger(__name__)


class ExtractionJournal:
    """Append-only journal of completed extraction windows, keyed by `(doc_id, window_index)`.

    Thread-safe: the parallel-window path records from worker threads, so both the in-memory index
    and the append are guarded by a lock. Ordering is irrelevant on disk -- the adapter reassembles
    windows by index -- so appends can interleave freely.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._done: dict[tuple[str, int], DocExtraction] = {}
        self._lock = threading.Lock()

    def load(self) -> int:
        """Load an existing journal (idempotent). Returns the number of journaled windows.

        A malformed line is skipped with a warning rather than aborting the resume: a truncated
        final line is the expected shape of a killed run.
        """
        if not self.path.is_file():
            return 0
        loaded = 0
        with self.path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    doc_id = str(record["doc_id"])
                    window_index = int(record["window_index"])
                    extraction = DocExtraction.model_validate(record["extraction"])
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    _LOG.warning(
                        "[ontology] skipping malformed extraction-journal line %s:%d (%s)",
                        self.path,
                        line_no,
                        exc,
                    )
                    continue
                self._done[(doc_id, window_index)] = extraction
                loaded += 1
        if loaded:
            _LOG.info(
                "[ontology] resuming: %d journaled extraction windows in %s", loaded, self.path
            )
        return loaded

    def get(self, doc_id: str, window_index: int) -> DocExtraction | None:
        """Return the journaled extraction for a window, or None when it must be re-extracted."""
        return self._done.get((doc_id, window_index))

    def record(
        self, doc_id: str, window_index: int, window_total: int, extraction: DocExtraction
    ) -> None:
        """Append one completed window and index it in memory. A window already present is not
        re-appended, so a rerun over the same journal stays append-idempotent."""
        key = (doc_id, window_index)
        with self._lock:
            if key in self._done:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "doc_id": doc_id,
                "window_index": window_index,
                "window_total": window_total,
                "extraction": extraction.model_dump(),
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._done[key] = extraction
