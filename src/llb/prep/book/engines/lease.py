"""Hold one engine's batch-scoped resources. A failed open is remembered until the batch ends.

Restarting a vLLM server per page spends longer loading than reading, so a stage opens an engine
once and closes it when the batch's pages are done. A server that refused to start fails the
remaining pages of the batch by the same named error, instead of paying the startup again.
"""

from llb.prep.book.engines.protocol import EngineAdapter, EngineRejected, EngineUnavailable


class EngineLease:
    """The engine currently held open for a batch, if any."""

    def __init__(self) -> None:
        self._current: EngineAdapter | None = None
        self._failed: dict[str, Exception] = {}

    def use(self, adapter: EngineAdapter) -> None:
        """Open `adapter` if it is not already the one held. Raise its remembered failure."""
        remembered = self._failed.get(adapter.engine_id)
        if remembered is not None:
            raise remembered
        if self._current is adapter:
            return
        self.close()
        try:
            adapter.open()
        except (EngineUnavailable, EngineRejected) as exc:
            self._failed[adapter.engine_id] = exc
            raise
        self._current = adapter

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None

    def reset(self) -> None:
        """Drop the remembered failure so the next batch may try the engine again."""
        self.close()
        self._failed.clear()
