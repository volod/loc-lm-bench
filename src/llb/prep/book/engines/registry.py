"""Name to adapter. The stage seam looks engines up here and nowhere else."""

from collections.abc import Iterable

from llb.prep.book.engines.protocol import EngineAdapter, EngineUnavailable


class AdapterRegistry:
    """The engines one run may name."""

    def __init__(self, adapters: Iterable[EngineAdapter]) -> None:
        self._by_id: dict[str, EngineAdapter] = {}
        for adapter in adapters:
            if adapter.engine_id in self._by_id:
                raise ValueError(f"engine {adapter.engine_id!r} registered twice")
            self._by_id[adapter.engine_id] = adapter

    def require(self, engine_id: str) -> EngineAdapter:
        """The adapter, or `EngineUnavailable` when the id is not registered."""
        try:
            return self._by_id[engine_id]
        except KeyError:
            known = ", ".join(sorted(self._by_id))
            raise EngineUnavailable(engine_id, f"unknown engine; known: {known}") from None

    def all(self) -> list[EngineAdapter]:
        return [self._by_id[key] for key in sorted(self._by_id)]


def default_adapter_registry() -> AdapterRegistry:
    """The fake engine plus every shortlisted adapter."""
    from llb.prep.book.engines.fake import FakeEngine
    from llb.prep.book.engines.shortlist import shortlist_adapters

    return AdapterRegistry((FakeEngine(), *shortlist_adapters()))


def describe_engines(registry: AdapterRegistry | None = None) -> list[str]:
    """One line per engine: id, the stages it serves, and ready or the named refusal."""
    lines = []
    for adapter in (registry or default_adapter_registry()).all():
        try:
            adapter.available()
        except EngineUnavailable as exc:
            state = f"refused: {exc.reason}"
        else:
            state = "ready"
        stages = ",".join(adapter.stages)
        lines.append(f"{adapter.engine_id}\t{stages}\t{state}")
    return lines
