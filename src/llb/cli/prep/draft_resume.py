"""Resume-state construction for ontology draft requests."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llb.cli.helpers import cli_error
from llb.cli.prep.draft_request import DraftRequest


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class DraftResumeBuilder:
    """Build a draft request from CLI values and authoritative journal metadata."""

    request: DraftRequest
    resume_dir: Path
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, request: DraftRequest) -> "DraftResumeBuilder":
        from llb.prep.ontology.pipeline.journaling import load_journal_meta

        if request.resume is None:
            raise ValueError("a resume directory is required")
        try:
            metadata = load_journal_meta(request.resume)
        except ValueError as exc:
            cli_error(str(exc))
        return cls(request=request, resume_dir=request.resume, metadata=metadata)

    def build(self) -> DraftRequest:
        extraction, drafting = self._stage_configs()
        request = replace(
            self.request,
            corpus_root=self.request.corpus_root or Path(str(self.metadata.get("corpus_root"))),
            out_dir=self.request.out_dir or self.resume_dir,
        )
        frontier_phases = self._frontier_phases(extraction, drafting)
        if frontier_phases:
            return replace(
                request,
                **self._frontier_values(frontier_phases, extraction, drafting),
            )
        return replace(request, **self._local_values(extraction))

    def _stage_configs(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        endpoint = _mapping(self.metadata.get("endpoint"))
        stages = _mapping(endpoint.get("stages"))
        return _mapping(stages.get("extraction")), _mapping(stages.get("drafting"))

    @staticmethod
    def _frontier_phases(extraction: Mapping[str, Any], drafting: Mapping[str, Any]) -> list[str]:
        stages = (("extraction", extraction), ("drafting", drafting))
        return [phase for phase, config in stages if config.get("kind") == "frontier"]

    def _frontier_values(
        self,
        phases: list[str],
        extraction: Mapping[str, Any],
        drafting: Mapping[str, Any],
    ) -> dict[str, Any]:
        frontier_stage = "both" if len(phases) == 2 else phases[0]
        frontier = extraction if extraction.get("kind") == "frontier" else drafting
        local = drafting if frontier_stage == "extraction" else extraction
        return {
            "model": self.request.model or str(frontier.get("model") or ""),
            "endpoint": "frontier",
            "frontier_stage": frontier_stage,
            "local_model": self.request.local_model or str(local.get("model") or "") or None,
            "backend": str(local.get("backend") or self.request.backend),
            "max_usd": self._optional_float("max_usd", frontier),
            "max_calls": self._optional_int("max_calls", frontier),
        }

    def _local_values(self, extraction: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model": self.request.model or str(extraction.get("model") or ""),
            "endpoint": "local",
            "backend": str(extraction.get("backend") or self.request.backend),
        }

    def _optional_float(self, key: str, config: Mapping[str, Any]) -> float | None:
        current = getattr(self.request, key)
        stored = config.get(key)
        return current if current is not None or stored is None else float(stored)

    def _optional_int(self, key: str, config: Mapping[str, Any]) -> int | None:
        current = getattr(self.request, key)
        stored = config.get(key)
        return current if current is not None or stored is None else int(stored)
