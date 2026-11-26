"""One shortlisted engine behind the adapter protocol.

The survey candidate says how the engine is installed and smoked. This adapter runs that same
route, then normalises the output into proposed blocks. Figure prose is dropped by the normaliser.
"""

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from llb.core.paths import resolve_data_dir
from llb.prep.book.engines.kinds import fit_proposals
from llb.prep.book.engines.launch import (
    log_tail,
    read_ollama,
    read_served,
    require_available,
    run_command,
)
from llb.prep.book.engines.parse_common import output_present, parse_directory, parse_text
from llb.prep.book.engines.protocol import (
    EngineRead,
    EngineRejected,
    EngineRequest,
    EngineUnavailable,
    Proposed,
)
from llb.prep.book.survey.candidates import (
    CANDIDATES,
    ROUTE_OLLAMA,
    SMOKE_SERVED,
    Candidate,
    Smoke,
)
from llb.prep.book.survey.vllm_server import ServerError, running_server, sent_size

_RECOGNITION = ("layout", "ocr", "formula", "table")
_DOCLING = "docling"


def with_ocr_language(engine_id: str, argv: Sequence[str], language: str | None) -> list[str]:
    """Append Docling's OCR language flag. Every other engine keeps its argv."""
    if engine_id != _DOCLING or not language:
        return list(argv)
    return [*argv, "--ocr-lang", language]


_PROCESS = "process"
_SERVED = "served"
_OLLAMA = "ollama"


@dataclass(frozen=True)
class AdapterSpec:
    """Which survey candidate, which parser, and a command when the smoke's own is not enough."""

    engine_id: str
    parser: str
    argv: tuple[str, ...] | None = None


class CandidateAdapter:
    """A shortlisted engine. Heavy code stays in its environment or its server."""

    def __init__(self, spec: AdapterSpec) -> None:
        self.spec = spec
        self.engine_id = spec.engine_id
        self._candidate = _candidate(spec.engine_id)
        self.stages = tuple(stage for stage in self._candidate.stages if stage in _RECOGNITION)
        # One command returns the whole page. Later stages reuse that reading.
        self.shares_reading = True
        self._stack: ExitStack | None = None
        self._base: str | None = None

    def available(self) -> None:
        require_available(self._candidate, resolve_data_dir())

    def open(self) -> None:
        """Start the vLLM server this engine needs, if it needs one and none is running."""
        if not self._needs_server() or self._base is not None:
            return
        self.available()
        smoke = self._smoke()
        server = smoke.server
        if server is None:
            raise EngineUnavailable(self.engine_id, "survey candidate names no server")
        stack = ExitStack()
        log_path = resolve_data_dir() / "cache" / "book-ocr-engines" / f"{self.engine_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            base = stack.enter_context(
                running_server(server, f"book-{self.engine_id}", resolve_data_dir(), log_path)
            )
        except (ServerError, TimeoutError) as exc:
            stack.close()
            raise EngineRejected(self.engine_id, str(exc)) from None
        self._base = base
        self._stack = stack

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._base = None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        self.available()
        own_server = self._needs_server() and self._base is None
        if own_server:
            self.open()
        try:
            proposals = self._proposals(request)
        finally:
            if own_server:
                self.close()
        return EngineRead(
            self.engine_id, self._model(request.model), request.variant, stage, tuple(proposals)
        )

    def _proposals(self, request: EngineRequest) -> list[Proposed]:
        mode = self._mode()
        if mode == _SERVED:
            text = read_served(self._candidate, self._base or "", request.image)
            return self._from_text(request, text)
        if mode == _OLLAMA:
            return self._from_text(request, read_ollama(self._candidate, request.image))
        return self._from_process(request)

    def _from_process(self, request: EngineRequest) -> list[Proposed]:
        smoke = self._smoke()
        values = require_available(self._candidate, resolve_data_dir())
        values.update(
            image=str(request.image.resolve()),
            pdf=str(request.pdf.resolve()),
            out=str(request.work_dir.resolve()),
        )
        if self._base:
            values["server"] = self._base
        code = run_command(
            self.engine_id,
            tuple(
                with_ocr_language(self.engine_id, self.spec.argv or smoke.argv, request.language)
            ),
            values,
            request.work_dir,
            resolve_data_dir(),
            smoke.timeout_s,
            smoke.env,
            smoke.fail_on,
        )
        if not output_present(self.spec.parser, request.work_dir):
            raise EngineRejected(self.engine_id, _no_output(code, request.work_dir))
        raw = parse_directory(self.spec.parser, request.work_dir, request.width, request.height)
        return fit_proposals(raw, request.width, request.height, request.width, request.height)

    def _from_text(self, request: EngineRequest, text: str) -> list[Proposed]:
        smoke = self._candidate.smoke
        max_side = smoke.max_image_side if smoke is not None else None
        sent_w, sent_h = sent_size(request.width, request.height, max_side)
        raw = parse_text(self.spec.parser, text, sent_w, sent_h)
        return fit_proposals(raw, sent_w, sent_h, request.width, request.height)

    def _mode(self) -> str:
        if self._candidate.route == ROUTE_OLLAMA:
            return _OLLAMA
        smoke = self._candidate.smoke
        if smoke is not None and smoke.kind == SMOKE_SERVED:
            return _SERVED
        return _PROCESS

    def _needs_server(self) -> bool:
        smoke = self._candidate.smoke
        return smoke is not None and smoke.server is not None

    def _smoke(self) -> Smoke:
        smoke = self._candidate.smoke
        if smoke is None:
            raise EngineUnavailable(self.engine_id, "survey candidate has no run command")
        return smoke

    def _model(self, override: str | None) -> str | None:
        if override:
            return override
        smoke = self._candidate.smoke
        if smoke is not None and smoke.model:
            return smoke.model
        return self._candidate.version


def _candidate(engine_id: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.id == engine_id:
            return candidate
    raise EngineUnavailable(engine_id, "not a surveyed candidate")


def _no_output(code: int, work_dir: Path) -> str:
    if code == 0:
        reason = "engine exited 0 but wrote no output file"
    else:
        reason = f"exit {code} and no usable output"
    tail = log_tail(work_dir)
    if not tail:
        return reason
    return f"{reason}; log tail: {tail}"
