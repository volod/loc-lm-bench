"""Per-candidate process isolation for the reranker bake-off.

Measured on the CUDA host: one candidate whose repository-supplied modelling code raises a
DEVICE-SIDE ASSERT poisons the CUDA context for the whole process, and every candidate loaded after
it fails with the same assert. Without isolation a roster of five reads as "four models do not run
on this host" when three of them run fine -- the bake-off would publish a host verdict that is an
artifact of the order the roster happened to be in.

So each candidate is loaded and scored in its OWN spawned process: the context dies with the child,
and the next candidate starts from a clean device. The parent keeps the lane exactly as it was --
the child is hidden behind the same `RerankScorer` callable -- and the VRAM baseline is read INSIDE
the child before its weights load, so a footprint is the candidate's own.

`spawn`, not `fork`: the parent has already initialized CUDA to retrieve the candidate pools, and
forking a process with a live CUDA context is unsupported.
"""

import logging
import multiprocessing as mp
import time
from multiprocessing.connection import Connection
from typing import Any

from llb.rag.rerank_bakeoff.families import resolve_convention
from llb.rag.rerank_bakeoff.models import LoadedScorer, ScorerLoadError, ScorerLoader

_LOG = logging.getLogger(__name__)

# A cold load pulls weights and compiles the first kernels; a scoring call is one batch. Generous
# enough for a cold 0.6B decoder on a laptop GPU, short enough that a hung child ends the run.
DEFAULT_LOAD_TIMEOUT_S = 900.0
DEFAULT_CALL_TIMEOUT_S = 300.0

_READY = "ready"
_SCORES = "scores"
_ERROR = "error"
_VRAM = "vram"
_STOP = "stop"


def _child_vram_mb(baseline: int | None) -> float | None:
    """Used VRAM above the child's own pre-load baseline (None when telemetry is unavailable)."""
    if baseline is None:
        return None
    try:
        from llb.executor.vram import nvml_reader

        return float(max(nvml_reader()() - baseline, 0))
    except (Exception, SystemExit):
        return None


def _read_baseline() -> int | None:
    try:
        from llb.executor.vram import nvml_reader

        return nvml_reader()()
    except (Exception, SystemExit):
        return None


def _serve(conn: Connection, model: str, options: dict[str, Any]) -> None:
    """Child entry point: load one candidate, then answer scoring calls until told to stop."""
    from llb.rag.rerank import CrossEncoderReranker

    baseline = _read_baseline()
    try:
        scorer = CrossEncoderReranker(
            model,
            device=options["device"],
            trust_remote_code=options["trust_remote_code"],
            model_kwargs={"torch_dtype": options["dtype"]},
            batch_size=options["batch_size"],
        )
        started = time.perf_counter()
        scorer("warmup", ["warmup passage"])
        load_seconds = time.perf_counter() - started
    except BaseException as exc:  # a device-side assert is not an Exception subclass everywhere
        conn.send((_ERROR, f"{type(exc).__name__}: {exc}"))
        return
    device = str(getattr(getattr(scorer, "_model", None), "device", None) or options["device"])
    conn.send((_READY, load_seconds, _child_vram_mb(baseline), device))
    while True:
        message = conn.recv()
        if message[0] == _STOP:
            return
        if message[0] == _VRAM:
            conn.send((_VRAM, _child_vram_mb(baseline)))
            continue
        try:
            conn.send((_SCORES, scorer(message[1], message[2])))
        except BaseException as exc:
            conn.send((_ERROR, f"{type(exc).__name__}: {exc}"))
            return


class _WorkerScorer:
    """A `RerankScorer` whose model lives in another process (see the module docstring)."""

    def __init__(self, process: "mp.process.BaseProcess", conn: Connection, model: str):
        self._process = process
        self._conn = conn
        self._model = model

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores: list[float] = self._request((_SCORES, question, texts), DEFAULT_CALL_TIMEOUT_S)
        return scores

    def read_vram(self) -> float | None:
        try:
            footprint: float | None = self._request((_VRAM,), DEFAULT_CALL_TIMEOUT_S)
            return footprint
        except ScorerLoadError:
            return None

    def _request(self, message: tuple[Any, ...], timeout: float) -> Any:
        """Send one message and return the child's payload, or raise what went wrong there."""
        if not self._process.is_alive():
            raise ScorerLoadError(f"{self._model}: the scoring process exited")
        try:
            self._conn.send(message)
            if not self._conn.poll(timeout):
                raise ScorerLoadError(f"{self._model}: no answer within {timeout:.0f}s")
            kind, *payload = self._conn.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            raise ScorerLoadError(f"{self._model}: the scoring process died ({exc})") from exc
        if kind == _ERROR:
            raise ScorerLoadError(f"{self._model}: {payload[0]}")
        return payload[0]

    def release(self) -> None:
        """Stop the child and wait for it, so its VRAM is gone before the next candidate loads."""
        try:
            if self._process.is_alive():
                self._conn.send((_STOP,))
        except (OSError, BrokenPipeError):
            pass
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._conn.close()


def isolated_loader(
    *,
    device: str | None = None,
    batch_size: int,
    dtype: str,
    load_timeout_s: float = DEFAULT_LOAD_TIMEOUT_S,
) -> ScorerLoader:
    """Bind a `ScorerLoader` that runs each candidate in its own process.

    The handshake IS the load: the child answers only after its weights are resident and it has
    scored one warmup pair, so a candidate that cannot run on this host raises `ScorerLoadError`
    here -- with the child's real error text -- and the lane records it as a skipped row.
    """

    def load(model: str) -> LoadedScorer:
        context = mp.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        options = {
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "trust_remote_code": resolve_convention(model).trust_remote_code,
        }
        process = context.Process(target=_serve, args=(child_conn, model, options), daemon=True)
        process.start()
        child_conn.close()  # the parent keeps only its own end
        if not parent_conn.poll(load_timeout_s):
            process.terminate()
            raise ScorerLoadError(f"{model}: did not load within {load_timeout_s:.0f}s")
        kind, *payload = parent_conn.recv()
        if kind == _ERROR:
            process.join(timeout=30)
            raise ScorerLoadError(f"{model}: {payload[0]}")
        load_seconds, vram_mb, resolved_device = payload
        _LOG.info(
            "[compare-rerankers] loaded %s in %.1fs on %s (isolated)",
            model,
            load_seconds,
            resolved_device,
        )
        scorer = _WorkerScorer(process, parent_conn, model)
        return LoadedScorer(
            scorer=scorer,
            device=resolved_device,
            load_seconds=load_seconds,
            vram_mb=vram_mb,
            read_vram=scorer.read_vram,
            release=scorer.release,
        )

    return load
