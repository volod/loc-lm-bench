"""Server-log lifecycle for launchers that run their backend as a subprocess.

A launcher writes its server log into the run's staging dir (`.<run>.tmp/vllm/vllm-<port>.log`),
which is removed as soon as the run fails. Inside a screen cell or a tuning trial that staging dir
is the ONLY copy: the cell is one of hundreds, nobody reproduces it by re-running the command
standalone, and the traceback names a path that stopped existing before anyone read it -- so the
one artifact that says WHY the engine exited is gone, and Optuna records a failed trial with no
cause. A FAILED launch therefore copies its log somewhere that outlives the staging dir, and the
error it raises names THAT path.

Successful launches keep nothing: the staging dir deleting a healthy cell's log is the temp dir
working as intended.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# Persistent per-host log root, a sibling of the run dirs rather than a child of any one of them,
# so a preserved log survives the staging dir AND the run dir the cell would have published.
LOG_DIR_PARTS = ("llb", "logs")
FAILED_PREFIX = "failed-"


def failed_log_dir(data_dir: Path | str | None = None) -> Path:
    """`$DATA_DIR/llb/logs` -- where a failed launch's server log is kept."""
    from llb.core.paths import resolve_data_dir

    return resolve_data_dir(data_dir).joinpath(*LOG_DIR_PARTS)


def _free_path(dest_dir: Path, stem: str, stamp: str) -> Path:
    """`failed-<stem>-<stamp>.log`, suffixed when two launches fail in the same second."""
    candidate = dest_dir / f"{FAILED_PREFIX}{stem}-{stamp}.log"
    attempt = 2
    while candidate.exists():
        candidate = dest_dir / f"{FAILED_PREFIX}{stem}-{stamp}-{attempt}.log"
        attempt += 1
    return candidate


def preserve_log(src: Path | None, dest_dir: Path) -> Path | None:
    """Copy `src` into `dest_dir` under a timestamped name; return where it now lives.

    None when there is nothing to keep, or when the copy itself fails -- a lost log is reported
    as "not preserved" by the caller and must never mask the launch failure that produced it.
    """
    if src is None or not src.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _free_path(dest_dir, src.stem, stamp)
        dest.write_bytes(src.read_bytes())
    except OSError:
        return None
    return dest


class ServerLog:
    """Open / close / preserve the log of a subprocess-backed launcher.

    Mixed into the launchers that spawn a server (vLLM, llama.cpp); the launchers that talk to an
    already-running daemon (Ollama) own no log and do not mix it in.
    """

    log_dir: Path | None = None
    log_path: Path | None = None
    # Where a failed launch's log is copied to. None resolves to `$DATA_DIR/llb/logs` at failure
    # time; `run_eval` sets it from the config so the copy lands under the run's own data root.
    failed_log_dir: Path | None = None
    # The surviving copy of the CURRENT launch attempt's log, once one has been made.
    failed_log_path: Path | None = None
    _log_handle: TextIO | None = None

    def open_log(self, name: str) -> int | TextIO:
        """Open this launch attempt's log in `log_dir` (truncating), or DEVNULL when unset.

        Clears the preserved-log marker: a relaunch overwrites the log it is about to reuse, so
        each attempt preserves its own copy rather than reporting the previous attempt's.
        """
        self.failed_log_path = None
        if self.log_dir is None:
            return subprocess.DEVNULL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / name
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        return self._log_handle

    def close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def preserve_failed_log(self) -> Path | None:
        """Keep this attempt's log outside the staging dir and remember where.

        Idempotent within one launch attempt: the launcher preserves on the way out of a failed
        `start()`, and the runner asks again while tearing the staging dir down -- one copy.
        Call it only after `close_log()`, so the copy holds everything the server wrote.
        """
        if self.failed_log_path is None:
            self.failed_log_path = preserve_log(
                self.log_path, self.failed_log_dir or failed_log_dir()
            )
        return self.failed_log_path

    def _log_note(self) -> str:
        """The `" (startup log: ...)"` suffix an error carries, naming the READABLE path.

        Empty only when the launcher wrote no log at all (`log_dir` unset -- DEVNULL): an error
        must never name a log that is not there.
        """
        kept = self.preserve_failed_log()
        if kept is not None:
            return f" (startup log: {kept})"
        if self.log_path is not None:
            return f" (startup log {self.log_path} could not be preserved)"
        return ""

    def annotate_launch_failure(self, exc: BaseException) -> BaseException:
        """Point `exc` at the surviving log and return it, so `raise` keeps type and traceback.

        Used for anything raised out of a launch, including errors this launcher did not build
        itself (a missing executable, an interrupt): the message is extended in place when it has
        one, else the path is attached as a note.
        """
        note = self._log_note()
        if not note:
            return exc
        if exc.args and isinstance(exc.args[0], str):
            exc.args = (f"{exc.args[0]}{note}", *exc.args[1:])
        else:
            exc.add_note(note.strip())
        return exc
