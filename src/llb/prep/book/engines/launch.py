"""Run a surveyed engine in its own environment, or read a page from a server it needs.

Nothing here imports an engine library into this process. A missing environment, binary, or
daemon is `EngineUnavailable`. A run that exits badly and writes nothing, or that silently falls
back to a weaker mode, is `EngineRejected` -- a segfault after a complete output file is not.
"""

import re
import subprocess
from pathlib import Path

from llb.prep.book.engines.protocol import EngineRejected, EngineUnavailable
from llb.prep.book.survey.candidates import Candidate
from llb.prep.book.survey.envs import tool_environment
from llb.prep.book.survey.ollama_smoke import read_page
from llb.prep.book.survey.smoke import _availability
from llb.prep.book.survey.vllm_server import ServerError, chat_content

RUNNERS = Path(__file__).resolve().parent / "runners"
LOG_NAME = "engine.log"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def require_available(candidate: Candidate, data_dir: Path) -> dict[str, str]:
    """Placeholder values for the engine's command, or a named refusal when it cannot run."""
    problem, values = _availability(candidate, data_dir)
    resolved = dict(values)
    resolved["runners"] = str(RUNNERS)
    if problem:
        raise EngineUnavailable(candidate.id, problem)
    return resolved


def run_command(
    engine: str,
    argv: tuple[str, ...],
    values: dict[str, str],
    work_dir: Path,
    data_dir: Path,
    timeout_s: float,
    extra_env: tuple[tuple[str, str], ...],
    fail_on: str | None,
) -> int:
    """Run the engine. Return its exit code when it wrote something, else refuse."""
    command = [part.format(**values) for part in argv]
    bin_dir = Path(values["bin"]) if values.get("bin") else None
    env = tool_environment(data_dir, bin_dir)
    env.update({key: template.format(**values) for key, template in extra_env})
    log_path = work_dir / LOG_NAME
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=work_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        raise EngineRejected(engine, f"timed out after {timeout_s:.0f}s") from None
    _refuse_fallback(engine, log_path, fail_on)
    return completed.returncode


def read_served(candidate: Candidate, base: str, image: Path) -> str:
    """One chat request against a server this process already started."""
    smoke = candidate.smoke
    if smoke is None or not smoke.model or not smoke.prompt:
        raise EngineRejected(candidate.id, "served engine has no model or prompt")
    try:
        return chat_content(
            base, smoke.model, smoke.prompt, image, smoke.timeout_s, smoke.max_image_side
        )
    except ServerError as exc:
        raise EngineRejected(candidate.id, str(exc)) from None


def read_ollama(candidate: Candidate, image: Path) -> str:
    """One Ollama page read. An empty or failed reading is a rejection, not a blank page."""
    smoke = candidate.smoke
    if smoke is None or not smoke.model or not smoke.prompt:
        raise EngineRejected(candidate.id, "ollama engine has no model or prompt")
    reading = read_page(smoke.model, smoke.prompt, image, smoke.timeout_s, None)
    if reading.error or not reading.text.strip():
        raise EngineRejected(candidate.id, reading.error or "empty response")
    return reading.text


def log_tail(work_dir: Path, limit: int = 400) -> str:
    """The last part of an engine log, with terminal color codes removed."""
    path = work_dir / LOG_NAME
    if not path.is_file():
        return ""
    text = _ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[-limit:]


def _refuse_fallback(engine: str, log_path: Path, fail_on: str | None) -> None:
    if not fail_on or not log_path.is_file():
        return
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if fail_on in log:
        raise EngineRejected(engine, f"refused a silent fallback ({fail_on})")
