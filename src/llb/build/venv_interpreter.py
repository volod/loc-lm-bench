"""Which interpreter a venv was built from, and whether it is still that interpreter.

`pyvenv.cfg` records `version_info` when the environment is created and never again, so it is the
one place an OS python upgrade shows up: the file still says 3.13.14 while `home` now resolves to
3.13.15. uv reads exactly this to decide an environment is stale and must be REPLACED, which is why
`llb.build.venv_state` reads it too -- before the sync, rather than discovering the replacement
afterwards.

`restamp` is the cheap way out of that, and it is bounded by the ABI. Within one `major.minor` a
CPython patch release keeps the ABI tag (`cp313`) and the `lib/python3.13/site-packages` layout, so
the venv is ALREADY running the patched interpreter through its `bin/python` symlink and every
compiled wheel in it still loads -- only the recorded string is behind. Recording the truth there
keeps a hardware-matched CUDA stack that a rebuild would spend gigabytes reinstalling. A MINOR move
is a different environment (new stdlib, new site-packages path, unloadable extensions), so it is
refused and left to the rebuild.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger(__name__)

PYVENV_CFG = "pyvenv.cfg"
VERSION_KEY = "version_info"
HOME_KEY = "home"
BASE_EXECUTABLE_KEY = "base-executable"
# CPython writes `3.13.14.final.0` and uv writes `3.13.14`; only the release triple is comparable.
VERSION_PARTS = 3
VERSION_PROBE = "import sys;print('.'.join(str(part) for part in sys.version_info[:3]))"
PROBE_TIMEOUT_S = 30

RESTAMP_OK = 0
RESTAMP_REFUSED = 1
RESTAMP_UNKNOWN = 2


@dataclass(frozen=True)
class Interpreter:
    """The interpreter a venv points at: where it is, what was recorded, what it runs today.

    Any of the three can be missing, and each absence means something different: no `path` is an
    interpreter that was removed, no `recorded` is a `pyvenv.cfg` with nothing comparable in it,
    and no `current` is an interpreter that will not run.
    """

    path: Path | None
    recorded: tuple[int, ...] | None
    current: tuple[int, ...] | None

    @property
    def moved(self) -> bool:
        return (
            self.recorded is not None and self.current is not None and self.recorded != self.current
        )

    @property
    def patch_move(self) -> bool:
        """A move inside one `major.minor` -- the ABI held, so the venv can be restamped."""
        return self.moved and self.recorded[:2] == self.current[:2]  # type: ignore[index]


def read_config(venv_dir: Path) -> dict[str, str]:
    """Parse `pyvenv.cfg`'s `key = value` lines; a missing or unreadable file reads as empty."""
    try:
        text = (venv_dir / PYVENV_CFG).read_text(encoding="utf-8")
    except OSError:
        return {}
    config: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            config[key.strip()] = value.strip()
    return config


def version_triple(text: str) -> tuple[int, ...] | None:
    """The leading release triple of a version string, or None when it is not one."""
    parts = text.split(".")[:VERSION_PARTS]
    if len(parts) != VERSION_PARTS or not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def base_interpreter(config: dict[str, str], venv_dir: Path) -> Path | None:
    """The interpreter this venv was built from, or None when it is no longer there.

    uv records `home` (a bin directory) plus `version_info`; CPython's own `venv` also writes
    `base-executable`. Both are honored, and the venv's own `bin/python` is the last resort so a
    relocated or hand-built venv still reports a version instead of a missing interpreter.
    """
    candidates: list[Path] = []
    executable = config.get(BASE_EXECUTABLE_KEY)
    if executable:
        candidates.append(Path(executable))
    home = config.get(HOME_KEY)
    if home:
        recorded = version_triple(config.get(VERSION_KEY, ""))
        if recorded is not None:
            candidates.append(Path(home) / f"python{recorded[0]}.{recorded[1]}")
        candidates.extend([Path(home) / "python3", Path(home) / "python"])
    candidates.append(venv_dir / "bin" / "python")
    # `is_file` follows the symlink, so a venv pointing at a removed interpreter finds nothing.
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def interpreter_version(interpreter: Path) -> tuple[int, ...] | None:
    """Ask an interpreter for its own release triple; None when it cannot be run."""
    try:
        probe = subprocess.run(
            [str(interpreter), "-c", VERSION_PROBE],
            capture_output=True,
            text=True,
            check=True,
            timeout=PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return version_triple(probe.stdout.strip())


def read_interpreter(config: dict[str, str], venv_dir: Path) -> Interpreter:
    """Resolve the recorded and the running version in one place, so both callers agree."""
    path = base_interpreter(config, venv_dir)
    return Interpreter(
        path=path,
        recorded=version_triple(config.get(VERSION_KEY, "")),
        current=interpreter_version(path) if path is not None else None,
    )


def stamp(venv_dir: Path, version: tuple[int, ...]) -> None:
    """Rewrite only the `version_info` line, leaving every other key uv wrote exactly as it was."""
    path = venv_dir / PYVENV_CFG
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = [
        f"{VERSION_KEY} = {format_version(version)}"
        if line.partition("=")[0].strip() == VERSION_KEY
        else line
        for line in lines
    ]
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def restamp(venv_dir: Path) -> int:
    """Record the interpreter's current version when the ABI held; refuse a minor move."""
    interpreter = read_interpreter(read_config(venv_dir), venv_dir)
    if interpreter.recorded is None or interpreter.current is None:
        _LOG.error("[venv] cannot read %s and its interpreter under %s", PYVENV_CFG, venv_dir)
        return RESTAMP_UNKNOWN
    if not interpreter.moved:
        _LOG.info("[venv] %s already records %s", PYVENV_CFG, format_version(interpreter.recorded))
        return RESTAMP_OK
    if not interpreter.patch_move:
        _LOG.error(
            "[venv] refusing to restamp %s: python %s -> %s is a MINOR move, so the site-packages "
            "layout and every compiled extension in this venv are wrong -- rebuild it with "
            "`make venv RECREATE_VENV=1`",
            venv_dir,
            format_version(interpreter.recorded),
            format_version(interpreter.current),
        )
        return RESTAMP_REFUSED
    stamp(venv_dir, interpreter.current)
    _LOG.info(
        "[venv] %s now records python %s (was %s); the ABI held, so the installed stack stands",
        PYVENV_CFG,
        format_version(interpreter.current),
        format_version(interpreter.recorded),
    )
    return RESTAMP_OK
