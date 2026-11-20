"""Install and locate the isolated survey environments, and the process environment they run in.

Each survey environment is a uv virtualenv under `$DATA_DIR/venvs/book-<env>/`, resolved with exact
top-level pins and `--exclude-newer` at the survey cutoff, and its full resolution is frozen next to
it (`llb-freeze.txt`) so a reader can see exactly what a smoke ran on. The shipped `.venv` is never
touched: these stacks each pin their own torch or ONNX Runtime build.

Every download a candidate makes -- uv wheels, Hugging Face and ModelScope weights, PaddleX models,
anything a tool writes under the XDG cache -- is pointed under `$DATA_DIR/cache/`, never the home
directory: a scanned-book stack pulls tens of gigabytes, and `rm -rf $DATA_DIR` must take it all.
"""

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from llb.prep.book.survey.candidates import ENVS, EXCLUDE_NEWER, EnvSpec

_LOG = logging.getLogger(__name__)

VENVS_DIRNAME = "venvs"
ENV_PREFIX = "book-"
FREEZE_NAME = "llb-freeze.txt"
CACHE_DIRNAME = "cache"
SURVEY_CACHE_DIRNAME = "book-survey"
UV_CACHE_DIRNAME = "uv"
INSTALL_TIMEOUT_S = 3600

# Every model cache a surveyed tool honours, under `$DATA_DIR/cache/book-survey/`.
_MODEL_CACHES = {
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "huggingface",
    "MODELSCOPE_CACHE": "modelscope",
    "PADDLE_PDX_CACHE_HOME": "paddlex",
    "TORCH_HOME": "torch",
    "MINERU_HOME": "mineru",  # MinerU ignores XDG and defaults to ~/.mineru
}
# Usage reporting off: the page under test may be a copyrighted scan.
_NO_TELEMETRY = {
    "MINERU_TELEMETRY_DISABLED": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "DISABLE_TELEMETRY": "1",
    # PaddleX otherwise probes every model hoster before each run.
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}


class SurveyEnvError(RuntimeError):
    """An environment cannot be installed or is not installed."""


@dataclass(frozen=True)
class InstalledEnv:
    name: str
    path: Path
    python: Path
    freeze: Path


def env_dir(data_dir: Path, name: str) -> Path:
    return data_dir / VENVS_DIRNAME / f"{ENV_PREFIX}{name}"


def installed_env(data_dir: Path, name: str) -> InstalledEnv | None:
    """The environment if its interpreter exists, else None."""
    path = env_dir(data_dir, name)
    python = path / "bin" / "python"
    if not python.exists():
        return None
    return InstalledEnv(name, path, python, path / FREEZE_NAME)


def uv_cache_dir(data_dir: Path) -> Path:
    """The uv cache survey installs use: an explicit `UV_CACHE_DIR`, else under `$DATA_DIR`."""
    explicit = os.environ.get("UV_CACHE_DIR")
    return Path(explicit) if explicit else data_dir / CACHE_DIRNAME / UV_CACHE_DIRNAME


def tool_environment(data_dir: Path, bin_dir: Path | None = None) -> dict[str, str]:
    """The process environment a smoke runs in: caches under `$DATA_DIR`, telemetry off."""
    env = dict(os.environ)
    cache = data_dir / CACHE_DIRNAME / SURVEY_CACHE_DIRNAME
    for variable, sub in _MODEL_CACHES.items():
        env[variable] = str(cache / sub)
        (cache / sub).mkdir(parents=True, exist_ok=True)
    env.update(_NO_TELEMETRY)
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    return env


def install_command(spec: EnvSpec, python: Path) -> list[str]:
    """The uv command that installs `spec` into the environment at `python`."""
    argv = ["uv", "pip", "install", "--python", str(python), "--exclude-newer", EXCLUDE_NEWER]
    for package in spec.undated:
        argv += ["--exclude-newer-package", f"{package}=false"]
    if spec.index:
        argv += ["--index", spec.index, "--index-strategy", "unsafe-best-match"]
    return [*argv, *spec.requirements]


def _run(argv: list[str], env: Mapping[str, str]) -> None:
    _LOG.info("[survey] %s", " ".join(argv))
    result = subprocess.run(argv, env=dict(env), timeout=INSTALL_TIMEOUT_S, check=False)
    if result.returncode != 0:
        raise SurveyEnvError(f"command failed ({result.returncode}): {' '.join(argv)}")


def install_env(data_dir: Path, name: str) -> InstalledEnv:
    """Create (or refresh) the named environment and freeze its resolution."""
    if name not in ENVS:
        raise SurveyEnvError(f"unknown survey environment {name!r}; known: {sorted(ENVS)}")
    if shutil.which("uv") is None:
        raise SurveyEnvError("uv not found -- install it from https://docs.astral.sh/uv/")
    spec = ENVS[name]
    path = env_dir(data_dir, name)
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(uv_cache_dir(data_dir))
    _run(["uv", "venv", "--allow-existing", "--python", spec.python, str(path)], env)
    python = path / "bin" / "python"
    _run(install_command(spec, python), env)
    frozen = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    freeze = path / FREEZE_NAME
    freeze.write_text(frozen, encoding="utf-8")
    return InstalledEnv(name, path, python, freeze)
