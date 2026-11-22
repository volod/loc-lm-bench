"""Run a shortlisted dewarper in its own environment. This process never imports it."""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llb.core.paths import resolve_data_dir
from llb.prep.book.preprocess.constants import PAGE_DEWARP_TIMEOUT_S, UVDOC_TIMEOUT_S, VARIANT_IMAGE
from llb.prep.book.preprocess.images import read_grey, write_png
from llb.prep.book.survey.envs import InstalledEnv, installed_env, tool_environment

log = logging.getLogger(__name__)
_RUNNERS = Path(__file__).resolve().parent / "runners"
_LOG_TAIL = 400


@dataclass(frozen=True)
class Warp:
    """A dewarped image and the dense variant-to-original maps that built it."""

    image: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray


def run_page_dewarp(grey: np.ndarray, data_dir: Path | None = None) -> Warp | str:
    """page-dewarp in the light environment, or the reason it produced nothing usable."""
    return _run(grey, "light", "page_dewarp_runner.py", PAGE_DEWARP_TIMEOUT_S, data_dir)


def run_uvdoc(grey: np.ndarray, data_dir: Path | None = None) -> Warp | str:
    """UVDoc in the paddle environment, or the reason it produced nothing usable."""
    return _run(grey, "paddle", "uvdoc_runner.py", UVDOC_TIMEOUT_S, data_dir)


def _run(
    grey: np.ndarray, env_name: str, runner: str, timeout_s: float, data_dir: Path | None
) -> Warp | str:
    root = data_dir if data_dir is not None else resolve_data_dir()
    env = installed_env(root, env_name)
    if env is None:
        return f"{env_name} environment is not installed"
    with tempfile.TemporaryDirectory(prefix="llb-dewarp-") as tmp:
        work = Path(tmp)
        source = work / "in.png"
        write_png(source, grey)
        command = [str(env.python), "-I", str(_RUNNERS / runner), str(source), str(work)]
        failed = _invoke(command, work, root, env, timeout_s)
        if failed:
            return failed
        return _load(work)


def _invoke(
    command: list[str], work: Path, data_dir: Path, env: InstalledEnv, timeout_s: float
) -> str | None:
    log_path = work / "dewarp.log"
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=work,
                env=tool_environment(data_dir, env.path / "bin"),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout_s:.0f}s"
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL:].strip()
        log.info("dewarper exit %s: %s", completed.returncode, tail)
        return tail or f"exit {completed.returncode}"
    return None


def _load(work: Path) -> Warp | str:
    image_path = work / VARIANT_IMAGE
    maps = work / "maps.npz"
    if not image_path.is_file():
        return "dewarper wrote no image"
    if not maps.is_file():
        return "no point map; the page is not covered"
    image = read_grey(image_path)
    with np.load(maps) as loaded:
        map_x = np.asarray(loaded["map_x"], dtype=np.float32)
        map_y = np.asarray(loaded["map_y"], dtype=np.float32)
    if map_x.shape != image.shape or map_y.shape != image.shape:
        return "point map does not match the dewarped image"
    return Warp(image, map_x, map_y)
