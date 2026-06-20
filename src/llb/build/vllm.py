"""Install prebuilt vLLM with uv or build one wheel from a local git checkout.

Registry/prebuilt packages and every dependency remain in uv's standard shared cache.
Only a deliberate source build from a clean git checkout is exported under DATA_DIR/wheels.
"""

import importlib.metadata as metadata
import logging
import os
import subprocess
import sys
from pathlib import Path

from llb.paths import PROJECT_ROOT, resolve_data_dir, resolve_project_path

_LOG = logging.getLogger(__name__)

DEFAULT_VLLM_SPEC = "vllm"
SOURCE_WHEEL_GLOB = "vllm-*.whl"
SOURCE_PACKAGE = "vllm"


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=capture, text=True)


def _uv_cache_dir() -> str:
    return _run(["uv", "cache", "dir"], capture=True).stdout.strip()


def _reject_implicit_source_spec(spec: str) -> None:
    source_prefixes = ("git+", "file:", "/", "./", "../")
    if spec.startswith(source_prefixes):
        raise ValueError(
            "source installs require VLLM_SOURCE_DIR=<clean-git-checkout>; "
            "VLLM_SPEC accepts only registry versions or prebuilt wheel URLs"
        )


def _install_prebuilt(python: Path, spec: str) -> None:
    _reject_implicit_source_spec(spec)
    _LOG.info("[build-vllm] mode: prebuilt")
    _LOG.info("[build-vllm] uv shared cache: %s", _uv_cache_dir())
    _LOG.info("[build-vllm] installing binary wheels only: %s", spec)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--only-binary",
            ":all:",
            spec,
        ]
    )


def _resolve_source_dir(value: str) -> Path:
    source_dir = resolve_project_path(value)
    if not source_dir.is_dir():
        raise ValueError(f"VLLM_SOURCE_DIR is not a directory: {source_dir}")
    return source_dir


def _git_output(source_dir: Path, *arguments: str) -> str:
    result = _run(["git", "-C", str(source_dir), *arguments], capture=True)
    return result.stdout.strip()


def _require_clean_git_checkout(source_dir: Path) -> str:
    try:
        inside = _git_output(source_dir, "rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"source directory is not a git checkout: {source_dir}") from exc
    if inside != "true":
        raise ValueError(f"source directory is not a git checkout: {source_dir}")
    checkout_root = Path(_git_output(source_dir, "rev-parse", "--show-toplevel")).resolve()
    if checkout_root != source_dir.resolve():
        raise ValueError(f"source directory must be the git checkout root: {source_dir}")
    status = _git_output(source_dir, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise ValueError("source checkout must be clean for a reproducible wheel")
    return _git_output(source_dir, "rev-parse", "--short=12", "HEAD")


def _install_build_requirements(python: Path, source_dir: Path) -> None:
    configured = os.environ.get("VLLM_BUILD_REQUIREMENTS")
    requirements = Path(configured) if configured else source_dir / "requirements" / "build.txt"
    if not requirements.is_absolute():
        requirements = PROJECT_ROOT / requirements
    if requirements.is_file():
        _LOG.info("[build-vllm] installing build requirements through uv: %s", requirements)
        _run(["uv", "pip", "install", "--python", str(python), "-r", str(requirements)])
    else:
        _LOG.info("[build-vllm] no build requirements file found; using active environment")


def _source_abi_key() -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "source builds require torch in the active environment; "
            "install build requirements first"
        ) from exc
    cuda = (torch.version.cuda or "cpu").replace(".", "")
    torch_version = torch.__version__.split("+")[0]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        accelerator = f"sm{major}{minor}"
    else:
        accelerator = "cpu"
    return (
        f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch_version}"
        f"_cu{cuda}_{accelerator}"
    )


def _cached_source_wheels(wheel_dir: Path) -> list[Path]:
    return sorted(wheel_dir.glob(SOURCE_WHEEL_GLOB))


def _build_source_wheel(python: Path, source_dir: Path, wheel_dir: Path) -> list[Path]:
    for wheel in _cached_source_wheels(wheel_dir):
        wheel.unlink()
    _run(
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--no-create-gitignore",
            "--python",
            str(python),
            "--out-dir",
            str(wheel_dir),
            str(source_dir),
        ]
    )
    return _cached_source_wheels(wheel_dir)


def _install_from_checkout(python: Path, source_value: str) -> None:
    source_dir = _resolve_source_dir(source_value)
    revision = _require_clean_git_checkout(source_dir)
    _install_build_requirements(python, source_dir)
    abi_key = _source_abi_key()
    wheel_dir = resolve_data_dir() / "wheels" / f"{SOURCE_PACKAGE}_{abi_key}_git{revision}"
    wheel_dir.mkdir(parents=True, exist_ok=True)

    max_jobs = os.environ.get("MAX_JOBS")
    if not max_jobs:
        raise RuntimeError("MAX_JOBS is required for source builds; use scripts/build_vllm.sh")
    _LOG.info("[build-vllm] mode: source checkout")
    _LOG.info("[build-vllm] source: %s@%s", source_dir, revision)
    _LOG.info("[build-vllm] MAX_JOBS=%s", max_jobs)
    _LOG.info("[build-vllm] source-built wheel cache: %s", wheel_dir)
    _LOG.info("[build-vllm] uv shared cache for dependencies: %s", _uv_cache_dir())

    wheels = _cached_source_wheels(wheel_dir)
    if os.environ.get("REBUILD_VLLM_WHEEL") == "1" or not wheels:
        wheels = _build_source_wheel(python, source_dir, wheel_dir)
    else:
        _LOG.info("[build-vllm] reusing source-built wheel: %s", wheels[0])
    if len(wheels) != 1:
        raise RuntimeError(f"expected one vLLM wheel in {wheel_dir}, found {len(wheels)}")

    # The direct wheel is local; all dependencies resolve through uv's shared cache.
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--only-binary",
            ":all:",
            str(wheels[0]),
        ]
    )


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _report_install() -> None:
    _LOG.info("[build-vllm] vllm==%s", _distribution_version("vllm"))
    flash_attention = _distribution_version("vllm-flash-attn")
    if flash_attention:
        _LOG.info("[build-vllm] flash-attn: bundled (vllm-flash-attn==%s)", flash_attention)
    else:
        _LOG.info("[build-vllm] flash-attn: vendored inside vLLM or selected at serve time")


def main() -> int:
    python = Path(sys.executable)
    source_dir = os.environ.get("VLLM_SOURCE_DIR")
    if source_dir:
        _install_from_checkout(python, source_dir)
    else:
        _install_prebuilt(python, os.environ.get("VLLM_SPEC", DEFAULT_VLLM_SPEC))
    _report_install()
    _LOG.info("[build-vllm] active attention backend is reported at serve time")
    _LOG.info("[build-vllm] serve: llb run-eval --backend vllm --model <hf-repo-id> --telemetry")
    return 0


if __name__ == "__main__":
    from llb.runtime import run

    raise SystemExit(run(main))
