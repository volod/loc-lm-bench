"""The canonical MAX_JOBS shell helper (AGENTS.md single source of truth)."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llb.backends.preflight_verdict import SAMPLER_NATIVE, SamplerVerdict, save_verdict
from llb.core.paths import PROJECT_ROOT

COMMON_SH = PROJECT_ROOT / "scripts" / "shared" / "common.sh"
BUILD_VLLM = COMMON_SH.parents[1] / "build_vllm.sh"
PYTHON_ABI_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
FAKE_VLLM_WHEEL = f"vllm-0.0.0-{PYTHON_ABI_TAG}-{PYTHON_ABI_TAG}-linux_x86_64.whl"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_toolchain(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    _write_executable(
        bin_dir / "uv",
        f"""#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$UV_LOG"
if [ "$1 $2" = "cache dir" ]; then
  echo /shared/uv-cache
elif [ "$1" = "build" ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--out-dir" ]; then
      shift
      mkdir -p "$1"
      : > "$1/{FAKE_VLLM_WHEEL}"
      break
    fi
    shift
  done
fi
""",
    )
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_executable(
        venv_bin / "python",
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n',
    )
    return bin_dir, uv_log


def _seed_sampler_preflight_verdict(data_dir: Path) -> None:
    """Pre-record a sampler verdict so the installer's preflight reuses it instead of probing.

    `llb.build.vllm.main` ends in the flashinfer sampler preflight, which the fake `uv` cannot
    intercept: it is an in-process probe that imports torch and JIT-builds + launches a flashinfer
    kernel on the real GPU. That probe, not any resolver call, is what makes an end-to-end
    installer run expensive on a CUDA host (5.3s of a 5.6s test). A verdict recorded with
    `driver: None` is current on every host (`verdict_is_current`), so the cache short-circuits the
    probe and these tests stay about wheel export and uv calls.
    """
    verdict: SamplerVerdict = {
        "sampler": SAMPLER_NATIVE,
        "flashinfer_version": None,
        "detail": "seeded by tests; the real probe needs a GPU",
        "checked_at": "2026-01-01T00:00:00+00:00",
        "driver": None,
        "pinned_version": None,
        "auto_pinned": False,
    }
    save_verdict(verdict, data_dir)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_max_jobs_is_a_positive_int():
    out = subprocess.run(
        ["bash", "-c", f". {COMMON_SH}; llb_max_jobs"],
        capture_output=True,
        text=True,
        check=True,
    )
    value = out.stdout.strip()
    assert value.isdigit() and int(value) >= 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_common_sh_exposes_helpers():
    out = subprocess.run(
        [
            "bash",
            "-c",
            f". {COMMON_SH}; type llb_max_jobs llb_load_env llb_python >/dev/null && echo ok",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "ok"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_prebuilt_vllm_uses_uv_shared_cache_without_project_wheelhouse(tmp_path):
    bin_dir, uv_log = _fake_toolchain(tmp_path)
    data_dir = tmp_path / "data"
    _seed_sampler_preflight_verdict(data_dir)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(tmp_path),
        "DATA_DIR": str(data_dir),
        "UV_LOG": str(uv_log),
        "VLLM_SPEC": "vllm==1.2.3",
    }

    subprocess.run(["bash", str(BUILD_VLLM)], env=env, check=True, capture_output=True, text=True)

    calls = uv_log.read_text(encoding="utf-8")
    assert "cache dir" in calls
    assert "pip install" in calls
    assert "--only-binary :all: vllm==1.2.3" in calls
    assert not (data_dir / "wheels").exists()


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="bash and git are required",
)
def test_source_vllm_exports_only_checkout_wheel(tmp_path):
    bin_dir, uv_log = _fake_toolchain(tmp_path)
    python_modules = tmp_path / "src"
    python_modules.mkdir()
    (python_modules / "torch.py").write_text(
        """__version__ = "2.9.1+cu128"

class _Version:
    cuda = "12.8"

class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_capability():
        return (8, 9)

version = _Version()
cuda = _Cuda()
""",
        encoding="utf-8",
    )
    source_dir = tmp_path / "vllm-source"
    source_dir.mkdir()
    (source_dir / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source_dir)], check=True)
    subprocess.run(["git", "-C", str(source_dir), "add", "pyproject.toml"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source_dir),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    data_dir = tmp_path / "data"
    _seed_sampler_preflight_verdict(data_dir)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(tmp_path),
        "DATA_DIR": str(data_dir),
        "UV_LOG": str(uv_log),
        "VLLM_SOURCE_DIR": str(source_dir),
    }

    subprocess.run(["bash", str(BUILD_VLLM)], env=env, check=True, capture_output=True, text=True)

    cached_files = list((data_dir / "wheels").rglob("*"))
    cached_wheels = [path for path in cached_files if path.is_file()]
    assert len(cached_wheels) == 1
    assert cached_wheels[0].name == FAKE_VLLM_WHEEL
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    assert cached_wheels[0].parent.name.startswith(f"vllm_{python_tag}_torch2.9.1_cu128_sm89_git")
    calls = uv_log.read_text(encoding="utf-8")
    assert "build --wheel --no-build-isolation --no-create-gitignore --python" in calls
