"""Local serving-runtime readiness checks shared by sweep and joint search."""

import shutil
from pathlib import Path


def local_backend_ready(backend: str, data_dir: Path) -> tuple[bool, str]:
    """Return whether the executable required by a resolved local backend is installed."""
    if backend == "vllm":
        from llb.backends.vllm_command import vllm_executable

        if vllm_executable():
            return True, ""
        return False, "vllm executable not found (run make build-vllm)"
    if backend == "llamacpp":
        built = data_dir / "llb" / "llamacpp" / "build" / "bin" / "llama-server"
        if built.exists() or shutil.which("llama-server"):
            return True, ""
        return False, "llama-server not found (run make build-llamacpp)"
    return True, ""
