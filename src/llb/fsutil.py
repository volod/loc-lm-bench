"""Small filesystem helpers shared across modules.

`atomic_write_text` is the single source of truth for "rewrite a whole file safely":
write a sibling temp file then `Path.replace` it into place, so a crash mid-write can
never leave a half-written file. Used for run manifests and the calibration worksheet
(which IS its own state, so every interactive edit rewrites it atomically).
"""

import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace ``path`` with UTF-8 text using a sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
