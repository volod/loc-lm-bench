"""Write a vault in one swap, so a reader never sees a half-rendered book.

The previous tree moves aside, the new tree takes its name, and the previous
tree is removed. A name the new render did not produce is gone with it.
"""

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from llb.prep.book.preprocess.images import encode_png, read_pixels
from llb.prep.book.render.constants import (
    ASSETS_DIRNAME,
    CONTENT_DIRNAME,
    METADATA_DIRNAME,
    REVIEW_DIRNAME,
    SOURCE_DIRNAME,
)

_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_DIRECTORIES = (
    CONTENT_DIRNAME,
    ASSETS_DIRNAME,
    REVIEW_DIRNAME,
    METADATA_DIRNAME,
    SOURCE_DIRNAME,
)


@dataclass(frozen=True)
class Crop:
    source: Path
    box: tuple[int, int, int, int]


type Asset = bytes | Path | Crop


def publish(dest: Path, files: dict[str, Asset], links: dict[str, Path]) -> None:
    """Replace `dest` with `files` and symlinks. `links` values are absolute targets."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    pixels = lru_cache(maxsize=1)(read_pixels)
    try:
        for name in _DIRECTORIES:
            (temp / name).mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            path = temp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, Path):
                shutil.copyfile(data, path)
            elif isinstance(data, Crop):
                x0, y0, x1, y1 = data.box
                path.write_bytes(encode_png(pixels(data.source)[y0:y1, x0:x1]))
            else:
                path.write_bytes(data)
            if path.suffix == ".pdf":
                os.chmod(path, _READ_ONLY)
        for rel, target in links.items():
            path = temp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        _swap(temp, dest)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _swap(temp: Path, dest: Path) -> None:
    trash = dest.with_name(f".{dest.name}.trash")
    if trash.exists():
        shutil.rmtree(trash)
    if dest.exists():
        os.replace(dest, trash)
    try:
        os.replace(temp, dest)
    except Exception:
        if trash.exists() and not dest.exists():
            os.replace(trash, dest)
        raise
    if trash.exists():
        shutil.rmtree(trash)
