"""Build a complete vault beside its destination, then install it."""

import fcntl
import os
import shutil
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from llb.prep.book.constants import PARTIAL_SUFFIX, TRASH_SUFFIX
from llb.prep.book.preprocess.images import encode_png, read_pixels
from llb.prep.book.render.constants import (
    ASSETS_DIRNAME,
    CONTENT_DIRNAME,
    METADATA_DIRNAME,
    REVIEW_DIRNAME,
    SOURCE_DIRNAME,
)
from llb.prep.book.render.swap import install_tree, recover_tree

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
    temp = dest.with_name(f".{dest.name}{PARTIAL_SUFFIX}")
    previous = dest.with_name(f".{dest.name}{TRASH_SUFFIX}")
    parent = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Lock the containing directory so a retry cannot remove an active publisher's staging.
        fcntl.flock(parent, fcntl.LOCK_EX)
        recover_tree(temp, dest, previous)
        _write_tree(temp, files, links)
        install_tree(parent, temp, dest, previous)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        os.close(parent)


def _write_tree(temp: Path, files: dict[str, Asset], links: dict[str, Path]) -> None:
    temp.mkdir(mode=0o700)
    pixels = lru_cache(maxsize=1)(read_pixels)
    for name in _DIRECTORIES:
        (temp / name).mkdir()
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
