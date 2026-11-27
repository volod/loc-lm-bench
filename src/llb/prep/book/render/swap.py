"""Install a staged vault while retaining a recoverable previous tree."""

import ctypes
import errno
import os
import shutil
from pathlib import Path

_RENAME_EXCHANGE = 2
_EXCHANGE_UNSUPPORTED = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}


def recover_tree(temp: Path, dest: Path, previous: Path) -> None:
    """Restore a displaced vault before discarding inactive staging."""
    for path in (dest, temp, previous):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"vault publication requires a directory at {path}")
    if previous.exists():
        if dest.exists():
            shutil.rmtree(previous)
        else:
            os.replace(previous, dest)
    if temp.exists():
        shutil.rmtree(temp)


def install_tree(parent: int, temp: Path, dest: Path, previous: Path) -> None:
    """Exchange directories atomically when supported, otherwise keep a rollback tree."""
    if not dest.exists():
        os.replace(temp, dest)
        return
    if _exchange(parent, temp, dest):
        shutil.rmtree(temp)
        return
    os.replace(dest, previous)
    try:
        os.replace(temp, dest)
    except Exception:
        os.replace(previous, dest)
        raise
    shutil.rmtree(previous)


def _exchange(parent: int, temp: Path, dest: Path) -> bool:
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        return False
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if (
        rename(parent, os.fsencode(temp.name), parent, os.fsencode(dest.name), _RENAME_EXCHANGE)
        == 0
    ):
        return True
    error = ctypes.get_errno()
    if error in _EXCHANGE_UNSUPPORTED:
        return False
    raise OSError(error, os.strerror(error), str(dest))
