"""Page-image measurements the engines need, with no image library."""

from pathlib import Path

_PNG = b"\x89PNG\r\n\x1a\n"
_IHDR = b"IHDR"


def png_size(path: Path) -> tuple[int, int]:
    """Width and height of a PNG, from its IHDR, in pixels."""
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) < 24 or not header.startswith(_PNG) or header[12:16] != _IHDR:
        raise ValueError(f"{path.name} is not a PNG")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
