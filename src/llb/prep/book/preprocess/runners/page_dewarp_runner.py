"""page-dewarp with its point map: `python -I page_dewarp_runner.py <image> <out-dir>`.

Writes `page.png` and `maps.npz`. `map_x` and `map_y` send a variant pixel back to the original.
Exits 2 when the page has no text spans to fit. This file imports nothing from the project, so
the light environment can run it with `python -I`.
"""

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    image_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    captured: dict[str, np.ndarray] = {}
    import page_dewarp.dewarp as dewarp

    real = dewarp.remap

    def wrapped(
        src: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *args: object, **kwargs: object
    ) -> np.ndarray:
        captured["map_x"] = np.asarray(map_x, dtype=np.float32)
        captured["map_y"] = np.asarray(map_y, dtype=np.float32)
        result = np.asarray(real(src, map_x, map_y, *args, **kwargs))
        captured["image"] = result
        return result

    dewarp.remap = wrapped
    from page_dewarp.image import WarpedImage
    from page_dewarp.options import Config

    config = Config(
        NO_BINARY=1,
        PAGE_MARGIN_X=0,
        PAGE_MARGIN_Y=0,
        DEBUG_LEVEL=0,
        OUTPUT_DIR=str(out_dir),
        OUTPUT_FORMAT="png",
    )
    warped = WarpedImage(str(image_path), config)
    if not getattr(warped, "written", False) or "image" not in captured:
        print("no spans", file=sys.stderr)
        return 2
    return _write(out_dir, captured)


def _write(out_dir: Path, captured: dict[str, np.ndarray]) -> int:
    import cv2

    image = captured["image"]
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        print("could not encode page.png", file=sys.stderr)
        return 2
    (out_dir / "page.png").write_bytes(encoded.tobytes())
    np.savez(out_dir / "maps.npz", map_x=captured["map_x"], map_y=captured["map_y"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
