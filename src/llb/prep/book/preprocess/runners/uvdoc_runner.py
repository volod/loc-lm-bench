"""UVDoc: `python -I uvdoc_runner.py <image> <out-dir>`.

Writes `page.png`. Writes `maps.npz` only when the model result exposes a point map. A result
with no map is still a successful process: the caller refuses the variant, because a geometric
variant with no map back to the original render is a crop. This file imports nothing from the
project, so the paddle environment can run it with `python -I`.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

_MAP_KEYS = ("map_x", "map_y", "point_map", "map", "grid")


def main() -> int:
    image_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    from paddleocr import TextImageUnwarping

    results = list(
        TextImageUnwarping(model_name="UVDoc", engine="transformers").predict(str(image_path))
    )
    if not results:
        print("no result", file=sys.stderr)
        return 2
    result = results[0]
    result.save_to_img(save_path=str(out_dir / "unwarped.png"))
    if not _write_page(out_dir):
        print("no output image", file=sys.stderr)
        return 2
    maps = _maps(result)
    if maps is not None:
        np.savez(out_dir / "maps.npz", map_x=maps[0], map_y=maps[1])
    else:
        print("no point map", file=sys.stderr)
    return 0


def _write_page(out_dir: Path) -> bool:
    import cv2

    image = cv2.imread(str(out_dir / "unwarped.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        return False
    (out_dir / "page.png").write_bytes(encoded.tobytes())
    return True


def _maps(result: Any) -> tuple[np.ndarray, np.ndarray] | None:
    payload = _payload(result)
    if not isinstance(payload, dict):
        return None
    direct = _pair(payload.get("map_x"), payload.get("map_y"))
    if direct is not None:
        return direct
    for key in _MAP_KEYS:
        found = _pair_from(payload.get(key))
        if found is not None:
            return found
    return None


def _payload(result: Any) -> Any:
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        return data["res"]
    return data if isinstance(data, dict) else None


def _pair_from(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(value, dict):
        return _pair(value.get("map_x", value.get("x")), value.get("map_y", value.get("y")))
    array = np.asarray(value) if value is not None else None
    if array is not None and array.ndim == 3 and array.shape[-1] == 2:
        return np.asarray(array[:, :, 0], dtype=np.float32), np.asarray(
            array[:, :, 1], dtype=np.float32
        )
    return None


def _pair(map_x: Any, map_y: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if map_x is None or map_y is None:
        return None
    xs, ys = np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32)
    if xs.shape != ys.shape or xs.ndim != 2:
        return None
    return xs, ys


if __name__ == "__main__":
    raise SystemExit(main())
