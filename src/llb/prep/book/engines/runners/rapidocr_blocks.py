"""RapidOCR blocks: `python -I rapidocr_blocks.py <image> <out-dir>` writes `raw_blocks.json`.

Runs inside the survey's light environment, which has RapidOCR and not this repo. Stdlib only,
apart from that import.
"""

import json
import sys
from pathlib import Path
from typing import Any


def main(image: str, out_dir: str) -> None:
    from rapidocr import RapidOCR

    result = RapidOCR()(image)
    boxes, texts, scores = _fields(result)
    blocks: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        quad = _quad(box)
        if quad is None:
            continue
        text = str(texts[index]) if index < len(texts) else ""
        score = _score(scores, index)
        blocks.append(
            {
                "kind": "body",
                "order": index,
                "x0": quad[0],
                "y0": quad[1],
                "x1": quad[2],
                "y1": quad[3],
                "text": text,
                "confidence": score,
            }
        )
    ordered = _reading_order(blocks)
    for index, item in enumerate(ordered):
        item["order"] = index
    Path(out_dir, "raw_blocks.json").write_text(
        json.dumps({"blocks": ordered}, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _fields(result: Any) -> tuple[list[Any], list[Any], list[Any]]:
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None and isinstance(result, tuple):
        boxes = result[0] if result else None
        texts = result[1] if len(result) > 1 else None
        scores = result[2] if len(result) > 2 else None
    return _as_list(boxes), _as_list(texts), _as_list(scores)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return [item for item in value]


def _quad(box: Any) -> tuple[float, float, float, float] | None:
    points = list(box)
    if not points:
        return None
    if _is_point(points[0]):
        xs = [_num(point[0]) for point in points]
        ys = [_num(point[1]) for point in points]
        return (min(xs), min(ys), max(xs), max(ys))
    if len(points) == 4:
        x0, y0, x1, y1 = (_num(value) for value in points)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return None


def _top_left(item: dict[str, Any]) -> tuple[float, float]:
    return (_num(item["y0"]), _num(item["x0"]))


def _reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top to bottom. Two text columns are read left then right, split by a wide line."""
    ordered = sorted(blocks, key=_top_left)
    if not _has_two_columns(ordered):
        return ordered
    result: list[dict[str, Any]] = []
    band: list[dict[str, Any]] = []
    for block in ordered:
        if _spans_gutter(block, ordered):
            if band:
                result.extend(_columns(band, ordered))
                band = []
            result.append(block)
            continue
        band.append(block)
    result.extend(_columns(band, ordered))
    return result


def _has_two_columns(blocks: list[dict[str, Any]]) -> bool:
    left = sum(1 for block in blocks if _column(block, blocks) == 0)
    right = sum(1 for block in blocks if _column(block, blocks) == 1)
    return left >= 3 and right >= 3


def _columns(band: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    left = [block for block in band if _column(block, blocks) == 0]
    right = [block for block in band if _column(block, blocks) == 1]
    return [*left, *right]


def _column(block: dict[str, Any], blocks: list[dict[str, Any]]) -> int:
    if _spans_gutter(block, blocks):
        return -1
    return 0 if _center(block) < _mid(blocks) else 1


def _spans_gutter(block: dict[str, Any], blocks: list[dict[str, Any]]) -> bool:
    margin = _span(blocks) * 0.05
    mid = _mid(blocks)
    return _num(block["x0"]) < mid - margin and _num(block["x1"]) > mid + margin


def _mid(blocks: list[dict[str, Any]]) -> float:
    left = min(_num(block["x0"]) for block in blocks)
    return left + _span(blocks) / 2


def _span(blocks: list[dict[str, Any]]) -> float:
    left = min(_num(block["x0"]) for block in blocks)
    right = max(_num(block["x1"]) for block in blocks)
    return max(right - left, 1.0)


def _center(block: dict[str, Any]) -> float:
    return (_num(block["x0"]) + _num(block["x1"])) / 2


def _num(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("a boolean is not a coordinate")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _is_point(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) >= 2
    except TypeError:
        return False


def _score(scores: list[Any], index: int) -> float | None:
    if index >= len(scores) or scores[index] is None:
        return None
    return _num(scores[index])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
