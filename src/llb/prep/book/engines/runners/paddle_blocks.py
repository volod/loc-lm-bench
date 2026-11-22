"""PaddleOCR-VL blocks: `python -I paddle_blocks.py <image> <out-dir>` writes `raw_blocks.json`.

Runs inside the survey's paddle environment. The Transformers engine needs no PaddlePaddle build
matched to the GPU. `tool.json` keeps the raw predict payload so a parser miss is visible.
"""

import json
import sys
from pathlib import Path
from typing import Any


def main(image: str, out_dir: str) -> None:
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        engine="transformers",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    results = list(pipeline.predict(image))
    payload = [_as_data(result) for result in results]
    Path(out_dir, "tool.json").write_text(_dump(payload), encoding="utf-8")
    rows = _rows(payload)
    Path(out_dir, "raw_blocks.json").write_text(_dump({"blocks": rows}), encoding="utf-8")


def _as_data(result: Any) -> Any:
    data = getattr(result, "json", None)
    if isinstance(data, dict):
        return data
    if isinstance(result, dict):
        return result
    return {"repr": repr(result)}


def _rows(payload: Any) -> list[dict[str, Any]]:
    found = _find_list(payload, "parsing_res_list")
    rows = []
    for index, item in enumerate(found):
        if not isinstance(item, dict):
            continue
        box = _box(item.get("block_bbox") or item.get("bbox"))
        if box is None:
            continue
        rows.append(
            {
                "kind": str(item.get("block_label") or item.get("label") or "unknown"),
                "order": index,
                "x0": box[0],
                "y0": box[1],
                "x1": box[2],
                "y1": box[3],
                "text": str(item.get("block_content") or item.get("content") or ""),
                "confidence": None,
            }
        )
    return rows


def _find_list(node: Any, key: str) -> list[Any]:
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, list):
            return value
        for child in node.values():
            found = _find_list(child, key)
            if found:
                return found
    if isinstance(node, list):
        for child in node:
            found = _find_list(child, key)
            if found:
                return found
    return []


def _box(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    if len(raw) == 4 and not isinstance(raw[0], (list, tuple)):
        x0, y0, x1, y1 = (float(value) for value in raw)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if isinstance(raw[0], (list, tuple)):
        xs = [float(point[0]) for point in raw]
        ys = [float(point[1]) for point in raw]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str) + "\n"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
