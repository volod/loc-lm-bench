"""PaddleOCR smoke: `python -I paddle_runner.py <unwarp|structure|vl> <image> <out-dir> [engine]`.

`unwarp` writes `unwarped.png` (UVDoc); `structure` and `vl` write `reading.md` (PP-StructureV3
and PaddleOCR-VL-1.6). The engine defaults to Transformers, which needs no PaddlePaddle build
matched to the GPU; `paddle` runs the PaddlePaddle framework instead.
"""

import sys
from pathlib import Path
from typing import Any

ENGINE = sys.argv[4] if len(sys.argv) > 4 else "transformers"


def _markdown(results: Any) -> str:
    parts: list[str] = []
    for result in results:
        markdown = result.markdown
        text = markdown.get("markdown_texts") if isinstance(markdown, dict) else markdown
        parts.append(str(text or ""))
    return "\n\n".join(parts)


def unwarp(image: str, out_dir: str) -> None:
    from paddleocr import TextImageUnwarping

    for index, result in enumerate(
        TextImageUnwarping(model_name="UVDoc", engine=ENGINE).predict(image)
    ):
        result.save_to_img(
            save_path=str(Path(out_dir, "unwarped.png" if index == 0 else f"unwarped_{index}.png"))
        )


def structure(image: str, out_dir: str) -> None:
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        engine=ENGINE, use_doc_orientation_classify=False, use_doc_unwarping=False
    )
    Path(out_dir, "reading.md").write_text(_markdown(pipeline.predict(image)), encoding="utf-8")


def vl(image: str, out_dir: str) -> None:
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        engine=ENGINE, use_doc_orientation_classify=False, use_doc_unwarping=False
    )
    Path(out_dir, "reading.md").write_text(_markdown(pipeline.predict(image)), encoding="utf-8")


MODES = {"unwarp": unwarp, "structure": structure, "vl": vl}

if __name__ == "__main__":
    MODES[sys.argv[1]](sys.argv[2], sys.argv[3])
