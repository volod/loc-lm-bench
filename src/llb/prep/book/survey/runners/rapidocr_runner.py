"""RapidOCR smoke: `python -I rapidocr_runner.py <image> <out-dir>` writes `reading.md`."""

import sys
from pathlib import Path


def main(image: str, out_dir: str) -> None:
    from rapidocr import RapidOCR

    result = RapidOCR()(image)
    lines = list(result.txts or ())
    Path(out_dir, "reading.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
