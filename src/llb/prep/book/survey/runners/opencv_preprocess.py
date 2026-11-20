"""OpenCV classical preprocessing smoke: `python -I opencv_preprocess.py <image> <out-dir>`.

Flattens the background by dividing the page by a large median-blurred estimate of it (which lifts
a gutter shadow without binarising), measures the skew as the rotation that maximises the variance
of the horizontal ink profile (text lines become sharp peaks when level), and rotates it out.
Writes `normalized.png` and `signals.json` (the measured skew in degrees).
"""

import json
import sys
from pathlib import Path

BACKGROUND_KERNEL = 61  # odd, wider than any glyph at 150 dpi
INK_THRESHOLD = 0.75  # a pixel darker than this fraction of its background counts as ink
MAX_SKEW_DEG = 5.0
SKEW_STEP_DEG = 0.1
PROFILE_SCALE = 0.5  # the search runs on a half-size ink mask


def main(image: str, out_dir: str) -> None:
    import cv2
    import numpy as np

    grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    background = cv2.medianBlur(grey, BACKGROUND_KERNEL).astype(np.float32) + 1.0
    flat = np.clip(grey.astype(np.float32) / background * 255.0, 0, 255).astype(np.uint8)
    ink = (flat < 255 * INK_THRESHOLD).astype(np.float32)
    small = cv2.resize(ink, None, fx=PROFILE_SCALE, fy=PROFILE_SCALE)
    sh, sw = small.shape
    best, skew = -1.0, 0.0
    for angle in np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + SKEW_STEP_DEG / 2, SKEW_STEP_DEG):
        matrix = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
        profile = cv2.warpAffine(small, matrix, (sw, sh)).sum(axis=1)
        score = float(np.var(profile))
        if score > best:
            best, skew = score, float(angle)
    h, w = flat.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    straight = cv2.warpAffine(flat, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    cv2.imwrite(str(Path(out_dir, "normalized.png")), straight)
    Path(out_dir, "signals.json").write_text(json.dumps({"skew_deg": round(skew, 2)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
