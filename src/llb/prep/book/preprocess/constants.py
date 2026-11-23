"""Thresholds and names for page signals and image variants.

A correction runs only when its signal crosses the threshold named here. A clean page, whose
signals sit on the other side of every threshold, keeps the original rendering and nothing else.
Blur is measured and never corrected: super-resolution is out of scope. Noise is measured because
denoising has to be gated by a signal the same way the others are.
"""

from typing import Final

SIGNAL_SCHEMA_ID: Final = "llb.book-ocr-page-signals"
MANIFEST_SCHEMA_ID: Final = "llb.book-ocr-variant-manifest"
SCHEMA_VERSION: Final = "1.0.0"
SIGNAL_RECIPE: Final = "2"
VARIANT_RECIPE: Final = "2"

ORIGINAL: Final = "original"
ROTATED: Final = "rotated"
DESKEW: Final = "deskew"
DEWARP: Final = "dewarp"
ILLUMINATION: Final = "illumination"
CONTRAST: Final = "contrast"
GUTTER: Final = "gutter"
DENOISE: Final = "denoise"
GEOMETRIC: Final = (ROTATED, DESKEW, DEWARP)
PHOTOMETRIC: Final = (ILLUMINATION, CONTRAST, GUTTER, DENOISE)
VARIANT_NAMES: Final = (ORIGINAL, *GEOMETRIC, *PHOTOMETRIC)

DEWARP_OPENCV: Final = "opencv"
DEWARP_PAGE: Final = "page-dewarp"
DEWARP_UVDOC: Final = "uvdoc"
DEWARP_AUTO: Final = "auto"
DEWARP_METHODS: Final = (DEWARP_OPENCV, DEWARP_PAGE, DEWARP_UVDOC, DEWARP_AUTO)

SIGNALS_FILENAME: Final = "signals.json"
MANIFEST_FILENAME: Final = "manifest.json"
VARIANT_IMAGE: Final = "page.png"
SIGNALS_STAGE: Final = "signals"
VARIANTS_STAGE: Final = "variants"

BACKGROUND_KERNEL: Final = 61
MAX_SKEW_DEG: Final = 8.0
SKEW_COARSE_STEP: Final = 0.5
SKEW_FINE_STEP: Final = 0.1
BAND_FRACTION: Final = 0.35
TILES: Final = 8
GRID_STEP: Final = 16

SKEW_CORRECT_DEG: Final = 1.5
CURVATURE_CORRECT_PX: Final = 3.0
# A fitted bow past this is a figure or a column edge, not curl. The planted synthetic bow is
# under 8 px; at 150 dpi this cap is about 7 mm.
MAX_LINE_BOW_PX: Final = 40.0
BOW_OUTLIER_RATIO: Final = 3.0
UNEVENNESS_CORRECT: Final = 0.04
CONTRAST_CORRECT: Final = 0.25
GUTTER_DARK_FRACTION: Final = 0.08
GUTTER_WIDTH_CORRECT: Final = 12
NOISE_CORRECT: Final = 8.0
POLARITY_MARGIN: Final = 8.0
SIDEWAYS_RATIO: Final = 6.0
BASELINE_FLIP: Final = 0.0
BASELINE_MIN_LINES: Final = 8
BASELINE_LINE_GLYPHS: Final = 8
LINE_CENTER_GAP: Final = 8
MIN_LINE_SPAN: Final = 80
PAPER_LIFT: Final = 40
PAPER_PAD: Final = 3
GUTTER_MAX_FRACTION: Final = 0.12
DOT_MIN_AREA: Final = 4
STROKE_INK_MAX: Final = 100
LINE_MAX_HEIGHT: Final = 80
BLEED_EDGE_FRACTION: Final = 0.2
BLEED_DENSITY: Final = 0.04
BLUR_SHARP_MIN: Final = 30.0

STROKE_MAX_HEIGHT: Final = 3
STROKE_MIN_WIDTH: Final = 8
DOT_MAX_AREA: Final = 24
DOT_MAX_SIDE: Final = 5
STROKE_WINDOW: Final = 3
STROKE_KEEP_FRACTION: Final = 0.7
STROKE_DARK_MAX: Final = 140
EDGE_FRACTION: Final = 0.05
COVER_CELL: Final = 16
CONTENT_WINDOW: Final = 4

CLAHE_CLIP: Final = 2.0
CLAHE_TILE: Final = 8
BILATERAL_D: Final = 5
BILATERAL_SIGMA: Final = 15.0
GUTTER_MAX_GAIN: Final = 2.5

PAGE_DEWARP_TIMEOUT_S: Final = 180
UVDOC_TIMEOUT_S: Final = 300
BOX_TOLERANCE_PX: Final = 4
