"""Thresholds for figure crops, caption association, and ornament rejection.

Lengths are fractions of the shorter page side, then clamped to pixels, so a
150 dpi handbook and a synthetic fixture use the same rule. A repeated badge
is small. An isolated speck is smaller. A caption may sit farther away than
the margin that keeps axis labels, but not so far that the crop crosses the page.
"""

from typing import Final

FIGURE_RECORD_FILENAME: Final = "figure_assets.json"
REASON_ORNAMENT: Final = "repeated-ornament"
REASON_NOISE: Final = "isolated-noise"

# Most-corrected kept geometric image first. Photometric variants are not crops.
GEOMETRIC_ORDER: Final = ("dewarp", "deskew", "rotated")
# Grid inverse is sampled. The crop grows by one step so a cell between samples stays in.
VARIANT_SAMPLE_STEP: Final = 8

# Union of source regions, then this band for labels, legends, and scale ticks.
MARGIN_FRAC: Final = 0.03
MARGIN_MIN_PX: Final = 16.0
MARGIN_MAX_PX: Final = 64.0

# Pieces of one drawing. Wider than this, they are two figures.
GROUP_GAP_FRAC: Final = 0.02
GROUP_GAP_MIN_PX: Final = 8.0
GROUP_GAP_MAX_PX: Final = 40.0

# An unnumbered caption this close can belong to the figure. A printed caption
# farther than this is still the caption in the note, but the image is not
# stretched across the page to reach it.
CAPTION_GAP_FRAC: Final = 0.10
CAPTION_GAP_MIN_PX: Final = 24.0
CAPTION_GAP_MAX_PX: Final = 160.0
# A second caption within this multiple of the nearest gap is ambiguous.
CAPTION_AMBIGUOUS_RATIO: Final = 1.25

# Same place on another page, and small: a badge or a rule, not a drawing.
BADGE_MAX_SIDE_FRAC: Final = 0.16
BADGE_MAX_AREA_FRAC: Final = 0.015
BADGE_IOU: Final = 0.45
BADGE_REPEAT_PAGES: Final = 2

# No number, no caption, and this small: detector noise.
NOISE_MAX_SIDE_FRAC: Final = 0.04
NOISE_MAX_AREA_FRAC: Final = 0.002
