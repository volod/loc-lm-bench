"""The injected engine CI runs. Its blocks are a function of the page size, not of the clock."""

from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed

FAKE_ENGINE = "fake"
FAKE_MODEL = "llb-fake-1"
_STAGES = ("layout", "ocr", "formula", "table")
# Vertical bands, as fractions of the page, top to bottom.
_BANDS = (
    ("running-header", 0.00, 0.06, "HEADER"),
    ("heading", 0.08, 0.16, "1.1 Section"),
    ("body", 0.18, 0.55, "The value is 0.05."),
    ("formula", 0.57, 0.70, "E = mc^2"),
    ("table", 0.72, 0.86, "a | b"),
    ("page-number", 0.90, 1.00, "1"),
)


class FakeEngine:
    """Deterministic blocks on every recognition stage. Always available."""

    engine_id = FAKE_ENGINE
    stages: tuple[str, ...] = _STAGES

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        blocks = tuple(
            _band(request.width, request.height, order, kind, top, bottom, text)
            for order, (kind, top, bottom, text) in enumerate(_blocks_for(stage))
        )
        return EngineRead(self.engine_id, FAKE_MODEL, request.variant, stage, blocks)


def _blocks_for(stage: str) -> tuple[tuple[str, float, float, str], ...]:
    if stage == "formula":
        return (_BANDS[3],)
    if stage == "table":
        return (_BANDS[4],)
    if stage == "ocr":
        return (_BANDS[0], _BANDS[2], _BANDS[5])
    return _BANDS


def _band(
    width: int, height: int, order: int, kind: str, top: float, bottom: float, text: str
) -> Proposed:
    y0 = top * height
    y1 = max(y0 + 1.0, min(float(height), bottom * height))
    return Proposed(kind, order, 0.0, y0, float(width), y1, text, None)
