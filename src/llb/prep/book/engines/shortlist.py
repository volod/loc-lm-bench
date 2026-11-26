"""Adapters for the survey shortlist. Each one is the survey's candidate, plus a parser.

Choosing among them is a later task. Qwen stays out: it is a correction model, and correction is
not this stage. Marker fast, PP-StructureV3, granite-docling, Chandra, and LightOnOCR were dropped
on the survey's own evidence and are not registered.
"""

from llb.prep.book.engines.candidate import AdapterSpec, CandidateAdapter

_DOCLING = (
    "{bin}/docling",
    "convert",
    "{pdf}",
    "--to",
    "json",
    "--ocr-mode",
    "full_page",
    "--enrich-formula",
    "--image-export-mode",
    "placeholder",
    "--output",
    "{out}",
)
_RAPID = ("{python}", "-I", "{runners}/rapidocr_blocks.py", "{image}", "{out}")
_PADDLE = ("{python}", "-I", "{runners}/paddle_blocks.py", "{image}", "{out}")


def _mineru(tier: str) -> tuple[str, ...]:
    """Middle JSON keeps each block's box. Markdown alone is a vertical band per line."""
    return (
        "{bin}/mineru-kit",
        "parse",
        "{pdf}",
        "-o",
        "{out}/page.json",
        "--format",
        "middle_json",
        "--tier",
        tier,
        "--ocr-mode",
        "ocr",
    )


def _vlm(preset: str) -> tuple[str, ...]:
    return (
        "{bin}/docling",
        "convert",
        "{pdf}",
        "--to",
        "json",
        "--pipeline",
        "vlm",
        "--vlm-model",
        preset,
        "--image-export-mode",
        "placeholder",
        "--output",
        "{out}",
    )


# engine id, parser, optional argv (None keeps the survey smoke command).
_SPECS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("rapidocr", "raw-json", _RAPID),
    ("mineru-basic", "files", _mineru("basic")),
    ("mineru-standard", "files", _mineru("standard")),
    ("docling", "docling-json", _DOCLING),
    ("docling-vlm-mineru2-pro", "docling-json", _vlm("mineru2_pro")),
    ("docling-vlm-surya-ocr2-5", "docling-json", _vlm("surya_ocr2_5")),
    ("paddleocr-vl", "raw-json", _PADDLE),
    ("marker-balanced", "files", None),
    ("dots-ocr", "dots", None),
    ("deepseek-ocr-2", "deepseek", None),
    ("olmocr-2", "olmocr", None),
    ("ollama-glm-ocr-latest", "markdown", None),
    ("ollama-deepseek-ocr-3b", "deepseek", None),
)


def shortlist_adapters() -> tuple[CandidateAdapter, ...]:
    """One adapter per shortlisted engine, in survey order."""
    return tuple(
        CandidateAdapter(AdapterSpec(engine_id, parser, argv)) for engine_id, parser, argv in _SPECS
    )
