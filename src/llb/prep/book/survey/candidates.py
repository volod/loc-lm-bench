"""The surveyed candidates for every recovery stage, and how each one installs and smokes.

One table, read whole on purpose: a candidate's row says which stages it serves, its version and
licence, whether it runs locally, its install route, the page classes it CLAIMS to handle (the
pilot measures them; this row only records the claim), and its smoke. A candidate that cannot run
here stays in the table with the reason, so the shortlist's absences are as auditable as its rows.

Install routes:

- `venv` -- an isolated virtualenv under `$DATA_DIR/venvs/book-<env>/`, resolved by uv with
  exact top-level pins and `--exclude-newer` at the survey cutoff, so a rerun resolves the same
  set. Every candidate here pins its own torch/onnxruntime stack, which is why none of them is a
  `pyproject.toml` extra (the `pdf-quality` precedent for Marker).
- `ollama` -- a model tag the local Ollama daemon serves.
- `system` -- a binary the host provides (`tesseract` comes with `make apt-deps`).
- `docker` -- a model served by the pinned vLLM image under the NVIDIA Container Toolkit, which
  every CUDA development host carries (`make venv` refuses one without it); the survey starts and
  stops the container itself (`vllm_server`).
- `none` -- surveyed and not installed: the row's `blocker` says why and what would reopen it.

Licence and locality are recorded as the upstream states them on the survey date; they are not a
licence determination (the spec leaves that to the operator).
"""

from dataclasses import dataclass, field

SURVEY_DATE = "2026-11-19"
# uv resolves every survey environment as of this instant, so a later rerun installs the same set.
EXCLUDE_NEWER = "2026-11-19T00:00:00Z"
SURVEY_PYTHON = "3.12"

STAGE_PREPROCESS = "preprocess"  # dewarp, deskew, illumination, background
STAGE_LAYOUT = "layout"  # regions and reading order
STAGE_OCR = "ocr"
STAGE_FORMULA = "formula"
STAGE_TABLE = "table"
STAGE_CONVERT = "convert"  # whole-page document-to-Markdown converter
STAGE_CORRECT = "correct"  # model-assisted correction (a vision model reading the region)
STAGES = (
    STAGE_PREPROCESS,
    STAGE_LAYOUT,
    STAGE_OCR,
    STAGE_FORMULA,
    STAGE_TABLE,
    STAGE_CONVERT,
    STAGE_CORRECT,
)

ROUTE_VENV = "venv"
ROUTE_OLLAMA = "ollama"
ROUTE_SYSTEM = "system"
ROUTE_DOCKER = "docker"
ROUTE_NONE = "none"

LOCAL = "local"
EXTERNAL = "external"

SMOKE_COMMAND = "command"
SMOKE_RUNNER = "runner"
SMOKE_OLLAMA = "ollama"
SMOKE_SERVED = "served"  # one OpenAI-compatible chat request to the smoke's vLLM server

VLLM_IMAGE = "vllm/vllm-openai:v0.25.1"  # the image Marker 2.2 / Surya 0.23 pin

# Page classes, the pilot's vocabulary (spec: Scanned book recovery).
BODY = "body"
TWO_COLUMN = "two-column"
CURVED = "curved"
SHADOWED = "shadowed"
FORMULA = "formula"
TABLE = "table"
FIGURE = "figure"
ROTATED = "rotated"


@dataclass(frozen=True)
class EnvSpec:
    """An isolated survey environment: its interpreter and its exact top-level pins.

    `index` adds a vendor package index; `undated` names packages served from it without upload
    dates, which `--exclude-newer` would otherwise drop -- their exact pin is what fixes them.
    """

    name: str
    requirements: tuple[str, ...]
    python: str = SURVEY_PYTHON
    index: str | None = None
    undated: tuple[str, ...] = ()


@dataclass(frozen=True)
class VllmServer:
    """A model served by the vLLM image for one smoke: its Hugging Face id and serve arguments."""

    model: str
    args: tuple[str, ...] = ()
    image: str = VLLM_IMAGE


@dataclass(frozen=True)
class Smoke:
    """How a candidate runs on one page.

    `argv` placeholders: `{bin}` the environment's bin directory, `{python}` its interpreter,
    `{runners}` the runner-script directory, `{image}` the page image, `{pdf}` the one-page PDF,
    `{out}` the candidate's output directory. `reading` globs the text output under `{out}`;
    `image_output` globs a preprocessed image, which is then read by Tesseract so a preprocessing
    candidate is judged by what it does to recognition. `fail_on` is a log pattern that marks
    a run failed although it exited 0 -- a tool that silently falls back to a weaker mode.
    `server` starts a vLLM container before the smoke and stops it after; its URL is `{server}`,
    usable in `argv` and in `env` (extra process variables, formatted the same way).
    """

    kind: str
    argv: tuple[str, ...] = ()
    reading: str | None = None
    image_output: str | None = None
    model: str | None = None
    prompt: str | None = None
    timeout_s: float = 900.0
    fail_on: str | None = None
    server: VllmServer | None = None
    env: tuple[tuple[str, str], ...] = ()
    max_image_side: int | None = (
        None  # a served read scales the page down to this, as the tool does
    )


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    stages: tuple[str, ...]
    version: str
    licence: str
    locality: str
    route: str
    source: str
    claims: tuple[str, ...]
    env: str | None = None
    smoke: Smoke | None = None
    blocker: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


ENVS: dict[str, EnvSpec] = {
    spec.name: spec
    for spec in (
        EnvSpec(
            "light",
            (
                "opencv-python-headless==5.1.0.95",
                "page-dewarp==0.3.4",
                "rapidocr==3.10.0",
                "onnxruntime==1.32.0",
                "ocrmypdf==17.14.2",
            ),
        ),
        EnvSpec("mineru", ("mineru==4.2.3",)),
        EnvSpec("docling", ("docling==2.150.1",)),
        EnvSpec("marker", ("marker-pdf==2.2.0",)),
        EnvSpec(
            "paddle",
            (
                "paddleocr[doc-parser]==3.7.0",
                "transformers==5.23.0",
                "torch==2.15.0",
                "torchvision==0.29.2",
                "onnxruntime-gpu==1.32.0",
            ),
        ),
        # The PaddlePaddle framework itself (CUDA 13.0 build, runs on sm_120), for the pipelines
        # whose Transformers port is incomplete.
        EnvSpec(
            "paddle-fw",
            ("paddleocr[doc-parser]==3.7.0", "paddlepaddle-gpu==3.4.0"),
            index="https://www.paddlepaddle.org.cn/packages/stable/cu130/",
            undated=("paddlepaddle-gpu",),
        ),
    )
}

_READ_PAGE = (
    "Transcribe this scanned book page exactly as printed, in reading order, as Markdown. "
    "Write formulas as LaTeX and tables as Markdown tables. Do not add, explain, summarise, "
    "correct, or translate anything; write only what the page shows."
)
_MARKDOWN = "**/*.md"
# Serve arguments sized for a 12 GiB card with nothing else resident.
_SURYA_SERVER = VllmServer(
    "datalab-to/surya-ocr-2.5",
    (
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "24576",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "8",
        "--enable-prefix-caching",
        "--mm-processor-kwargs",
        '{"min_pixels": 3136, "max_pixels": 6291456}',
    ),
)
_DOTS_LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's "
    "bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', "
    "'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)
_OLMOCR_PROMPT = (  # olmocr.prompts.build_no_anchoring_v4_yaml_prompt
    "Attached is one page of a document that you must process. Just return the plain text "
    "representation of this document as if you were reading it naturally. Convert equations to "
    "LateX and tables to HTML.\nIf there are any figures or charts, label them with the following "
    "markdown syntax ![Alt text describing the contents of the figure]"
    "(page_startx_starty_width_height.png)\nReturn your output as markdown, with a front matter "
    "section on top specifying values for the primary_language, is_rotation_valid, "
    "rotation_correction, is_table, and is_diagram parameters."
)

CANDIDATES: tuple[Candidate, ...] = (
    # --- preprocessing -------------------------------------------------------------------------
    Candidate(
        id="opencv",
        name="OpenCV classical preprocessing (deskew, background normalisation)",
        stages=(STAGE_PREPROCESS,),
        version="opencv-python-headless 5.1.0.95",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="light",
        source="https://github.com/opencv/opencv-python",
        claims=(SHADOWED, ROTATED),
        smoke=Smoke(
            SMOKE_RUNNER,
            ("{python}", "-I", "{runners}/opencv_preprocess.py", "{image}", "{out}"),
            image_output="normalized.png",
            timeout_s=120,
        ),
        notes=("Deskew by minimum-area rectangle of ink, background by a large median blur.",),
    ),
    Candidate(
        id="page-dewarp",
        name="page-dewarp (cubic sheet model, text-line fit)",
        stages=(STAGE_PREPROCESS,),
        version="0.3.4",
        licence="MIT",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="light",
        source="https://github.com/lmmx/page-dewarp",
        claims=(CURVED, ROTATED),
        smoke=Smoke(
            SMOKE_COMMAND,
            ("{bin}/page-dewarp", "-nb", "1", "-x", "0", "-y", "0", "-o", "{out}", "{image}"),
            image_output="*_thresh.png",
            timeout_s=300,
        ),
        notes=(
            "Binarises by default; `-nb 1` keeps grey, which the spec requires. Its default page "
            "margins (50 x 20 px) crop text at both edges and shrink the page; `-x 0 -y 0` keeps "
            "every line at about the native size.",
        ),
    ),
    Candidate(
        id="uvdoc",
        name="UVDoc text-image unwarping (PaddleOCR module)",
        stages=(STAGE_PREPROCESS,),
        version="paddleocr 3.7.0, UVDoc",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="paddle",
        source="https://github.com/PaddlePaddle/PaddleOCR",
        claims=(CURVED,),
        smoke=Smoke(
            SMOKE_RUNNER,
            ("{python}", "-I", "{runners}/paddle_runner.py", "unwarp", "{image}", "{out}"),
            image_output="unwarped.png",
        ),
    ),
    Candidate(
        id="scantailor-advanced",
        name="ScanTailor Advanced",
        stages=(STAGE_PREPROCESS,),
        version="1.2.1",
        licence="GPL-3.0",
        locality=LOCAL,
        route=ROUTE_NONE,
        source="https://github.com/ScanTailor-Advanced/scantailor-advanced",
        claims=(CURVED, SHADOWED, ROTATED),
        blocker=(
            "GUI-only: the Advanced fork ships no batch command-line interface (only an AppImage "
            "and a .deb), so it cannot run as a per-page stage. Reopens if a CLI returns."
        ),
    ),
    Candidate(
        id="unpaper",
        name="unpaper (OCRmyPDF --clean)",
        stages=(STAGE_PREPROCESS,),
        version="7.0.0 (Ubuntu noble)",
        licence="GPL-2.0",
        locality=LOCAL,
        route=ROUTE_NONE,
        source="https://github.com/unpaper/unpaper",
        claims=(SHADOWED, ROTATED),
        blocker=(
            "Needs `sudo apt install unpaper`, unavailable to the agent on this host; it also "
            "binarises by design, which the spec forbids by default. Reopens as an OCRmyPDF "
            "`--clean` variant once installed."
        ),
    ),
    # --- line OCR ------------------------------------------------------------------------------
    Candidate(
        id="tesseract",
        name="Tesseract (LSTM)",
        stages=(STAGE_OCR,),
        version="5.3.4 (Ubuntu noble; upstream 5.5.4)",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_SYSTEM,
        source="https://github.com/tesseract-ocr/tesseract",
        claims=(BODY, TWO_COLUMN),
        smoke=Smoke(
            SMOKE_COMMAND,
            ("tesseract", "{image}", "{out}/page", "--psm", "3", "-l", "eng"),
            reading="page.txt",
            timeout_s=120,
        ),
    ),
    Candidate(
        id="ocrmypdf",
        name="OCRmyPDF (Tesseract with deskew and page rotation)",
        stages=(STAGE_OCR, STAGE_PREPROCESS),
        version="17.14.2",
        licence="MPL-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="light",
        source="https://github.com/ocrmypdf/OCRmyPDF",
        claims=(BODY, TWO_COLUMN, ROTATED),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/ocrmypdf",
                "--deskew",
                "--rotate-pages",
                "--sidecar",
                "{out}/text.txt",
                "-l",
                "eng",
                "{pdf}",
                "{out}/ocr.pdf",
            ),
            reading="text.txt",
            timeout_s=300,
        ),
        notes=(
            "Its product is a searchable PDF, which the spec does not produce; only its "
            "sidecar text and deskew are of use.",
        ),
    ),
    Candidate(
        id="rapidocr",
        name="RapidOCR (PP-OCRv6 det/rec on ONNX Runtime)",
        stages=(STAGE_OCR,),
        version="3.10.0, PP-OCRv6 small",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="light",
        source="https://github.com/RapidAI/RapidOCR",
        claims=(BODY, ROTATED),
        smoke=Smoke(
            SMOKE_RUNNER,
            ("{python}", "-I", "{runners}/rapidocr_runner.py", "{image}", "{out}"),
            reading="reading.md",
            timeout_s=300,
        ),
    ),
    # --- whole-page converters (layout + OCR + formula + table) --------------------------------
    Candidate(
        id="mineru-basic",
        name="MinerU basic tier (ONNX layout, OCR, formula, table)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="4.2.3",
        licence="MinerU Open Source License (Apache-2.0 with additional conditions)",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="mineru",
        source="https://github.com/opendatalab/MinerU",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/mineru-kit",
                "parse",
                "{pdf}",
                "-o",
                "{out}/page.md",
                "--tier",
                "basic",
                "--ocr-mode",
                "ocr",
            ),
            reading="page.md",
        ),
    ),
    Candidate(
        id="mineru-standard",
        name="MinerU standard tier (VLM on llama.cpp)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="4.2.3",
        licence="MinerU Open Source License; MinerU2.5 VLM weights AGPL-3.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="mineru",
        source="https://github.com/opendatalab/MinerU",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/mineru-kit",
                "parse",
                "{pdf}",
                "-o",
                "{out}/page.md",
                "--tier",
                "standard",
                "--ocr-mode",
                "ocr",
            ),
            reading="page.md",
            timeout_s=1800,
        ),
    ),
    Candidate(
        id="docling",
        name="Docling standard pipeline (Heron layout, OCR, TableFormer, formula enrichment)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="2.150.1",
        licence="MIT (models: Apache-2.0 / CDLA)",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="docling",
        source="https://github.com/docling-project/docling",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/docling",
                "convert",
                "{pdf}",
                "--to",
                "md",
                "--ocr-mode",
                "full_page",
                "--enrich-formula",
                "--image-export-mode",
                "placeholder",
                "--output",
                "{out}",
            ),
            reading=_MARKDOWN,
            timeout_s=1800,
        ),
    ),
    # Docling's VLM pipeline runs these document VLMs on Transformers in the same environment,
    # which is the one route here for models whose own packaging wants vLLM in Docker.
    *(
        Candidate(
            id=f"docling-vlm-{preset.replace('_', '-')}",
            name=f"{label} via the Docling VLM pipeline (Transformers)",
            stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
            version=f"docling 2.150.1, {weights}",
            licence=licence,
            locality=LOCAL,
            route=ROUTE_VENV,
            env="docling",
            source=f"https://huggingface.co/{weights}",
            claims=(BODY, TWO_COLUMN, FORMULA, TABLE),
            smoke=Smoke(
                SMOKE_COMMAND,
                (
                    "{bin}/docling",
                    "convert",
                    "{pdf}",
                    "--to",
                    "md",
                    "--pipeline",
                    "vlm",
                    "--vlm-model",
                    preset,
                    "--image-export-mode",
                    "placeholder",
                    "--output",
                    "{out}",
                ),
                reading=_MARKDOWN,
                timeout_s=1800,
            ),
        )
        for preset, label, weights, licence in (
            (
                "granite_docling",
                "granite-docling 258M",
                "ibm-granite/granite-docling-258M",
                "Apache-2.0",
            ),
            (
                "surya_ocr2_5",
                "Surya 2.5 0.7B",
                "datalab-to/surya-2.5",
                "weights modified OpenRAIL-M (revenue cap)",
            ),
            (
                "chandra_ocr2",
                "Chandra 2 5.3B",
                "datalab-to/chandra-ocr-2",
                "weights modified OpenRAIL-M (revenue cap, no use competing with the vendor API)",
            ),
            (
                "lightonocr3_4b",
                "LightOnOCR-3 4.5B",
                "lightonai/LightOnOCR-3-4B",
                "Apache-2.0",
            ),
            (
                "mineru2_pro",
                "MinerU2.5-Pro",
                "opendatalab/MinerU2.5-Pro",
                "AGPL-3.0 weights",
            ),
        )
    ),
    Candidate(
        id="marker-fast",
        name="Marker fast mode (Indra ONNX layout and OCR, CPU)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_TABLE, STAGE_CONVERT),
        version="2.2.0",
        licence="Apache-2.0 code; weights modified OpenRAIL-M (revenue cap)",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="marker",
        source="https://github.com/datalab-to/marker",
        claims=(BODY, TWO_COLUMN, TABLE),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/marker_single",
                "{pdf}",
                "--mode",
                "fast",
                "--force_ocr",
                "--output_dir",
                "{out}",
            ),
            reading=_MARKDOWN,
            timeout_s=1800,
        ),
    ),
    Candidate(
        id="marker-balanced",
        name="Marker balanced mode (Surya 2.5 VLM for formulas, tables, scans)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="2.2.0 with surya-ocr 0.23",
        licence="Apache-2.0 code; weights modified OpenRAIL-M (revenue cap)",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="marker",
        source="https://github.com/datalab-to/marker",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_COMMAND,
            (
                "{bin}/marker_single",
                "{pdf}",
                "--mode",
                "balanced",
                "--force_ocr",
                "--output_dir",
                "{out}",
            ),
            reading=_MARKDOWN,
            timeout_s=1800,
            fail_on="falling back to Indra",
            server=_SURYA_SERVER,
            env=(("SURYA_INFERENCE_URL", "{server}/v1"),),
        ),
        notes=(
            "Left alone it spawns vLLM in Docker as root, mounting the home directory's model "
            "cache; the survey serves Surya 2.5 itself and attaches Marker through "
            "SURYA_INFERENCE_URL. Without a server it exits 0 after silently falling back to "
            "fast mode, so the smoke fails it on that log line.",
        ),
    ),
    Candidate(
        id="pp-structurev3",
        name="PP-StructureV3 (PP-DocLayout, PP-OCR, PP-FormulaNet, SLANet tables)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="paddleocr 3.7.0",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="paddle",
        source="https://github.com/PaddlePaddle/PaddleOCR",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_RUNNER,
            ("{python}", "-I", "{runners}/paddle_runner.py", "structure", "{image}", "{out}"),
            reading="reading.md",
            timeout_s=1800,
        ),
    ),
    Candidate(
        id="paddleocr-vl",
        name="PaddleOCR-VL-1.6 (0.9B document VLM, Transformers engine)",
        stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
        version="paddleocr 3.7.0, PaddleOCR-VL-1.6",
        licence="Apache-2.0",
        locality=LOCAL,
        route=ROUTE_VENV,
        env="paddle",
        source="https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6",
        claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
        smoke=Smoke(
            SMOKE_RUNNER,
            ("{python}", "-I", "{runners}/paddle_runner.py", "vl", "{image}", "{out}"),
            reading="reading.md",
            timeout_s=1800,
        ),
    ),
    *(
        Candidate(
            id=f"{base}-paddle-engine",
            name=f"{label} on the PaddlePaddle 3.4 framework engine",
            stages=(STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
            version="paddleocr 3.7.0, paddlepaddle-gpu 3.4.0 (cu130)",
            licence="Apache-2.0",
            locality=LOCAL,
            route=ROUTE_VENV,
            env="paddle-fw",
            source="https://github.com/PaddlePaddle/PaddleOCR",
            claims=(BODY, TWO_COLUMN, FORMULA, TABLE, FIGURE),
            smoke=Smoke(
                SMOKE_RUNNER,
                (
                    "{python}",
                    "-I",
                    "{runners}/paddle_runner.py",
                    mode,
                    "{image}",
                    "{out}",
                    "paddle",
                ),
                reading="reading.md",
                timeout_s=1800,
            ),
        )
        for base, label, mode in (
            ("pp-structurev3", "PP-StructureV3", "structure"),
            ("paddleocr-vl", "PaddleOCR-VL-1.6", "vl"),
        )
    ),
    # --- vLLM-only document VLMs, served from the pinned vLLM image ------------------------------
    *(
        Candidate(
            id=cid,
            name=f"{label} served by vLLM in Docker",
            stages=stages,
            version=f"{model}, {VLLM_IMAGE}",
            licence=licence,
            locality=LOCAL,
            route=ROUTE_DOCKER,
            source=f"https://huggingface.co/{model}",
            claims=claims,
            smoke=Smoke(
                SMOKE_SERVED,
                model=model,
                prompt=prompt,
                timeout_s=900,
                server=VllmServer(model, args),
                max_image_side=max_side,
            ),
        )
        for cid, label, model, licence, stages, claims, prompt, args, max_side in (
            (
                "dots-ocr",
                "dots.ocr 3B (layout JSON)",
                "rednote-hilab/dots.ocr",
                "MIT",
                (STAGE_LAYOUT, STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
                (BODY, TWO_COLUMN, FORMULA, TABLE),
                _DOTS_LAYOUT_PROMPT,
                (
                    "--trust-remote-code",
                    "--max-model-len",
                    "16384",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--max-num-seqs",
                    "4",
                ),
                None,
            ),
            (
                "olmocr-2",
                "olmOCR-2 7B (FP8)",
                "allenai/olmOCR-2-7B-1025-FP8",
                "Apache-2.0",
                (STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
                (BODY, FORMULA, TABLE),
                _OLMOCR_PROMPT,
                (  # at 0.92 the 12k context lacked 0.01 GiB of KV cache on a 12 GiB card
                    "--max-model-len",
                    "12288",
                    "--gpu-memory-utilization",
                    "0.95",
                    "--max-num-seqs",
                    "2",
                ),
                1288,  # olmOCR's own pipeline renders pages at 1288 px on the longest side
            ),
            (
                "deepseek-ocr-2",
                "DeepSeek-OCR-2 3.4B",
                "deepseek-ai/DeepSeek-OCR-2",
                "Apache-2.0",
                (STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
                (BODY, FORMULA, TABLE),
                "<|grounding|>Convert the document to markdown.",
                (
                    "--trust-remote-code",
                    "--max-model-len",
                    "8192",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--max-num-seqs",
                    "2",
                    "--no-enable-prefix-caching",
                    "--mm-processor-cache-gb",
                    "0",
                ),
                None,
            ),
        )
    ),
    # --- surveyed, not installed: why, and what would reopen each ------------------------------
    Candidate(
        id="pix2tex",
        name="pix2tex / LaTeX-OCR (formula crops to LaTeX)",
        stages=(STAGE_FORMULA,),
        version="0.1.4 (last release 2025-01-18)",
        licence="MIT",
        locality=LOCAL,
        route=ROUTE_NONE,
        source="https://github.com/lukas-blecher/LaTeX-OCR",
        claims=(FORMULA,),
        blocker=(
            "Unmaintained since January 2025 and superseded by the formula recognisers inside "
            "the converters smoked here (PP-FormulaNet, UniMERNet in MinerU, Docling "
            "CodeFormula) and by the document VLMs, which all emit LaTeX. Not installed."
        ),
    ),
    *(
        Candidate(
            id=service_id,
            name=name,
            stages=(STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CONVERT),
            version="hosted API",
            licence="commercial terms of service",
            locality=EXTERNAL,
            route=ROUTE_NONE,
            source=source,
            claims=(BODY, FORMULA, TABLE),
            blocker=(
                "External service: every page image of a copyrighted book would leave the host, "
                "which the operator's redistribution class and the egress boundary refuse by "
                "default. Justified only if the pilot shows a local pipeline missing formula or "
                "table fidelity the service measurably recovers on the same pages."
            ),
        )
        for service_id, name, source in (
            ("mathpix", "Mathpix Convert API", "https://mathpix.com/docs/convert"),
            ("datalab-api", "Datalab hosted Marker/Chandra API", "https://www.datalab.to"),
            ("mistral-ocr", "Mistral OCR API", "https://docs.mistral.ai/capabilities/document_ai"),
            (
                "azure-document-intelligence",
                "Azure AI Document Intelligence",
                "https://learn.microsoft.com/azure/ai-services/document-intelligence/",
            ),
            ("google-document-ai", "Google Document AI", "https://cloud.google.com/document-ai"),
            ("aws-textract", "Amazon Textract", "https://aws.amazon.com/textract/"),
        )
    ),
    # --- vision models served by Ollama (page reading and region-level correction) -------------
    *(
        Candidate(
            id=f"ollama-{tag.replace(':', '-')}",
            name=f"{label} via Ollama",
            stages=(STAGE_OCR, STAGE_FORMULA, STAGE_TABLE, STAGE_CORRECT),
            version=tag,
            licence=licence,
            locality=LOCAL,
            route=ROUTE_OLLAMA,
            source=f"https://ollama.com/library/{tag.split(':')[0]}",
            claims=(BODY, FORMULA, TABLE),
            smoke=Smoke(SMOKE_OLLAMA, model=tag, prompt=prompt, timeout_s=900),
        )
        for tag, label, licence, prompt in (
            ("glm-ocr:latest", "GLM-OCR 0.9B", "MIT", "Text Recognition:"),
            (
                "deepseek-ocr:3b",
                "DeepSeek-OCR 3B",
                "MIT",
                "<|grounding|>Convert the document to markdown.",
            ),
            ("qwen3-vl:8b", "Qwen3-VL 8B", "Apache-2.0", _READ_PAGE),
            ("qwen2.5vl:7b", "Qwen2.5-VL 7B", "Apache-2.0", _READ_PAGE),
        )
    ),
)
