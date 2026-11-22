# Book Recovery: Tool Selection

Which recovery tools install and run on this host, and therefore which ones the pilot
(`book-pilot-pipeline-comparison`) compares. The field moves faster than any package description,
so nothing here is chosen by reputation: every candidate was installed from its current release
into its own environment and run on the same synthetic page with a known transcription and on one
real page of the target handbook. This page records the survey of 2026-11-19 and its 2026-11-20
extension to the models that need vLLM in Docker; the pilot's measured choice per page class will
be recorded here too, and it decides. The survey only bounds what it may choose from -- the
[selection for the pilot](#selection-for-the-pilot) says which candidates that is. A run selects
one of them per stage ([block ledger](block-ledger.md)); which one serves a page class is still
the pilot's decision.

Operator usage: [book-ocr guide, recovery-tool survey](../../../guides/data-prep/book-ocr.md#recovery-tool-survey).

## Where it lives

| Module | Owns |
| --- | --- |
| `src/llb/prep/book/survey/candidates.py` | The registry: every candidate's stages, version, licence, locality, install route, claimed page classes, smoke command, and, for one not installed, the blocker and what would reopen it; the six survey environments and their exact pins |
| `src/llb/prep/book/survey/envs.py` | Environment install (`uv venv` plus `uv pip install --exclude-newer`), the freeze file, and the process environment a smoke runs in (every model cache under `$DATA_DIR/cache/book-survey/`, telemetry off) |
| `src/llb/prep/book/survey/page_content.py`, `fixture.py` | The synthetic smoke page: content and transcription, then the seeded degradation into a 150 dpi greyscale JPEG in an image-only PDF; a real page rendered at native resolution |
| `src/llb/prep/book/survey/measure.py` | One measured run: wall time, peak RSS of the process tree, peak VRAM held by the tree's PIDs and the device-wide rise over the pre-run baseline; the wait for VRAM to come back |
| `src/llb/prep/book/survey/ollama_smoke.py` | A page read by an Ollama vision model, with Ollama's own placement (`/api/ps`) as the VRAM reading and `keep_alive: 0` so it unloads at once |
| `src/llb/prep/book/survey/vllm_server.py`, `served.py` | A model served from the pinned vLLM image for one smoke: the Docker preflight (`nvidia` runtime, room for the image), the container started as the invoking user and always stopped, its log kept, the page read through the OpenAI-compatible API |
| `src/llb/prep/book/survey/smoke.py`, `results.py`, `report.py`, `score.py` | One candidate on one page (status, measurements, reading), the survey walk and its `survey.json` / `survey.md`, the smoke reading against the transcription, and the rescore from saved readings |
| `src/llb/prep/book/survey/runners/` | Scripts that run INSIDE a candidate environment (OpenCV preprocessing, RapidOCR, PaddleOCR), standard library plus the tool only |
| `src/llb/cli/book/survey.py`, `make/data-prep/book.mk` | `llb book-ocr survey list / install / smoke / rescore`; `make book-survey-list`, `book-survey-install`, `book-survey-smoke`, `book-survey-rescore` |

Tests: `tests/llb/prep/book/test_survey_fixture_score.py` (the page is byte-identical on rerun,
image-only, and degraded; the scorer's classes) and `test_survey_registry_smoke.py` (every row
complete, every pin exact and resolved at the cutoff, every cache under `$DATA_DIR`, and the smoke
orchestrator over injected candidates: scored, silently fallen back, empty, hung, not installed,
blocked, rescored) and `test_survey_served.py` (the vLLM route over a faked Docker CLI: no
`nvidia` runtime, no room for the image, a layout-JSON reading, a server that never comes up or
runs out of KV cache).

## How a candidate is installed

Every candidate that pins its own torch, ONNX Runtime, or PaddlePaddle build installs into an
isolated uv virtualenv under `$DATA_DIR/venvs/book-<env>/`, never into `.venv` and never as a
`pyproject.toml` extra -- they would fight over one environment, which is the `pdf-quality` /
Marker precedent. Each environment resolves exact top-level pins with `--exclude-newer
2026-11-19T00:00:00Z`, so a rerun installs the same set, and freezes its full resolution to
`llb-freeze.txt`. Ollama candidates are model tags; Tesseract is the system package; models that
ship only for vLLM are served from the pinned `vllm/vllm-openai:v0.25.1` image (below).

| Environment | Pins | Holds |
| --- | --- | --- |
| `light` | `opencv-python-headless==5.1.0.95`, `page-dewarp==0.3.4`, `rapidocr==3.10.0`, `onnxruntime==1.32.0`, `ocrmypdf==17.14.2` | CPU preprocessing and line OCR (60 packages, 0.9 GB) |
| `mineru` | `mineru==4.2.3` | MinerU's ONNX `basic` tier and its llama.cpp VLM `standard` tier (0.7 GB plus 2 GB of models) |
| `docling` | `docling==2.150.1` | Docling's standard pipeline and its VLM pipeline on Transformers (torch 2.15.0+cu132, 5.8 GB) |
| `marker` | `marker-pdf==2.2.0` | Marker with Surya 0.23 (torch 2.15.0, 5.8 GB) |
| `paddle` | `paddleocr[doc-parser]==3.7.0`, `transformers==5.23.0`, `torch==2.15.0`, `torchvision==0.29.2`, `onnxruntime-gpu==1.32.0` | PaddleOCR pipelines on the Transformers engine, UVDoc (6.2 GB) |
| `paddle-fw` | `paddleocr[doc-parser]==3.7.0`, `paddlepaddle-gpu==3.4.0` from the PaddlePaddle CUDA 13.0 index | The same pipelines on the PaddlePaddle framework engine |

The PaddlePaddle index publishes no upload dates, so `--exclude-newer` would drop every wheel; that
one package is exempted (`--exclude-newer-package paddlepaddle-gpu=false`) and its exact pin is
what fixes it. torch 2.15.0 ships CUDA 13.2 kernels that run on this host's sm_120 (Blackwell) GPU;
no source build was needed for any candidate.

**vLLM in Docker.** Every development host is a CUDA host with the NVIDIA Container Toolkit, and
`make venv` refuses one without Docker's `nvidia` runtime
([dev setup](../../../guides/development/dev-setup.md)). The survey serves a vLLM-only model from
`vllm/vllm-openai:v0.25.1` -- the image Marker 2.2 pins, 27.8 GB unpacked -- in its OWN container:
started as the invoking user (`USER`, `HOME`, and torch's compile cache set inside, since the uid
has no passwd entry in the image), Hugging Face downloads under `$DATA_DIR/cache/book-survey/`,
bound to 127.0.0.1, and stopped and removed when the smoke ends, after its log is kept. A tool's
own spawn is not used: Surya's runs as root and mounts the home directory's model cache. Before
pulling, the survey refuses a Docker data root with less than 30 GB free rather than fill its
filesystem. vLLM reserves a fixed fraction of the card (`--gpu-memory-utilization`, 0.85 here), so a
served candidate's VRAM reading is that reservation, not the model's need.

**Every cache goes under `$DATA_DIR`.** The candidate stacks and their models run to tens of
gigabytes. The first installs here used uv's default cache in the home directory and filled the root
filesystem to 99% (4.5 GB free) before MinerU's first model download failed with `No space left on
device`; the survey therefore points uv at `$DATA_DIR/cache/uv` and every model cache a tool honours
-- `HF_HOME`, `XDG_CACHE_HOME`, `MODELSCOPE_CACHE`, `PADDLE_PDX_CACHE_HOME`, `TORCH_HOME`, and
`MINERU_HOME`, which MinerU needs because it ignores XDG and writes `~/.mineru` -- at
`$DATA_DIR/cache/book-survey/`. Ollama's model store is the daemon's own and stays where it is. Two
paths are hard-coded and cannot be moved, and both stay empty: PaddlePaddle creates
`~/.cache/paddle/dataset`, and Surya creates a server lock file under `~/.cache/datalab`.

## The smoke

**The synthetic page** is a 6 x 8 in page drawn with PyMuPDF's built-in Times and Symbol fonts: a
running header, a numbered heading, two columns of body text with decimals, minus signs, Greek
letters, a micro sign, and cross-references, a display formula with subscripts and an equation
number, a ruled table with a negative value and a thousands separator, a figure with a caption, and
a page number. It is then skewed 1.2 degrees, curled toward the gutter, shadowed on the gutter side,
set on grey paper, blurred, given seeded noise, and saved as a 150 dpi greyscale JPEG inside an
image-only PDF -- the shape of the target scan, byte-identical on every host, and no page of a real
book in git.

**The real page** is PDF page 300 of the target handbook: a two-column page with a labelled orbit
diagram and its caption, five numbered display equations with subscripts and Greek letters, inline
math, a margin badge, gutter curl, and shadow. It was read by every candidate, and its outputs stay
under the run directory; there is no transcription, so it is read by eye, never scored.

**What a smoke reading is, and is not.** It is not the pilot's metric (that is
`book-recovery-quality-metrics`); it says whether a tool read the page at all and kept the tokens a
misreading corrupts. `word recall` is the share of reference words found, order-free; `char
agreement` is one minus the character edit distance over the reference length, in order, so text
the page does not print -- an invented figure description -- costs it; each of 11 dangerous tokens
(`0.05`, `−3.2`, `1.07`, `12.5`, `α`, `β`, `Δv`, `1,240.5`, `−98.2`, `(3-4)`, `μm/s`) is `exact`,
`hyphen_minus` (a hyphen or dash where a minus is printed -- a substitution class, not a pass), or
`missing`. Markup is read as it typesets: a hyphen inside LaTeX math IS a minus sign, `I_{sp}` is
`Isp`, a Markdown table rule is not text. Preprocessing candidates are read through Tesseract, so
they are judged by what they do to recognition. One page is one page: these are smoke readings, not
estimates of anything.

## Shortlist per stage

Measured on 2026-11-19 (vLLM-served rows: 2026-11-20) on the RTX PRO 3000 Blackwell laptop GPU
(12,227 MiB, driver 620.70.06, CUDA 13.5 runtime), Intel Core Ultra 7 265HX, 62 GiB RAM, one
candidate at a time with VRAM returned to the idle baseline between candidates. VRAM is the peak
held by the tool's own processes (for Ollama, the daemon's resident placement); RAM is the peak
resident set of the tool's process tree (for Ollama, the model bytes held outside the GPU). Seconds
include model load, because a one-page run pays it every time.

Columns: **syn.** = synthetic page as `seconds, VRAM GB, RAM GB; word recall / char agreement /
dangerous tokens kept of 11`; **p300** = real PDF page 300 as `seconds, VRAM GB`, with what reading
it by eye showed. A dash is a reading the host cannot give or that does not apply.

### Page preprocessing (dewarp, deskew, illumination)

Read through Tesseract; Tesseract on the raw synthetic page is 0.865 / 0.594 / 5.

| Candidate | Version, licence, route | syn. | p300 | Verdict |
| --- | --- | --- | --- | --- |
| UVDoc (PaddleOCR module) | paddleocr 3.7.0, Apache-2.0; `paddle` env, Transformers engine | 5.1 s, 0.3, 1.7; 0.901 / 0.606 / 5 | 6.4 s, 0.4; curl and shadow flattened, whole page kept at the native 1241 x 1755 | **shortlisted** (learned dewarp); cut the synthetic page number off the bottom edge |
| page-dewarp | 0.3.4, MIT; `light`, CPU | 1.0 s, 0, 0.1; 0.865 / 0.596 / 4 | 4.3 s; with default margins cropped text at both edges and shrank the page to 960 x 1648 | **shortlisted** (CPU dewarp) with `-nb 1 -x 0 -y 0`, which keeps the whole page at 1264 x 1776 |
| OpenCV background division + projection-profile deskew | opencv-python-headless 5.1.0.95, Apache-2.0; `light`, CPU | 0.2 s, 0, 0.01; 0.910 / 0.616 / 5; skew measured 1.3 deg against 1.2 deg planted | 0.3 s; skew 0.0 deg, background flattened | **shortlisted** (signals, deskew, illumination) |

### Line OCR

| Candidate | Version, licence, route | syn. | p300 | Verdict |
| --- | --- | --- | --- | --- |
| RapidOCR (PP-OCRv6 small, ONNX) | 3.10.0, Apache-2.0; `light`, CPU | 2.0 s, 0, 0.2; 0.964 / 0.504 / 9 | 2.9 s; clean lines, but in detection order: diagram labels and both columns interleave | **shortlisted** as a line recogniser behind a layout engine; best word recall of any line engine |
| Tesseract (LSTM, `eng`) | 5.3.4 (upstream 5.5.4), Apache-2.0; system | 0.3 s, 0, 0.04; 0.865 / 0.594 / 5 -- every minus sign and Greek letter lost (`α` as `o`, `Δv` as `Av`) | 1.2 s; formulas unreadable (`ω` as `@`) | baseline: the existing lane's engine, not a candidate to beat it |
| OCRmyPDF (Tesseract, deskew, rotation) | 17.14.2, MPL-2.0; `light` | 2.2 s, 0, 0.1; 0.892 / 0.745 / 5 | 2.6 s; as Tesseract | not shortlisted: Tesseract with deskew; its product is a searchable PDF the spec does not make |

### Layout, formula, table, and whole-page conversion

| Candidate | Version, licence, route | syn. | p300 | Verdict |
| --- | --- | --- | --- | --- |
| MinerU `standard` (MinerU2.5-Pro VLM on llama.cpp, ONNX layout) | 4.2.3, MinerU Open Source License, VLM weights AGPL-3.0; `mineru` | 4.5 s, 2.8, 1.0; 0.892 / 0.913 / 8 | 9.9 s, 2.8; every equation as LaTeX with `\tag{10-28}` and the rest, inline `ω_E` right; running header and page number left out of the Markdown as page furniture; on both runs of this page it segfaulted (exit -11) at teardown after writing a complete reading | **shortlisted** |
| MinerU2.5-Pro through Docling's VLM pipeline | docling 2.150.1, AGPL-3.0 weights; `docling` | 12.1 s, 2.9, 3.0; 0.946 / 0.988 / 10 | 19.9 s, 3.5; as MinerU `standard`, equation numbers as `\tag{}`, running header and page number kept | **shortlisted** (the same model; best synthetic reading of any converter) |
| MinerU `basic` (ONNX pipeline) | 4.2.3; `mineru`, ONNX on the GPU through WebGPU | 8.2 s, 1.7, 1.6; 0.910 / 0.861 / 9 -- the table's three rows merged into one | 13.1 s, 3.2; display formulas good, inline `ω_E` read as a circled `w` | **shortlisted** (no VLM; table structure weak) |
| Docling standard (Heron layout, OCR, TableFormer, CodeFormula enrichment) | 2.150.1, MIT; `docling` | 9.7 s, 1.9, 2.7; 0.919 / 0.903 / 9 -- table perfect, display formula classified as a heading so never enriched, `Δ` written as U+2206 | 17.6 s, 2.9; formulas enriched, but alignment `&` emitted outside any environment (invalid LaTeX) and words from the neighbouring column ("spake", "e r that") pulled into an equation | **shortlisted** (best table structure of the pipelines; the existing lane's family) |
| PaddleOCR-VL-1.6 (0.9B, PP-DocLayoutV3) | paddleocr 3.7.0, Apache-2.0; `paddle`, Transformers engine | 8.4 s, 3.2, 2.8; 0.883 / 0.902 / 9 | 15.8 s, 3.2; text, table, and formulas good, figure kept as an image region with no invented text; every equation number dropped; `v` and `ν` mixed on one page | **shortlisted** |
| Surya 2.5 (0.7B) through Docling | docling 2.150.1, weights modified OpenRAIL-M; `docling` | 13.2 s, 2.0, 2.6; 0.928 / 0.750 / 9 | 32.4 s, 2.3; best reading order and equations of the page readers, equation numbers kept; invents a figure description; `°` written as U+FFFD | **shortlisted** (licence: free below a revenue cap) |
| LightOnOCR-3 4B through Docling | docling 2.150.1, Apache-2.0; `docling` | 30.4 s, 9.4, 9.8; 0.919 / 0.834 / 8 -- invented a data table for the chart | 57.5 s, 9.6; equations good, one display block split into three; invents a figure description | **shortlisted** (fits with 2.6 GB spare; invents content) |
| Chandra 2 (5.3B) through Docling | docling 2.150.1, weights modified OpenRAIL-M, no use competing with the vendor API; `docling` | 29.9 s, 10.8, 9.8; 0.928 / 0.182 / 9 -- the agreement is low because of a long invented figure description | 71.6 s, 10.8; equations excellent; a paragraph of invented figure labels | not shortlisted: 10.8 of 12.2 GB, the slowest, and the most invented text; its gated-delta-rule layers ran on Transformers' slow reference kernels (`flash-linear-attention` and `causal_conv1d` absent) |
| PP-StructureV3 on the Transformers engine | paddleocr 3.7.0, Apache-2.0; `paddle` | 12.5 s, 4.4, 2.6; 0.595 / 0.742 / 8 -- table emitted empty | 19.2 s, 9.8; words run together and letters dropped ("Geomryof DeartureaArir") | not shortlisted on this engine: the port is incomplete (see the PaddlePaddle-engine run) |
| granite-docling 258M through Docling | docling 2.150.1, Apache-2.0; `docling` | 72.3 s, 1.3, 2.2; 0.036 / 0.042 / 0 -- read the heading and stopped | 23.7 s, 1.3; read the page, but `ω` as `0` and `ω_M T_H` as one subscript | not shortlisted |
| Marker fast (Indra ONNX, CPU) | 2.2.0, Apache-2.0 code, weights modified OpenRAIL-M; `marker` | 5.0 s, 0, 1.2; 0.865 / 0.896 / 7 -- `−3.2` lost its sign, `α` as `&`, `β` as `ß`, formula garbage | 6.7 s; formulas garbage, `1.0000` read as `15000` | not shortlisted: silent value changes |
| Marker balanced | 2.2.0 | see [vLLM in Docker](#document-vlms-served-by-vllm-in-docker) | | |

The two PaddleOCR pipelines again on the PaddlePaddle 3.4 framework engine (`paddle-fw`), to tell an
incomplete Transformers port from the models themselves:

| Candidate | syn. | p300 | Verdict |
| --- | --- | --- | --- |
| PP-StructureV3, PaddlePaddle engine | 147.7 s (first use, includes model download), 4.0, 3.2; 0.739 / 0.820 / 9 -- text clean, table cells read but rows collapsed | 31.2 s, 6.4; letters and digits dropped from body text (`355.45` read as `55.45`), a sentence lost | not shortlisted on either engine |
| PaddleOCR-VL-1.6, PaddlePaddle engine | 37.7 s, 8.2, 3.3; 0.883 / 0.856 / 9 | 27.8 s, 8.2; the same reading as on Transformers | not shortlisted on this engine: the Transformers engine reads the same in 3.2 GB |

### Vision models served by Ollama (page reading and region-level correction)

Read with the same prompt-per-model, temperature 0, thinking off, and the model unloaded after each
page. VRAM is Ollama's own placement; each model was entirely on the GPU (no CPU-resident bytes).

| Candidate | Tag, licence | syn. | p300 | Verdict |
| --- | --- | --- | --- | --- |
| GLM-OCR 0.9B | `glm-ocr:latest`, MIT | 3.5 s, 2.8; 0.937 / 0.922 / **11** -- the only reader to keep every dangerous token, minus signs written inside math | 8.2 s, 2.8; coverage unstable: one run returned only the right column, a second (differing only in the `think` flag) both columns, without equation numbers and with a term of (10-28) missing | **shortlisted** for region reading behind a layout engine, which is how its own SDK runs it; never as a full-page reader |
| DeepSeek-OCR 3B | `deepseek-ocr:3b`, MIT | 5.1 s, 7.5; 0.739 / 0.457 / 7 | 9.9 s, 7.5; both columns, every equation number, each block labelled with its box (`equation[[140, 383, 489, 450]]`) under the grounding prompt | **shortlisted**: a page reader whose grounding output maps onto ledger blocks; weak on the synthetic page |
| Qwen2.5-VL 7B | `qwen2.5vl:7b`, Apache-2.0 | 7.2 s, 6.1; 0.955 / 0.958 / 9 | 25.2 s, 6.1; both columns, but the right column first and every equation number dropped | **shortlisted** for region-level correction |
| Qwen3-VL 8B | `qwen3-vl:8b`, Apache-2.0 | 12.7 s, 7.4; 0.973 / **0.995** / 9 -- the best synthetic reading of all | failed: it reasons inside `<think>` although `think: false` is sent, and spent 4,096 tokens, then 12,288 in a direct retry (218 s), without writing the page | **shortlisted** for region-level correction only, where a crop is small; never full pages on this tag |

### Document VLMs served by vLLM in Docker

Each smoke starts its own container from `vllm/vllm-openai:v0.25.1` and stops it after; **load** is
the server's start before the first request -- 140-200 s every launch even with the weights cached
(startup, compilation, CUDA graph capture), so a book run must keep one server up per batch, never
per page. VRAM is the device-wide rise, dominated by vLLM's fixed reservation of the card
(`--gpu-memory-utilization`), not the model's need; RAM inside the container is not read (`-`).

| Candidate | Model, licence | syn. | p300 | Verdict |
| --- | --- | --- | --- | --- |
| Marker balanced (Surya 2.5 0.7B served; Marker attached by `SURYA_INFERENCE_URL`) | `datalab-to/surya-ocr-2.5`, Marker 2.2.0; Apache-2.0 code, weights modified OpenRAIL-M | load 197 s; 27.5 s, 10.1, 0.8; 0.928 / 0.917 / 9 | 31.3 s, 10.3; **the most complete real-page reading of the survey**: both columns in order, every equation whole with its number (the `= v_E(t_0)` term others drop, `t_1` not `t_I`), `°` correct; the figure kept as an image file with its caption and no invented text | **shortlisted** -- the strongest page reader here; the same model through Docling invents figure text, Marker's pipeline does not |
| dots.ocr 3B (layout JSON) | `rednote-hilab/dots.ocr`, MIT | load 191 s; 19.4 s, 10.8; 0.982 / **0.997** / 9 -- best synthetic agreement of any page reader | 33.1 s, 11.3; header, page number, and figure label kept as blocks, every equation number, the figure left as a region with no text; dropped `= v_E(t_0)` from (10-28) and read `t_1` as `t_I` | **shortlisted**: layout categories and boxes per block in one call, the shape the block ledger wants |
| DeepSeek-OCR-2 3.4B | `deepseek-ai/DeepSeek-OCR-2`, Apache-2.0 | load 142 s; 21.5 s, 11.2; 0.775 / 0.535 / 8 -- table cells run together inside a bare `<table>` | 26.3 s, 11.2; complete, every equation number, (10-28) whole, each block labelled with its box | **shortlisted** for layout regions and formulas; not for tables |
| olmOCR-2 7B (FP8) | `allenai/olmOCR-2-7B-1025-FP8`, Apache-2.0 | load 176 s; 12.6 s, 11.7; 0.910 / 0.796 / **11** -- true minus signs throughout; the table caption and page furniture dropped | 29.4 s, 11.7; complete and in order, no invented figure text, but every equation number dropped | **shortlisted** with care: fits only at a 0.95 reservation and at its own 1288 px input (at 0.92 the 12k context lacked 0.01 GiB of KV cache; at the native 1755 px height the vision encoder ran out of memory) |

## Selection for the pilot

What `book-pilot-pipeline-comparison` compares, per stage, against the existing PDF lane (Docling
with Tesseract). The order inside a stage is the survey's reading, not a ranking the pilot inherits.

| Stage | Candidates the pilot compares | Why these |
| --- | --- | --- |
| Preprocessing | UVDoc; page-dewarp (`-nb 1 -x 0 -y 0`); OpenCV signals, deskew, illumination | the only local dewarp methods that keep the page whole; OpenCV for the measured signals |
| Page reading (layout, order, text, formula) | Marker `balanced` (Surya 2.5 on vLLM); MinerU `standard`; MinerU2.5-Pro; dots.ocr; PaddleOCR-VL-1.6; DeepSeek-OCR-2; olmOCR-2 | complete real-page readings with LaTeX formulas; each fails differently (dropped term, lost equation numbers, table structure, VRAM margin) |
| Tables | Docling standard (TableFormer); MinerU `standard`; PaddleOCR-VL-1.6; Marker `balanced`; dots.ocr | exact synthetic table structure |
| Line OCR behind a layout engine | RapidOCR (PP-OCRv6) | best line recall, CPU only |
| Region-level correction | Qwen3-VL 8B; Qwen2.5-VL 7B; GLM-OCR | strongest region readers that fit beside nothing else; GLM-OCR the only perfect dangerous-token reader |

Dropped from the comparison on survey evidence: Tesseract and OCRmyPDF as candidates (they remain
the baseline), Marker `fast`, PP-StructureV3 (both engines), granite-docling, Chandra 2, and
LightOnOCR-3 (invented figure data, and 9.4 GB for a reading the smaller models match).

## Page classes

A full run routes an engine by the class of the page in front of it, and the pilot is a seeded
sample of those classes. The `classes` stage reads `render` and `signals` and writes one
`classes.json` per page (`llb.book-ocr-page-classes` 1.0.0), published at
`pages/<pNNNN>/classes.json`. The first class a page matches is its primary, and that is what
`by_page_class` sees. A page can carry several classes; the pilot may select it for a later one.
`normal` is written only when nothing else matched.

Content is measured on a copy whose long side is at most 720 px, after the page is turned
upright. The stage reads ink and rules, not characters. The cuts below, and the page's placement
rotation, are in the stage's cache key, so moving a cut reclassifies the pages it affects and
leaves the render and the signal files alone.

| Order | Class | A page matches when |
| --- | --- | --- |
| 1 | `severe_curvature` | `line_curvature_px` is at least 3, the same bow the dewarp gate treats as worth unbending |
| 2 | `dense_formulas` | At least three text lines have an empty gap and a short mark at the right, the page has at least twelve text lines, the share of such lines is at least 0.06, and there are at most two columns. A gap full of faint dots is an index leader and does not count |
| 3 | `large_tables` | A run of at least ten horizontal rules, with the gaps a line apart, and at least four vertical rules |
| 4 | `labelled_diagrams` | One drawing covers 2.5% to 18% of the page and has 8 to 70 letter-sized components on it |
| 5 | `small_footnotes` | A short rule in the bottom fifth, 10% to 32% of the page width, has at least ten glyphs under it, and the band from that rule to the bottom is 4% to 15% of the page height |
| 6 | `gutter_corner` | Six or more glyph-sized components sit in a corner, or the page has a neighbour-page bleed. Gutter width is not a class: a dark edge is the common case on this scan |
| 7 | `grey_background` | Background unevenness is at least 0.20, or contrast is 0.35 or less |
| 8 | `landscape` | The page box is landscape, the scan is placed at 90 or 270 degrees, or the content is measured at 90 or 270. Upside down is not landscape |
| 9 | `multi_column` | Two or more text columns |
| 10 | `normal` | Nothing above matched |

`make book-ocr-pilot-select` reads a finished `classes` stage and writes
`<run-dir>/pilot/selection.json` (`llb.book-ocr-pilot-selection` 1.0.0). The default is 25 pages
at seed 0. A smaller size is accepted, so a test can ask for one page. Quotas are equal across
the classes that have a member, and the remainder goes to the earlier classes. Inside a class
the strongest pages form a head four times the quota, and the seed draws from that head. A
footnote's strength is how close its rule sits to the bottom. A page is selected once. A class
with no member is named in `empty_classes`; a class that had members but received no page is
named in `unfilled_classes`. Both use the prose labels (`severe curvature`, `dense formulas`,
`large or irregular tables`, `labelled diagrams`, `small footnotes`, `gutter and corner text`,
`grey or shadowed background`, `landscape`, `multi-column layout`, `normal`). `--out` copies the
same file. The handbook list is
`samples/book_ocr/space-mission-engineering-the-new-smad/pilot_pages.json`: the source checksum,
the PDF page numbers, the classes, and the signals. It holds no page image and no recovered text.

How a routed engine stage consumes the class is [the block ledger](block-ledger.md#engines).
The modules are `src/llb/prep/book/classes/`, `src/llb/prep/book/stages/classes.py`, and
`src/llb/cli/book/pilot.py`. Tests are `tests/llb/prep/book/test_page_classes.py` and
`tests/llb/prep/book/test_page_class_stage.py`.

### Evidence

**What ran.** On 2026-11-22, on the GeForce RTX 5050 laptop (8 GB), CPU only. OpenCV measured
the content signals and the GPU was idle. The input was every page of the 1,067-page *Space
Mission Engineering* scan (SHA-256
`89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`, 163,709,872 bytes), through
`render`, `signals`, and `classes`. `make book-ocr-pilot-select` then drew 25 pages at seed 0.

**The reading.** Every declared class has at least one page, and the pilot of 25 fills all ten.
Membership is how many pages carry the class; primary is how many route on it.

| Class | Members | Primary | Pilot pages |
| --- | --- | --- | --- |
| severe curvature | 26 | 26 | 154, 351, 765 |
| dense formulas | 41 | 38 | 258, 977, 979 |
| large or irregular tables | 59 | 58 | 768, 963, 966 |
| labelled diagrams | 143 | 120 | 350, 426, 473 |
| small footnotes | 72 | 56 | 53, 102, 131 |
| gutter and corner text | 40 | 34 | 316, 845 |
| grey or shadowed background | 78 | 56 | 638, 1006 |
| landscape | 45 | 31 | 132, 135 |
| multi-column layout | 667 | 425 | 1007, 1017 |
| normal | 223 | 223 | 4, 18 |

Checked against the scan: pages 258, 977, and 979 are display equations (977 and 979 are the
Kepler-equation appendix), and pages 53, 102, and 131 each have a footnote. Page 131 is placed
upside down, which is not the landscape class. Pages 1007 and 1017 are three-column index pages.
They rank ahead of ordinary two-column body pages because the strength is the column count.

**What would overturn it.** The cuts were read on this one greyscale handbook. A different scan
can move a page across a cut. The stage does not read characters. PDF page 1043 of this scan is
a dense numeric parameter table and is not `large_tables`, because its horizontal rules do not
form a run of ten. A conversion-factor table can still match `dense_formulas`, because each row
has an empty gap before a short right-hand mark. A footnote with no rule is missed, and a short
rule that is not a footnote can be counted. Changing a threshold reclassifies, because the cuts
are in the cache key.

The task record is
[0009](../../records/0009-scanned-book-recovery-book-pilot-page-selection.md).

## Not installed, and why

Recorded with the blocker and what would reopen each, so the shortlist's absences are auditable
(`make book-survey-list` prints them with the rest).

| Candidate | Stage | Why not here | Reopens when |
| --- | --- | --- | --- |
| ScanTailor Advanced 1.2.1 (GPL-3.0) | preprocess | GUI-only: no batch command-line interface, only an AppImage and a .deb | a CLI returns |
| unpaper 7.0.0 (GPL-2.0) | preprocess | needs `sudo apt install`, unavailable to the agent; binarises by design | installed, as an OCRmyPDF `--clean` variant |
| pix2tex 0.1.4 (MIT) | formula | unmaintained since January 2025; every converter and VLM smoked here emits LaTeX | never, unless the others fail on formula crops |
| `gemma4:31b` (vision, 19 GB on this host) | correction | larger than the 12 GiB card; it would run partly on the CPU | a larger card |

## External services

No external service was installed or called. Each would send every page image of a copyrighted book
off the host, which the operator's declared redistribution class and the
[egress boundary](../../../design/spec.md#data-egress-boundary) refuse by default: Mathpix, the
Datalab hosted API (Marker/Chandra), Mistral OCR, Azure AI Document Intelligence, Google Document
AI, and Amazon Textract. What would justify one is a pilot result, not a reputation: a page class on
which every local pipeline misses formula or table fidelity that the service measurably recovers on
the same pilot pages, for a book whose redistribution class permits the egress.

## Findings the next tasks inherit

Each of these is something a later task's contract already covers; the survey supplies the
instance it must handle.

- **Page-level VLMs invent figure content.** Surya 2.5 (through Docling), Chandra 2, and
  LightOnOCR-3 each wrote a description of the figure the page does not print ("Technical
  illustration: Line graph showing a linear increase..."), and LightOnOCR-3 also emitted a data
  table of the synthetic chart with values that appear nowhere on the page. On the real page
  Chandra 2 wrote a paragraph naming diagram labels the scan does not show. The spec forbids
  generated descriptions and captions, so a VLM adapter must keep a figure as its image region and
  drop whatever the model writes for it (`book-block-ledger-and-engine-adapters`,
  `book-figure-and-caption-assets`). PaddleOCR-VL-1.6,
  MinerU, Docling's standard pipeline, Marker `balanced`, dots.ocr, DeepSeek-OCR-2, and olmOCR-2
  emit the figure as an image region only -- for Surya 2.5 the pipeline around the model decides.
- **A dropped term reads as a complete equation.** dots.ocr and GLM-OCR both wrote (10-28)'s third
  line as `v_spc(t_1) = v_E(t_1) + omega_E(t_1 - t_0)`, silently losing `= v_E(t_0)`; the result
  still parses. Formula fidelity has to compare terms, not parse validity.
- **The minus sign is the most common dangerous substitution.** Every engine except GLM-OCR and
  olmOCR-2 wrote
  `−3.2` or `−98.2` with an ASCII hyphen or an en dash outside math, and Tesseract, OCRmyPDF, and
  Marker's fast mode dropped the sign altogether. A hyphen-minus is ambiguous, not wrong, but a
  lost sign is a silent value change; the metrics report them apart (`book-recovery-quality-metrics`).
- **Latin/Greek lookalikes come in two kinds.** Docling's standard pipeline wrote `∆` (U+2206
  INCREMENT) for the printed Greek `Δ`, and granite-docling read `ω` as `0`. On the real page the
  formula variable is a Times-italic glyph that MinerU and Paddle's formula model read as Greek `ν`
  and the VLMs read as Latin `v` -- PaddleOCR-VL and PP-StructureV3 even mix both on the one page;
  the text defines it as a longitude, and the pilot's reference transcription has to settle it.
- **Equation numbers are dropped by some readers.** PaddleOCR-VL-1.6 lost every equation number on
  both pages (`(3-4)` and `(10-28)` to `(10-32)`); MinerU writes them as `\tag{}`, the Datalab and
  LightOn VLMs as `\quad (n)`.
- **Dewarping can cut content.** page-dewarp's default page margins cropped text at both edges of
  the real page and shrank it; `-x 0 -y 0` keeps everything. UVDoc kept the real page whole but cut
  the page number off the bottom of the synthetic page. The stroke-preservation and content guard
  of `book-page-signals-and-variants` must check a dewarped variant for lost lines, not only lost
  strokes.
- **Preprocessing does not fix recognition.** Through Tesseract, the OpenCV, page-dewarp, and UVDoc
  variants of the synthetic page kept 4-5 of 11 dangerous tokens, the same as the raw page:
  Tesseract loses minus signs and Greek letters whatever the geometry.
- **A silent fallback looks like success.** Without the NVIDIA Container Toolkit, Marker's
  `balanced` mode cannot start its VLM server (`unknown or invalid runtime name: nvidia`), logs
  "falling back to Indra full-page OCR", and exits 0 with fast-mode output. The survey fails it on
  that line; with its own Surya server attached the same mode gave the best real-page reading.
- **A served model's fit is a configuration, not a size.** olmOCR-2's 8.3B FP8 weights fit the
  12 GiB card only with vLLM reserving 0.95 of it and with pages at the 1288 px its own pipeline
  uses: at 0.92 vLLM refused to start (`0.66 GiB KV cache is needed, which is larger than the
  available KV cache memory (0.65 GiB)`), and the real page at native size ran the vision encoder
  out of memory mid-request (HTTP 500). The smoke reports both as `over-vram-budget`.
- **Starting a vLLM server costs minutes.** Every launch took 140-200 s with the weights already
  cached; a stage that served one model per page would spend more time starting than reading.
- **An engine port can be quietly incomplete.** PP-StructureV3 on PaddleOCR's Transformers engine
  returned an empty synthetic table (`Resampling is not supported in SLANeXt`) and on the real page
  ran words together and dropped letters ("Geomryof DeartureaArir"). On the PaddlePaddle engine the
  synthetic text was clean and the table cells present, so part of that was the port -- but the
  real page still lost letters and a digit there (`355.45` as `55.45`).
- **Exit status is not output status.** MinerU `standard` wrote a complete reading of the real page
  and then segfaulted (exit -11) at teardown, on both runs; on the synthetic page it exited 0. An
  adapter has to validate what an engine wrote, not trust or distrust its exit code alone.
- **`think: false` is not always honoured.** The `qwen3-vl:8b` tag reasons inside `<think>` anyway;
  on the dense real page it spent its entire budget (4,096 tokens, then 12,288) reasoning and wrote
  nothing. Ollama keeps that reasoning out of `response`, so the result is an empty reading, which
  the smoke records as a failure rather than a blank page.
- **A page reader's coverage can change between identical-looking calls.** GLM-OCR read the
  synthetic page fully, but on the two-column real page one run returned only the right column,
  silently, and another that differed only in the `think` flag returned both. It is a region reader
  -- its own SDK runs a layout detector first -- and belongs behind one. Qwen2.5-VL read the right
  column before the left. Coverage has to be checked per page, which is what the block ledger's
  page-to-block accounting is for.
- **The degree sign is a character, too.** Surya 2.5 wrote `100.46°` as `100.46` plus U+FFFD.

## Evidence

**What ran.** On 2026-11-19 on the host above: `make book-survey-install` for the six environments
(the `light` environment was also deleted and rebuilt from scratch through the same target: 60
packages in 9 s from the warm cache, with its freeze file), then `make book-survey-smoke` over all
34 registry rows on the synthetic page and on PDF page 300 of the 1,067-page target handbook scan,
one candidate at a time, followed by a second smoke of the rows the first one changed -- page-dewarp
with zero margins, MinerU `standard` again, PP-StructureV3 and PaddleOCR-VL on the PaddlePaddle
engine, and the four Ollama models with thinking off. Synthetic scores are the rescored ones
(`make book-survey-rescore`): the first scorer version split LaTeX scripts into words and read a
math-mode hyphen as a hyphen, which under-read every VLM. Ollama 0.32.15 served the vision models.

On 2026-11-20, after the NVIDIA Container Toolkit 1.20.2 was installed and Docker's data root had
room, `make book-survey-smoke` ran the four vLLM-served rows (Marker `balanced`, dots.ocr,
DeepSeek-OCR-2, olmOCR-2) on the same two pages from `vllm/vllm-openai:v0.25.1`, then twice more
for the rows its first pass failed: DeepSeek-OCR-2 after the request stopped asking for a fixed
8,192-token budget its 8,192-token context could not hold (HTTP 400), and olmOCR-2 at a 0.95
reservation and its own 1288 px input (see the findings).

**The reading.** Every stage has at least one local candidate that installs and runs within the
12 GiB card, so no stage is a negative result:

- **Preprocessing:** UVDoc (GPU, 0.4 GB) and page-dewarp (CPU, with zero margins) for curl; OpenCV
  for skew, illumination, and the page signals. None of them changed what Tesseract recovers: the
  dangerous losses are recognition, not geometry, at this degradation.
- **Layout and reading order:** the converters' own layout models (MinerU's, PP-DocLayoutV3 in
  PaddleOCR-VL, Docling's Heron) and the page VLMs; line engines alone interleave columns.
- **Line OCR:** RapidOCR's PP-OCRv6 (best word recall of the line engines, 0.964, on the CPU) behind
  a layout engine; Tesseract remains the baseline it must beat.
- **Formula:** MinerU (both tiers), MinerU2.5-Pro, Surya 2.5, LightOnOCR-3, PaddleOCR-VL, Docling's
  enrichment, DeepSeek-OCR, and Qwen2.5-VL all emit LaTeX for the real page's five display
  equations; they differ on equation numbers and on `v` against `ν`, which is what the pilot's
  formula metric must see.
- **Table:** Docling's TableFormer, MinerU `standard`, PaddleOCR-VL, and the page VLMs read the
  synthetic table exactly; MinerU `basic` merged its rows, PP-StructureV3 on Transformers emitted it
  empty.
- **Page readers served by vLLM:** Marker `balanced` (Surya 2.5) gave the most complete real-page
  reading of the survey, dots.ocr the best synthetic agreement of any page reader with layout boxes
  per block, DeepSeek-OCR-2 boxes and complete formulas but no table structure, and olmOCR-2 true
  minus signs within the narrowest VRAM margin.
- **Correction:** Qwen3-VL 8B (7.4 GB) read the synthetic page best of any candidate and Qwen2.5-VL
  7B (6.1 GB) close behind, each leaving about 5 GB of the card free; both are region readers for
  `book-evidence-constrained-correction`, not page readers, and GLM-OCR (2.8 GB) is the cheap
  region reader for the same stage.

**Boundaries.** One synthetic page and one real page are a smoke, not a measurement: the numbers
say a tool runs here, at what cost, and whether it kept the tokens a misreading corrupts on these
two pages, never which tool is best on the book. The VRAM figures are single-page peaks on a
12,227 MiB card with nothing else resident; a batch run may hold more. Timings include model load
and, for a tool's first use, model download. The real page has no transcription, so its readings
were compared by eye.

**What would overturn it.** A shortlisted candidate failing to reinstall from its pins on this host,
or exceeding the card on a pilot page; a not-installed row's blocker going away; or a new release
of a candidate changing its smoke reading -- the survey re-runs from its registry with
`make book-survey-smoke`.
