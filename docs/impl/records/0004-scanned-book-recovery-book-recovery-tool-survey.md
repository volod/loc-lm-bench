# book-recovery-tool-survey

## Work contract

```markdown
#### book-recovery-tool-survey

The field for every recovery stage moves faster than any package description: page dewarping and
illumination correction, layout and reading-order analysis, OCR, formula-to-LaTeX, table structure,
document-to-Markdown converters, and the vision models Ollama serves. Pinning one by reputation would
decide the pilot before running it. Survey current versions from official documentation, source
repositories, current benchmarks, and papers, and prove which candidates actually install and run on
the 16 GiB CUDA host without disturbing the project environment.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: RUN NEEDED -- each candidate's install and one-page smoke runs on the CUDA host.
- Research: yes
- Dependencies: none. The `pdf-quality` extra and the Marker isolation note in `pyproject.toml` are
  the precedent for keeping a hardware-matched torch stack out of the base environment
  ([dev setup](../guides/development/dev-setup.md)).
- User-visible outcome: a recorded shortlist per stage -- candidate, version, licence, local or
  external, measured VRAM and RAM on this host, the page classes it claims to handle, its install
  route (optional extra, isolated environment, or Ollama model), and its one-page smoke result on a
  synthetic scan -- so the pilot compares only tools that demonstrably run here.
- Scope boundary: in scope -- the operator's indicative candidates (PaddleOCR and PaddleOCR-VL with
  PP-Structure, MinerU, Marker, Docling, Surya, OCRmyPDF/Tesseract, ScanTailor-derived processing,
  OpenCV preprocessing, dedicated formula recognisers, Ollama-served vision models) plus any
  stronger current candidate the survey finds; an optional extra or isolated environment for each
  shortlisted one; every external service listed with its privacy implication and the quality gain
  that would justify it. Out of scope -- choosing the pipeline (`book-pilot-pipeline-comparison`),
  wiring candidates into the ledger, and any candidate whose licence forbids local use.
- Data and artifact paths: optional extras in `pyproject.toml`; smoke outputs under
  `$DATA_DIR/book-ocr/<run_timestamp>/survey/`.
- Execution path: install each candidate into its declared extra or environment, capping any source
  build with `llb_max_jobs()`; a one-page smoke on a synthetic fixture page under the GPU isolation
  contract of [heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md).
- Acceptance gates: `make ci` green with every candidate absent from the base install; each
  shortlisted candidate has a reproducible install command and a smoke result on this host; a
  candidate that fails to install or exceeds the VRAM budget is recorded with the failure rather
  than dropped silently; a stage no local candidate serves is a negative result that names what
  would reopen it.
- Documentation target: `docs/impl/current/book-recovery/tool-selection.md`, and
  [dev setup](../guides/development/dev-setup.md) for the install routes.
```

## Implementation and decisions

The survey is a tool, not a one-off: `llb book-ocr survey` (`list`, `install`, `smoke`, `rescore`)
behind `make book-survey-*`, over one registry (`src/llb/prep/book/survey/candidates.py`) that holds
every candidate the survey considered -- installed or not -- with its stages, version, licence,
locality, install route, claimed page classes, smoke command, and, for one not installed, the
blocker and what would reopen it. Where it lives, how a smoke is measured and read, and the
shortlist: [tool selection](../current/book-recovery/tool-selection.md).

Decisions the contract left open:

- **Isolated environments, not `pyproject.toml` extras.** The contract allowed either. Every
  shortlisted stack pins its own torch, ONNX Runtime, or PaddlePaddle build (Docling and Marker
  each pull torch 2.15.0+cu132; the native Paddle engine needs a vendor-index wheel), so none can
  share `.venv` without moving what `make ci` runs on. They follow the `encoders-legacy` precedent:
  uv virtualenvs under `$DATA_DIR/venvs/book-<env>/`. No extra was added and `uv.lock` is unchanged.
- **Reproducible without a lock file.** Each environment resolves exact top-level pins with
  `--exclude-newer` at the survey cutoff and freezes the full resolution beside it. The
  PaddlePaddle index publishes no upload dates, so that one package is exempted by name.
- **Every cache on the data volume.** uv's cache and every model cache a tool honours go under
  `$DATA_DIR/cache/` (see the finding below on why).
- **One synthetic page with a transcription, plus one real page read by eye.** The contract asked
  for a smoke on a synthetic scan; the operator also pointed the survey at the target handbook, so
  PDF page 300 (formulas, figure, two columns, curl) was read by every candidate too. Its outputs
  stay under the run directory; nothing from the book enters git.
- **A smoke score, deliberately not the pilot metric.** Word recall, character agreement, and 11
  dangerous tokens read as exact, hyphen-for-minus, or missing -- enough to tell "reads the page and
  keeps values" from "does not", and no more. The metric harness is
  `book-recovery-quality-metrics`. The scorer reads LaTeX as it typesets; its first version split
  `I_{sp}` into words and read a math-mode hyphen as a hyphen, which under-read every VLM, so
  `rescore` recomputes the scores from saved readings without a GPU.
- **A silent fallback is a failure.** Marker's `balanced` mode exits 0 after falling back to its
  CPU mode when it cannot start its VLM server; a candidate row may declare a log line that fails
  the smoke (`fail_on`).
- **VLMs through Docling's Transformers backend.** Surya 2.5, Chandra 2, LightOnOCR-3, and
  MinerU2.5-Pro ship for vLLM in Docker, which needs the NVIDIA Container Toolkit this host lacks;
  Docling 2.150's VLM pipeline runs each of them on Transformers in one environment, which is how
  they were smoked here.
- **The host is a 12 GiB card, not the 16 GiB the contract names.** The budget every VRAM reading
  is held to is the measured 12,227 MiB of this RTX PRO 3000 Blackwell laptop GPU.

## Acceptance evidence

Status: accepted.

Survey on 2026-11-19 on the RTX PRO 3000 Blackwell laptop host (12,227 MiB GPU, 62 GiB RAM, Ollama
0.32.15), packages resolved as of the 2026-11-19 cutoff. Each gate:

- **`make ci` green with every candidate absent from the base install.** No `pyproject.toml` extra
  and no `uv.lock` change: every candidate lives in a `$DATA_DIR/venvs/book-<env>/` environment, an
  Ollama tag, or the system Tesseract. The 15 new tests (`tests/llb/prep/book/test_survey_*.py`)
  import nothing from any candidate stack. On 2026-11-19, `make ci`: every check green (format,
  ruff, mypy, complexity, shell lint, spec-plan, artifact contracts); its test suite 4,913 passed,
  49 failed, 4 errors -- every failure in the `rag`, `eval`, `ontology`, and `linkage` test
  modules, importing an extra this host's venv lacks (`faiss` 29, `langgraph` 10, `rdflib`,
  `splink`), the same count and cause as the previous task's record; none touches this work.
  `make ci-github` (the base-install gate): green, 4,902 passed, 6 skipped. `make lint-md` green.
- **Each shortlisted candidate has a reproducible install command and a smoke result on this host.**
  `make book-survey-install` built all six environments from exact pins with `--exclude-newer` and
  froze each; `light` was deleted and rebuilt from scratch through the same target (60 packages,
  9 s). `make book-survey-smoke` ran 34 registry rows on the synthetic page and on PDF page 300 of
  the target scan, then a second smoke of the eight rows the first one changed (page-dewarp with
  zero margins, MinerU `standard`, both PaddleOCR pipelines on the PaddlePaddle engine, the four
  Ollama models with thinking off). Every shortlisted row has its version, licence, route, VRAM,
  RAM, timing, synthetic score, and real-page reading in
  [tool selection](../current/book-recovery/tool-selection.md#shortlist-per-stage).
- **A candidate that fails to install or exceeds the VRAM budget is recorded with the failure.**
  Nothing exceeded the card: the largest was Chandra 2 at 10.8 of 12.2 GB. Recorded failures:
  Marker `balanced` (silent fallback, no NVIDIA Docker runtime), MinerU `standard` (segfault at
  teardown on the real page after a complete reading), Qwen3-VL 8B on the real page (an empty
  reading after its whole token budget went to reasoning), granite-docling on the synthetic page
  (read the heading only). Not installed, with blockers: ScanTailor Advanced, unpaper, olmOCR-2,
  dots.ocr, DeepSeek-OCR-2, pix2tex, the six external services, and models larger than the card --
  [not installed, and why](../current/book-recovery/tool-selection.md#not-installed-and-why).
- **A stage no local candidate serves is a negative result.** None: preprocessing, layout, OCR,
  formula, table, conversion, and correction each have at least one shortlisted local candidate, so
  this research task closes positive and needs no future-research section.

## Handoff

`make plan-status` before: 61 tasks (36 agent, 25 human). After: 60 (35 agent, 25 human). Only
`scanned-book-recovery` moved: this task left its group; its registry row stays `planned`, its link
text now naming the survey. Removing it releases the last `Requires` of
`book-block-ledger-and-engine-adapters` (now the agent priority task) and
`book-page-signals-and-variants`; each gained a one-line `Dependencies` pointer to
[tool selection](../current/book-recovery/tool-selection.md). The pilot
(`book-pilot-pipeline-comparison`) records its measured choice on the same page.

What the work surfaced, routed:

- **Chores done now.** The scorer's LaTeX reading and the `rescore` command; `think: false` on
  Ollama requests; the `fail_on` silent-fallback check; the exit code in a failure's detail;
  `MINERU_HOME` and the uv cache moved under `$DATA_DIR` after the first installs filled the root
  filesystem. The cache entries those installs left in the home directory (11.8 GiB of uv archives
  created by this session, 2 GB of MinerU models, three empty tool directories) were removed or
  moved to `$DATA_DIR/cache/`; the Ollama tags pulled for the survey (`glm-ocr:latest`,
  `deepseek-ocr:3b`, `qwen3-vl:8b`, 15.6 GB in the daemon's store) are kept, as the shortlist uses
  them.
- **More work under an existing capability: none new.** Each finding -- invented figure content,
  minus-sign and Latin/Greek substitutions, dropped equation numbers, unstable page coverage,
  content-cutting dewarps, an exit code that disagrees with the output -- falls inside the contract
  of `book-block-ledger-and-engine-adapters`, `book-page-signals-and-variants`,
  `book-recovery-quality-metrics`, or `book-figure-and-caption-assets`; the topic page names which.
- **No capability the spec does not describe.** Installing the NVIDIA Container Toolkit (which
  would reopen the vLLM-served candidates) is a host change outside the agent's reach and is
  recorded as a reopening condition, not a task.
