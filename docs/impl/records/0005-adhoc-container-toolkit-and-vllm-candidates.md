# Container Toolkit Requirement and the vLLM-Served Recovery Candidates

## Work contract

Direct request, after the recovery-tool survey (record
[0004](0004-scanned-book-recovery-book-recovery-tool-survey.md)) recorded four candidates as blocked
by the missing NVIDIA Container Toolkit: every development computer is a CUDA host, so require the
toolkit on a CUDA host; the operator installed it on this host; rerun the
[tool selection](../current/book-recovery/tool-selection.md) candidates that needed it, compare them
with the others, and make the selection with the additional tools in it.

## Implementation and decisions

- **The requirement.** `make venv` (`scripts/setup_venv.sh`, `llb_require_container_toolkit`)
  refuses a host whose `nvidia-smi` lists a GPU while `docker info` lists no `nvidia` runtime, and
  prints the install and `nvidia-ctk runtime configure` steps; a host without a GPU (CI) is not
  checked, and `LLB_REQUIRE_CONTAINER_TOOLKIT=0` skips it deliberately.
  [Dev setup](../../guides/development/dev-setup.md) states the rule and the room Docker's data root
  needs.
- **A `docker` route in the survey.** `src/llb/prep/book/survey/vllm_server.py` serves a candidate
  from the pinned `vllm/vllm-openai:v0.25.1` image in the survey's own container -- run as the
  invoking user, downloads under `$DATA_DIR/cache/book-survey/`, bound to 127.0.0.1, stopped and
  removed after its log is kept -- and `served.py` runs a smoke against it: one OpenAI-compatible
  chat request (`served` smoke kind) or a tool attached to it (`Smoke.env`, e.g. Marker's
  `SURYA_INFERENCE_URL`). The result types moved to `results.py` so `smoke.py` stays one concern.
  Tests: `tests/llb/prep/book/test_survey_served.py`.
- **Not the tools' own spawn.** Surya's vLLM spawn runs the container as root and mounts the home
  directory's Hugging Face cache, which would put root-owned weights back on the root filesystem;
  the survey serves Surya 2.5 itself and attaches Marker `balanced` to it.
- **A disk guard before any pull.** The image is 27.8 GB unpacked. Twice during this work Docker's
  data root sat on the root filesystem with 15 GB free (first a `data-root` key nested inside
  `runtimes.nvidia`, where Docker ignores it; then a directory under the home directory, still on
  `/`). The survey refused both times by name ("is not pulled and the Docker data root ... has
  15 GB free (needs 30 GB)") instead of filling the filesystem; the run went ahead once the
  operator freed the root filesystem to 162 GB.
- **Two container fixes the first attempts surfaced.** `--rm` deleted a crashed container before
  its log could be read, so the container is removed only after `docker logs` is kept; and the
  invoking uid has no passwd entry in the image, which broke torch's compile-cache path
  (`getpwuid(): uid not found: 1000`), so `USER`, `LOGNAME`, and `TORCHINDUCTOR_CACHE_DIR` are set
  inside.
- **KV-cache refusal is a VRAM-budget result.** vLLM refuses to start when its reservation leaves
  too little KV cache for the configured context; the smoke now reports that as
  `over-vram-budget`, not a generic failure.
- **Requests that respect the served context.** The first DeepSeek-OCR-2 request asked for a fixed
  8,192-token budget its 8,192-token context could not hold (HTTP 400); a served read now sends no
  `max_tokens`, so vLLM allows what the context leaves, and a rejected request carries the server's
  reason. olmOCR-2 reads at 1288 px on the longest side (`Smoke.max_image_side`), the size its own
  pipeline renders.

## Acceptance evidence

Status: accepted.

- **The requirement.** On this host `llb_require_container_toolkit` reports the `nvidia` runtime
  present; a test container (`nvidia/cuda:12.6.0-base-ubuntu24.04`) sees the RTX PRO 3000 Blackwell
  GPU (12,227 MiB, driver 620.70.06). `make shell-lint-gate` green.
- **The rerun.** On 2026-11-20, Container Toolkit 1.20.2, `vllm/vllm-openai:v0.25.1` (27.8 GB,
  pulled in 5 min), `make book-survey-smoke` ran Marker `balanced`, dots.ocr, DeepSeek-OCR-2, and
  olmOCR-2 on the synthetic page and on PDF page 300 of the target scan, then retried the two whose
  first pass failed. All four now run on the 12 GiB card. On the synthetic page (word recall /
  character agreement / dangerous tokens of 11): dots.ocr 0.982 / 0.997 / 9 -- the best agreement
  of any page reader in the survey -- Marker `balanced` 0.928 / 0.917 / 9, olmOCR-2 0.910 / 0.796 /
  11 (true minus signs), DeepSeek-OCR-2 0.775 / 0.535 / 8 (table structure lost). On the real page
  Marker `balanced` gave the most complete reading of the survey: every equation whole with its
  number and the figure kept as an image with no invented text. Each server start took 140-200 s
  with the weights cached. Readings, VRAM, and the reasons each candidate fails where it does:
  [tool selection, vLLM in Docker](../current/book-recovery/tool-selection.md#document-vlms-served-by-vllm-in-docker).
- **The selection.** [Selection for the pilot](../current/book-recovery/tool-selection.md#selection-for-the-pilot)
  now names Marker `balanced`, dots.ocr, DeepSeek-OCR-2, and olmOCR-2 among the page readers the
  pilot compares; the four left the "not installed" table.
- **Gates.** On 2026-11-20, `make ci`: every check green (format, ruff, mypy, complexity -- after
  `_availability` was split for the new route -- shell lint, spec-plan, artifact contracts); its
  test suite 4,926 passed, 49 failed, 4 errors, every failure an extra this host's venv lacks
  (`faiss`, `langgraph`, `rdflib`, `splink`), the same baseline as record 0004; `make ci-github`
  green, 4,915 passed, 6 skipped; `make lint-md` green.

## Handoff

No plan task moves: `make plan-status` counts 60 tasks (35 agent, 25 human) before and after, and
no capability's group or registry row changed. The selection feeds `book-pilot-pipeline-comparison`
and `book-block-ledger-and-engine-adapters` through the tool-selection page, which both already
link. Surfaced and done now: the Docker disk guard, the container's log kept before removal, the
uid fix inside the image, the KV-cache refusal and a mid-request out-of-memory read as
`over-vram-budget`, the context-sized request, and the per-candidate input size. Nothing new routed
to the plan: serving one model per batch rather than per page (the 140-200 s start) is inside the
engine-adapter contract already.
