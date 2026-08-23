# Dev setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13) on any host. On **Debian/Ubuntu**,
`make venv` installs OS packages from [scripts/apt/](../../../scripts/apt/) (`sudo apt-get` when
needed).

    make venv     # apt + .venv + package + extras + .env
    make test     # unit tests
    make ci       # lint (ruff) + types + complexity/shell gates + tests (GitHub CI)
    make          # list targets

`make venv` installs every Python extra below so a fresh checkout can run every command without a
follow-up `uv pip install`. It is a larger one-time download; for a lean install trim it,
e.g. `make venv EXTRAS=dev` (or `EXTRAS=rag,eval` for the RAG core path).

Versions come from **`uv.lock`**, via `uv sync --inexact`. GitHub CI syncs the same lock
(`--locked`), so `make ci` locally and the workflow run the same ruff, mypy, and library versions
-- an unpinned install on either side is how a lint or type error lands in CI that no local run can
reproduce. Two consequences:

- Editing dependencies in `pyproject.toml` makes the next `make venv` refresh `uv.lock`.
  **Commit the updated lock**: CI syncs `--locked` and fails on a stale one.
  `make venv VENV_LOCKED=1` reproduces that failure locally instead of relocking.
- `--inexact` leaves packages the lock does not name in place, so vLLM and any separately installed
  extra survive a re-run. It is **not** a promise about `torch`: torch IS in the lock's resolution
  (via sentence-transformers), so every sync installs the LOCKED torch over the one vLLM pinned.
  `make venv` says so and reinstalls vLLM afterwards to put it back -- `VENV_INSTALL_VLLM=0`
  included, since a skipped reinstall leaves a torch the serving stack cannot use.

**`make venv` states which it is doing: reuse or rebuild.** `.venv/pyvenv.cfg` records the python
version at creation time, so an OS upgrade (say 3.13.14 -> 3.13.15) leaves it behind the real one,
and uv then treats the environment as stale and REPLACES it -- discarding vLLM, flashinfer, and the
CUDA wheels, and putting the lock's torch back instead of the one vLLM pinned. `make venv` checks
that before syncing and refuses a rebuild nobody asked for, naming what it would discard:

    make venv-restamp          # patched python, same 3.x line: record it and keep the stack
    make venv RECREATE_VENV=1  # accept the rebuild (the vLLM reinstall is then forced)

Prefer the restamp: a CPython patch release keeps the ABI, so the venv already runs the new
interpreter and only its recorded version is stale. A MINOR move (3.13 -> 3.14) is refused there --
that venv really does need rebuilding. `LLB_VENV_STALE_GUARD=report` syncs anyway and `=off` skips
the check; a rebuild that discarded vLLM reinstalls it even under `VENV_INSTALL_VLLM=0`.

**Add an extra with `make install-extras`, never a bare `uv pip install`.** uv's pip interface has
no lockfile, so `uv pip install -e ".[review]"` re-resolves the WHOLE requirement set and takes the
newest version each specifier admits -- on a clean venv the `dev` extra alone lands ten declared
packages off the lock, `ruff` and `mypy` among them, which is the split verdict the pins exist to
prevent. `make install-extras` runs the same install under a `uv.lock`-derived constraint:

    make install-extras EXTRAS=review,pdf-quality   # add extras, held to uv.lock
    make lock-drift                                 # name anything already off the lock

`make lock-drift` prints each off-lock package with the extra that declares it and the single
`make install-extras EXTRAS=...` that puts them all back; it exits non-zero when anything drifted,
and `LLB_EXTRAS_LOCK_GUARD=report` downgrades that to a warning while deliberately testing an
upgrade. Only packages `pyproject.toml` declares are constrained -- vLLM/torch stay exactly where
`scripts/build_vllm.sh` put them, since they are hardware-matched and trail the lock on purpose.

`make venv` resolves uv's package link mode per host. If this checkout and uv's shared cache are on
different devices, it sets `UV_LINK_MODE=copy` to avoid failed cross-device hardlinks; otherwise it
uses uv's default. For one-off `uv` commands, load the same resolver first:

    source scripts/shared/common.sh
    llb_load_env
    uv run --extra dev python -m pytest

Set `SKIP_APT=1` when apt is unavailable (macOS, minimal CI images) -- the Python venv still
builds; only the OS package step is skipped.

## Apt dependencies (Debian/Ubuntu)

Lists live under [scripts/apt/](../../../scripts/apt/). Install manually with:

    make apt-deps                      # production profile (default)
    make apt-deps APT_PROFILE=dev    # dev-only packages (currently none)
    make apt-deps APT_PROFILE=all    # production + dev

`make venv` installs **production** packages always, and **dev** packages when `EXTRAS` includes
`dev` (the default full install). Use `APT_DRY_RUN=1` to print missing packages without
installing.

| Profile | Packages | Used for |
| ------- | -------- | -------- |
| **production** | `git`, `make`, `curl` | Makefile, git vLLM builds, HTTP probes |
| **dev** | *(none)* | reserved for a future dev-only OS package |

Production packages are safe on eval/GPU hosts. The dev list is empty, so
`make apt-deps APT_PROFILE=dev` prints `nothing to install`; GitHub CI does not run `make venv`
and installs no apt packages at all.

**No apt package backs the shell lint.** The `dev` **extra** ships a pinned `shellcheck-py` wheel
(it bundles the real binary), so `.venv/bin/shellcheck` exists wherever `make ci` can run, and the
gate uses *that* binary only -- there is no `PATH` fallback, because a distro `shellcheck` is
releases behind the pin and would give the same commit a different verdict per host (see
[host validation](../../impl/current/host-validation.md#code-quality-checks)). A host whose venv
lacks the `dev` extra cannot run `make ci` regardless; `make venv EXTRAS=dev` is the fix.

The installer uses `apt-get install --no-upgrade` so a small dev package does not pull in pending
kernel or NVIDIA DKMS upgrades. If apt still exits non-zero because of **unrelated** broken
packages on the host, `make venv` continues when the requested profile packages are verified
installed.

### Apt troubleshooting (broken dpkg / NVIDIA DKMS)

If `apt install` fails with errors about `linux-headers-*`, `nvidia-dkms-*`, or
`Sub-process /usr/bin/dpkg returned an error code (1)` while installing an unrelated package,
the requested tool may still be installed. Check it by name, e.g. for `curl`:

    curl --version
    dpkg -s curl | grep ^Status

When `Status: install ok installed`, the tool is usable even though apt reported errors
configuring kernel/NVIDIA packages that were already pending on the system (this is also why
`make venv` continues once it verifies the requested packages).

To repair the host package manager (run when convenient; may take several minutes):

    sudo dpkg --configure -a
    sudo apt-get -f install

NVIDIA DKMS "already installed at version ... override by specifying --force" usually means
the kernel modules are already present under `/lib/modules/<kernel>/kernel/nvidia-595/`
but DKMS status shows `built` instead of `installed`. The GPU may still work on the
running kernel (`nvidia-smi`); only dpkg configuration is stuck.

Register the built modules with DKMS (safe when versions match; requires sudo):

    sudo dkms install nvidia/595.71.05 -k "$(uname -r)" --force
    # Repeat for each half-configured HWE kernel, e.g.:
    sudo dkms install nvidia/595.71.05 -k 6.17.0-29-generic --force
    sudo dpkg --configure -a
    sudo apt-get -f install

Verify: `dkms status` should show `installed` for each kernel, and
`dpkg -l | awk '$1 ~ /^(iF|iU|iH)$/'` should print nothing.

If `--force` still fails, inspect with `dkms status` and consider removing unused old
HWE kernels (`sudo apt autoremove --purge`) after the running kernel is healthy. Full
driver reinstall is a last resort on Ubuntu (`ubuntu-drivers`/NVIDIA docs).

This is independent of loc-lm-bench.

## Python extras (what each group provides)

The groups installed by `make venv` (and what `EXTRAS=` selects from):

| Extra | Pulls | For |
|-------|-------|-----|
| `dev` | pytest, ruff, mypy, radon, complexipy, pymarkdownlnt, optuna | tests, lint, code quality, lightweight search fakes |
| `goldset` | datasets | `ingest_squad --hf-dataset` |
| `rag` | faiss-cpu, sentence-transformers, langchain, DeepEval | index + judge |
| `rag-chroma` | chromadb | Chroma vector-store adapter |
| `rag-qdrant` | qdrant-client | Qdrant vector-store adapter |
| `rag-lancedb` | lancedb | LanceDB vector-store adapter, opt-in |
| `eval` | langgraph | retrieve -> generate eval graph (`run-eval`) |
| `track` | mlflow, duckdb, pyarrow, optuna | tracking + config search |
| `board` | streamlit | leaderboard |
| `prep` | litellm | frontier-API prep utils |
| `telemetry` | nvidia-ml-py, psutil | GPU/host telemetry |
| `finetune` | peft, trl, accelerate, datasets, optuna | LoRA training and hparam search |
| `pdf-quality` | Docling, Unstructured, MarkItDown OCR/layout helpers | scanned-PDF recovery and parser probes |

`make venv` includes the Chroma and Qdrant vector-store extras so the full local suite runs their
live adapter checks without skips. GitHub CI installs only `.[dev]` (it never runs `make venv`), so
the lint+test job stays light and never pulls the heavy/eval deps. On CUDA hosts, `make venv`
also runs the repo-managed vLLM binary-wheel installer (`VENV_INSTALL_VLLM=auto`); use
`VENV_INSTALL_VLLM=0 make venv` for a lean environment. vLLM / torch / flash-attn remain
hardware-matched and are installed through `scripts/build_vllm.sh`, not as plain pyproject deps.
CrewAI remains a dedicated environment because its pins conflict with the dev/RAG/vector lanes.

`pdf-quality` is opt-in because OCR/layout packages are large. `make apt-deps` installs the system
helpers used by that path: `poppler-utils`, `libmagic-dev`, `tesseract-ocr`, `tesseract-ocr-eng`,
and `tesseract-ocr-ukr`. Install the Python extra on transform hosts with:

    make install-extras EXTRAS=pdf-quality

Marker is not part of `pdf-quality` because it pulls a hardware-matched torch stack. Install
`marker-pdf` only in the dedicated CUDA transform environment when benchmarking it explicitly.

## Heavy compilation (ninja / cmake / CUDA)

Any installation that compiles C++/CUDA from source (git+, --no-binary, --no-build-isolation)
MUST cap parallelism using the formula MAX_JOBS=min(cpu_core_num//2, RAM // 14)

Do not inline the formula -- the helpers are the single source of truth. The canonical helper is
`llb_max_jobs()` in `scripts/shared/common.sh` (source it; see `scripts/build_vllm.sh` for usage).

Only wheels deliberately built from a local git checkout (flash-attn forks, vLLM forks, xformers
forks, etc.) may be exported under
`$DATA_DIR/wheels/<package-name>_<abi-key>_git<revision>/`. The key MUST encode the
ABI-relevant dimensions (Python, torch, CUDA, GPU compute capability) and the exact git
revision; source checkouts must be clean before building.

Registry wheels, prebuilt wheels, and all ordinary build/runtime dependencies MUST be installed
directly with `uv` and left in uv's standard shared cache. Never use `pip wheel` or a
dependency-resolving wheelhouse under `$DATA_DIR/wheels`; that directory contains only
intentional local-source build outputs.

## Conventions

- Runtime output under `.data/` (gitignored); secrets in `.env` (gitignored).
- Resolve paths from the project root; never hardcode absolute home paths.
- ASCII in logs/comments; UTF-8 only in data payloads.
