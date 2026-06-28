# Dev setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13) on any host. On **Debian/Ubuntu**,
`make venv` installs OS packages from [scripts/apt/](../../scripts/apt/) (`sudo apt-get` when
needed).

    make venv     # apt + .venv + package + extras + .env
    make test     # unit tests
    make ci       # lint (ruff) + tests (GitHub CI)
    make          # list targets

`make venv` installs every Python extra below so a fresh checkout can run every command without a
follow-up `uv pip install`. It is a larger one-time download; for a lean install trim it,
e.g. `make venv EXTRAS=dev` (or `EXTRAS=rag,eval` for the RAG core path).

Set `SKIP_APT=1` when apt is unavailable (macOS, minimal CI images) -- the Python venv still
builds; only the OS package step is skipped.

## Apt dependencies (Debian/Ubuntu)

Lists live under [scripts/apt/](../../scripts/apt/). Install manually with:

    make apt-deps                      # production profile (default)
    make apt-deps APT_PROFILE=dev    # dev-only packages (shellcheck)
    make apt-deps APT_PROFILE=all    # production + dev

`make venv` installs **production** packages always, and **dev** packages when `EXTRAS` includes
`dev` (the default full install). Use `APT_DRY_RUN=1` to print missing packages without
installing.

| Profile | Packages | Used for |
| ------- | -------- | -------- |
| **production** | `git`, `make`, `curl` | Makefile, git vLLM builds, HTTP probes |
| **dev** | `shellcheck` | `scripts/code_quality.sh` shell lint |

Production packages are safe on eval/GPU hosts. Dev packages are optional for contributors;
GitHub CI does not run `make venv` and does not install them.

The installer uses `apt-get install --no-upgrade` so a small dev package (for example
`shellcheck`) does not pull in pending kernel or NVIDIA DKMS upgrades. If apt still exits
non-zero because of **unrelated** broken packages on the host, `make venv` continues when
the requested profile packages are verified installed.

### Apt troubleshooting (broken dpkg / NVIDIA DKMS)

If `apt install` fails with errors about `linux-headers-*`, `nvidia-dkms-*`, or
`Sub-process /usr/bin/dpkg returned an error code (1)` while installing an unrelated package,
the dev tool may still be installed. Check:

    shellcheck --version
    dpkg -s shellcheck | grep ^Status

When `Status: install ok installed`, you can use `scripts/code_quality.sh` even though apt
reported errors configuring kernel/NVIDIA packages that were already pending on the system.

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
| `dev` | pytest, ruff, mypy, radon, complexipy, pymarkdownlnt | tests, lint, code quality |
| `goldset` | datasets | `ingest_squad --hf-dataset` |
| `rag` | faiss-cpu, sentence-transformers, langchain, DeepEval | index + judge |
| `eval` | langgraph | retrieve -> generate eval graph (`run-eval`) |
| `track` | mlflow, duckdb, pyarrow, optuna | tracking + config search |
| `board` | streamlit | leaderboard |
| `prep` | litellm | frontier-API prep utils |
| `telemetry` | nvidia-ml-py, psutil | GPU/host telemetry |

GitHub CI installs only `.[dev]` (it never runs `make venv`), so the lint+test job stays
light and never pulls the heavy/eval deps. vLLM / torch / flash-attn are hardware-matched
(host CUDA/GPU) and installed via a separate path per [AGENTS.md](../../AGENTS.md), never as
plain deps.

## Conventions

- Runtime output under `.data/` (gitignored); secrets in `.env` (gitignored).
- Resolve paths from the project root; never hardcode absolute home paths.
- ASCII in logs/comments; UTF-8 only in data payloads.
