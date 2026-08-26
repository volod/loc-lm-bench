# Host Validation

Host validation is the repeatable checklist for a CUDA workstation, plus the gates that decide
whether a change may land at all. It complements CI, which avoids network, model downloads, and
GPU-dependent paths -- and two of the pages below are about exactly that boundary: what the
non-slow tier promises, and how each promise is enforced instead of assumed.

This page is the AREA INDEX; each subject lives in its own page under
[`host-validation/`](host-validation/).

## Validating a host

| Page | What it answers |
| --- | --- |
| [Host acceptance paths](host-validation/acceptance-paths.md) | The cells to run on a fresh CUDA host -- core RAG, one per backend, robust-backend checks, category smoke, judge, platform matrix -- and the properties each should show |
| [Runtime version floors](host-validation/runtime-version-floors.md) | Why a correct host can still be unable to serve a roster entry, where an artifact's floor is read from, and what a host below one sees at resolution, preparation, and launch |
| [RTX PRO 3000 Blackwell 12 GiB acceptance](host-validation/gpu-tier-acceptance.md) | What the 12 GiB tier acceptance run measured, the configuration gap it exposed, and the encoder readings taken beside it |

## Gating a change

| Page | What it answers |
| --- | --- |
| [Quality gate](host-validation/quality-gate.md) | What `make ci`, `make test`, and `scripts/code_quality.sh` each check, the source-size refactor behind the soft limit, and why `.gitignore` is part of the gate |
| [The no-GPU and no-download tier guard](host-validation/no-gpu-tier-guard.md) | How the non-slow tier's two promises are enforced: denying a child the device, covering every spawn entry point, and what the denial still misses |
| [Interpreter reach coverage](host-validation/interpreter-reach-coverage.md) | How the guard's coverage claim is re-measured against the running interpreter, its stdlib, its installed packages, and the trees a `.pth` file adds -- and what each scan costs |
| [Complexity and shell gates](host-validation/complexity-and-shell-gates.md) | The Radon/Complexipy thresholds and the ShellCheck-plus-symbols scan, both pinned and both enforced, plus the informational sweep beside them |
