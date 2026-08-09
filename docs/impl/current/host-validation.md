# Host Validation

Host validation is the repeatable checklist for a CUDA workstation. It complements CI, which avoids
network, model downloads, and GPU-dependent paths.

## Core RAG Path

```bash
make validate-goldset
make build-index
make validate-retrieval RAG_K=10
make run-eval MODEL=<fitting-ua-model> BACKEND=ollama LIMIT=20 TELEMETRY=1
```

Expected properties:

- the committed fixture validates;
- retrieval clears the configured recall gate;
- `run-eval` writes a manifest and per-case scores;
- telemetry records throughput and peak VRAM when NVML is reachable;
- MLflow mirroring does not replace the canonical bundle.

## Backend Paths

Run one small cell for each backend available on the host:

```bash
llb run-eval --backend ollama --model <ollama-tag> --telemetry --limit 20
llb run-eval --backend vllm --model <hf-repo> --telemetry \
  --max-model-len 8192 --gpu-memory-utilization 0.80 --evict --limit 20
llb run-eval --backend llamacpp --model <gguf-source> --telemetry \
  --max-model-len 8192 --gpu-layers -1 --limit 20
```

Check that each backend records the same manifest shape. For vLLM, inspect contention and sampler
fields. For llama.cpp, inspect served context and `n_gpu_layers`.

On 12 GiB CUDA hosts, pin embeddings to CPU before a vLLM probe so the embedder does not compete
with the served model for the last few hundred MiB. Use the generated config so the offloaded 12B
target carries its `cpu_offload_gb` and `kv_offloading_size_gb` settings into `run-eval`:

```bash
make gen-serving-config
LLB_EMBED_DEVICE=cpu llb run-eval \
  --config "$DATA_DIR/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml" \
  --evict --limit 1
```

## Robust Backend Checks

```bash
llb list-models --trust-config
llb preflight-vllm --force
llb detect-gpu-vram
llb gen-serving-config
```

When testing VRAM contention, prefer `--evict` or `--wait` before manual process intervention. The
contention guard should abort before launching a doomed vLLM server when headroom is insufficient.

## RTX PRO 3000 Blackwell 12 GiB Acceptance

The 2026-07-28 acceptance run used an NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
(12,227 MiB, compute capability 12.0), driver 610.43.02, PyTorch 2.11.0+cu130, and vLLM 0.24.0.
`make detect-gpu-vram` selected tier 12, and `make gen-serving-config` persisted the detected GPU
identity and memory rather than an override-only tier.

The host run exposed and fixed one shared configuration gap. The generated serving YAML and the
`validate-retrieval` / `run-eval` CLI plus make paths now carry `corpus_root`; before that change,
the published gold set could be paired with the unrelated default corpus and a freshly rebuilt
store immediately read as stale. Regression coverage lives in
`tests/llb/inference/test_inference_generate.py`,
`tests/llb/rag/test_validate_retrieval_cli.py`, and
`tests/llb/eval/test_run_eval_cli.py`.

Acceptance results:

- The 250-item published gold set passed validation. A CPU-pinned e5-base rebuild wrote 311 chunks,
  and `make validate-retrieval RAG_K=10` scored the 82-item final split at recall@10 0.976 and MRR
  0.838.
- The generated Gemma 4 12B vLLM config ran one item with embeddings on CPU. The contention guard
  accepted 0.90 utilization with 11,696 MiB free; native sampling, Triton attention, Marlin W4A16,
  16 GiB CPU weight offload, and 32 GiB KV offload served a 16,384-token context at 3.32 tok/s and
  11,511 MiB peak VRAM. The load took 246.07 seconds, chiefly CUDA-graph capture. FlashInfer 0.6.12
  could not supply its sampler on SM 12.0 and the recorded native-sampler fallback worked. Artifact:
  `$DATA_DIR/run-eval/20260728T065519.474285Z-2f08bcd131d7/`.
- The 20-item Ollama path used the Ukrainian MamayLM Gemma 3 12B Q4_K_M model with CUDA embeddings.
  It scored objective 0.406, reliability 1.0, retrieval recall@5 0.900 / MRR 0.787, 39.16 tok/s,
  and 9,932 MiB peak VRAM. Artifact:
  `$DATA_DIR/run-eval/20260728T075902.053333Z-7e94edc3fe16/`.
- llama.cpp was not an available backend on this host (`llama-server` was absent), so no llama.cpp
  cell was claimed.
- The repository gate selects only current implementation coverage: obsolete unpublished-artifact
  compatibility checks were removed rather than skipped. It passes 2,226 tests with 43
  opt-in/slow tests deselected and zero runtime skips. Ruff format/check, mypy, Markdown lint, and
  the code-quality report also passed. `ollama ps` was empty after the evidence runs.

The recent paired embedder, context-ablation, and local drafting evidence reruns are recorded in
[RAG core](rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty),
[RAG core](rag-core/embedders.md#blackwell-encoder-throughput-decomposition),
[RAG core](rag-core/context-ablation.md#context-ablation-evidence), and
[data prep](data-prep/drafting-lanes.md#sequential-local-qwengemma-draft-comparison).

Encoder throughput on this host (2026-07-29): `EMBED_ENCODER_THROUGHPUT=1` over the 311-chunk UA
fixture at the 80 W power limit. Warm CUDA rates are ~638 chunks/s for e5-small, 208 for e5-base,
~62 for e5-large and BGE-M3, and ~334 for the paraphrase model; cold load (~5.7 s) dominated the
earlier one-pass rates, but the e5-base vs large spread survives warm measurement. Prefer warm
chunks/s for host cost columns. e5-small is the named cheap CUDA alternative (~3.05x base, lower
peak VRAM) when quality is flat; the paired verdict still RETAINs e5-base on n=82. Artifacts:
`$DATA_DIR/encoder-throughput/20260729T131520.054732Z-1d36908e745c/` (full roster) and
`$DATA_DIR/encoder-throughput/20260729T133400.407347Z-c79df0776706/` (VRAM after release fix).
See [RAG core](rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small).

## Category Smoke Path

Run representative category commands with committed samples:

```bash
llb bench-security --model <model> --backend <backend>
llb bench-tooling --model <model> --backend <backend>
llb bench-agentic --model <model> --backend <backend>
llb bench-summarization --model <model> --backend <backend>
llb bench-structured --model <model> --backend <backend>
llb bench-text-analysis --bundle samples/text_analysis_bundle_uk \
  --model <model> --backend <backend>
```

Each category should write a tier-specific manifest and per-case score series under
`$DATA_DIR/<category>/<run>/`.

## Judge Path

```bash
llb judge-smoke --judge-model <judge> --judge-base-url <url>
make calibration-score
```

Use the smoke check before long judged category or RAG runs. Use the calibration score to decide
whether `JUDGE_RHO` is admissible for the run.

## Platform Matrix

```bash
make platform-matrix
```

Use this only after the individual backend paths are known to work. The matrix compares backend
serve paths for a common logical model base, not arbitrary unrelated checkpoints.

## Quality Gate

Run the repository checks after host-specific validation:

```bash
make ci
make lint-md
scripts/code_quality.sh
```

`scripts/code_quality.sh` always prints the largest tracked Python files and largest tracked
non-Python files. Root-file, markdown, shell, and complexity sections are quiet when clean and
appear only when they have findings, missing optional tools, or failures; a shell-lint or
complexity finding also exits the sweep non-zero (see
[Code quality checks](#code-quality-checks)).

`make test` is the full local precommit flow when slow tests are acceptable.

### Code quality checks

`make ci` checks Ruff formatting and lint, mypy, the acceptance-gate inventory, the complexity and
shell-lint gates below, and the non-slow pytest suite. `make test` adds the full local test flow and
Markdown lint; `make lint-md` also runs `make lint-doc-links` (`llb.quality.doc_links`), which
resolves every relative docs link -- file plus `#anchor` -- so the three-level
current-implementation tree cannot rot into unfindable pages. `scripts/code_quality.sh` is the
wider sweep: it reports long source files and runs both gates, so maintainers can split code at
functional seams. The ~250-line source-file target is soft; cohesive schemas and regular lookup
families may remain whole.

**Every check above only sees files the repo can see, so `.gitignore` is part of the gate.** The
packaging rules are anchored to the repo root (`/build/`, `/dist/`, `/lib/`, `/var/`, `/target/`,
...) because an unanchored directory rule matches that name at ANY depth. Unanchored `build/` hid
`tests/llb/build/test_build_helper.py` -- the only test of `llb_max_jobs`, the canonical
parallelism cap AGENTS.md names for heavy CUDA builds -- from every clone: pytest collected it and
`make ci` ran it on the one box that held the file, while `git ls-files` did not know it existed
and GitHub CI never ran it. Nothing failed, which is the whole problem. The rule now matches only
where setuptools/uv actually write (heavy build trees go under `$DATA_DIR`, covered by `.data/`),
the two `!src/llb/build/` negations that used to patch the rule one package at a time are gone, and
the file is committed. `tests/llb/quality/test_ignored_sources.py` holds the invariant: no `.py`,
`.sh`, or `.md` under `src/`, `tests/`, or `scripts/` may be ignored, and the packaging rules must
still catch root build output. The same blindness reaches the shell gate, which scans
`git ls-files --cached --others --exclude-standard` -- a script under an ignored tree is not linted
either. `git status --ignored --short` is the manual read.

**All four of those build tests run in the non-slow suite**, so `make ci`, `make ci-github`, and
GitHub CI verify the vLLM installer's behavior -- the prebuilt path installs `--only-binary :all:`
through uv's shared cache and writes no project wheelhouse, and the source path exports exactly one
ABI-keyed wheel from a clean checkout -- not only the two `llb_max_jobs` / `common.sh` assertions.
Two of them used to carry `@pytest.mark.slow` because an end-to-end run cost ~5.6s on this host.
The cost was not a resolver call the fake `uv` fails to intercept: it was the flashinfer sampler
preflight that `llb.build.vllm.main` ends in, an in-process probe that imports torch and
JIT-builds + launches a kernel on the real GPU (5.3s measured; the other run was fast only by
accident, because its fake `torch` fixture shadows the real one). GitHub CI would never have paid
it -- torch is absent there, so the probe raises `ImportError` and the verdict is `native` in
milliseconds -- which is exactly why the marker cost coverage where the code is trusted from while
buying nothing. `_seed_sampler_preflight_verdict` in `tests/llb/build/test_build_helper.py` now
pre-records a verdict with `driver: None`, which `verdict_is_current` accepts on every host, so the
installer reuses the cached verdict and the tests stay about wheels and uv calls: 5.6s -> 0.08s and
0.52s -> 0.14s. A unit suite that reaches the GPU is the general hazard here -- an end-to-end run of
an installer entrypoint executes everything that entrypoint does, including probes no shell stub
sits in front of.

**So the tier's no-GPU promise is a check now, not a convention.** `tests/conftest.py` wraps every
test in an autouse guard (`llb.quality.gpu_guard`) that watches this process and denies the device
to the children of an unmarked test. The watch snapshots two effects before the test and reads them
again after: `torch.cuda.is_initialized()`, and whether `flashinfer` is in `sys.modules` (its first
sampling call is the JIT build). An unmarked test that flips either one fails at
teardown, naming the test, what it did, and the ways out. Both reads come out of `sys.modules` and
neither imports anything, so the guard no-ops where torch is absent -- GitHub CI installs no torch
and must not start -- and costs two dict lookups per test: the whole non-slow suite runs in 101-102s
with it and 105s under `LLB_GPU_GUARD=off`, which is run-to-run noise on this box. Importing torch
is deliberately not a
finding: `import torch`, `torch.cuda.is_available()`, and a test's own fake `torch` module all leave
`is_initialized()` False, which is what keeps the many tests that pull torch in without touching the
device green.

**It refuses rather than reports, decided on the evidence.** All tests of the non-slow suite
are clean under it on this CUDA host, so a finding is a new violation on the commit that adds it
rather than an entry in a backlog nobody drains -- the cheap moment to enforce, the same reasoning
the complexity gates were taken at. A probe run on this host shows both halves of the verdict: a
test that runs `torch.rand(2, device="cuda")` and one that calls `_default_flashinfer_probe()`
(10.3s, almost all of it the JIT build) each fail, while a third that only imports torch and calls
`is_available()` passes. The escape hatches are markers: `slow`, because the intrinsically expensive
tier is where a real backend run belongs, and `gpu_env` for a quick test that must reach the device
anyway. `LLB_GPU_GUARD=report` downgrades a finding to a warning and `off` disables it; an
unrecognized value is refused rather than read as off, so a typo'd knob cannot quietly disable the
check it was aimed at.

**A CHILD process is denied the device rather than observed**, because there is nothing in this
process to observe it by. For the duration of an unmarked test, every spawn entry point is swapped
for one that starts the child with an empty `CUDA_VISIBLE_DEVICES` -- whatever environment the
caller passed. `subprocess.Popen` is the seam that `run` / `call` / `check_output` all reach for at
call time. That closes the shape that motivated the guard: `test_build_helper.py` drives
`scripts/build_vllm.sh` through `subprocess.run`, so its flashinfer probe lives in a python no
in-process fixture can inspect. Remove the seeded verdict as an experiment and the prebuilt-installer
test costs 5.99s under `LLB_GPU_GUARD=off` -- the child JIT-builds the sampler kernel on the real
GPU -- against 0.81s with the denial, where the child finds no device and the probe returns `native`
in milliseconds. It passes either way. The seeded verdict stays regardless: it is what keeps the
cost off a run with the guard disabled, and it states the intent at the call site.

**Child-only is the mechanism, not a shortcut.** Setting `CUDA_VISIBLE_DEVICES` in the pytest
process would poison the session: `torch.cuda.is_available()` caches, and on torch 2.11 it keeps
reporting False after the variable is restored (measured -- `False` while denied, still `False`
restored, though `device_count()` recovers to 1), so the first unmarked test to ask would take the
GPU away from every `slow` / `gpu_env` test after it. Patching the seams leaves this process
untouched, which is checkable end to end: an unmarked test's child reads `is_available() == False`,
a `gpu_env` test's child reads True, and the parent still reads True afterwards. Only the refusing
mode denies -- `report` exists to let a run through while SAYING what it did, and a denial says
nothing to anyone. All 2955 non-slow tests pass with it live, so no test's child needed a device.

**The denial covers every spawn entry point, not only `subprocess`.** `llb.quality.gpu_guard_spawn`
owns that half: `spawn_seams()` names each entry point and the replacement installed at it, and
`denied_children()` is the context the autouse fixture enters. Three argument shapes cover the
surface -- an entry point that TAKES an environment has it rewritten (`subprocess.Popen`,
`os.execve`, `os.execvpe`, `os.posix_spawn`, `os.posix_spawnp`); one that takes none reaches its
`*e` sibling with a denied one (`os.execv` -> `os.execve`, `os.execvp` -> `os.execvpe`) or, for
`os.system`, carries `export CUDA_VISIBLE_DEVICES=''` on a line in front of the command; and a FORK
(`os.fork`, `os.forkpty`) applies the denial inside the child, where poisoning the variable costs
nothing because the child is not the session. Ten seams cover more than they name, because the rest
of the `os` spawn surface is written in Python on top of those names and resolves them as module
globals at call time: `execl` / `execlp` go through `execv` / `execvp`, `execle` / `execlpe` through
`execve` / `execvpe`, the whole `spawnv*` / `spawnl*` family through `_spawnvef` (which forks and
then calls `execv` / `execve`), and `multiprocessing` under the default `fork` start method reaches
`os.fork` directly.

**Widening the patch beat re-execing pytest, on evidence from the repo.** The other candidate was a
pytest that re-execs itself once with an empty `CUDA_VISIBLE_DEVICES` and hands the device back only
to the marked tiers. It cannot work here: `make test` runs the WHOLE suite -- `slow` and unmarked
together -- in one pytest process, and `gpu_env` is exempt from `slow` (`NOT_SLOW` deselects only
`slow` and `opt_in_env`), so even `make ci`'s non-slow suite runs device-needing tests in the same
process as the tests being guarded. A process-wide denial has no per-test granularity, and per-test
granularity is the guard; handing the device back mid-session is not available either, for the
caching reason above.

**Residual: attribution and reach.** A CUDA context exists for the rest of the session once opened,
so the first unmarked test to open one is named and a later one that would have runs unobserved. The
denial still misses four paths, all stated in the `gpu_guard_spawn` docstring: `multiprocessing`
under the `spawn` or `forkserver` start method, where `multiprocessing.util.spawnv_passfds` calls
`_posixsubprocess.fork_exec` with no environment list -- neither `subprocess.Popen` nor an `os`
entry point (`fork` is the Linux default and IS covered); a native extension that calls `fork(2)` /
`execve(2)` / `system(3)` in C without coming back through `os`; a child that sets
`CUDA_VISIBLE_DEVICES` back itself, since the denial is a default and not a sandbox; and the
`os.system` mechanism being POSIX-shell-specific. As before, it hides the device from the CUDA
runtime, not from NVML: `nvidia-smi` still lists the GPU under an empty `CUDA_VISIBLE_DEVICES`, so a
child that only ASKS whether hardware exists still gets yes -- it just cannot open a context.

**The coverage claim is re-checked against the running interpreter, not against the one it was
written on.** Both halves above are CPython-specific -- ten seams cover the families only because
`os.py` builds them in Python, and the residual list is accurate only because `multiprocessing`
defaults to `fork` -- and a Python upgrade can move either without failing anything, since a name the
seam set never heard of is indistinguishable from one it deliberately excluded.
`llb.quality.gpu_guard_spawn_surface` closes that by ENUMERATING the process-starting names the
running interpreter exposes (a rule, not a list: every `os` name in the exec / fork / spawn /
posix_spawn families plus `system` / `popen` / `startfile`, and every public non-exception callable
of `subprocess` -- 30 names on Python 3.13) and declaring each one in one of four states: a seam
`spawn_seams()` patches, a delegation to another declared name, a residual with its reason, or not an
entry point at all (`subprocess.CompletedProcess`). A delegation is CHECKED rather than believed:
`delegation_is_live` reads the callable's code object and asks whether the target is still a name it
resolves at call time, so `os.spawnv` rewritten in C -- or pointed somewhere else -- stops being
covered loudly. `llb.quality.gpu_guard_spawn_surface_audit` refuses six shapes: an undeclared name, a
delegation the interpreter no longer makes, a delegation chain that ends outside the seam set, a name
declared a seam that `spawn_seams()` does not patch, a patched seam no declaration names, and -- the
`multiprocessing` half -- a start method the interpreter offers that is undeclared, or a DEFAULT
start method that is a declared residual, which is exactly what Python 3.14 does to the `fork`
default this denial rests on. A declaration naming nothing on this host (`os.startfile`,
`subprocess.STARTUPINFO`) is reported by `absent_declarations`, never refused: that is a host
difference, the same reason the seam builder tolerates a missing attribute. The default start method
is read WITHOUT resolving it (`get_start_method(allow_none=True)`, else the documented default-first
head of `get_all_start_methods()`), because resolving it would make a later `set_start_method` raise
for the whole session.

**Two modules is the right enumerated surface, and that is a measurement now.** Everything else in
the stdlib that starts a child was covered only because the helper it calls resolves an `os` /
`subprocess` name -- a sentence, not a check. `llb.quality.gpu_guard_spawn_reach` reads the stdlib
instead: every `*.py` under the stdlib root is parsed and its process-starting CALL SITES are
resolved through that module's own imports (`os.fork`, `from subprocess import Popen`,
`import os as operating`), against an alphabet taken from the declared surface plus the C modules
under it -- `posix` / `nt`, which `os` re-exports, and `_posixsubprocess` / `_winapi`, which
`subprocess` and `multiprocessing` call below any patchable name. On CPython 3.13 that finds **25
stdlib modules that start a child, 23 of which resolve a declared name** (`pty.py` ->
`os.fork` / `os.forkpty` / `os.execlp`, `asyncio/unix_events.py`, `socketserver.py`,
`http/server.py`, `webbrowser.py`, `uuid.py`, `venv/__init__.py`, `platform.py`, `ctypes/util.py`,
`ensurepip`, `imaplib.py`, the idlelib trio, and the rest). The three that do not are the ones
already on the record and are declared as `DECLARED_REACHERS`: `subprocess.py` itself, whose
`_posixsubprocess` / `_winapi` starts are reached only from inside the patched `Popen`, and
`multiprocessing/util.py` + `multiprocessing/popen_spawn_win32.py`, which are the POSIX and Windows
halves of the `spawn` / `forkserver` residual. `llb.quality.gpu_guard_spawn_reach_audit` refuses a
module reaching an undeclared name, an excuse whose seam is no longer patched, and -- the failure
mode a source scan invites -- a scan that read NO source, which says where the tree is rather than
what is in it (`SpawnScan.files_read` is what tells "read and quiet" apart from "never read"; the
unmeasured middle between those is `audit_read_coverage`, below). CPython's own regression suite
(`test/`, `*/tests/`, `idlelib/idle_test`) is excluded by a stated rule: a corpus that starts
children on purpose, costing 4s and one extra declaration to include. The
stdlib scan is ~0.9s on this host (631 files read, 372 parsed), run once per session by a
module-scoped fixture in `tests/llb/quality/test_gpu_guard_spawn_reach.py`, which also drives it over
fabricated trees for the cases the host cannot produce (an aliased import, a local helper that merely
shares a name with a spawn entry point, a file that will not parse, a module reaching past every
patchable name). The per-file half -- resolving a source buffer's call sites through its own imports
-- is `llb.quality.gpu_guard_spawn_source`, shared by both scans.

**The scan says what it FAILED to read, so the result is about the stdlib rather than about
whichever files this host shipped.** A file count says how much was read, not what was missed: a
module that ships without source is never parsed and reports exactly like a module that starts no
children, so a host could pass having read half the library.
`llb.quality.gpu_guard_spawn_reach_coverage` measures the reading against `sys.stdlib_module_names`
-- the interpreter's own list of what its standard library contains -- using the top-level names the
pass parsed (`SpawnScan.modules_read`), and classifies every declared name no source was read for.
Most of that list has no `.py` by construction, which is why the classification is the deliverable
and not the refusal: `compiled` (in `sys.builtin_module_names`), `extensions` (a shared object under
the root or its `lib-dynload`, matched on `importlib.machinery.EXTENSION_SUFFIXES`), `declared`
(`SOURCELESS_STDLIB_MODULES`: the two frozen bootstrap modules, plus the Windows and macOS names the
list carries because it is documented platform-independent -- `_winapi`, `nt`, `msvcrt`, `winreg`,
`winsound`, `_overlapped`, `_wmi`, `_scproxy`), `compiled_only` (a `.pyc` under the root with no
`.py` beside it), and `absent` (nothing under the root at all). **On this host, of 290 declared
names: 184 read as source, 61 compiled in, 35 extensions, 10 declared sourceless, 0 compiled-only,
0 absent** -- every unread name accounted for, and the fields partition the list so a name cannot
fall between two of them. `gpu_guard_spawn_reach_audit.audit_read_coverage` refuses ONLY the
compiled-only class, decided on that evidence: it is the frozen / zipped / source-stripped layout,
where the module is importable on this host, can start a child, and was not parsed. `absent` is
recorded and not refused, because a `python3-minimal` or split-package host (Debian ships `tkinter`
apart) cannot import what it does not have, so the claim holds vacuously for it -- and refusing
either class of by-construction absence is the naive gate that fails on every host.
`read_coverage_message` renders the whole breakdown, and is the assertion message the stdlib
coverage test fails with. The scan's excluded segments do not hide anything here: none of `test` /
`tests` / `idle_test` / `site-packages` is a name `sys.stdlib_module_names` carries, which is
asserted rather than assumed. Residual: the measurement is by top-level module, so a package that
ships half its submodules as source is read as read -- the sharp version of that would need the
same per-module list one level down, which CPython does not publish.

**The installed packages are read the same way, for the one question that can differ there.** This
repo runs on dependencies that start children constantly (torch dataloader workers, vLLM engine
processes, uv, the build scripts), and each was covered only by that same unstated assumption.
`installed_spawn_reaches` reads the venv's site-packages with a narrower alphabet --
`below_the_seams()`: `posix`, `_posixsubprocess`, `_winapi` -- because a dependency calling
`subprocess.Popen` says nothing the declaration does not already say, while scanning for the covered
names too means parsing 7420 files instead of 301 (measured). A one-off full-alphabet pass over this
host's 40119 site-packages files found **362 packages that start a child and exactly 5 files that go
below the seams**, in two packages: `joblib`'s vendored `loky` (3 files -- `backend/fork_exec.py` ->
`_posixsubprocess.fork_exec`, plus `_winapi.CreateProcess` in `backend/popen_loky_win32.py` and
`backend/resource_tracker.py`) and `multiprocess` (2 files -- a `dill`-based fork of
`multiprocessing`, carrying that module's residual verbatim in `util.py` and
`popen_spawn_win32.py`). Both are private copies of a residual already on the record and neither is
closable from here, so both are declared in `DECLARED_PACKAGE_REACHERS` -- by PACKAGE rather than by
file, since a release moves its modules and the decision an operator makes is about the dependency. A
THIRD package arriving is what `audit_installed_reach` refuses; an excuse is looked up as the exact
path first and then the top-level package, so the stdlib and package tables read through one lookup.
`nt` is deliberately absent from the installed alphabet: it is the Windows twin of names `os`
re-exports, and its two-letter module name matches too much text to prefilter on, so including it
would cost a full-tree parse on every host for a platform whose denial mechanism is already a
residual.

**A package excuse carries the reach it was MEASURED against, so a widened vendored backend arrives
as a line to re-read.** Package granularity is the right unit for surviving a release bump and the
wrong unit for a residual: it excuses every module in the package, so a future `joblib` that starts
children a second way, from a file the reason never saw, would be covered by a line written about
`loky`. Narrowing it back to per-file declarations would reintroduce the churn the package unit
exists to avoid, so each declaration is a `PackageReacher` instead -- the `SpawnCoverage` reason plus
the primitives and the file count it was written on (`joblib`: 3 files,
`multiprocess`: 2, both through `_posixsubprocess.fork_exec` + `_winapi.CreateProcess`). A
declaration cannot be added without that record: `PackageReacher.__post_init__` refuses an empty
primitive list or a zero file count. `gpu_guard_spawn_reach_audit.outgrown_reachers` then reports a
declared package that reaches a primitive its excuse was not measured on, or starts children from
more files than it was (naming those files), and `audit_installed_reach` includes those findings, so
the widening turns the suite red on the release that introduces it rather than passing under the old
reason. Growth only: a package reaching the same way from FEWER files -- a dropped backend, a slimmer
build -- is not a decision to revisit, and an excuse that matches nothing at all is already what
`absent_reachers` reports. What this does NOT do is close either vendored residual or check the
declarations of third-party packages per file; both remain what they were. Residual: the record is a
COUNT and a primitive set, not the file identities, so a release that renames one backend while
dropping another reaches the same way from the same number of files and stays quiet -- naming the
paths is the per-file churn the package unit exists to avoid.

**The site-packages cases are `slow` and the stdlib ones are not, decided on the measured cost.** The
stdlib is ~600 files that ship with the interpreter; site-packages is whatever is installed -- 40119
files and 556 MB here, 2.2s warm and disk-bound cold -- and it changes only when the lock file does,
which is a `make test` moment rather than a `make ci` one. The mechanism itself (the package-level
excuse, the narrow alphabet, a call that goes around `os` through `posix`) is pinned over fabricated
trees in the non-slow tier, so `make ci` still covers the code and only the 40k-file read is
deferred. Residual: the scan reads SOURCE, so a dynamic import, a call through an object attribute,
and anything a compiled extension does below Python stay invisible -- the last being the same
native-extension residual the denial itself carries.

Coverage is five files. `tests/llb/quality/test_gpu_guard.py` is the observation half plus the
suite wiring: the state reads over a fake module table, and the fixture body driven against the live
process. `test_gpu_guard_spawn.py` puts a recorder behind each seam and asserts what it was passed,
including the positional-`env` `Popen` shape, the `os.system` command text, and the `os.execl` /
`os.execlp` delegation that the four exec seams rely on. `test_gpu_guard_spawn_children.py` is the
end-to-end half: it starts a REAL child through `subprocess.run`, `os.system`, `os.popen`,
`os.spawnv`, `os.spawnlp`, `os.posix_spawn`, `os.posix_spawnp`, a raw `os.fork`, and a `fork`-context
`multiprocessing.Process`, and reads back what each child saw -- `""` under the denial and the
parent's own value without it, so each assertion is about the denial rather than about a host that
never had a device. `test_gpu_guard_spawn_surface.py` is the re-check: one assertion that this
interpreter's surface is the declared one, and the rest driving the audit against FABRICATED
interpreters (a Python that grew a spawn function, one that rewrote a delegation in C, one whose
default start method is a residual), because those cases cannot be produced by the host.

**Both complexity thresholds are enforced, not reported.** `scripts/complexity_gate.sh` runs the
Radon D-or-worse scan and the Complexipy scan at `COGNITIVE_MAX=15` over `src` and `tests`, and
exits non-zero as soon as either prints a row -- so a function that crosses a threshold turns the
build red on its own commit instead of surfacing in a later sweep. `ci-checks` runs it beside
`llb.quality.acceptance_gates --check` (so it runs in `make ci`, `make ci-github`, and GitHub CI);
`make complexity-gate` runs it alone after a refactor. Neither tool signals through its exit status
in this configuration (radon always exits 0, complexipy is asked for a plain listing with
`--ignore-complexity`), so the gate treats printed output as the failure. Both scans always run --
one pass shows every peak. The thresholds are the shipped numbers: a finding is split, not
accommodated by raising the maximum.

The scans, thresholds, and labels live once in `scripts/shared/complexity.sh`, sourced by both the
gate and `scripts/code_quality.sh`, so the sweep fails on exactly what CI fails on and prints it
identically (block reporting is `llb_print_block` / `llb_report_if_output` / `llb_fail_if_output`
in `scripts/shared/common.sh`).

**Shell scripts are gated the same way.** `scripts/shell_lint_gate.sh` (`make shell-lint-gate`,
also inside `ci-checks`) runs three scans over every tracked-or-new `*.sh` in the repo
(`git ls-files --cached --others --exclude-standard`, so a script is linted before its first commit
and nothing under a gitignored tree is scanned; a staged delete is dropped):

- `bash -n` syntax, which needs nothing installed and therefore always runs.
- `shellcheck -x -P SCRIPTDIR -S warning`, over the whole set in one invocation so findings stay
  repo-relative.
- the same run restricted to `SC1090,SC1091` at `-S style`: every `# shellcheck source=` directive
  must resolve.
- `llb.quality.shell_symbols`: every function a tracked `*.sh` defines carries the `llb_` prefix,
  and every `llb_*` name a caller uses has a definition in that caller's declared scope.

**The lint is cross-file, and the third scan is what keeps it that way.** `-x` follows a sourced
file, so a caller is checked against what `scripts/shared/{common,complexity,shell_lint}.sh`
actually define instead of in isolation -- the axis these scripts grew along. `-P SCRIPTDIR` is
what makes `-x` work here: a `source=` path resolves relative to the CWD by default, so running
from the project root, every `source=shared/common.sh` directive in `scripts/` resolved to nothing
and `-x` followed nothing (9 such directives). An unresolved directive is reported at INFO,
*below* the severity floor, so the lint pass alone cannot tell "followed and clean" from "never
followed" -- hence the dedicated scan. The proof that following now happens is in the tree: the two
gate entrypoints set `LLB_REPORT_PREFIX` for `common.sh` to read, which un-followed shellcheck
called an unused variable (SC2034); both `# shellcheck disable` workarounds are gone and the gate
stays green. A path genuinely computed at run time is annotated in the script with a reasoned
`# shellcheck disable=SC1091` rather than by dropping the scan.

**Functions are checked by `llb.quality.shell_symbols`, because the linter does not model them.**
`-x` shares VARIABLE knowledge across a followed source, not function definitions, so a call to a
helper that no longer exists passes every scan above and fails as `command not found` on an
operator's host. The module collects `llb_*` definitions over a caller's declared scope and names
any `llb_*` it uses with no definition there, for tracked-or-new `*.sh` **and** `Makefile`/`*.mk`
(the recipes that `source "$(PROJECT_ROOT)/scripts/shared/common.sh"` are call sites too). Scope is
what a caller DECLARES, never what happens to be loaded at run time:

- a `# shellcheck source=` directive (resolved script-dir first, then repo root, as shellcheck
  resolves it -- and the scan above already proves these land),
- a make recipe's literal `$(PROJECT_ROOT)/...` source,
- a `# llb-requires: <sibling>` line, for a shared module whose functions assume the entrypoint
  sourced a sibling. `scripts/shared/{complexity,shell_lint}.sh` carry it for `common.sh`: they
  call `llb_fail_if_output` and never source it themselves, which is a contract their headers
  stated in prose and now state in a line the check reads.

What counts as a call is a name in command or argument position -- including one handed to a runner
(`llb_fail_if_output "$LABEL" llb_cyclomatic_scan`), which breaks identically when the definition
goes. Prose in a comment, an expansion operand (`${name#llb_prefix}`), a `$llb_var`, and a path
segment are not calls. A call built by `eval` or through a variable is out of reach and stays out of
scope. Coverage is `tests/llb/quality/test_shell_symbols.py`, whose first assertion is the shipped
tree itself.

**The prefix is a rule, not a convention, because the check has no other way to find a call.**
Keying on `llb_*` is what separates a call from an English word, so the scan's coverage used to be
whatever share of the tree happened to follow the convention: 35 of the 92 functions defined in
tracked `*.sh`. The uncovered 57 were the most cross-file-coupled code here -- `scripts/quickstart.sh`
plus the eight `scripts/quickstart/*.sh` fragments, which are sourced into one namespace and share a
vocabulary (`resolve_path`, `run_target`, `prompt_yes_no`, `make_with_data_dir`, ...) that no scan
resolved. Two rules could have closed it, and the tree now carries the first:

- **Adopt the prefix** (taken). All 57 are renamed `llb_*`, and `unprefixed_definitions` refuses a
  function defined in a tracked `*.sh` without it, so the coverage cannot decay back. Three names
  needed more than a prefix, because `source` makes the namespace flat: `main` in `quickstart.sh` is
  `llb_quickstart_main`, and the two different `usage` bodies are `llb_quickstart_usage` and
  `llb_apt_usage`.
- **Drop the prefix and treat a name as a call when it is DEFINED somewhere in the scanned set**
  (rejected). It cannot deliver the outcome the check exists for. A rename removes the old name from
  the defined set, so the stale call site stops looking like a call at the exact commit that broke
  it; the rule only catches a helper that moved out of scope while still being defined elsewhere,
  which is the narrower half of the failure. It also has a false-call surface the prefix does not:
  the uncovered names include bare words (`main`, `usage`, `result`, `heading`), and a scan with no
  command-position parsing reads `KNOWLEDGE_CUTOFF_REVISION ?= main` in `make/config.mk` as a call --
  the one false positive a shipped-tree trial of the rule produced.

The scope declarations were the shared cost either rule had to pay, not a cost of the rename: the
fragments are sourced BY `quickstart.sh` and never source each other, so all 217 of their calls were
out of scope under either rule until each fragment named its siblings with `# llb-requires:`
(`track_c.sh` declares `helpers.sh`, `model_select.sh`, `pdf_draft.sh`, `track_b.sh`; the other
seven are shorter). A function defined inside a make recipe is exempt from the prefix -- `wants_backend`
and `record_failure` in `make/eval/categories-platform.mk` live and die in one `bash -c` and never
enter the sourced namespace -- which is why the prefix scan reads `*.sh` only.

**Residual: a name that never carried the prefix is still invisible.** The check finds a call whose
helper was renamed, moved, or deleted; it cannot find a call to an EXTERNAL command that does not
exist, because a name defined nowhere is indistinguishable from a binary the host is expected to
have without real command-position parsing. That case stays with the `command -v` guards the scripts
already carry.

A MISSING shellcheck fails the gate rather than skipping it -- a linter that reports itself as fine
when it never ran is worse than no linter, and the old sweep did exactly that on a host without the
apt package. `LLB_SHELLCHECK_OPTIONAL=1` downgrades that to a printed skip for a lean venv, and the
pass line then says `shellcheck NOT run` rather than claiming the tree is clean. That escape hatch
is rarely needed: shellcheck now ships in the `dev` extra as the pinned `shellcheck-py` wheel
(the real binary, upper-bounded like ruff so a new release cannot redden CI on unchanged scripts),
so `.venv/bin/shellcheck` exists wherever `make ci` runs, GitHub CI included -- no apt step in the
workflow. Scans, severity, and the missing-binary policy live once in
`scripts/shared/shell_lint.sh`, sourced by both the gate and the sweep; every scan runs before any
of them fails.

**Exactly one binary decides the verdict: `$LLB_SHELLCHECK`, default `.venv/bin/shellcheck`.** The
`command -v shellcheck` fallback is gone. It was what made a green run host-dependent: a distro
package is releases behind the pin (this dev box ships 0.9.0 against the pinned 0.11.0) and
shellcheck ADDS checks between releases, so the same commit could pass on the fallback host and
fail in CI -- the split verdict a pinned linter exists to prevent, moved one level down. The gap is
real, not theoretical: `out="$(echo hi >/tmp/f)"` draws SC2327 (warning) + SC2328 (error) from
0.11.0 and nothing at all from 0.9.0, so a script written that way lints clean on the fallback host.
On the shipped tree the two versions happen to agree today (both scans clean at `-S warning` and at
`-S style -i SC1090,SC1091`), which is why the fix was cheap to take before a divergence landed.
Requiring the wheel costs no workflow -- a venv without the `dev` extra cannot run `make ci` anyway
-- and the missing-binary block now names the path it looked at, since "MISSING" is otherwise
puzzling on a host that has `shellcheck` on `PATH`. `LLB_SHELLCHECK` still overrides the path for a
venv living elsewhere. Consequently the `dev` apt profile is now **empty**
(`scripts/apt/dev.packages` keeps the file and the comment; the profile stays for a future dev-only
OS package) and [dev setup](../../guides/development/dev-setup.md#apt-dependencies-debianubuntu)
no longer lists `shellcheck` as a fallback. Residual: the pin is a RANGE (`>=0.10,<0.12`), so two
hosts resolving the extra months apart can still land on different wheels; `uv.lock` pins
0.11.0.1 for anyone installing through the lock, and tightening the range itself is a separate
call. Resolution behavior is covered by `tests/llb/quality/test_shell_lint_resolution.py`.

Everything else in the sweep stays informational -- in particular the `.py`/`.sh` line-count
report, which backs a target AGENTS.md keeps SOFT on purpose and which has legitimate offenders.
The maintainability-index section (Radon MI grade C) is also still a report.

The D-grade cyclomatic-complexity cleanup keeps orchestration separate from validation, state
accumulation, and presentation. Ontology dedup now uses an embedded-candidate value object and
named matching/report helpers; the multi-hop expansion audit uses a check accumulator that builds
the final report. Retrieval validation passes an immutable request into
`cli/rag/retrieval_validation.py`, autonomous verification scoring lives in
`auto_rag/verification_auto.py`, and query-prep dependency checks are table-driven. The query
robustness integration test uses a module-level morphology-loader callable and named assertion /
artifact phases. The repository-wide Radon D-or-worse scan is empty; focused coverage lives in the
ontology, auto-RAG verification, query-prep, and query-robustness test suites.

The cognitive-complexity cleanup extends that separation across backend planning, review
workflows, conflict resolution and filtering, query robustness, incremental refresh, ontology
expansion, retrieval fusion, and reporting. Complex branches now live in named policy,
validation, selection, and rendering helpers. Stateful assembly uses
`rag/refresh/merge_assembly.py`; tree leaf filtering, robustness recovery, and hybrid-store
retrieval use focused owner modules. Launcher and morphology closures are module-level callable
adapters. Focused verification covers the affected backend, review, conflict, evaluation, graph,
ontology, retrieval, and scoring paths.

The agentic/bench lanes grew their own peaks after that pass and were cleaned the same way, with
three patterns doing most of the work:

- **A named check per contract area.** Every `validate_*_design` is now a sequence of `_check_*`
  calls in the order a reader of the design file meets them (identity, ledger, roster, wording,
  sampling, gates). `validate_channel_authority_design` went from F(54) to a body of nine calls.
  Typed field reads moved to `bench/agentic_design_fields.py`, so a rule reads as the contract
  (`as_int(matrix, "n_tasks") < 6`) instead of as a cast.
- **A record for what a step accumulates.** `run_episode` was eleven mutable counters threaded
  through one 130-line loop; it is now `_EpisodeTally` (counting plus the one place an episode
  ENDS), `_ControllerSeam` (how the prompt is serialized, guarded, and sent), and named steps for
  the repair round and the tool call. The loop body is the cycle and nothing else: F(47)/93 became
  C(11)/11. `_CaseMeans`, `_ChannelGates`, `_ComparisonSettings`, and `_PlacementContext` play the
  same role in run metrics, the channel-authority reading, retrieval comparison, and the seeded
  placement run.
- **No closures over a caller's locals.** The controller-authority seed run's nested `harness` /
  `record` / `unused_complete` closures became module-level functions taking an explicit
  `_PlacementContext`, leaving a one-line protocol adapter.

Two oversized modules split at their functional seams in the same pass --
`rag/encoder_throughput.py` into measurement / `_summary` / `_report` / `_profile`, and
`bench/agentic_context_sweep.py` into `_model` (axes, grids, cells) / `_verdict` (pairing and the
pin-or-expose cut) / the runner -- with call sites repointed at the real submodule rather than a
re-export shim.

The residual band was then finished the same way, and **both complexity scans are now silent**: the
Radon D-or-worse scan is empty (19 functions before, up to F(54)) and the Complexipy scan at the
shipped maximum of 15 is empty (51 functions before, up to 93). Two shapes carried the last 21:

- **A run entry point is a plan plus named steps.** `run_constant_sweep`, `run_agentic_loop_policy`,
  `run_bakeoff`, and `run_draft` now resolve their contract first (`_scored_cells` caching identical
  cells, `_validate_study_design` / `_study_analysis` for whichever prospective study is running,
  `_ScoredCandidates` accumulating rows/vectors/stores together, `_ResolvedDraft` carrying the two
  fields a resume fills in) and then read as the sequence of steps they are.
- **A CLI command drives cells from a record, not from a closure.** The two repeat-feedback commands
  and the crossover restatement built their per-cell run as a nested function closing over a dozen
  locals; each is now a `_Plan` / `_ResolvedDraft` record plus a module-level step bound with
  `functools.partial`, so a cell cannot read a different temperature or policy than its neighbour.

The rest are the same named-check and reading-record patterns applied to `_judge`, `decide_verdict`,
`build_multi_reviewer_worksheets`, `pair_against_shipped`, `aggregate_safe_verdict`,
`collapse_reading`, `_refuse_cycles`, and the remaining `validate_*` / `analyze_*` readings. The
longest tracked source file is 500 lines.
