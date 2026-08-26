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
- The generated Gemma 4 12B vLLM config (`google/gemma-4-12B-it-qat-w4a16-ct`, seed 13, k=5) ran
  ONE item with embeddings on CPU -- a serving smoke, not a quality reading. The contention guard
  accepted 0.90 utilization with 11,696 MiB free of 12,227 MiB (weight floor 7,817 MiB, not
  derated); native sampling, Triton attention, Marlin W4A16, 16 GiB CPU weight offload, and 32 GiB
  KV offload served the full requested 16,384-token context at 3.32 tok/s steady, 51.99 W mean, and
  11,511 MiB peak VRAM. The load took 246.07 seconds, chiefly CUDA-graph capture. FlashInfer 0.6.12
  could not supply its sampler on SM 12.0 and the recorded native-sampler fallback worked. The one
  item scored objective 0.200 with recall@5 1.0 and reliability 1.0; retrieve took 11.75 s against
  1.44 s of generation. Reading: the 12 GiB tier SERVES a 12B W4A16 model at its declared window
  with headroom to spare, and 3.32 tok/s is the price of the CPU/KV offload it needs to do so. n=1
  supports the serving claim and nothing about quality. Lookup key: run
  `serving-12gb-gemma-4-12b-vllm`, run id `2f08bcd131d7`.
- The 20-item Ollama path used the Ukrainian MamayLM Gemma 3 12B Q4_K_M model
  (`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`, seed 13, k=5) with CUDA
  embeddings on the 20-item final split. It scored objective 0.406, reliability 1.0, retrieval
  recall@5 0.900 / MRR 0.787, 39.16 tok/s steady at 79.42 W mean (0.493 tokens/W), and 9,932 MiB
  peak VRAM, with retrieve 0.53 s against generate 1.44 s per item. Reading: the Ollama lane is the
  usable everyday path on this tier -- ~12x the vLLM lane's throughput at 2 GiB less peak VRAM,
  because a Q4_K_M GGUF fits without the offload the W4A16 config needed. Objective 0.406 on n=20 is
  a smoke figure, not a leaderboard one. Lookup key: run `rag-eval`, run id `7e94edc3fe16`.
- llama.cpp was not an available backend on this host (`llama-server` was absent), so no llama.cpp
  cell was claimed.
- The repository gate selects only current implementation coverage: obsolete unpublished-artifact
  compatibility checks were removed rather than skipped. It passes 2,226 tests with 43
  opt-in/slow tests deselected and zero runtime skips. Ruff format/check, mypy, Markdown lint, and
  the code-quality report also passed. `ollama ps` was empty after the evidence runs.

The recent paired embedder, context-ablation, and local drafting evidence reruns are recorded in
[RAG core](rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty),
[RAG core](rag-core/embedders.md#blackwell-encoder-throughput-decomposition),
[RAG core](rag-core/context-ablation-evidence.md), and
[data prep](data-prep/drafting-lanes.md#sequential-local-qwengemma-draft-comparison).

Encoder throughput on this host (2026-07-29): `EMBED_ENCODER_THROUGHPUT=1` over the 311-chunk UA
fixture at the 80 W power limit. Warm CUDA rates are ~638 chunks/s for e5-small, 208 for e5-base,
~62 for e5-large and BGE-M3, and ~334 for the paraphrase model; cold load (~5.7 s) dominated the
earlier one-pass rates, but the e5-base vs large spread survives warm measurement. Prefer warm
chunks/s for host cost columns. e5-small is the named cheap CUDA alternative (~3.05x base, lower
peak VRAM) when quality is flat; the paired verdict still RETAINs e5-base on n=82. Lookup keys: run
id `1d36908e745c` (full roster) and `c79df0776706` (VRAM after the release fix); the per-encoder
numbers those two carry are tabulated on the RAG-core page linked below rather than repeated here.
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

`scripts/code_quality.sh` always prints the largest repository-visible Python files and non-Python
files, including new non-ignored files before they are staged. Root-file, markdown, shell, and
complexity sections are quiet when clean and
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

The source-size refactor separates the most frequently edited long modules at their functional
boundaries:

- agent context policy vocabulary, transcript state, and bounded summarization live in
  `bench/agentic/context_policy.py`, `context.py`, and `context_summary.py`;
- context comparison models/pairing, task-kind analysis, recommendation rendering, and persistence
  live in `bench/context_policy/report.py` plus the `_kind`, `_recommendation`, and `_persist`
  modules; `context_policy/report_persist.py` also writes the per-policy bundles
  (`persist_policy_bundles`), so `agentic_context.py` is the sweep alone;
- the episode loop delegates prompt/compaction assembly, controller transport/repair, and mutable
  tally state to `agentic/episode_prompt.py`, `episode_controller.py`, and `episode_state.py`;
- retrieval comparison contracts, resolved settings, and optional output rows live in
  `rag/comparison/models.py`, `comparison/settings.py`, and `comparison/rows.py`;
- controller-authority run contracts and snapshot-isolation proof live in
  `controller_authority/model.py` and `controller_authority/snapshot.py`;
- embedding adoption reason clauses live in `rag/embedding_bakeoff/reason.py`;
- policy-change replay geometry and design loading live in
  `bench/policy_change/geometry.py`;
- embedding CLI validation/output persistence and persisted agentic comparison commands live in
  `cli/rag/compare_embeddings_output.py` and `cli/bench/categories/agentic_compare.py`.

Callers and tests import each symbol from its owning module; the split adds no compatibility
re-export layer. `scripts/code_quality.sh` now prints production `.py`/`.sh` soft-limit findings
separately from the all-files list, whose longer scenario-ledger tests remain visible without being
mistaken for production modules. No tracked shell file currently exceeds the 250-line soft target.
The context, report, episode, retrieval, authority, and embedding verdict modules named above are
all at or below 250 lines after the split. Focused regression suites, Ruff, and full-source mypy
pass.

The two production modules still above 300 lines are cohesive soft-limit exceptions:
`quality/gpu_guard/surface.py` keeps one exhaustive declaration table beside the interpreter
enumeration that audits it, while `quality/gpu_guard/reach/coverage.py` keeps the documented
stdlib classification partition beside its coverage record and filesystem classifiers. Splitting
either lookup/classifier family would add navigation without creating an independent functional
owner. The remaining production findings are within 47 lines of the soft target and stay visible
for future seam-driven cleanup.

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
test in an autouse guard (`llb.quality.gpu_guard.guard`) that watches this process and denies the
device to the children of an unmarked test. The watch snapshots two effects before the test and
reads them again after: `torch.cuda.is_initialized()`, and whether `flashinfer` is in `sys.modules`
(its first sampling call is the JIT build). An unmarked test that flips either one fails at
teardown, naming the test, what it did, and the ways out. Both reads come out of `sys.modules` and
neither imports anything, so the guard no-ops where torch is absent -- GitHub CI installs no torch
and must not start -- and costs two dict lookups per test: the whole non-slow suite runs in 101-102s
with it and 105s under `LLB_GPU_GUARD=off`, which is run-to-run noise on this box. Importing torch
is deliberately not a finding: `import torch`, `torch.cuda.is_available()`, and a test's own fake
`torch` module all leave `is_initialized()` False, which is what keeps the many tests that pull
torch in without touching the device green.

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

**The tier's no-download promise is enforced along the same axis.** The second root autouse fixture
wraps every unmarked test in `llb.quality.download_guard`, which replaces
`socket.socket.connect` and `connect_ex` for the test's lifetime. A connection to a non-loopback
destination raises `DownloadGuardError` before the original connector runs, naming the test,
destination, and remedies. This effect-level seam catches a cold-cache `from_pretrained`,
`hf_hub_download`, or ordinary HTTP client without importing or patching any of them. Loopback IPv4
and IPv6, `localhost`, IPv4-mapped loopback, and Unix-domain sockets remain available, so local fake
servers do not need an exemption. `slow` declares integration work; `network_env` is the narrow
escape hatch for a quick test that legitimately connects elsewhere. `LLB_DOWNLOAD_GUARD=report`
warns and allows the connector, `off` disables the guard, and an unknown mode is refused.

Refusal is again the evidence-backed default: the complete `make ci` run is clean with the guard
live (3068 passed, 64 deselected), without changing any existing marker. The focused contract in
`tests/llb/quality/test_download_guard.py` proves that refusal happens before a connector runs,
report mode passes through, the root fixture honors `network_env`, and a real TCP client can reach a
fake server bound to `127.0.0.1`. The implementation adds no dependency to the base GitHub CI
environment.

**A CHILD process is denied the device rather than observed**, because there is nothing in this
process to observe it by. For the duration of an unmarked test, every spawn entry point is swapped
for one that starts the child with an empty `CUDA_VISIBLE_DEVICES` -- whatever environment the
caller passed. `subprocess.Popen` is the seam that `run` / `call` / `check_output` all reach for at
call time. That closes the shape that motivated the guard: `test_build_helper.py` drives
`scripts/build_vllm.sh` through `subprocess.run`, so its flashinfer probe lives in a python no
in-process fixture can inspect. Remove the seeded verdict as an experiment and the
prebuilt-installer test costs 5.99s under `LLB_GPU_GUARD=off` -- the child JIT-builds the sampler
kernel on the real GPU -- against 0.81s with the denial, where the child finds no device and the
probe returns `native` in milliseconds. It passes either way. The seeded verdict stays regardless:
it is what keeps the cost off a run with the guard disabled, and it states the intent at the call
site.

**Child-only is the mechanism, not a shortcut.** Setting `CUDA_VISIBLE_DEVICES` in the pytest
process would poison the session: `torch.cuda.is_available()` caches, and on torch 2.11 it keeps
reporting False after the variable is restored (measured -- `False` while denied, still `False`
restored, though `device_count()` recovers to 1), so the first unmarked test to ask would take the
GPU away from every `slow` / `gpu_env` test after it. Patching the seams leaves this process
untouched, which is checkable end to end: an unmarked test's child reads `is_available() == False`,
a `gpu_env` test's child reads True, and the parent still reads True afterwards. Only the refusing
mode denies -- `report` exists to let a run through while SAYING what it did, and a denial says
nothing to anyone. All 3050 non-slow tests pass with it live, so no test's child needed a device.

**The denial covers every spawn entry point, not only `subprocess`.** `llb.quality.gpu_guard.spawn`
owns that half: `spawn_seams()` names each entry point and the replacement installed at it, and
`denied_children()` is the context the autouse fixture enters. Four argument shapes cover the
surface -- an entry point that TAKES an environment has it rewritten (`subprocess.Popen`,
`os.execve`, `os.execvpe`, `os.posix_spawn`, `os.posix_spawnp`); one that takes none reaches its
`*e` sibling with a denied one (`os.execv` -> `os.execve`, `os.execvp` -> `os.execvpe`) or, for
`os.system`, carries `export CUDA_VISIBLE_DEVICES=''` on a line in front of the command; and a FORK
(`os.fork`, `os.forkpty`) applies the denial inside the child, where poisoning the variable costs
nothing because the child is not the session. The eleventh seam is
`multiprocessing.util.spawnv_passfds`: its replacement routes through the already-denied
`os.posix_spawn`. Its file actions first copy every requested descriptor into the lowest temporary
numbers that are not themselves requested, close every other slot in that dense prefix, apply
`POSIX_SPAWN_CLOSEFROM` above it, restore the requested descriptor numbers, and close the
temporaries. The dense prefix is material: it preserves the spawn data pipes and resource-tracker
descriptor, closes an explicitly inheritable descriptor that was NOT listed, and covers a new
descriptor another thread opens after the actions are built without reading `/proc/self/fd`.
Ten `os` / `subprocess` seams still cover more than they name because
the rest of the `os` spawn surface resolves those names as module globals at call time: `execl` /
`execlp` go through `execv` / `execvp`, `execle` / `execlpe` through `execve` / `execvpe`, and the
whole `spawnv*` / `spawnl*` family through `_spawnvef`. `multiprocessing(fork)` reaches `os.fork`;
`multiprocessing(spawn)` and the initial forkserver launch reach the new helper seam.

**Forkserver state is kept per-test too.** A forkserver inherits its environment once, then forks
all later children from that long-lived process. Rewriting only its launch would therefore miss an
unmarked test when a visible-device server already existed, and would leak a denied server into a
later `gpu_env` test. `gpu_guard_spawn_multiprocessing.stop_forkserver` uses CPython's test
lifecycle hook before and after each `denied_children()` context: the denied test gets a fresh
denied server, and the next exempt test gets a fresh visible-device server. The real-child matrix in
`tests/llb/quality/test_gpu_guard_spawn_children.py` exercises all three start methods with and
without the seams. Both `spawn` and `forkserver` exit successfully and write their environment; that
successful bootstrap is the end-to-end evidence that the passed data and resource-tracker
descriptors survived. `make test` is the standard repository verification; the complete quality
package passes with these cases included.

**The public close-all primitive is a capability boundary.** This host's Python 3.13 exposes
`POSIX_SPAWN_CLOSEFROM`; Python 3.12 exposes only one-descriptor close actions. Enumerating the
currently open descriptors on 3.12 leaves an open-fd race, while emitting one close action for
every number below this host's 1,048,576 descriptor ceiling is not a viable per-child operation.
`supports_descriptor_closure` therefore makes the public helper a seam only when `CLOSEFROM` is
available. On Python 3.12, `spawn` / `forkserver` remains an explicit non-default residual and the
original `spawnv_passfds` stays installed, retaining its `close_fds=True` semantics; the surface and
stdlib-reach audits derive the same residual from that capability check. The real-child descriptor
case keeps a non-inheritable listed report pipe and deliberately marks a second pipe inheritable:
Python 3.13's route reports the device denied and the second pipe closed, while the 3.12 residual
reports the device visible and the second pipe still closed by the original helper. The focused
spawn, surface, and stdlib-reach suite passes under both the host's Python 3.13 environment and an
isolated Python 3.12 environment.

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
denial still misses five paths, all stated in the `gpu_guard_spawn` docstring: explicit `spawn` /
`forkserver` on a Python without `POSIX_SPAWN_CLOSEFROM`; a caller reaching the private
`_posixsubprocess.fork_exec` directly instead of the patched public helper; a native extension that
calls `fork(2)` / `execve(2)` / `system(3)` in C without coming back through `os`; a child that sets
`CUDA_VISIBLE_DEVICES` back itself, since the denial is a default and not a sandbox; and the
`os.system` mechanism being POSIX-shell-specific. As before, it hides the device from the CUDA
runtime, not from NVML: `nvidia-smi` still lists the GPU under an empty `CUDA_VISIBLE_DEVICES`, so a
child that only ASKS whether hardware exists still gets yes -- it just cannot open a context.

**The coverage claim is re-checked against the running interpreter, not against the one it was
written on.** Both halves above are CPython-specific -- the `os` seams cover the families only
because `os.py` builds them in Python, and the multiprocessing coverage depends on the public POSIX
helper above its private C call -- and a Python upgrade can move either without failing anything,
since a name the seam set never heard of is indistinguishable from one it deliberately excluded.
`llb.quality.gpu_guard.surface` closes that by ENUMERATING the process-starting names the running
interpreter exposes (a rule, not a list: every `os` name in the exec / fork / spawn / posix_spawn
families plus `system` / `popen` / `startfile`, and every public non-exception callable of
`subprocess`, plus `multiprocessing.util.spawnv_passfds` -- 31 names on Python 3.13) and declaring
each one in one of four states: a seam `spawn_seams()` patches, a delegation to another declared
name, a residual with its reason, or not an entry point at all (`subprocess.CompletedProcess`). A
delegation is CHECKED rather than believed: `delegation_is_live` reads the callable's code object
and asks whether the target is still a name it resolves at call time, so `os.spawnv` rewritten in C
-- or pointed somewhere else -- stops being covered loudly. `llb.quality.gpu_guard.surface_audit`
refuses six shapes: an undeclared name, a delegation the interpreter no longer makes, a delegation
chain that ends outside the seam set, a name declared a seam that `spawn_seams()` does not patch, a
patched seam no declaration names, and -- the `multiprocessing` half -- a start method the
interpreter offers that is undeclared, or a DEFAULT start method that is a platform residual because
the POSIX helper is unavailable. A declaration naming nothing on this host (`os.startfile`,
`subprocess.STARTUPINFO`) is reported by `absent_declarations`, never refused: that is a host
difference, the same reason the seam builder tolerates a missing attribute. The default start method
is read WITHOUT resolving the parent: `get_start_method(allow_none=True)` supplies an already-set
method, while an unresolved parent asks a disposable child interpreter to resolve its own context.
That answer is checked against the documented default-first head of `get_all_start_methods()`; a
disagreement raises rather than letting the audit judge the wrong method. The child result is cached
per executable and parent process, so repeated surface reads pay for one child, and the parent stays
free to call `set_start_method` later.

**Two broad modules plus one exact helper is the right enumerated surface, and that is a measurement
now.** Everything else in the stdlib that starts a child was covered only because the helper it
calls resolves an `os` / `subprocess` name or the exact `multiprocessing` helper -- a sentence, not
a check. `llb.quality.gpu_guard.reach.scan` reads the stdlib instead: every `*.py` under the stdlib
root is parsed and its process-starting CALL SITES are resolved through that module's own imports
(`os.fork`, `from subprocess import Popen`, `import os as operating`), against an alphabet taken
from the declared surface plus the C modules under it -- `posix` / `nt`, which `os` re-exports, and
`_posixsubprocess` / `_winapi`, which `subprocess` and `multiprocessing` call below any patchable
name. On CPython 3.13 that finds **25 stdlib modules that start a child, 23 of which resolve a
declared name** (`pty.py` -> `os.fork` / `os.forkpty` / `os.execlp`, `asyncio/unix_events.py`,
`socketserver.py`, `http/server.py`, `webbrowser.py`, `uuid.py`, `venv/__init__.py`, `platform.py`,
`ctypes/util.py`, `ensurepip`, `imaplib.py`, the idlelib trio, and the rest). The three that do not
are the ones already on the record and are declared as `DECLARED_REACHERS`: `subprocess.py` itself,
whose `_posixsubprocess` / `_winapi` starts are reached only from inside the patched `Popen`, and
`multiprocessing/util.py`, whose low-level start is behind the new public helper seam, and
`multiprocessing/popen_spawn_win32.py`, which remains the Windows residual.
`llb.quality.gpu_guard.reach.audit` refuses a module reaching an undeclared name, an excuse whose
seam is no longer patched, and -- the failure mode a source scan invites -- a scan that read NO
source, which says where the tree is rather than what is in it (`SpawnScan.files_read` is what tells
"read and quiet" apart from "never read"; the unmeasured middle between those is
`audit_read_coverage`, below). CPython's own regression suite (`test/`, `*/tests/`,
`idlelib/idle_test`) is excluded by a stated rule: a corpus that starts children on purpose, costing
4s and one extra declaration to include. The stdlib scan is ~0.9s on this host (631 files read, 372
parsed), run once per session by a module-scoped fixture in
`tests/llb/quality/test_gpu_guard_spawn_reach.py`, which also drives it over fabricated trees for
the cases the host cannot produce (an aliased import, a local helper that merely shares a name with
a spawn entry point, a file that will not parse, a module reaching past every patchable name). The
per-file half -- resolving a source buffer's call sites through its own imports -- is
`llb.quality.gpu_guard.spawn_source`, shared by both scans.

**The scan says what it FAILED to read, so the result is about the stdlib rather than about
whichever files this host shipped.** A file count says how much was read, not what was missed: a
module that ships without source is never parsed and reports exactly like a module that starts no
children, so a host could pass having read half the library.
`llb.quality.gpu_guard.reach.coverage` measures the reading against `sys.stdlib_module_names`
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
asserted rather than assumed.

**One level down, the package directories are the list CPython does not publish.** That
classification is per TOP-LEVEL name, because `sys.stdlib_module_names` is the only list of its kind
the interpreter ships -- so a package that ships its `__init__.py` and not its submodules counts as
read, and the very layout the measurement exists for hides there: `multiprocessing/__init__.py`
present with `multiprocessing/util.py` stripped reads exactly like a complete package.
`compiled_only_submodules` needs no published list, because the interpreter leaves the evidence on
disk -- inside every package directory the scan walked, a `__pycache__` entry whose source file is
not beside the package is the same compiled-only finding, named `multiprocessing.util` rather than
`multiprocessing`. Which source an entry claims is `cached_source`, and that rule is sharper than it
looks: PEP 3147 names a cache `<stem>.<tag>.pyc` and neither half is one dot-separated component,
since `optuna` ships alembic revisions as `v3.0.0.a.py` and pytest writes rewritten caches under
`cpython-313-pytest-9.1`. Splitting on the running interpreter's own `cache_tag` reads both right
(and the `.opt-1` of an optimized cache); `importlib.util.source_from_cache` answers neither, as it
refuses any name past three dots. A cache with no tag to split on -- one written by another
interpreter version, or the tagless `pkg/__pycache__/util.pyc` -- falls back to the PEP's shape. The
stdlib exercises none of this (0 either way, before and after), and site-packages exercises all of
it: reading the stem to the first dot called four `optuna` sources stripped, and reading the tag
back from the last called 397 modules stripped across the venv, every one of them sitting on disk.
The vocabulary
and the decision are reused rather than duplicated: `audit_read_coverage` raises these as the same
`unread-module` problem, a `.py` with no `.pyc` is nothing (caching is incidental), and a name in
neither list is simply not shipped -- the `absent` half of the same evidence-based split. A cached
`__init__` is left to the package NAME that already classifies it, so a stripped package is one
finding and not two, and the directories the scan skipped are skipped here through the scan's own
`is_excluded` rather than a second copy of the rule. The field sits outside the six-way partition
deliberately: its entries are dotted names the declared list does not contain. **On this host: 0
compiled-only submodules** (the walk costs 0.024s), pinned alongside the name-level counts, with
fabricated trees pinning both directions -- a stripped `pkg/util.py` and a nested
`pkg/sub/deep.py` refused, a package whose submodules all ship source clean.

**A stdlib that ships as an ARCHIVE is refused, because every read above is directory-shaped.**
`compiled_only`, `compiled_only_submodules`, and the scan's own `rglob("*.py")` all read filenames
under the root, which is the layout every source-stripped install this host can produce and not the
only layout CPython ships: a stdlib imported from a zip (`pythonXY.zip` on `sys.path`, what an
embedded or single-file build carries) has no package directory to walk at all. Measured over
fabricated archives before deciding anything, the untreated reading reported: with the root IS the
zip, or the root holding it beside `lib-dynload`, 0 files read -- so `audit_spawn_reach` refused the
tree as `unscanned`, while the coverage line called every declared name `absent` and
`audit_read_coverage` passed clean, the one check that speaks about the stdlib saying the stdlib is
not there; and with a MIXED layout -- part of the library as source on disk, the rest in the archive
-- files ARE read, so nothing was refused anywhere, an archived `subprocess` was reported `absent`
("this host does not ship it", for a module the interpreter imports on demand and which starts
children), and an archived `multiprocessing/util.pyc` produced no submodule finding at all.
`llb.quality.gpu_guard.reach.archive` closes both on the archive's own evidence: `zipfile`
reads the name list -- importing nothing, disassembling no `.pyc` -- and the names become the
`archived` bucket of the partition plus `archived_submodules` beside `compiled_only_submodules`,
which `audit_read_coverage` refuses as the same `unread-module` problem. The reading is REFUSED
rather than counted as read, deliberately: a name list says the module is there and says nothing
about whether it starts a child, so "not measured here" is the only honest statement this scan can
make about an archive. Three places an archive can sit are read -- the root itself, an archive
inside it, and the sibling under the exact name CPython puts on `sys.path` (looked up by name and
not by glob, since the parent of the stdlib directory is a shared library directory on most hosts)
-- and only a candidate that exists and opens as a zip counts, so the placeholder `sys.path` entry
of an ordinary source install contributes nothing. Source and cached entries count alike, because a
`.py` inside an archive is as unread as a `.pyc`; an entry that is a directory, a data file, a
non-identifier stem, or under an excluded segment names no module; an archived name whose source the
directory tree also carries is not a finding (an archive shipped beside a full source tree is copies
of what was read); and a submodule of a package that is itself archived is left to that package's
one finding, the rule a cached `__init__` is already handled by. **On this host: 0 archives found,
0 archived, 0 archived submodules** -- the `/usr/lib/python313.zip` entry CPython names does not
exist here -- and the lookup costs 0.0002s. The stdlib half still refuses rather than parsing INTO
an archive, and that stays a decision about the layouts that ship: the embeddable builds carry
`.pyc` only, so there is nothing to parse there. The dependency half is where an archive does carry
source, and it is read rather than refused -- below.

**The installed packages are read the same way, for the one question that can differ there.** This
repo runs on dependencies that start children constantly (torch dataloader workers, vLLM engine
processes, uv, the build scripts), and each was covered only by that same unstated assumption.
`llb.quality.gpu_guard.reach.installed.installed_spawn_reaches` reads the venv's site-packages with
a narrower alphabet -- `below_the_seams()`: `posix`, `_posixsubprocess`, `_winapi` -- because a
dependency calling `subprocess.Popen` says nothing the declaration does not already say, while
scanning for the covered names too means parsing 7420 files instead of 301 (measured). A one-off
full-alphabet pass over this host's 40119 site-packages files found **362 packages that start a
child and exactly 5 files that go below the seams**, in two packages: `joblib`'s vendored `loky` (3
files -- `backend/fork_exec.py` -> `_posixsubprocess.fork_exec`, plus `_winapi.CreateProcess` in
`backend/popen_loky_win32.py` and `backend/resource_tracker.py`) and `multiprocess` (2 files -- a
`dill`-based fork of `multiprocessing`, carrying a private copy of the low-level bypass in `util.py`
and the Windows residual in `popen_spawn_win32.py`). Neither private implementation reaches the
stdlib helper seam, so both remain declared in `DECLARED_PACKAGE_REACHERS` -- by PACKAGE rather than
by file, since a release moves its modules and the decision an operator makes is about the
dependency. A THIRD package arriving is what
`gpu_guard_spawn_reach_installed_audit.audit_installed_reach` refuses; an excuse is looked up as the
exact path first and then the top-level package, so the stdlib and package tables read through one
lookup. `nt` is deliberately absent from the installed alphabet: it is the Windows twin of names
`os` re-exports, and its two-letter module name matches too much text to prefilter on, so including
it would cost a full-tree parse on every host for a platform whose denial mechanism is already a
residual.

**A package excuse carries the reach it was MEASURED against, so a widened vendored backend arrives
as a line to re-read.** Package granularity is the right unit for surviving a release bump and the
wrong unit for a residual: it excuses every module in the package, so a future `joblib` that starts
children a second way, from a file the reason never saw, would be covered by a line written about
`loky`. Narrowing it back to per-file declarations would reintroduce the churn the package unit
exists to avoid, so each declaration is a `PackageReacher` instead -- the `SpawnCoverage` reason
plus the primitives and the file count it was written on (`joblib`: 3 files, `multiprocess`: 2, both
through `_posixsubprocess.fork_exec` + `_winapi.CreateProcess`). A declaration cannot be added
without that record: `PackageReacher.__post_init__` refuses an empty primitive list or a zero file
count. `gpu_guard_spawn_reach_installed_audit.outgrown_reachers` then reports a declared package
that reaches a primitive its excuse was not measured on, or starts children from more files than it
was (naming those files), and `audit_installed_reach` includes those findings, so the widening turns
the suite red on the release that introduces it rather than passing under the old reason. Growth
only: a package reaching the same way from FEWER files -- a dropped backend, a slimmer build -- is
not a decision to revisit, and an excuse that matches nothing at all is already what
`absent_reachers` reports. What this does NOT do is close either vendored residual or check the
declarations of third-party packages per file; both remain what they were. Residual: the record is a
COUNT and a primitive set, not the file identities, so a release that renames one backend while
dropping another reaches the same way from the same number of files and stays quiet -- naming the
paths is the per-file churn the package unit exists to avoid.

**A dependency that ships ZIPPED is parsed out of its archive, where a zipped stdlib is refused.**
The installed scan was directory-shaped in exactly the way the stdlib scan had been: a dependency
with no package directory to walk -- a zipped egg, a `--zip-ok` install, any `sys.path` entry that
is an archive rather than a directory -- was parsed by nothing and reported by nothing, so
`audit_installed_reach` returned clean for a venv half of which it never opened. That is worse here
than one tree over, because site-packages is where the packages that start children constantly live.
`llb.quality.gpu_guard.reach.installed_archive` reads the archives on the import path -- `sys.path`
entries that open as a zip, plus `*.egg` / `*.zip` under the scan root, minus the stdlib's own
`pythonXY.zip`, which the stdlib half already accounts for and which reporting here would be the
same finding twice. And the read-or-refuse call the stdlib half deferred is taken the OTHER way,
because the evidence differs: a zip-shipped stdlib is `.pyc`-only (what the embeddable builds
carry), while a zip-shipped dependency carries `.py` -- `bdist_egg` zips the source tree -- which
the tests establish rather than assume by fabricating an egg-shaped archive, importing it through
`zipimport` on this interpreter, and reading its source back out with `zipfile`. So a `.py` entry is
parsed out of the archive (the same bytes through the same parser and the same import resolution as
a file on disk) and counted as read, with `ModuleReach.container` naming the zip it came from; the
reach it finds is weighed against the same `DECLARED_PACKAGE_REACHERS` excuse a file on disk would
be, since the top-level package of `pkg/backend/start.py` is the same `pkg` either way. Both halves
fold into ONE `SpawnScan` (`with_archives`) so `files_read` adds up over the whole import path -- a
venv that ships only zipped is then a scan that read source, not a tree refused as `unscanned` while
its source sat in a zip nobody opened. What is left over is refused by
`gpu_guard_spawn_reach_installed_audit.unread_archived_packages` as the same `unread-module`
problem: a module an archive ships compiled with no source anywhere -- not in that archive, not in
another one on the same path, not as a copy in the directory tree. Per PACKAGE, because that is the
unit the excuses are written at and an operator acts on, so a `.pyc`-only egg is one line naming the
modules it hid rather than one line per module; and a package the declarations already name is not
refused at all, because the declaration IS the decision that it starts children and that this is
accepted. **On this host: 0 archives on the import path, 0 unread archived** -- every dependency
here installs as a directory tree -- and the discovery costs 0.0013s. Residual: an archive is only
as readable as its entries, so a `.pyc`-only dependency is still a refusal rather than a
measurement, which is the same statement the stdlib half makes and for the same reason.

**The tree this repo itself ships is read too, because a `.pth` file is the third kind of
import-path entry.** An archive is not the only thing on the path that is not the scanned directory:
a `.pth` file adds other DIRECTORIES to it, and that is not an exotic layout -- it is how this repo
is installed. `__editable__.llb-0.1.0.pth` holds one line, `<repo>/src`, so `llb`'s own modules were
parsed by neither scan while every dependency around them was held to the question, and the code an
unmarked test runs the most was the one tree nobody asked it of.
`llb.quality.gpu_guard.reach.installed_sites` reads those files with `site.addpackage`'s own rule --
a line starting with the word `import` plus a space or a tab is CODE the interpreter runs, a comment
or a blank line is nothing, anything else is a path resolved against the file's directory -- and
`installed_spawn_reaches` folds the resulting trees into the same `SpawnScan` (`SpawnScan.sites`
records which), so `files_read` and `modules_read` now add up over the whole import path. The `.pth`
files are read rather than `sys.path`, deliberately: `sys.path` would answer too, and would answer
wrong under pytest, which puts the repo root and the test directories on it, so a scan of those
walks the venv it is trying to describe. One entry is left alone for a stated reason: a path INSIDE
the scan root, which the directory pass already walked (`nvidia-cutlass-dsl` ships one, making
`cutlass` importable out of a subdirectory of site-packages -- reading it again would count those
files twice and report one file under two package names, `cutlass` here and the `nvidia_cutlass_dsl`
its distribution publishes there, which is the name an excuse would be written at). A reach found in
an added tree carries it as `ModuleReach.container`, so the finding names the file an operator has
to open rather than a path that reads like site-packages and is not.

**Executable `.pth` lines are now resolved or refused, never silently skipped.** Setuptools' common
flat-layout form writes `import __editable___pkg_finder; __editable___pkg_finder.install()` and
keeps the exposed names and targets in the generated finder's `MAPPING`. The static decoder in
`llb.quality.gpu_guard.reach.installed_finder` parses that exact installer statement and
reads only an `ast.literal_eval`-compatible mapping assignment; it never imports the finder or
executes either file. Package-directory targets are scanned under their mapped import name, and
single-file module targets are read directly, so the pass does not expose or scan unrelated
siblings from the source parent. Any other executable line is retained as `<pth-name>:<line>` in
`SpawnScan.unread_path_entries`, and
`gpu_guard_spawn_reach_installed_audit.unread_path_entries` emits an `unread-path-entry` finding
that says the pass cannot know which trees the line adds. Fabricated coverage in
`tests/llb/quality/test_gpu_guard_spawn_reach_installed.py` pins literal-path, generated-finder,
single-file mapping, non-execution, and unresolved-line behavior; run it through `make ci`, while
the real import-path assertions remain in the slow tier. This venv has no generated editable finder
to decode: its direct `<repo>/src` line is still read, while `_virtualenv.pth:1` and
`distutils-precedence.pth:1` are explicitly reported as the two unresolved bootstrap hooks.

**On this host: two literal entries, one under the root, so ONE tree scanned -- `<repo>/src`, 931
files in 0.04s, and no reach below the seams at all.** That is the answer to whether this repo's own
source needs a declaration like a dependency's: it starts children in fifteen modules (`backends/*`,
`build/vllm.py`, `cli/ui.py`, `executor/*`, `tracking/server.py`, and the rest) and every one of
them goes through `subprocess.run` / `subprocess.call` / `subprocess.Popen`, which the denial
patches -- so it is held to exactly the question a dependency is held to, by the same
`audit_installed_reach`, and needs no excuse to pass it.

**The installed scan says what it FAILED to read, so "no dependency goes below the seams" names the
venv it was read from.** The stdlib half accounts for every declared name it read no source for; the
installed half had only the degenerate end -- an empty read, plus a `files_read` assertion -- which
is exactly the check the stdlib half outgrew, because a file count says how much was read and not
what was missed. A dependency installed with its sources stripped is parsed by nothing and reported
by nothing: the directory-tree twin of the archive case above.
`llb.quality.gpu_guard.reach.installed_coverage` weighs the scan against the union of the
top-level names `importlib.metadata.packages_distributions()` publishes and the names every
resolved filesystem entry actually provides. The metadata half is read through
`importable_top_level_names` because some distributions record a path (`nvidia/cusparselt`,
`sentencepiece/__init__`); the filesystem half is read without imports by
`gpu_guard_spawn_reach_installed_paths.provided_top_level_names`. `SpawnScan.path_entries` retains
the optional import name from a generated finder mapping, while an ordinary directory entry is
enumerated from its immediate packages, modules, caches, and extensions. In-root `.pth` entries are
recorded too even though their files are not counted or parsed twice, which is what brings
`cutlass` from `nvidia_cutlass_dsl/python_packages` into the declared surface. Every name no source
was read for is classified against the entry or entries that provide it, and as one tree over the
classification is the deliverable: `extensions` (the name resolves to a shared object --
an extension module installed under the name itself, or a directory shipping objects and no Python,
which is what the `nvidia-*` wheels are), `namespace` (a directory with no module of its own: an
implicit namespace package, a PEP 561 `-stubs` directory, a data directory like `include` or
`schemas`), `compiled_only` (a cached module with no source beside it), `archived` (nothing in the
tree and an archive on the import path carries it), and `absent` (nothing the pass read provides
it). **On this venv, of 424 provided or metadata-published top-level names: 406 read as source,
6 namespace, 0 compiled-only, 0 archived, 2 absent** -- the measurement costs 0.9s on top of the
2.2s scan, and the fields partition the list so a name cannot fall between two. The three names
missing from distribution metadata -- `OleFileIO_PL`, `_virtualenv`, and `cutlass` -- are all
accounted for as read; the namespace list includes filesystem-provided `cpp` and `saxonc` too.
`gpu_guard_spawn_reach_installed_audit.audit_installed_read_coverage` refuses ONLY `compiled_only`,
decided on that evidence: it is the stripped tree, where the module is importable here and the scan
did not read it. `extensions` and `namespace` have no source by construction, and a gate refusing
either is the naive one that fails on any host with a CUDA wheel installed. `archived` is left to
`unread_archived_packages`, which already refuses those names at this same granularity -- reporting
them here too would be one finding wearing two names. `absent` is an ANSWER rather than an artifact
of reading one root, now that the pass reads the tree, its archives, and the directories a `.pth`
adds: what is left is a distribution recording a submodule as a top-level name, which
`tree-sitter-*` (`_binding`) and `xxhash` (`_xxhash`) both do here. It is still reported and not
refused -- a name nothing provides cannot start a child, and refusing two third-party metadata
quirks is the naive gate again. The refusal is
grouped per PACKAGE, the way the archive one is, and the submodule level joins its own package's
line -- so a stripped dependency is one line naming the modules it hid, and a package the
declarations already name is not refused at all. `compiled_only_submodules` is reused unchanged from
the stdlib coverage for the scan root; the per-entry extension is deliberately top-level, the unit
the task and distribution declarations use. `installed_read_coverage_message` renders the
breakdown and every resolved entry as the assertion message the venv test fails with. Fabricated
cases pin an unrecorded source package, a cached-only package in an external `.pth` tree, and an
in-root entry under its actual import name; `make ci` runs those cases, and the live-path assertion
remains in the slow tier.

**The site-packages cases are `slow` and the stdlib ones are not, decided on the measured cost.** The
stdlib is ~600 files that ship with the interpreter; site-packages is whatever is installed -- 40119
files and 556 MB here, 2.2s warm and disk-bound cold -- and it changes only when the lock file does,
which is a `make test` moment rather than a `make ci` one. The mechanism itself (the package-level
excuse, the narrow alphabet, a call that goes around `os` through `posix`, a source-carrying archive,
a `.pyc`-only one, a `.pth`-added tree) is pinned over fabricated trees, eggs, and `.pth` files in
the non-slow tier, so
`make ci` still covers the code and only the 40k-file read is
deferred. The split follows the same seam the source does: `test_gpu_guard_spawn_reach.py` is the
stdlib pass and `test_gpu_guard_spawn_reach_installed.py` the dependency one. Residual: the scan
reads SOURCE, so a dynamic import, a call through an object attribute,
and anything a compiled extension does below Python stay invisible -- the last being the same
native-extension residual the denial itself carries.

Coverage is six files. `tests/llb/quality/test_gpu_guard.py` is the observation half plus the
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
default start method is a residual), because those cases cannot be produced by the host. Its default
reader cases also prove that the parent remains unresolved, an existing choice avoids a child, a
child/order mismatch is refused, and two reads start the child only once. The standard `make ci`
gate runs the focused proof; all 3,041 non-slow tests pass, including all 22 cases in that suite.

**Both function-complexity thresholds are enforced, not reported.** `scripts/complexity_gate.sh`
runs the Radon D-or-worse scan and the Complexipy scan at `COGNITIVE_MAX=15` over `src` and
`tests`, and
exits non-zero as soon as either prints a row -- so a function that crosses a threshold turns the
build red on its own commit instead of surfacing in a later sweep. `ci-checks` runs it beside
`llb.quality.acceptance_gates --check` (so it runs in `make ci`, `make ci-github`, and GitHub CI);
`make complexity-gate` runs it alone after a refactor. Neither tool signals through its exit status
in this configuration (radon always exits 0, complexipy is asked for a plain listing with
`--ignore-complexity`), so the gate treats printed output as the failure. Both scans always run --
one pass shows every peak. The thresholds are the shipped numbers: a finding is split, not
accommodated by raising the maximum.

**Both scanners are pinned exactly** (`radon==6.0.1`, `complexipy==6.0.0`), for the reason the
shell-lint gate pins its ShellCheck wheel: the tool version is half of the verdict. Complexipy
rescored this unchanged tree across its 5.6.1 -> 6.0.0 boundary in both directions --
`persist_run` 18 -> 14 and `research_conflict_nulls_cmd` 16 -> 15 (dropping to the maximum),
`fetch_manifest` 11 -> 13 (climbing) -- so a floating requirement let the same commit fail the gate
on the host that resolved 5.6.1 and pass on the host that resolved 6.0.0. The pin is what makes
`COGNITIVE_MAX=15` mean one thing. Upgrading either scanner is a deliberate edit: change the pin in
`pyproject.toml`, refresh `uv.lock`, then run `make complexity-gate` plus `make ci` and split
whatever the new algorithm surfaces.

The scans, thresholds, and labels live once in `scripts/shared/complexity.sh`, sourced by both the
gate and `scripts/code_quality.sh`, so the sweep fails on exactly what CI fails on and prints it
identically (block reporting is `llb_print_block` / `llb_report_if_output` / `llb_fail_if_output`
in `scripts/shared/common.sh`). The same shared module owns the informational maintainability-index
scan so its scope cannot drift from the two gates.

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
stays green.

**A path computed at run time is annotated `# shellcheck source=/dev/null` with its reason, and
that choice is the difference between one verdict and one verdict per host.** `common.sh` sources
`$PROJECT_ROOT/.env`, an untracked operator file, and `-x` FOLLOWS a sourced path where it exists
while reporting SC1091 where it does not -- so the source-directive scan was silent on every dev
box (which has a `.env`) and failed the gate in a fresh GitHub checkout (which does not). That is
the same split verdict the pinned binary exists to prevent, arriving through the tree instead of
through the linter, so the fix says once that there is nothing static to follow rather than letting
the linter decide per host; `disable=SC1091` would have hidden the message while leaving the
following itself host-dependent. `test_the_source_directive_scan_is_clean_in_a_checkout_that_has_no_env_file`
in `tests/llb/quality/test_shell_lint_resolution.py` is the regression: it copies the tracked `*.sh`
into a `git init` tmp tree with no `.env`, runs the real `llb_shellcheck_sources_scan` against it,
and requires empty output -- the side a dev box cannot otherwise see (1.5s, non-slow, since a
CI-only failure has to be caught in `make ci`).

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

`llb-requires` scope is also minimal at the file boundary. `unused_requirements` reports the
declaration line when the declaring caller names none of the `llb_*` functions its sibling defines,
and the command exits non-zero so `make shell-lint-gate` refuses the stale declaration. One direct
reference justifies the whole sibling; this remains a file-level contract rather than a per-symbol
import list. Real `source` directives are not subject to this rule because a sourced file can supply
variables or intentional side effects without a function call.

The gate decision came from the shipped tree rather than an exception policy: all 19 current
`llb-requires` declarations support direct calls, including the four declarations in `track_c.sh`.
No declaration exists only to document load order. A downstream fragment owns its own sibling
requirements, and transitive scope already follows them, so adding a load-order-only declaration to
an otherwise unrelated caller would widen the symbol contract without resolving one of that
caller's names. The shipped-tree assertion and synthetic stale/used declaration cases live in
`tests/llb/quality/test_shell_symbols.py`.

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
is rarely needed: shellcheck now ships in the `dev` extra as the exactly pinned
`shellcheck-py==0.11.0.1` wheel (the real 0.11.0 binary), so `.venv/bin/shellcheck` exists wherever
`make ci` runs, GitHub CI included -- no apt step in the workflow. Scans, severity, and the
missing-binary policy live once in
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
no longer lists `shellcheck` as a fallback. The exact project requirement is intentional and is the
outer of two guarantees. `make venv` and GitHub CI both install through `uv sync` (the local
target with `--inexact`, the workflow with `--locked`), so `uv.lock` already pins the wheel for
every venv either one builds; the `==` requirement in `pyproject.toml` extends the same pin to an
install that bypasses the lock -- plain pip, or a bare `uv pip install` run instead of
`make install-extras` ([overview](overview.md#an-extras-install-respects-uvlock)). A host
resolving the extra today and one resolving it months later still install the same wheel. An
upgrade therefore costs one deliberate pin edit in `pyproject.toml`, `uv lock`, and verification
with `make shell-lint-gate` plus `make ci`; the lock and the fresh-install requirement move
together.
The binary resolution behavior remains covered by
`tests/llb/quality/test_shell_lint_resolution.py`.

Everything else in the sweep stays informational -- in particular the `.py`/`.sh` line-count report,
which backs a target AGENTS.md keeps SOFT on purpose and which has legitimate offenders. The
maintainability-index section is deliberately a report too. It now scans only `src` and `tests`, at
grade C by default (`LLB_MI_MIN_GRADE=C`), rather than sweeping the repository root. The decision
not to add it to `llb_complexity_gate` is evidence-based: the C-only scan is empty, while the next
band contains one file, `tests/llb/bench/context_policy/test_agentic_context.py`, at B (10.66). That
file is large (728 lines, 563 source lines) but regular: Radon finds 48 blocks with average
cyclomatic complexity A (4.15), and its two highest blocks are only B (10). It is already the kind
of volume case covered by the soft line target, so making the nearby MI boundary hard would turn
that soft policy into an indirect gate rather than add an independent complexity signal. Focused
coverage in `tests/llb/quality/test_maintainability_report.py` pins both the `src tests` arguments
and the informational exit behavior when Radon prints a C row.

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
  Typed field reads moved to `bench/agentic/design_fields.py`, so a rule reads as the contract
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
`rag/encoders/throughput.py` into measurement / `_summary` / `_report` / `_profile`, and
`bench/context_policy/sweep.py` into `_model` (axes, grids, cells) / `_verdict` (pairing and the
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
