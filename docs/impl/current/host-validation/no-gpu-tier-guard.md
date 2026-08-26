# The No-GPU And No-Download Tier Guard

The non-slow tier promises two things a reader cannot verify by reading it: that no test
opens a CUDA context, and that none downloads a model. Both are enforced by autouse guards
that REFUSE rather than report, and what follows is how each denial works and where it ends.

**The tier's no-GPU promise is a check, not a convention.** `tests/conftest.py` wraps every
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

## A child is denied the device, not observed

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

## Every spawn entry point

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

## Residual: attribution and reach

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
