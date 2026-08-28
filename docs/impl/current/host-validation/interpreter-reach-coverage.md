# Interpreter Reach Coverage

The [tier guard](no-gpu-tier-guard.md) covers a set of process-starting seams. That coverage is a
claim about the RUNNING interpreter and its installed packages, so it is re-measured rather than
asserted --
this page is what the audit reads, what it refuses to read, and what each scan costs.

**The coverage claim is re-checked against the running interpreter, not against the one it was
written on.** Both halves of that guard are CPython-specific -- the `os` seams cover the families only
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

## The enumerated surface

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

## Reading the stdlib

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

## Installed packages and vendored backends

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

## The trees a `.pth` file adds

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

## What the scans cost

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
