"""The child-starting call sites of the STDLIB, read rather than assumed.

The name surface covers `os` and `subprocess`; everything else is covered only if the helper it
calls resolves a name in those two. This file is where that stops being a sentence: one pass over
the stdlib this interpreter ships asserts that every module which starts a child does it through a
DECLARED name, and that every declared name it read no source for is accounted for by a construction
that has none. The same question of this venv's DEPENDENCIES is
`test_gpu_guard_spawn_reach_installed.py`, which is `slow` where these are not: the stdlib is ~600
files that ship with the interpreter (0.9s), while site-packages is whatever is installed.

The rest drives the scan over fabricated trees and archives, because a tree that reaches past every
patchable name cannot be produced on demand -- and because the ways a source scan goes wrong (an
aliased import, a local function that shares a name with a spawn entry point, a file that will not
parse) are worth pinning where they can be written out in four lines.
"""

import sys
import zipfile
from collections.abc import Mapping
from importlib.util import cache_from_source
from pathlib import Path

import pytest

from llb.quality import gpu_guard_spawn_reach as reach
from llb.quality import gpu_guard_spawn_reach_archive as archive
from llb.quality import gpu_guard_spawn_reach_audit as reach_audit
from llb.quality import gpu_guard_spawn_reach_coverage as coverage
from llb.quality import gpu_guard_spawn_surface as surface
from llb.quality import gpu_guard_spawn_surface_audit as audit


@pytest.fixture(scope="module")
def stdlib_scan() -> reach.SpawnScan:
    """The real stdlib scan, run once for the four cases that read it (0.9s on this host)."""
    return reach.stdlib_spawn_reaches()


def _stdlib(tmp_path: Path, modules: Mapping[str, str]) -> Path:
    """A fabricated stdlib tree: relative path -> source."""
    root = tmp_path / "stdlib"
    for relative, source in modules.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _paths(scan: reach.SpawnScan) -> dict[str, tuple[str, ...]]:
    return {module.path: module.primitives for module in scan.reaches}


def _archive(path: Path, entries: Mapping[str, str]) -> Path:
    """A fabricated zip-imported stdlib: archive entry path -> contents (never read)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as opened:
        for name, contents in entries.items():
            opened.writestr(name, contents)
    return path


def test_this_stdlib_starts_children_only_through_names_the_denial_declares(stdlib_scan):
    """The claim the name surface rested on, now a result: no undeclared way in."""
    findings = reach_audit.audit_spawn_reach(stdlib_scan)
    assert findings == (), audit.surface_message(findings)


def test_the_modules_that_reach_past_the_declared_surface_are_exactly_the_declared_ones(
    stdlib_scan,
):
    """The evidence behind "two modules is the right enumerated surface"."""
    past = {
        module.path
        for module in stdlib_scan.reaches
        if any(name not in surface.DECLARED_SPAWN_SURFACE for name in module.primitives)
    }
    assert past == set(reach.DECLARED_REACHERS)


def test_every_excused_module_is_one_the_scan_still_finds(stdlib_scan):
    """An excuse that outlives what it excused is an excuse nobody re-reads."""
    assert reach_audit.absent_reachers(stdlib_scan) == ()


def test_the_scan_reads_the_stdlib_that_actually_ships_here(stdlib_scan):
    """A guard against the scan quietly reading nothing: these three are stable across versions."""
    found = _paths(stdlib_scan)
    assert "os.forkpty" in found["pty.py"]
    assert "subprocess.Popen" in found["asyncio/unix_events.py"]
    assert "_posixsubprocess.fork_exec" in found["multiprocessing/util.py"]


def test_the_stdlib_scan_accounts_for_every_module_it_read_no_source_for(stdlib_scan):
    """The claim is about the stdlib, so it has to say which stdlib it was read from: every name the
    interpreter declares is either read or excused by a construction that has no source."""
    measured = coverage.stdlib_read_coverage(stdlib_scan)
    message = coverage.read_coverage_message(measured)
    assert reach_audit.audit_read_coverage(stdlib_scan, measured) == (), message
    # A partition of the declared list: no name falls between two buckets or into both.
    assert len(measured.read) + len(measured.unread) == len(sys.stdlib_module_names), message
    assert {"multiprocessing", "subprocess", "asyncio", "pty", "os"} <= set(measured.read), message
    assert "_posixsubprocess" in measured.compiled + measured.extensions, message
    # And one level down, where the declared list stops: this host strips no submodule of a package
    # it otherwise ships, so "read as source" holds for `multiprocessing/util.py` too.
    assert measured.compiled_only_submodules == (), message
    # This host ships its stdlib as a directory tree, so the archive reading finds nothing to
    # account for -- the `sys.path` zip entry CPython names is a placeholder that does not exist.
    assert (measured.archives, measured.archived, measured.archived_submodules) == ((), (), ()), (
        message
    )


def test_the_scan_excludes_no_directory_this_interpreter_calls_a_stdlib_module():
    """The exclusion rule and the coverage measurement meet here: a skipped directory that IS a
    declared module would read as unread source rather than as a stated omission."""
    assert not (reach._EXCLUDED_SEGMENTS & set(sys.stdlib_module_names))


def test_a_module_with_a_pyc_and_no_source_is_refused_rather_than_read_as_quiet(tmp_path):
    """The middle the file count cannot see: a frozen, zipped, or stripped layout, where the module
    is importable, can start a child, and was never parsed."""
    root = _stdlib(
        tmp_path,
        {
            "quiet.py": "import json\n\ndef go():\n    return json.dumps({})\n",
            "stripped/__pycache__/__init__.cpython-313.pyc": "",
        },
    )
    scan = reach.stdlib_spawn_reaches(root)
    measured = coverage.stdlib_read_coverage(scan, names=["quiet", "stripped"])
    assert (measured.read, measured.compiled_only) == (("quiet",), ("stripped",))
    findings = reach_audit.audit_read_coverage(scan, measured)
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("stripped", reach_audit.PROBLEM_UNREAD_MODULE)
    ]
    assert "stripped" in coverage.read_coverage_message(measured)
    # A cached `__init__` names its PACKAGE, which is a declared name and is classified as one --
    # reporting `stripped.__init__` beside it would be the same finding twice.
    assert measured.compiled_only_submodules == ()


def test_a_stripped_submodule_of_a_package_the_scan_read_is_refused(tmp_path):
    """The level the declared list stops at: `sys.stdlib_module_names` names `pkg` and never
    `pkg.util`, so a package that ships its `__init__.py` and not its submodules reads as read.
    The interpreter leaves the evidence in the package directory instead."""
    root = _stdlib(
        tmp_path,
        {
            "pkg/__init__.py": "import json\n",
            "pkg/__pycache__/__init__.cpython-313.pyc": "",
            "pkg/__pycache__/util.cpython-313.pyc": "",
            "pkg/sub/__init__.py": "import json\n",
            "pkg/sub/__pycache__/deep.cpython-313.pyc": "",
        },
    )
    scan = reach.stdlib_spawn_reaches(root)
    measured = coverage.stdlib_read_coverage(scan, names=["pkg"])
    assert (measured.read, measured.compiled_only) == (("pkg",), ())
    assert measured.compiled_only_submodules == ("pkg.sub.deep", "pkg.util")
    findings = reach_audit.audit_read_coverage(scan, measured)
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("pkg.sub.deep", reach_audit.PROBLEM_UNREAD_MODULE),
        ("pkg.util", reach_audit.PROBLEM_UNREAD_MODULE),
    ]
    assert "pkg.util" in coverage.read_coverage_message(measured)


def test_a_package_whose_submodules_all_ship_source_is_clean(tmp_path):
    """The other direction, and the naive gate this avoids: a `.py` with no `.pyc` is nothing --
    caching is incidental -- and a submodule that is in neither list is simply not shipped."""
    root = _stdlib(
        tmp_path,
        {
            "pkg/__init__.py": "import json\n",
            "pkg/__pycache__/__init__.cpython-313.pyc": "",
            "pkg/util.py": "import json\n",
            "pkg/__pycache__/util.cpython-313.pyc": "",
            "bare/__init__.py": "import json\n",
            "bare/helper.py": "import json\n",
        },
    )
    scan = reach.stdlib_spawn_reaches(root)
    measured = coverage.stdlib_read_coverage(scan, names=["pkg", "bare"])
    assert measured.compiled_only_submodules == ()
    assert reach_audit.audit_read_coverage(scan, measured) == ()


def test_a_cached_submodule_is_matched_to_the_source_name_that_actually_wrote_it(tmp_path):
    """Neither half of `<stem>.<tag>.pyc` is one dot-separated component, and both shapes ship in
    this repo's venv: `optuna` names its alembic revisions `v1.2.0.a.py`, so a stem read to the
    first dot loses most of it, and pytest writes its rewritten caches under a tag of its own
    (`cpython-313-pytest-9.1`), so a tag read back from the last dot eats into the module name.
    Either misreading reports source that is sitting right there as stripped -- four `optuna`
    modules, then 397 across the venv, before the split was anchored on the interpreter's own
    `cache_tag`."""
    dotted = "pkg/v1.2.0.a.py"
    rewritten = f"pkg/__pycache__/helper.{sys.implementation.cache_tag}-pytest-9.1.pyc"
    root = _stdlib(
        tmp_path,
        {
            "pkg/__init__.py": "import json\n",
            dotted: "import json\n",
            cache_from_source(dotted): "",
            "pkg/helper.py": "import json\n",
            rewritten: "",
            # The same two shapes with no source beside them, which IS the finding -- so the rule is
            # pinned as reading the name rather than as never reporting a dotted one.
            cache_from_source("pkg/v9.9.9.a.py"): "",
        },
    )
    measured = coverage.stdlib_read_coverage(reach.stdlib_spawn_reaches(root), names=["pkg"])
    assert measured.compiled_only_submodules == ("pkg.v9.9.9.a",)


def test_a_cached_submodule_of_another_interpreter_reads_against_the_source_beside_it(tmp_path):
    """A cache this interpreter cannot import from is not evidence that a module was stripped: the
    tag it does carry is not the one to split on, so the PEP's shape is what is left to read."""
    root = _stdlib(
        tmp_path,
        {
            "pkg/__init__.py": "import json\n",
            "pkg/util.py": "import json\n",
            "pkg/__pycache__/util.cpython-311.pyc": "",
        },
    )
    measured = coverage.stdlib_read_coverage(reach.stdlib_spawn_reaches(root), names=["pkg"])
    assert measured.compiled_only_submodules == ()


def test_a_cached_submodule_the_scan_never_walked_is_not_measured_either(tmp_path):
    """The exclusion rule is the scan's, so the two halves cannot disagree about which directories
    the statement covers."""
    root = _stdlib(
        tmp_path,
        {
            "quiet.py": "import json\n",
            "test/__pycache__/support.cpython-313.pyc": "",
            "site-packages/vendored/__pycache__/child.cpython-313.pyc": "",
        },
    )
    measured = coverage.stdlib_read_coverage(reach.stdlib_spawn_reaches(root), names=["quiet"])
    assert measured.compiled_only_submodules == ()


def test_a_stdlib_that_ships_as_an_archive_is_refused_rather_than_read_as_absent(tmp_path):
    """The layout both directory-shaped reads are blind to. Nothing is under the root to compare, so
    every name used to fall through to `absent` -- recorded as "this host does not ship it" for a
    module the interpreter imports on demand. The scan refuses the tree as unscanned as well, and
    that is the point: the two answers now agree instead of one saying the stdlib is not there."""
    root = _archive(
        tmp_path / archive.ZIP_STDLIB_NAME,
        {"os.pyc": "", "subprocess.pyc": "", "multiprocessing/__init__.pyc": ""},
    )
    scan = reach.stdlib_spawn_reaches(root)
    assert scan.files_read == 0
    assert [(finding.name, finding.problem) for finding in reach_audit.audit_spawn_reach(scan)] == [
        (str(root), reach_audit.PROBLEM_UNSCANNED)
    ]
    measured = coverage.stdlib_read_coverage(scan, names=["multiprocessing", "os", "subprocess"])
    assert (measured.archived, measured.absent) == (
        ("multiprocessing", "os", "subprocess"),
        (),
    )
    assert measured.archives == (str(root),)
    findings = reach_audit.audit_read_coverage(scan, measured)
    assert [finding.problem for finding in findings] == [reach_audit.PROBLEM_UNREAD_MODULE] * 3
    assert "3 archived" in coverage.read_coverage_message(measured)


def test_an_archive_beside_the_stdlib_root_is_the_one_this_interpreter_would_import_from(tmp_path):
    """Where CPython actually puts it: `<prefix>/lib/pythonXY.zip` sits BESIDE the stdlib directory
    and on `sys.path`, so a root-relative walk alone would never reach it. Looked up by that exact
    name rather than by glob, because the parent is a shared library directory on most hosts."""
    _archive(tmp_path / archive.ZIP_STDLIB_NAME, {"subprocess.pyc": ""})
    _archive(tmp_path / "unrelated.zip", {"payload/thing.pyc": ""})
    root = _stdlib(tmp_path, {"quiet.py": "import json\n"})
    measured = coverage.stdlib_read_coverage(
        reach.stdlib_spawn_reaches(root), names=["quiet", "subprocess"]
    )
    assert (measured.read, measured.archived) == (("quiet",), ("subprocess",))
    assert measured.archives == (str(tmp_path / archive.ZIP_STDLIB_NAME),)


def test_a_stdlib_half_on_disk_and_half_in_an_archive_is_refused_at_both_levels(tmp_path):
    """The case nothing caught: files ARE read, so the scan is not degenerate and no check fires,
    while a `subprocess` only the archive carries reads as absent and an archived
    `multiprocessing/util.pyc` produces no submodule finding at all -- a clean coverage line over a
    library half of which was never read."""
    _archive(
        tmp_path / "stdlib" / "python313.zip",
        {"subprocess.pyc": "", "multiprocessing/util.pyc": "", "multiprocessing/__init__.pyc": ""},
    )
    root = _stdlib(
        tmp_path, {"os.py": "import json\n", "multiprocessing/__init__.py": "import json\n"}
    )
    scan = reach.stdlib_spawn_reaches(root)
    assert scan.files_read == 2
    measured = coverage.stdlib_read_coverage(scan, names=["os", "multiprocessing", "subprocess"])
    assert (measured.read, measured.archived, measured.absent) == (
        ("multiprocessing", "os"),
        ("subprocess",),
        (),
    )
    # `multiprocessing` itself is read, so the level the declared list stops at is where its
    # stripped `util` shows up -- the archive's own name list in place of the package directory.
    assert measured.archived_submodules == ("multiprocessing.util",)
    findings = reach_audit.audit_read_coverage(scan, measured)
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("subprocess", reach_audit.PROBLEM_UNREAD_MODULE),
        ("multiprocessing.util", reach_audit.PROBLEM_UNREAD_MODULE),
    ]


def test_an_archived_name_the_directory_tree_also_carries_as_source_is_not_a_finding(tmp_path):
    """The naive gate this avoids: an archive shipped BESIDE a full source tree carries copies of
    what the scan already read, and a copy of a module that was read is not an unread module."""
    _archive(
        tmp_path / archive.ZIP_STDLIB_NAME,
        {"os.pyc": "", "pkg/util.pyc": "", "pkg/sub/__init__.pyc": "", "pkg/gone.pyc": ""},
    )
    root = _stdlib(
        tmp_path,
        {
            "os.py": "import json\n",
            "pkg/__init__.py": "import json\n",
            "pkg/util.py": "import json\n",
            "pkg/sub/__init__.py": "import json\n",
        },
    )
    measured = coverage.stdlib_read_coverage(reach.stdlib_spawn_reaches(root), names=["os", "pkg"])
    assert (measured.read, measured.archived) == (("os", "pkg"), ())
    # Only the one submodule the archive carries and the tree does not.
    assert measured.archived_submodules == ("pkg.gone",)


def test_a_submodule_of_an_archived_package_is_left_to_that_packages_own_finding(tmp_path):
    """A zip-shipped stdlib carries thousands of submodules under names that are already refused;
    reporting both is the same finding twice, the rule a cached `__init__` is handled by."""
    root = _archive(
        tmp_path / archive.ZIP_STDLIB_NAME,
        {"multiprocessing/__init__.pyc": "", "multiprocessing/util.pyc": ""},
    )
    measured = coverage.stdlib_read_coverage(
        reach.stdlib_spawn_reaches(root), names=["multiprocessing"]
    )
    assert (measured.archived, measured.archived_submodules) == (("multiprocessing",), ())


def test_an_archive_entry_that_is_not_an_importable_module_names_nothing(tmp_path):
    """Read as zipimport would: a directory entry, a data file, a suffix no importer loads, and a
    stem that is not an identifier are not modules -- and the scan's own exclusion rule applies to
    the entry paths, so the two halves cannot disagree about which directories are covered."""
    path = _archive(
        tmp_path / archive.ZIP_STDLIB_NAME,
        {
            "pkg/": "",
            "LICENSE.txt": "",
            "pkg/data.json": "",
            "pkg/lib.so": "",
            "not-an-identifier.pyc": "",
            "test/support.pyc": "",
            "site-packages/vendored/child.pyc": "",
            "pkg/real.pyc": "",
        },
    )
    assert archive.archived_modules([path]) == ("pkg.real",)
    assert archive.archived_modules([tmp_path / "missing.zip", _stdlib(tmp_path, {})]) == ()


def test_a_declared_module_this_host_does_not_ship_is_recorded_rather_than_refused(tmp_path):
    """`sys.stdlib_module_names` is what CPython contains, not what this host installed: a
    split-package or python3-minimal layout cannot import what is not there."""
    root = _stdlib(tmp_path, {"quiet.py": "import json\n"})
    scan = reach.stdlib_spawn_reaches(root)
    measured = coverage.stdlib_read_coverage(scan, names=["quiet", "tkinter"])
    assert measured.absent == ("tkinter",)
    assert reach_audit.audit_read_coverage(scan, measured) == ()


def test_a_module_that_has_no_source_by_construction_is_not_an_unread_one(tmp_path):
    """Two thirds of the declared list is compiled in, an extension, or a name of another platform;
    a gate that refused those would fail on every host."""
    root = _stdlib(
        tmp_path,
        {"quiet.py": "import json\n", f"speedy{coverage.EXTENSION_SUFFIXES[0]}": ""},
    )
    scan = reach.stdlib_spawn_reaches(root)
    measured = coverage.stdlib_read_coverage(scan, names=["sys", "_winapi", "speedy", "quiet"])
    assert measured.compiled == ("sys",)
    assert measured.declared == ("_winapi",)
    assert measured.extensions == ("speedy",)
    assert (measured.compiled_only, measured.absent) == ((), ())


def test_the_alphabet_is_taken_from_the_declared_surface_and_the_c_modules_under_it():
    alphabet = reach.spawn_primitives()
    assert {"fork", "posix_spawn", "system"} <= alphabet["os"]
    assert "Popen" in alphabet["subprocess"]
    # `os` re-exports these, so a caller can reach them without going through `os` at all.
    assert alphabet["posix"] == alphabet["os"]
    assert alphabet["_posixsubprocess"] == frozenset({"fork_exec"})
    # A record is not an entry point, so it is not in the alphabet either.
    assert "CompletedProcess" not in alphabet["subprocess"]


def test_a_module_that_starts_a_child_through_a_covered_name_is_found_and_passes(tmp_path):
    root = _stdlib(
        tmp_path,
        {
            "plain.py": "import os\n\ndef go():\n    os.fork()\n",
            "aliased.py": "import os as operating\n\ndef go():\n    operating.forkpty()\n",
            "imported.py": (
                "from subprocess import Popen as Runner\n\ndef go():\n    Runner(['true'])\n"
            ),
        },
    )
    scan = reach.stdlib_spawn_reaches(root)
    assert _paths(scan) == {
        "plain.py": ("os.fork",),
        "aliased.py": ("os.forkpty",),
        "imported.py": ("subprocess.Popen",),
    }
    assert reach_audit.audit_spawn_reach(scan) == ()


def test_a_local_name_that_only_looks_like_a_spawn_is_not_read_as_one(tmp_path):
    """The scan resolves through the module's own imports, so a same-named helper is not a hit."""
    root = _stdlib(tmp_path, {"own.py": "def fork():\n    return 0\n\ndef go():\n    fork()\n"})
    scan = reach.stdlib_spawn_reaches(root)
    assert (scan.files_read, scan.reaches) == (1, ())


def test_a_module_that_reaches_past_every_patchable_name_is_refused(tmp_path):
    root = _stdlib(
        tmp_path,
        {
            "covered.py": "import os\n\ndef go():\n    os.fork()\n",
            "below.py": "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n",
        },
    )
    findings = reach_audit.audit_spawn_reach(reach.stdlib_spawn_reaches(root))
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("below.py", reach_audit.PROBLEM_UNCOVERED_REACH)
    ]
    assert "_posixsubprocess.fork_exec" in findings[0].detail


def test_an_excused_module_carries_its_reason_instead_of_a_finding(tmp_path):
    root = _stdlib(
        tmp_path, {"below.py": "import _winapi\n\ndef go():\n    _winapi.CreateProcess()\n"}
    )
    excused = {"below.py": surface.SpawnCoverage(surface.COVERAGE_RESIDUAL, reason="Windows only")}
    assert reach_audit.audit_spawn_reach(reach.stdlib_spawn_reaches(root), reachers=excused) == ()


def test_an_excuse_that_points_at_no_patched_seam_is_refused(tmp_path):
    """`subprocess.py` is excused because `Popen` is patched; take the seam away and it is not."""
    root = _stdlib(
        tmp_path,
        {"below.py": "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"},
    )
    excused = {
        "below.py": surface.SpawnCoverage(
            surface.COVERAGE_THROUGH, through="subprocess.Popen", reason="behind the seam"
        )
    }
    unpatched = surface.ObservedSurface(
        names=(), seams=(), start_methods=("fork",), default_start_method="fork", modules={}
    )
    findings = reach_audit.audit_spawn_reach(
        reach.stdlib_spawn_reaches(root), reachers=excused, surface=unpatched
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("below.py", audit.PROBLEM_UNREACHED)
    ]


def test_cpythons_own_regression_suite_is_left_out_of_the_scan(tmp_path):
    """A corpus that starts children on purpose, and no llb path imports it."""
    below = "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"
    root = _stdlib(
        tmp_path,
        {
            "test/test_spawning.py": below,
            "idlelib/idle_test/test_run.py": below,
            "site-packages/vendored/child.py": below,
            "kept.py": "import os\n\ndef go():\n    os.fork()\n",
        },
    )
    assert _paths(reach.stdlib_spawn_reaches(root)) == {"kept.py": ("os.fork",)}


def test_a_file_the_scan_cannot_parse_is_skipped_rather_than_failing(tmp_path):
    root = _stdlib(
        tmp_path,
        {
            "broken.py": "import os\ndef go(:\n    os.fork()\n",
            "fine.py": "import os\n\ndef go():\n    os.fork()\n",
        },
    )
    assert _paths(reach.stdlib_spawn_reaches(root)) == {"fine.py": ("os.fork",)}


def test_a_tree_the_scan_could_not_read_is_refused_rather_than_read_as_clean(tmp_path):
    """A scan that silently reads no source is the one way this check could pass for free."""
    findings = reach_audit.audit_spawn_reach(reach.stdlib_spawn_reaches(tmp_path))
    assert [(finding.name, finding.problem) for finding in findings] == [
        (str(tmp_path), reach_audit.PROBLEM_UNSCANNED)
    ]


def test_a_tree_that_was_read_and_starts_no_children_is_clean(tmp_path):
    """The distinction the file count buys: read-and-quiet is not the same as never read."""
    root = _stdlib(tmp_path, {"quiet.py": "import json\n\ndef go():\n    return json.dumps({})\n"})
    scan = reach.stdlib_spawn_reaches(root)
    assert scan.files_read == 1
    assert reach_audit.audit_spawn_reach(scan) == ()
