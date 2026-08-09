"""The child-starting call sites of the library and its dependencies, read rather than assumed.

The name surface covers `os` and `subprocess`; everything else is covered only if the helper it
calls resolves a name in those two. This file is where that stops being a sentence: one pass over
the stdlib this interpreter ships asserts that every module which starts a child does it through a
DECLARED name, and one pass over this venv's site-packages asserts that no dependency goes below the
seams outside a declared package -- nor from more files, or through more primitives, than the
package's excuse was measured against.

The site-packages cases are `slow` and the stdlib ones are not, decided on the measured cost: the
stdlib is ~600 files that ship with the interpreter (0.9s), while site-packages is whatever is
installed -- 40119 files and 556 MB on this host, 2.2s warm and disk-bound cold -- and it changes
only when the lock file does, which is a `make test` moment rather than a `make ci` one.

The rest drives both scans over fabricated trees, because a tree that reaches past every patchable
name cannot be produced on demand -- and because the ways a source scan goes wrong (an aliased
import, a local function that shares a name with a spawn entry point, a file that will not parse)
are worth pinning where they can be written out in four lines.
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from llb.quality import gpu_guard_spawn_reach as reach
from llb.quality import gpu_guard_spawn_reach_audit as reach_audit
from llb.quality import gpu_guard_spawn_reach_coverage as coverage
from llb.quality import gpu_guard_spawn_surface as surface
from llb.quality import gpu_guard_spawn_surface_audit as audit


@pytest.fixture(scope="module")
def stdlib_scan() -> reach.SpawnScan:
    """The real stdlib scan, run once for the four cases that read it (0.9s on this host)."""
    return reach.stdlib_spawn_reaches()


@pytest.fixture(scope="module")
def installed_scan() -> reach.SpawnScan:
    """The same over this venv's site-packages, narrowed to reaches below the seams (2.2s)."""
    return reach.installed_spawn_reaches()


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


@pytest.mark.slow
def test_this_venvs_packages_go_below_the_seams_only_where_a_package_declares_it(installed_scan):
    """The dependencies an unmarked test actually drives, held to the same question."""
    findings = reach_audit.audit_installed_reach(installed_scan)
    assert findings == (), audit.surface_message(findings)


@pytest.mark.slow
def test_the_packages_that_go_below_the_seams_are_a_subset_of_the_declared_ones(installed_scan):
    """A subset, not equality: a host that never installed `joblib` is not a finding, and a THIRD
    package arriving is what the audit above refuses."""
    reaching = {module.path.split("/")[0] for module in installed_scan.reaches}
    assert reaching <= set(reach.DECLARED_PACKAGE_REACHERS)


@pytest.mark.slow
def test_the_declared_packages_still_reach_only_what_their_excuses_were_measured_on(installed_scan):
    """A package excuse survives a release bump, which is also how a widened vendored backend would
    arrive unread; the measurement is what makes the widening say so."""
    findings = reach_audit.outgrown_reachers(installed_scan)
    assert findings == (), audit.surface_message(findings)


@pytest.mark.slow
def test_the_installed_scan_read_the_venv_rather_than_an_empty_path(installed_scan):
    assert installed_scan.files_read > 100


def test_the_installed_alphabet_is_only_what_goes_below_the_seams():
    """A dependency calling `subprocess.Popen` says nothing the declaration does not already say,
    and looking for it means parsing the whole tree."""
    alphabet = reach.below_the_seams()
    assert set(alphabet) == {"posix", "_posixsubprocess", "_winapi"}
    assert "fork" in alphabet["posix"]


def _excused(files: int, *primitives: str) -> reach.PackageReacher:
    """A package excuse measured against `files` files reaching `primitives`."""
    return reach.PackageReacher(
        surface.SpawnCoverage(
            surface.COVERAGE_RESIDUAL, reason="a private copy of the multiprocessing residual"
        ),
        primitives=primitives or ("_posixsubprocess.fork_exec",),
        files=files,
    )


def test_a_package_is_excused_as_a_whole_because_a_release_moves_its_files(tmp_path):
    """`joblib` vendors `loky` at a path its next release may rename; the decision is the package."""
    below = "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"
    root = _stdlib(
        tmp_path, {"declared/backend/fork_exec.py": below, "undeclared/backend/start.py": below}
    )
    findings = reach_audit.audit_installed_reach(
        reach.installed_spawn_reaches(root), reachers={"declared": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("undeclared/backend/start.py", reach_audit.PROBLEM_UNCOVERED_REACH)
    ]


def test_a_declared_package_that_starts_children_from_a_new_file_is_reported(tmp_path):
    """The failure package granularity invites: a line written about one backend excusing the one
    the next release adds. The excuse still covers it -- the growth is what arrives to be re-read."""
    below = "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"
    root = _stdlib(
        tmp_path, {"vendored/backend/fork_exec.py": below, "vendored/backend/added.py": below}
    )
    findings = reach_audit.audit_installed_reach(
        reach.installed_spawn_reaches(root), reachers={"vendored": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("vendored", reach_audit.PROBLEM_OUTGROWN_REACH)
    ]
    assert "vendored/backend/added.py" in findings[0].detail


def test_a_declared_package_that_reaches_a_primitive_its_excuse_never_saw_is_reported(tmp_path):
    """A second WAY down, from the same file count -- a POSIX-only excuse grown a Windows half."""
    root = _stdlib(
        tmp_path,
        {"vendored/win.py": "import _winapi\n\ndef go():\n    _winapi.CreateProcess()\n"},
    )
    findings = reach_audit.outgrown_reachers(
        reach.installed_spawn_reaches(root), {"vendored": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("vendored", reach_audit.PROBLEM_OUTGROWN_REACH)
    ]
    assert "_winapi.CreateProcess" in findings[0].detail


def test_a_declared_package_that_reaches_the_same_way_from_fewer_files_is_not_growth(tmp_path):
    """Shrinking is not a decision to revisit: a dropped backend, or a host that vendors less."""
    root = _stdlib(
        tmp_path,
        {
            "vendored/util.py": (
                "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"
            )
        },
    )
    measured = _excused(3, "_posixsubprocess.fork_exec", "_winapi.CreateProcess")
    assert (
        reach_audit.outgrown_reachers(reach.installed_spawn_reaches(root), {"vendored": measured})
        == ()
    )


def test_a_package_excuse_must_record_the_reach_it_was_measured_against():
    """The record is the narrowing, so a declaration cannot be added without one."""
    residual = surface.SpawnCoverage(surface.COVERAGE_RESIDUAL, reason="vendored multiprocessing")
    with pytest.raises(ValueError, match="measured against"):
        reach.PackageReacher(residual, primitives=(), files=1)
    with pytest.raises(ValueError, match="measured against"):
        reach.PackageReacher(residual, primitives=("_posixsubprocess.fork_exec",), files=0)


def test_the_installed_scan_reads_a_call_that_goes_around_os_and_ignores_one_that_does_not(
    tmp_path,
):
    root = _stdlib(
        tmp_path,
        {
            "around.py": "import posix\n\ndef go():\n    posix.fork()\n",
            "through.py": "import subprocess\n\ndef go():\n    subprocess.Popen(['true'])\n",
        },
    )
    assert _paths(reach.installed_spawn_reaches(root)) == {"around.py": ("posix.fork",)}


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
