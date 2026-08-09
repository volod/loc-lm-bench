"""The DEPENDENCIES an unmarked test drives, read over the whole import path.

One pass over this venv's site-packages asserts that no dependency goes below the seams outside a
declared package -- nor from more files, or through more primitives, than the package's excuse was
measured against. The rest drives the same scan over fabricated trees and fabricated ARCHIVES,
because neither a package that reaches past every patchable name nor a venv that ships zipped can be
produced on demand here.

The site-packages cases are `slow` and the fabricated ones are not, decided on the measured cost:
site-packages is whatever is installed -- 40119 files and 556 MB on this host, 2.2s warm and
disk-bound cold -- and it changes only when the lock file does, which is a `make test` moment rather
than a `make ci` one.
"""

import sys
import zipfile
from collections.abc import Mapping
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source
from pathlib import Path

import pytest

from llb.quality import gpu_guard_spawn_reach as reach
from llb.quality import gpu_guard_spawn_reach_archive as archive
from llb.quality import gpu_guard_spawn_reach_audit as reach_audit
from llb.quality import gpu_guard_spawn_reach_installed as installed
from llb.quality import gpu_guard_spawn_reach_installed_archive as installed_archive
from llb.quality import gpu_guard_spawn_reach_installed_audit as installed_audit
from llb.quality import gpu_guard_spawn_reach_installed_coverage as coverage
from llb.quality import gpu_guard_spawn_surface as surface
from llb.quality import gpu_guard_spawn_surface_audit as audit

# A file that reaches past every patchable name -- the one thing the installed scan looks for.
BELOW = "import _posixsubprocess\n\ndef go():\n    _posixsubprocess.fork_exec()\n"
# What an extension module is named here, taken from the interpreter rather than spelled out, so a
# fabricated pure-extension dependency is one on whatever platform runs the suite.
EXTENSION = EXTENSION_SUFFIXES[0]


def _tree(tmp_path: Path, modules: Mapping[str, str]) -> Path:
    """A fabricated site-packages tree: relative path -> source."""
    root = tmp_path / "site-packages"
    for relative, source in modules.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _egg(path: Path, entries: Mapping[str, str]) -> Path:
    """A fabricated zip-shipped distribution: archive entry path -> contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as opened:
        for name, contents in entries.items():
            opened.writestr(name, contents)
    return path


def _paths(scan: reach.SpawnScan) -> dict[str, tuple[str, ...]]:
    return {module.path: module.primitives for module in scan.reaches}


@pytest.fixture(scope="module")
def installed_scan() -> reach.SpawnScan:
    """The same over this venv's site-packages, narrowed to reaches below the seams (2.2s)."""
    return installed.installed_spawn_reaches()


@pytest.mark.slow
def test_this_venvs_packages_go_below_the_seams_only_where_a_package_declares_it(installed_scan):
    """The dependencies an unmarked test actually drives, held to the same question."""
    findings = installed_audit.audit_installed_reach(installed_scan)
    assert findings == (), audit.surface_message(findings)


@pytest.mark.slow
def test_the_packages_that_go_below_the_seams_are_a_subset_of_the_declared_ones(installed_scan):
    """A subset, not equality: a host that never installed `joblib` is not a finding, and a THIRD
    package arriving is what the audit above refuses."""
    reaching = {module.path.split("/")[0] for module in installed_scan.reaches}
    assert reaching <= set(installed.DECLARED_PACKAGE_REACHERS)


@pytest.mark.slow
def test_the_declared_packages_still_reach_only_what_their_excuses_were_measured_on(installed_scan):
    """A package excuse survives a release bump, which is also how a widened vendored backend would
    arrive unread; the measurement is what makes the widening say so."""
    findings = installed_audit.outgrown_reachers(installed_scan)
    assert findings == (), audit.surface_message(findings)


@pytest.mark.slow
def test_the_installed_scan_accounts_for_every_published_name_it_read_no_source_for(installed_scan):
    """The below-the-seams verdict is about this venv, so it has to say which venv it was read from:
    every top-level name the installed distributions publish is either read or excused by a
    construction that has no source. The file count this replaces said how much was read and not
    what was missed, which is the same middle the stdlib half outgrew."""
    measured = coverage.installed_read_coverage(installed_scan)
    message = coverage.installed_read_coverage_message(measured)
    assert installed_audit.audit_installed_read_coverage(installed_scan, measured) == (), message
    # A partition of the published list: no name falls between two buckets or into both.
    published = coverage.importable_top_level_names()
    assert len(measured.read) + len(measured.unread) == len(published), message
    assert installed_scan.files_read > 100, message
    assert {"joblib", "numpy", "pytest", "torch"} <= set(measured.read), message
    # And one level down, where the published list stops: no dependency here ships a package whose
    # `__init__.py` is present and whose submodules are not.
    assert measured.compiled_only_submodules == (), message


def test_the_installed_alphabet_is_only_what_goes_below_the_seams():
    """A dependency calling `subprocess.Popen` says nothing the declaration does not already say,
    and looking for it means parsing the whole tree."""
    alphabet = installed.below_the_seams()
    assert set(alphabet) == {"posix", "_posixsubprocess", "_winapi"}
    assert "fork" in alphabet["posix"]


def _excused(files: int, *primitives: str) -> installed.PackageReacher:
    """A package excuse measured against `files` files reaching `primitives`."""
    return installed.PackageReacher(
        surface.SpawnCoverage(
            surface.COVERAGE_RESIDUAL, reason="a private copy of the multiprocessing residual"
        ),
        primitives=primitives or ("_posixsubprocess.fork_exec",),
        files=files,
    )


def test_a_package_is_excused_as_a_whole_because_a_release_moves_its_files(tmp_path):
    """`joblib` vendors `loky` at a path its next release may rename; the decision is the package."""
    root = _tree(
        tmp_path, {"declared/backend/fork_exec.py": BELOW, "undeclared/backend/start.py": BELOW}
    )
    findings = installed_audit.audit_installed_reach(
        installed.installed_spawn_reaches(root), reachers={"declared": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("undeclared/backend/start.py", reach_audit.PROBLEM_UNCOVERED_REACH)
    ]


def test_a_declared_package_that_starts_children_from_a_new_file_is_reported(tmp_path):
    """The failure package granularity invites: a line written about one backend excusing the one
    the next release adds. The excuse still covers it -- the growth is what arrives to be re-read."""
    root = _tree(
        tmp_path, {"vendored/backend/fork_exec.py": BELOW, "vendored/backend/added.py": BELOW}
    )
    findings = installed_audit.audit_installed_reach(
        installed.installed_spawn_reaches(root), reachers={"vendored": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("vendored", installed_audit.PROBLEM_OUTGROWN_REACH)
    ]
    assert "vendored/backend/added.py" in findings[0].detail


def test_a_declared_package_that_reaches_a_primitive_its_excuse_never_saw_is_reported(tmp_path):
    """A second WAY down, from the same file count -- a POSIX-only excuse grown a Windows half."""
    root = _tree(
        tmp_path,
        {"vendored/win.py": "import _winapi\n\ndef go():\n    _winapi.CreateProcess()\n"},
    )
    findings = installed_audit.outgrown_reachers(
        installed.installed_spawn_reaches(root), {"vendored": _excused(1)}
    )
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("vendored", installed_audit.PROBLEM_OUTGROWN_REACH)
    ]
    assert "_winapi.CreateProcess" in findings[0].detail


def test_a_declared_package_that_reaches_the_same_way_from_fewer_files_is_not_growth(tmp_path):
    """Shrinking is not a decision to revisit: a dropped backend, or a host that vendors less."""
    root = _tree(
        tmp_path,
        {"vendored/util.py": BELOW},
    )
    measured = _excused(3, "_posixsubprocess.fork_exec", "_winapi.CreateProcess")
    assert (
        installed_audit.outgrown_reachers(
            installed.installed_spawn_reaches(root), {"vendored": measured}
        )
        == ()
    )


def test_a_package_excuse_must_record_the_reach_it_was_measured_against():
    """The record is the narrowing, so a declaration cannot be added without one."""
    residual = surface.SpawnCoverage(surface.COVERAGE_RESIDUAL, reason="vendored multiprocessing")
    with pytest.raises(ValueError, match="measured against"):
        installed.PackageReacher(residual, primitives=(), files=1)
    with pytest.raises(ValueError, match="measured against"):
        installed.PackageReacher(residual, primitives=("_posixsubprocess.fork_exec",), files=0)


def test_the_installed_scan_reads_a_call_that_goes_around_os_and_ignores_one_that_does_not(
    tmp_path,
):
    root = _tree(
        tmp_path,
        {
            "around.py": "import posix\n\ndef go():\n    posix.fork()\n",
            "through.py": "import subprocess\n\ndef go():\n    subprocess.Popen(['true'])\n",
        },
    )
    assert _paths(installed.installed_spawn_reaches(root)) == {"around.py": ("posix.fork",)}


def test_a_zipped_dependency_is_importable_here_and_readable_out_of_its_archive(tmp_path):
    """The evidence the read-or-refuse decision rests on. The stdlib half refuses an archive because
    the layouts that ship zipped carry `.pyc` only; a zipped egg carries `.py`, so the source is
    right there, and this is the case stated rather than assumed: the fabricated distribution
    imports through `zipimport` and `zipfile` hands back the same bytes a parser would read."""
    egg = _egg(tmp_path / "zipped_dep-1.0-py3.13.egg", {"zipped_dep/__init__.py": "VALUE = 1\n"})
    sys.path.insert(0, str(egg))
    try:
        module = __import__("zipped_dep")
        assert (module.VALUE, type(module.__loader__).__name__) == (1, "zipimporter")
    finally:
        sys.path.remove(str(egg))
        sys.modules.pop("zipped_dep", None)
    reading = installed_archive.archive_reaches(egg, installed.below_the_seams(), ())
    assert (reading.read, reading.unread) == (("zipped_dep",), ())


def test_a_zipped_dependency_that_ships_source_is_parsed_rather_than_refused(tmp_path):
    """The half the directory scan never opened: a package whose source is inside the archive is
    read out of it, so the reach it carries is a finding on the same terms as one on disk."""
    egg = _egg(
        tmp_path / "vendored-1.0.egg",
        {"vendored/__init__.py": "", "vendored/backend/start.py": BELOW, "EGG-INFO/PKG-INFO": ""},
    )
    scan = installed.installed_spawn_reaches(_tree(tmp_path, {}), archives=[egg])
    assert (scan.files_read, scan.unread_archived) == (2, ())
    assert _paths(scan) == {"vendored/backend/start.py": ("_posixsubprocess.fork_exec",)}
    findings = installed_audit.audit_installed_reach(scan, reachers={})
    assert [(finding.problem, finding.name) for finding in findings] == [
        (reach_audit.PROBLEM_UNCOVERED_REACH, f"vendored/backend/start.py (in {egg})")
    ]


def test_a_zipped_dependency_rides_the_same_package_excuse_a_directory_one_does(tmp_path):
    """The excuse is keyed on the package, and an archive entry names the same package a directory
    does -- so shipping zipped neither loses the excuse nor earns a second finding."""
    egg = _egg(tmp_path / "vendored.zip", {"vendored/backend/start.py": BELOW})
    scan = installed.installed_spawn_reaches(_tree(tmp_path, {}), archives=[egg])
    assert (
        installed_audit.audit_installed_reach(scan, reachers={"vendored": _excused(1)}),
        installed_audit.outgrown_reachers(scan, {"vendored": _excused(1)}),
    ) == ((), ())


def test_an_archived_package_that_ships_no_source_is_refused_as_unread(tmp_path):
    """What is left after parsing: a `.pyc`-only entry is importable here and says nothing about
    whether it starts a child, which is the stdlib archive refusal one tree over."""
    egg = _egg(
        tmp_path / "stripped.egg",
        {"stripped/__init__.pyc": "", "stripped/util.pyc": "", "stripped/backend/start.pyc": ""},
    )
    scan = installed.installed_spawn_reaches(_tree(tmp_path, {}), archives=[egg])
    assert scan.unread_archived == ("stripped", "stripped.backend.start", "stripped.util")
    findings = installed_audit.unread_archived_packages(scan, reachers={})
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("stripped", reach_audit.PROBLEM_UNREAD_MODULE)
    ]
    assert "stripped.util" in findings[0].detail and str(egg) in findings[0].detail


def test_an_unread_archived_package_the_declaration_already_names_is_not_a_second_finding(tmp_path):
    """The declaration IS the decision that this package starts children and that it is accepted;
    refusing it as unmeasured would be the same finding twice."""
    egg = _egg(tmp_path / "vendored.egg", {"vendored/util.pyc": ""})
    scan = installed.installed_spawn_reaches(_tree(tmp_path, {}), archives=[egg])
    assert scan.unread_archived == ("vendored.util",)
    assert installed_audit.unread_archived_packages(scan, {"vendored": _excused(1)}) == ()


def test_a_module_the_directory_tree_or_another_archive_carries_as_source_is_not_unread(tmp_path):
    """Two ways a compiled entry is a copy of something that WAS read: the installed source tree
    beside the archive, and a second archive on the same path that ships the source."""
    stripped = _egg(tmp_path / "stripped.egg", {"ondisk/util.pyc": "", "elsewhere/deep/go.pyc": ""})
    sourced = _egg(tmp_path / "sourced.egg", {"elsewhere/deep/go.py": "import json\n"})
    root = _tree(tmp_path, {"ondisk/__init__.py": "", "ondisk/util.py": "import json\n"})
    scan = installed.installed_spawn_reaches(root, archives=[stripped, sourced])
    assert scan.unread_archived == ()
    assert installed_audit.audit_installed_reach(scan, reachers={}) == ()


def test_a_venv_that_ships_only_zipped_is_read_rather_than_refused_as_unscanned(tmp_path):
    """The degenerate end the file count exists for, now that the count adds up over the whole
    import path: an empty directory beside a source-carrying archive is a scan that read source."""
    egg = _egg(tmp_path / "vendored.egg", {"vendored/__init__.py": "import json\n"})
    scan = installed.installed_spawn_reaches(_tree(tmp_path, {}), archives=[egg])
    assert (scan.files_read, scan.archives, scan.modules_read) == (1, (str(egg),), ("vendored",))
    assert installed_audit.audit_installed_reach(scan, reachers={}) == ()


def test_an_archive_on_the_import_path_and_one_under_the_root_are_both_read(tmp_path):
    """The two places a zipped distribution sits: added to `sys.path` by a `.pth`, or sitting in
    site-packages before anything has added it. A directory entry and a file that is not a zip
    contribute nothing, which is what keeps an ordinary source install out of every count."""
    onpath = _egg(tmp_path / "elsewhere" / "onpath.egg", {"onpath/__init__.py": ""})
    root = _tree(tmp_path, {"plain/__init__.py": ""})
    under = _egg(root / "under.zip", {"under/__init__.py": ""})
    (root / "notazip.egg").write_text("this is not an archive\n")
    found = installed_archive.installed_archives(root, [str(onpath), str(root), "", "/nope.zip"])
    assert found == (onpath, under)


def test_the_stdlib_archive_is_left_to_the_check_that_speaks_about_the_stdlib(tmp_path):
    """`gpu_guard_spawn_reach_archive` accounts for a zipped stdlib; reporting it here as an unread
    dependency would be the same finding twice, from the scan that has no business saying it."""
    stdlib_zip = _egg(tmp_path / archive.ZIP_STDLIB_NAME, {"subprocess.pyc": ""})
    root = _tree(tmp_path, {"plain/__init__.py": ""})
    assert installed_archive.installed_archives(root, [str(stdlib_zip)]) == ()


def test_an_archive_that_ships_a_module_twice_is_read_from_its_source(tmp_path):
    """A zipped egg carrying `pkg/util.py` beside `pkg/util.pyc` is a module that CAN be read, and
    preferring the source to the name list is the whole point of reading into the archive."""
    egg = _egg(tmp_path / "both.egg", {"both/util.pyc": "", "both/util.py": BELOW})
    reading = installed_archive.archive_reaches(
        egg, installed.below_the_seams(), reach.module_triggers(installed.below_the_seams())
    )
    assert (reading.read, reading.unread) == (("both.util",), ())
    assert _paths(reach.SpawnScan("", 0, (), reading.reaches)) == {
        "both/util.py": ("_posixsubprocess.fork_exec",)
    }


def test_an_archive_that_cannot_be_read_contributes_nothing_rather_than_failing(tmp_path):
    """The same tolerance the directory scan gives an unreadable file: a truncated or absent archive
    is not evidence, and a scan that raises on one is a scan nobody can run."""
    broken = tmp_path / "broken.egg"
    broken.write_text("not a zip\n")
    scan = installed.installed_spawn_reaches(
        _tree(tmp_path, {"plain/__init__.py": ""}), archives=[broken, tmp_path / "missing.egg"]
    )
    assert (scan.archives, scan.unread_archived, scan.files_read) == ((), (), 1)


def test_a_stripped_dependency_is_refused_where_a_pure_extension_one_is_not(tmp_path):
    """The directory-tree twin of the `.pyc`-only egg, and the naive gate beside it. A dependency
    whose modules ship cached with no source is importable here and was parsed by nothing, which is
    the refusal; a dependency that ships shared objects (`nvidia-*` wheels ship little else) or a
    directory with no module in it at all has no source BY CONSTRUCTION, and refusing either would
    fail on any host with a CUDA wheel installed."""
    root = _tree(
        tmp_path,
        {
            "sourced/__init__.py": "import json\n",
            cache_from_source("stripped/__init__.py"): "",
            cache_from_source("stripped/util.py"): "",
            f"accel{EXTENSION}": "",
            f"cuda/lib/libkernels{EXTENSION}": "",
            "shipped/data/rows.json": "{}\n",
        },
    )
    scan = installed.installed_spawn_reaches(root)
    measured = coverage.installed_read_coverage(
        scan, names=["accel", "cuda", "shipped", "sourced", "stripped"]
    )
    assert (measured.read, measured.compiled_only) == (("sourced",), ("stripped",))
    assert (measured.extensions, measured.namespace) == (("accel", "cuda"), ("shipped",))
    findings = installed_audit.audit_installed_read_coverage(scan, measured, reachers={})
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("stripped", reach_audit.PROBLEM_UNREAD_MODULE)
    ]
    # One line per PACKAGE, as the archive refusal is: the stripped submodule joins the name's own
    # finding rather than earning a second one.
    assert "stripped.util" in findings[0].detail
    assert "stripped" in coverage.installed_read_coverage_message(measured)


def test_a_stripped_submodule_of_a_dependency_the_scan_read_joins_that_packages_line(tmp_path):
    """The level the published list stops at: `packages_distributions()` names `pkg` and never
    `pkg.util`, so a dependency that ships its `__init__.py` and strips the rest reads as read."""
    root = _tree(
        tmp_path, {"pkg/__init__.py": "import json\n", cache_from_source("pkg/util.py"): ""}
    )
    scan = installed.installed_spawn_reaches(root)
    measured = coverage.installed_read_coverage(scan, names=["pkg"])
    assert (measured.read, measured.compiled_only) == (("pkg",), ())
    assert measured.compiled_only_submodules == ("pkg.util",)
    findings = installed_audit.audit_installed_read_coverage(scan, measured, reachers={})
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("pkg", reach_audit.PROBLEM_UNREAD_MODULE)
    ]
    assert "pkg.util" in findings[0].detail


def test_a_stripped_package_the_declaration_already_names_is_not_a_second_finding(tmp_path):
    """The declaration IS the decision that this package starts children and that it is accepted,
    which is the rule the archive refusal applies to the same shape one layout over."""
    root = _tree(tmp_path, {cache_from_source("vendored/__init__.py"): ""})
    scan = installed.installed_spawn_reaches(root)
    measured = coverage.installed_read_coverage(scan, names=["vendored"])
    assert measured.compiled_only == ("vendored",)
    assert (
        installed_audit.audit_installed_read_coverage(scan, measured, {"vendored": _excused(1)})
        == ()
    )


def test_a_published_name_this_tree_does_not_carry_is_reported_not_refused(tmp_path):
    """The two ways a name goes unread without this tree stripping anything. An editable install or
    a second site directory provides it from elsewhere on the import path, and an archive provides
    it from a zip -- which `unread_archived_packages` already refuses at this same granularity, so
    refusing it here would be one finding wearing two names."""
    egg = _egg(tmp_path / "stripped.egg", {"zipped/util.pyc": ""})
    root = _tree(tmp_path, {"plain/__init__.py": "import json\n"})
    scan = installed.installed_spawn_reaches(root, archives=[egg])
    measured = coverage.installed_read_coverage(scan, names=["editable", "plain", "zipped"])
    assert (measured.read, measured.archived, measured.absent) == (
        ("plain",),
        ("zipped",),
        ("editable",),
    )
    assert measured.archives == (str(egg),)
    assert installed_audit.audit_installed_read_coverage(scan, measured, reachers={}) == ()
    assert [finding.name for finding in installed_audit.unread_archived_packages(scan, {})] == [
        "zipped"
    ]


def test_a_distribution_that_records_a_path_publishes_the_top_level_name_it_installs():
    """`packages_distributions()` reads each distribution's own record of what it installed, and a
    few write a path there -- `sentencepiece/__init__`, `nvidia/cusparselt` on this host. The scan
    reports top-level names, so the two are only comparable once the path is read as one."""
    assert coverage.importable_top_level_names(
        {
            "joblib": ["joblib"],
            "nvidia/cusparselt": ["nvidia-cusparselt"],
            "sentencepiece": ["sentencepiece"],
            "sentencepiece/__init__": ["sentencepiece"],
        }
    ) == ("joblib", "nvidia", "sentencepiece")


@pytest.mark.slow
def test_this_venvs_import_path_carries_no_unread_archive(installed_scan):
    """The counterpart of the stdlib archive assertion: this host installs its dependencies as
    directory trees, so the archive half finds nothing to account for."""
    assert (installed_scan.archives, installed_scan.unread_archived) == ((), ())
