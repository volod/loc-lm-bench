"""The stdlib's own child-starting call sites, read rather than assumed.

The name surface covers `os` and `subprocess`; everything else in the stdlib is covered only if the
helper it calls resolves a name in those two. This file is where that stops being a sentence: one
pass over the stdlib this interpreter ships asserts that every module which starts a child does it
through a DECLARED name, and that the only exceptions are the ones on the record.

The rest drives the scan over fabricated trees, because a stdlib that reaches past every patchable
name cannot be produced on demand -- and because the ways a source scan goes wrong (an aliased
import, a local function that shares a name with a spawn entry point, a file that will not parse)
are worth pinning where they can be written out in four lines.
"""

from collections.abc import Mapping
from pathlib import Path

import pytest

from llb.quality import gpu_guard_spawn_reach as reach
from llb.quality import gpu_guard_spawn_reach_audit as reach_audit
from llb.quality import gpu_guard_spawn_surface as surface
from llb.quality import gpu_guard_spawn_surface_audit as audit


@pytest.fixture(scope="module")
def stdlib_reaches() -> tuple[reach.ModuleReach, ...]:
    """The real scan, run once for the three cases that read it (0.9s on this host)."""
    return reach.stdlib_spawn_reaches()


def _stdlib(tmp_path: Path, modules: Mapping[str, str]) -> Path:
    """A fabricated stdlib tree: relative path -> source."""
    root = tmp_path / "stdlib"
    for relative, source in modules.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _paths(reaches: tuple[reach.ModuleReach, ...]) -> dict[str, tuple[str, ...]]:
    return {module.path: module.primitives for module in reaches}


def test_this_stdlib_starts_children_only_through_names_the_denial_declares(stdlib_reaches):
    """The claim the name surface rested on, now a result: no undeclared way in."""
    findings = reach_audit.audit_spawn_reach(stdlib_reaches)
    assert findings == (), audit.surface_message(findings)


def test_the_modules_that_reach_past_the_declared_surface_are_exactly_the_declared_ones(
    stdlib_reaches,
):
    """The evidence behind "two modules is the right enumerated surface"."""
    past = {
        module.path
        for module in stdlib_reaches
        if any(name not in surface.DECLARED_SPAWN_SURFACE for name in module.primitives)
    }
    assert past == set(reach.DECLARED_REACHERS)


def test_every_excused_module_is_one_the_scan_still_finds(stdlib_reaches):
    """An excuse that outlives what it excused is an excuse nobody re-reads."""
    assert reach_audit.absent_reachers(stdlib_reaches) == ()


def test_the_scan_reads_the_stdlib_that_actually_ships_here(stdlib_reaches):
    """A guard against the scan quietly reading nothing: these three are stable across versions."""
    found = _paths(stdlib_reaches)
    assert "os.forkpty" in found["pty.py"]
    assert "subprocess.Popen" in found["asyncio/unix_events.py"]
    assert "_posixsubprocess.fork_exec" in found["multiprocessing/util.py"]


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
    reaches = reach.stdlib_spawn_reaches(root)
    assert _paths(reaches) == {
        "plain.py": ("os.fork",),
        "aliased.py": ("os.forkpty",),
        "imported.py": ("subprocess.Popen",),
    }
    assert reach_audit.audit_spawn_reach(reaches) == ()


def test_a_local_name_that_only_looks_like_a_spawn_is_not_read_as_one(tmp_path):
    """The scan resolves through the module's own imports, so a same-named helper is not a hit."""
    root = _stdlib(tmp_path, {"own.py": "def fork():\n    return 0\n\ndef go():\n    fork()\n"})
    assert reach.stdlib_spawn_reaches(root) == ()


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


def test_a_tree_that_yields_nothing_reads_as_unscanned_rather_than_clean(tmp_path):
    """A scan that silently reads no source is the one way this check could pass for free."""
    findings = reach_audit.audit_spawn_reach(reach.stdlib_spawn_reaches(tmp_path))
    assert [(finding.name, finding.problem) for finding in findings] == [
        ("<stdlib>", reach_audit.PROBLEM_UNSCANNED)
    ]
