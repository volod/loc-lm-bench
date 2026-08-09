"""The spawn surface is re-checked against the running interpreter, not against the one it was
written on.

Two halves. The first is one assertion about THIS Python: every process-starting name it exposes is
declared, every declared delegation is one it still makes, and its default `multiprocessing` start
method is covered. That is the assertion a Python upgrade turns red -- 3.14 moves the default off
`fork`, and a later one could move `os.spawnv` into C.

The second half drives the audit against fabricated interpreters, because the interesting cases
cannot be produced by the host: a Python that grew a spawn function, one that rewrote a delegation,
one whose seam set and declarations drifted apart. The fabricated modules below are real functions
with real code objects, so the delegation check reads them exactly as it reads `os`.
"""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from llb.quality import gpu_guard_spawn_surface as surface
from llb.quality import gpu_guard_spawn_surface_audit as audit


# The fabricated interpreter. `execl` resolves `execv` at call time the way `os.execl` does, and
# `spawnl` is one hop further out, so a chain has something to walk.
def execv(*args: object) -> None:
    """Stands in for a patched entry point."""


def execl(*args: object) -> None:
    """Written on top of `execv`, which it resolves as a global when called."""
    execv(*args)


def spawnl(*args: object) -> None:
    """Two hops from the seam: it names `execl`, which names `execv`."""
    execl(*args)


def recursing(*args: object) -> None:
    """Names `execl` while standing in for it -- a declaration that points back at itself."""
    execl(*args)


class _NativeEntryPoint:
    """A C entry point: callable, with no code object to read a delegation out of."""

    def __call__(self, *args: object) -> None:
        """Never called -- the missing `__code__` is the whole point."""


_SEAM = surface.SpawnCoverage(surface.COVERAGE_SEAM)
_PATCHED = ("os.fork", "os.execv")
# The `fork` start method, covered the way the shipped declaration covers it.
_FORK_COVERED = {"fork": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.fork")}


def _fake_os(**entry_points: object) -> SimpleNamespace:
    return SimpleNamespace(**entry_points)


def _declared(
    extra: Mapping[str, surface.SpawnCoverage] | None = None,
) -> dict[str, surface.SpawnCoverage]:
    """The two patched names, declared -- plus whatever the case under test adds or overrides."""
    return {name: _SEAM for name in _PATCHED} | dict(extra or {})


def _surface(
    names: tuple[str, ...],
    seams: tuple[str, ...] = _PATCHED,
    *,
    start_methods: tuple[str, ...] = ("fork",),
    default: str = "fork",
    modules: object = None,
) -> surface.ObservedSurface:
    return surface.ObservedSurface(
        names=names,
        seams=seams,
        start_methods=start_methods,
        default_start_method=default,
        modules=modules
        if modules is not None
        else {"os": _fake_os(execv=execv, execl=execl, spawnl=spawnl)},
    )


def _reported(findings: tuple[audit.SurfaceFinding, ...]) -> list[tuple[str, str]]:
    return [(finding.name, finding.problem) for finding in findings]


def test_this_interpreters_spawn_surface_is_the_one_the_denial_declares():
    """The check the whole module exists for: a Python that moves either half arrives as this."""
    findings = audit.audit_spawn_surface()
    assert findings == (), audit.surface_message(findings)


def test_the_enumeration_is_a_rule_over_the_families_rather_than_a_list():
    names = set(surface.interpreter_spawn_names())
    assert {"os.execl", "os.execve", "os.spawnv", "os.fork", "os.popen", "os.system"} <= names
    assert {"subprocess.Popen", "subprocess.run", "subprocess.getoutput"} <= names
    assert "multiprocessing.util.spawnv_passfds" in names
    # Waiting on a child, killing one, or a constant is not a way to START one.
    assert not {"os.waitpid", "os.kill", "os.abort", "subprocess.PIPE"} & names


def test_an_exception_class_is_not_read_as_a_subprocess_entry_point():
    assert "subprocess.CalledProcessError" not in surface.interpreter_spawn_names()


def test_a_delegation_is_read_off_the_interpreter_rather_than_believed():
    assert surface.delegation_is_live("os.execl", "os.execv")
    assert surface.delegation_is_live("os.spawnv", "os.execv")
    assert surface.delegation_is_live("os.popen", "subprocess.Popen")
    assert surface.delegation_is_live("subprocess.check_call", "subprocess.call")


def test_a_delegation_the_implementation_does_not_make_reads_as_dead():
    """`os.system` reaches no exec name, and `os.fork` is C -- neither could carry the claim."""
    assert not surface.delegation_is_live("os.system", "os.execv")
    assert not surface.delegation_is_live("os.fork", "os.execv")


def test_a_name_the_interpreter_grows_arrives_as_an_undeclared_finding():
    """The upgrade case: a spawn function the seam set never heard of is not silently residual."""
    grown = _fake_os(execv=execv, execveat=_NativeEntryPoint())
    names = surface.interpreter_spawn_names(grown, SimpleNamespace(__all__=()), SimpleNamespace())
    findings = audit.audit_spawn_surface(_surface(names), _declared(), _FORK_COVERED)
    assert _reported(findings) == [("os.execveat", audit.PROBLEM_UNDECLARED)]


def test_a_declared_delegation_the_interpreter_stopped_making_is_refused():
    """A family rewritten in C keeps its name and loses its coverage; the claim has to notice."""
    native = {"os": _fake_os(execv=execv, execl=_NativeEntryPoint())}
    findings = audit.audit_spawn_surface(
        _surface(("os.execl", "os.execv"), modules=native),
        _declared(
            {"os.execl": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.execv")}
        ),
        _FORK_COVERED,
    )
    assert _reported(findings) == [("os.execl", audit.PROBLEM_UNPINNED)]


def test_a_delegation_chain_is_walked_to_the_seam_that_carries_it():
    findings = audit.audit_spawn_surface(
        _surface(("os.spawnl", "os.execl", "os.execv")),
        _declared(
            {
                "os.execl": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.execv"),
                "os.spawnl": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.execl"),
            }
        ),
        _FORK_COVERED,
    )
    assert findings == ()


def test_a_chain_that_ends_outside_the_seam_set_is_refused():
    """`os.execl` does reach `os.execv`; taking `os.execv` out of the seam set breaks the chain."""
    findings = audit.audit_spawn_surface(
        _surface(("os.execl", "os.execv"), seams=("os.fork",)),
        _declared(
            {
                "os.execv": surface.SpawnCoverage(
                    surface.COVERAGE_RESIDUAL, reason="not patched in this fabricated interpreter"
                ),
                "os.execl": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.execv"),
            }
        ),
        _FORK_COVERED,
    )
    assert _reported(findings) == [("os.execl", audit.PROBLEM_UNREACHED)]


def test_a_declaration_that_points_back_at_itself_reaches_no_seam():
    findings = audit.audit_spawn_surface(
        _surface(("os.execl",), modules={"os": _fake_os(execl=recursing)}),
        _declared(
            {"os.execl": surface.SpawnCoverage(surface.COVERAGE_THROUGH, through="os.execl")}
        ),
        _FORK_COVERED,
    )
    assert _reported(findings) == [("os.execl", audit.PROBLEM_UNREACHED)]


def test_a_name_declared_a_seam_that_the_denial_does_not_patch_is_refused():
    findings = audit.audit_spawn_surface(
        _surface(("os.execv", "os.execl")), _declared({"os.execl": _SEAM}), _FORK_COVERED
    )
    assert _reported(findings) == [("os.execl", audit.PROBLEM_SEAM_UNPATCHED)]


def test_a_seam_the_denial_patches_and_no_declaration_names_is_refused():
    """The other direction: a seam added to `spawn_seams` without a line in the declared surface."""
    findings = audit.audit_spawn_surface(
        _surface(("os.execv",), seams=(*_PATCHED, "os.forkpty")), _declared(), _FORK_COVERED
    )
    assert _reported(findings) == [("os.forkpty", audit.PROBLEM_SEAM_UNDECLARED)]


def test_a_start_method_the_interpreter_grows_is_refused():
    findings = audit.audit_spawn_surface(
        _surface(("os.execv",), start_methods=("fork", "vfork")), _declared(), _FORK_COVERED
    )
    assert _reported(findings) == [("multiprocessing(vfork)", audit.PROBLEM_UNDECLARED)]


def test_a_python_whose_default_start_method_is_a_declared_residual_is_refused():
    """The audit still refuses any platform-specific start method left declared residual."""
    residual_methods = {
        "forkserver": surface.SpawnCoverage(
            surface.COVERAGE_RESIDUAL, reason="not covered on this fabricated platform"
        ),
        "fork": _FORK_COVERED["fork"],
    }
    findings = audit.audit_spawn_surface(
        _surface(("os.execv",), start_methods=("forkserver", "fork"), default="forkserver"),
        _declared(),
        residual_methods,
    )
    assert _reported(findings) == [("multiprocessing(forkserver)", audit.PROBLEM_START_METHOD)]
    assert "keeps the device" in findings[0].detail


def test_a_covered_start_method_whose_seam_is_gone_is_refused():
    findings = audit.audit_spawn_surface(
        _surface(("os.execv",), seams=("os.execv",)), _declared(), _FORK_COVERED
    )
    assert _reported(findings) == [("multiprocessing(fork)", audit.PROBLEM_UNREACHED)]


def test_a_residual_declaration_has_to_say_why():
    """The vocabulary is only worth having if a residual carries its reason."""
    with pytest.raises(ValueError, match="say why"):
        surface.SpawnCoverage(surface.COVERAGE_RESIDUAL)
    with pytest.raises(ValueError, match="say why"):
        surface.SpawnCoverage(surface.COVERAGE_NOT_A_SPAWN)


def test_a_delegation_declaration_has_to_name_what_it_is_covered_through():
    with pytest.raises(ValueError, match="covered through"):
        surface.SpawnCoverage(surface.COVERAGE_THROUGH)


def test_the_default_start_method_is_read_from_a_child_without_fixing_the_parent_context():
    """`get_start_method()` RESOLVES the default, after which `set_start_method` raises -- a check
    must not be what pins the suite's start method."""

    def refuse(allow_none: bool = False) -> str | None:
        assert allow_none, "the audit must not resolve the default context"
        return None

    unresolved = SimpleNamespace(
        get_start_method=refuse, get_all_start_methods=lambda: ["fork", "spawn"]
    )
    assert surface.default_start_method(unresolved, lambda: "fork") == "fork"
    resolved = SimpleNamespace(
        get_start_method=lambda allow_none=False: "spawn", get_all_start_methods=lambda: ["fork"]
    )
    assert (
        surface.default_start_method(
            resolved, lambda: pytest.fail("an already-set context needs no child")
        )
        == "spawn"
    )


def test_a_child_default_that_disagrees_with_the_documented_ordering_is_refused():
    unresolved = SimpleNamespace(
        get_start_method=lambda allow_none=False: None,
        get_all_start_methods=lambda: ["fork", "spawn"],
    )
    with pytest.raises(RuntimeError, match="child interpreter reported 'spawn'.*'fork' first"):
        surface.default_start_method(unresolved, lambda: "spawn")


def test_the_child_default_is_resolved_only_once(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **options: object) -> SimpleNamespace:
        calls.append((command, options))
        return SimpleNamespace(stdout="fork\n")

    monkeypatch.setattr(surface.subprocess, "run", run)
    surface._child_default_start_method.cache_clear()
    try:
        assert surface._child_default_start_method("python-under-test") == "fork"
        assert surface._child_default_start_method("python-under-test") == "fork"
    finally:
        surface._child_default_start_method.cache_clear()

    assert calls == [
        (
            ["python-under-test", "-c", surface._DEFAULT_START_METHOD_SCRIPT],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_a_declaration_this_host_has_no_name_for_is_reported_and_not_refused():
    """`os.startfile` on POSIX, `os.fork` on Windows: a host difference, not a stale claim."""
    observed = _surface(_PATCHED)
    declared = _declared(
        {
            "os.startfile": surface.SpawnCoverage(
                surface.COVERAGE_RESIDUAL, reason="Windows only, and the shell mechanism is POSIX"
            )
        }
    )
    assert audit.absent_declarations(observed, declared) == ("os.startfile",)
    assert audit.audit_spawn_surface(observed, declared, _FORK_COVERED) == ()


def test_the_message_names_every_finding_and_the_way_out():
    message = audit.surface_message(
        [audit.SurfaceFinding("os.execveat", audit.PROBLEM_UNDECLARED, "no declaration covers it")]
    )
    assert "os.execveat" in message
    assert audit.PROBLEM_UNDECLARED in message
    assert "spawn_seams" in message
