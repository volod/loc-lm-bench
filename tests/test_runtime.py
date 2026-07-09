"""Shared CLI runtime: Ctrl-C + crash handling for every entrypoint."""

import click
import pytest

from llb.core.runtime import INTERRUPT_EXIT, run, run_typer


def test_run_returns_normal_exit_codes():
    assert run(lambda: 0) == 0
    assert run(lambda: None) == 0  # main() returning None -> 0
    assert run(lambda: 3) == 3


def test_run_handles_ctrl_c():
    def boom():
        raise KeyboardInterrupt

    assert run(boom) == INTERRUPT_EXIT


def test_run_logs_and_returns_1_on_crash():
    def boom():
        raise ValueError("nope")

    assert run(boom) == 1


def test_run_preserves_systemexit_code():
    def code_2():
        raise SystemExit(2)

    def code_none():
        raise SystemExit()

    assert run(code_2) == 2
    assert run(code_none) == 0


class _FakeApp:
    """Stands in for a Typer app: raises whatever it is told when invoked."""

    def __init__(self, exc=None):
        self.exc = exc

    def __call__(self, standalone_mode=True):
        if self.exc is not None:
            raise self.exc
        return None


class _ReturningApp:
    """A Typer app driven with `standalone_mode=False`, which RETURNS `typer.Exit(code)` as an int.

    This is click's real contract and the one `_FakeApp` cannot express: a command that does
    `raise typer.Exit(code=2)` surfaces here as a returned `2`, never as a raised exception.
    """

    def __init__(self, code):
        self.code = code

    def __call__(self, standalone_mode=True):
        assert standalone_mode is False, "run_typer must drive the app in non-standalone mode"
        return self.code


def test_run_typer_translates_abort_to_130():
    with pytest.raises(SystemExit) as ei:
        run_typer(_FakeApp(click.exceptions.Abort()))
    assert ei.value.code == INTERRUPT_EXIT


def test_run_typer_translates_keyboard_interrupt():
    with pytest.raises(SystemExit) as ei:
        run_typer(_FakeApp(KeyboardInterrupt()))
    assert ei.value.code == INTERRUPT_EXIT


def test_run_typer_preserves_typer_exit_code():
    with pytest.raises(SystemExit) as ei:
        run_typer(_FakeApp(click.exceptions.Exit(2)))
    assert ei.value.code == 2


def test_run_typer_exits_on_a_returned_failure_code():
    """`raise typer.Exit(code=2)` reaches run_typer as a RETURN value, not an exception."""
    with pytest.raises(SystemExit) as ei:
        run_typer(_ReturningApp(2))
    assert ei.value.code == 2


def test_run_typer_returned_zero_stays_success():
    """`--help` returns 0 through the same path and must not become a failure."""
    assert run_typer(_ReturningApp(0)) is None


def test_run_typer_normal_completion_does_not_exit():
    assert run_typer(_FakeApp()) is None
