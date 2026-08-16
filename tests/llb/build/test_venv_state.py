"""`make venv` must not call a REPLACE a reuse (llb.build.venv_state).

`uv sync --inexact` leaves packages the lock does not name alone -- until the system interpreter
the venv points at is patched, at which point uv finds the recorded `pyvenv.cfg` version behind the
real one and replaces the whole environment. The target printed `reusing .venv` either way, and the
replacement discards the hardware-matched vLLM/torch stack. These cover the decision over written
`pyvenv.cfg` fixtures, the pricing of a rebuild, the refusal and its remedies, and the two things
the make target's shell reads back: the action and the forced vLLM reinstall.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llb.build import lock_guard, lock_reader, venv_plan, venv_state
from llb.core import env
from llb.core.paths import PROJECT_ROOT
from tests.llb.build._venv_fixtures import (
    OTHER_MINOR,
    PATCHED_AWAY,
    RUNNING_VERSION,
    write_project,
    write_venv,
)

SETUP_VENV_SH = PROJECT_ROOT / "scripts" / "setup_venv.sh"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return write_project(tmp_path)


def test_reuse_when_the_recorded_version_still_matches_the_interpreter(tmp_path):
    action, reason, restampable = venv_plan.decide(write_venv(tmp_path / ".venv"))

    assert action == venv_plan.REUSE
    assert RUNNING_VERSION in reason and restampable is False


def test_rebuild_when_the_system_python_was_patched_underneath(tmp_path):
    """The whole finding: same interpreter path, moved version -- uv replaces rather than updates."""
    venv_dir = write_venv(tmp_path / ".venv", version_info=PATCHED_AWAY)

    action, reason, restampable = venv_plan.decide(venv_dir)

    assert action == venv_plan.REBUILD
    assert PATCHED_AWAY in reason and RUNNING_VERSION in reason
    assert restampable is True


def test_a_minor_move_rebuilds_with_no_restamp_offered(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv", version_info=OTHER_MINOR)

    action, _, restampable = venv_plan.decide(venv_dir)

    assert action == venv_plan.REBUILD and restampable is False


def test_create_when_there_is_no_venv_yet(tmp_path):
    assert venv_plan.decide(tmp_path / ".venv")[0] == venv_plan.CREATE


def test_reuse_when_pyvenv_cfg_carries_no_comparable_version(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv")
    (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    action, reason, _ = venv_plan.decide(venv_dir)

    assert action == venv_plan.REUSE and "no readable version" in reason


def test_rebuild_when_the_interpreter_the_venv_points_at_is_gone(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv", home=str(tmp_path / "removed-python"))

    action, reason, _ = venv_plan.decide(venv_dir)

    assert action == venv_plan.REBUILD and "gone" in reason


def test_venv_versions_reads_the_dist_info_names_not_the_interpreter(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv", packages={"flashinfer-python": "0.6.12"})

    # The wheel escapes the hyphen to an underscore on disk; the reader canonicalizes it back.
    assert lock_reader.venv_versions(venv_dir) == {"flashinfer-python": "0.6.12"}


def test_unreproduced_separates_a_dropped_package_from_a_downgraded_one():
    losses = venv_plan.unreproduced(
        {"vllm": "0.24.0", "torch": "2.11.0", "numpy": "2.4.6"},
        {"torch": {"2.12.1"}, "numpy": {"2.4.6"}},
    )

    # numpy is reproduced exactly, so it is not a loss; the other two are, for different reasons.
    assert venv_state.describe(losses[0]) == "torch 2.11.0 -> 2.12.1 from the lock"
    assert venv_state.describe(losses[1]) == "vllm 0.24.0 (not in the lock)"


def test_plan_prices_a_rebuild_in_the_packages_the_sync_will_not_restore(project):
    venv_dir = write_venv(
        project / ".venv",
        version_info=PATCHED_AWAY,
        packages={"vllm": "0.24.0", "torch": "2.11.0", "numpy": "2.4.6"},
    )

    plan = venv_plan.plan_venv(venv_dir, root=project)

    assert plan.action == venv_plan.REBUILD
    assert {loss.name for loss in plan.hardware_matched} == {"vllm", "torch"}
    assert plan.force_vllm is True


def test_a_matching_venv_whose_stack_the_lock_does_not_touch_is_priced_at_nothing(project):
    venv_dir = write_venv(project / ".venv", packages={"vllm": "0.24.0", "numpy": "2.4.6"})

    plan = venv_plan.plan_venv(venv_dir, root=project)

    # vLLM is not in the lock and numpy matches it, so this sync moves nothing under the stack.
    assert plan.action == venv_plan.REUSE
    assert plan.losses == () and plan.force_vllm is False


def test_a_reuse_still_forces_the_reinstall_when_the_sync_re_pins_torch(project):
    """Measured on the CUDA host: no rebuild, and `uv sync` still moved torch 2.13.0 -> 2.12.1.

    `--inexact` only promises not to REMOVE what the lock does not name. torch is inside the
    resolution (sentence-transformers pulls it in), so a plain reuse installs the LOCK's torch over
    the one vLLM pinned -- and with `VENV_INSTALL_VLLM=0` nothing puts it back.
    """
    venv_dir = write_venv(project / ".venv", packages={"vllm": "0.24.0", "torch": "2.13.0"})

    plan = venv_plan.plan_venv(venv_dir, root=project)

    assert plan.action == venv_plan.REUSE
    assert [loss.name for loss in plan.repinned] == ["torch"]
    assert plan.force_vllm is True


def test_a_reuse_forces_nothing_when_vllm_is_not_installed(project):
    """A CPU-only venv has no stack to protect, so the lock's torch is simply the right torch."""
    venv_dir = write_venv(project / ".venv", packages={"torch": "2.13.0"})

    plan = venv_plan.plan_venv(venv_dir, root=project)

    assert plan.repinned and plan.force_vllm is False


def test_the_reuse_message_names_the_re_pin_that_forces_the_reinstall(project):
    venv_dir = write_venv(project / ".venv", packages={"vllm": "0.24.0", "torch": "2.13.0"})

    lines = venv_state.report_lines(
        venv_plan.plan_venv(venv_dir, root=project), venv_dir, lock_guard.GUARD_REFUSE
    )

    assert lines[0].startswith("reusing")
    assert "re-pins torch 2.13.0 -> 2.12.1 from the lock" in lines[1]
    assert "vLLM reinstall is forced" in lines[1]


def test_recreate_is_a_rebuild_that_still_forces_the_vllm_reinstall(project):
    venv_dir = write_venv(project / ".venv", packages={"vllm": "0.24.0"})

    plan = venv_plan.plan_venv(venv_dir, root=project, requested=True)

    # The version matches, so only the explicit request makes this a rebuild -- and a rebuild still
    # discards vLLM, so skipping the reinstall would leave the lock's torch with nothing to match.
    assert plan.action == venv_plan.REBUILD and plan.force_vllm is True


def test_the_refusal_fires_only_on_a_rebuild_nobody_asked_for(project):
    venv_dir = write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"vllm": "0.24.0"})
    stale = venv_plan.plan_venv(venv_dir, root=project)

    assert venv_plan.refuses(stale, lock_guard.GUARD_REFUSE) is True
    # `report` and an explicit RECREATE_VENV are the two ways to say "yes, replace it".
    assert venv_plan.refuses(stale, lock_guard.GUARD_REPORT) is False
    requested = venv_plan.plan_venv(venv_dir, root=project, requested=True)
    assert venv_plan.refuses(requested, lock_guard.GUARD_REFUSE) is False


def test_a_rebuild_without_a_hardware_matched_stack_is_not_refused(project):
    """A CPU-only venv rebuilds from the lock alone, so there is nothing to warn anyone about."""
    venv_dir = write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"numpy": "2.4.6"})

    plan = venv_plan.plan_venv(venv_dir, root=project)

    assert plan.action == venv_plan.REBUILD
    assert venv_plan.refuses(plan, lock_guard.GUARD_REFUSE) is False


def test_report_lines_say_which_of_the_two_is_about_to_happen(project):
    venv_dir = write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"vllm": "0.24.0"})

    rebuild = venv_state.report_lines(
        venv_plan.plan_venv(venv_dir, root=project), venv_dir, lock_guard.GUARD_REFUSE
    )
    reuse = venv_state.report_lines(
        venv_plan.plan_venv(write_venv(project / ".other")), venv_dir, lock_guard.GUARD_REFUSE
    )

    assert rebuild[0].startswith("REBUILDING") and "vllm 0.24.0 (not in the lock)" in rebuild[1]
    assert "REFUSING" in rebuild[-3] and "venv-restamp" in rebuild[-2]
    assert "RECREATE_VENV=1" in rebuild[-1]
    assert reuse[0].startswith("reusing") and len(reuse) == 1


def test_a_minor_move_refusal_offers_only_the_rebuild(project):
    """Restamping across minor versions would be a lie, so it is not offered as a way out."""
    venv_dir = write_venv(project / ".venv", version_info=OTHER_MINOR, packages={"vllm": "0.24.0"})

    lines = venv_state.report_lines(
        venv_plan.plan_venv(venv_dir, root=project), venv_dir, lock_guard.GUARD_REFUSE
    )

    assert "REFUSING" in lines[-2] and "venv-restamp" not in " ".join(lines)


def test_a_lock_match_is_not_enough_when_only_an_unsynced_extra_declares_it(project):
    """Measured on the CUDA host: `bitsandbytes` matched the lock exactly and still vanished.

    `uv sync` installs the extras it was ASKED for, and `make venv`'s default set has no
    `finetune`, so the package the lock carries is simply not part of that resolution.
    """
    venv_dir = write_venv(
        project / ".venv",
        version_info=PATCHED_AWAY,
        packages={"bitsandbytes": "0.49.2", "numpy": "2.4.6"},
    )

    plan = venv_plan.plan_venv(venv_dir, root=project, synced_extras={"dev"})

    # numpy is a base dependency, so every sync restores it; bitsandbytes depends on the extras.
    assert [(loss.name, loss.owners) for loss in plan.at_risk] == [("bitsandbytes", ("finetune",))]
    assert plan.losses == ()


def test_syncing_the_declaring_extra_puts_the_package_out_of_danger(project):
    venv_dir = write_venv(
        project / ".venv", version_info=PATCHED_AWAY, packages={"bitsandbytes": "0.49.2"}
    )

    plan = venv_plan.plan_venv(venv_dir, root=project, synced_extras={"dev", "finetune"})

    assert plan.at_risk == ()


def test_the_at_risk_line_names_the_owning_extra_and_the_install_that_restores_it(project):
    venv_dir = write_venv(
        project / ".venv",
        version_info=PATCHED_AWAY,
        packages={"vllm": "0.24.0", "bitsandbytes": "0.49.2"},
    )

    lines = venv_state.report_lines(
        venv_plan.plan_venv(venv_dir, root=project, synced_extras={"dev"}),
        venv_dir,
        lock_guard.GUARD_REPORT,
    )

    assert "also at risk: bitsandbytes 0.49.2 (finetune)" in lines[2]
    assert "make install-extras EXTRAS=finetune" in lines[2]


def test_loss_summary_counts_what_it_does_not_list(project):
    venv_dir = write_venv(
        project / ".venv",
        version_info=PATCHED_AWAY,
        packages={"vllm": "0.24.0", **{f"nvidia-cuda-{index}": "13.0" for index in range(9)}},
    )

    summary = venv_state.loss_summary(venv_plan.plan_venv(venv_dir, root=project))

    assert summary.startswith("vllm 0.24.0 (not in the lock)") and summary.endswith("(+9 more)")


def test_main_prints_the_plan_for_the_shell_to_read(project, capsys, monkeypatch):
    monkeypatch.delenv(env.VENV_STALE_GUARD, raising=False)
    write_venv(project / ".venv", packages={"vllm": "0.24.0"})

    assert venv_state.main(["--venv", str(project / ".venv"), "--root", str(project)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"{venv_state.ACTION_KEY}=reuse",
        f"{venv_state.FORCE_VLLM_KEY}=0",
    ]


def test_main_refuses_with_its_own_exit_code_and_prints_no_plan(project, capsys, monkeypatch):
    monkeypatch.delenv(env.VENV_STALE_GUARD, raising=False)
    write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"vllm": "0.24.0"})

    status = venv_state.main(["--venv", str(project / ".venv"), "--root", str(project)])

    # The shell must not eval a plan for a run that was refused, so stdout stays empty.
    assert status == venv_state.REFUSED_EXIT and capsys.readouterr().out == ""


def test_main_restamps_the_venv_the_make_target_points_at(project, monkeypatch):
    monkeypatch.delenv(env.VENV_STALE_GUARD, raising=False)
    venv_dir = write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"vllm": "0.24.0"})

    assert venv_state.main(["--venv", str(venv_dir), "--restamp"]) == 0

    # And the same venv now reads as a reuse, which is the whole point of the remedy.
    assert venv_plan.plan_venv(venv_dir, root=project).action == venv_plan.REUSE


def test_guard_off_checks_nothing_and_says_so(project, capsys, monkeypatch):
    monkeypatch.setenv(env.VENV_STALE_GUARD, lock_guard.GUARD_OFF)
    write_venv(project / ".venv", version_info=PATCHED_AWAY, packages={"vllm": "0.24.0"})

    assert venv_state.main(["--venv", str(project / ".venv"), "--root", str(project)]) == 0

    assert f"{venv_state.ACTION_KEY}={venv_plan.UNCHECKED}" in capsys.readouterr().out


def _fake_host(
    tmp_path: Path, *, version_info: str, packages: dict[str, str] | None = None
) -> tuple[Path, Path]:
    """A project root the make target's script can run against: fake uv, fake vLLM installer."""
    write_project(tmp_path)
    write_venv(
        tmp_path / ".venv",
        version_info=version_info,
        packages=packages if packages is not None else {"vllm": "0.24.0", "torch": "2.11.0"},
    )
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    python = venv_bin / "python"
    python.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8"
    )
    python.chmod(0o755)

    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    build_vllm = scripts / "build_vllm.sh"
    build_vllm.write_text('#!/usr/bin/env bash\nprintf "build_vllm\\n" >> "$UV_LOG"\n', "utf-8")
    build_vllm.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$UV_LOG"\n'
        'if [ "$1 $2" = "cache dir" ]; then echo /shared/uv-cache; fi\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    return bin_dir, tmp_path / "uv.log"


def _run_setup_venv(root: Path, bin_dir: Path, log: Path, **overrides: str):
    return subprocess.run(
        ["bash", str(SETUP_VENV_SH)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PROJECT_ROOT": str(root),
            "VENV": str(root / ".venv"),
            "UV_LOG": str(log),
            "UV_SYNC_ARGS": "--inexact --extra dev",
            "VENV_INSTALL_VLLM": "0",
            **overrides,
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_venv_refuses_a_stale_rebuild_before_uv_runs(tmp_path):
    """The acceptance shape: the venv is still whole when the operator reads the refusal."""
    bin_dir, log = _fake_host(tmp_path, version_info=PATCHED_AWAY)

    result = _run_setup_venv(tmp_path, bin_dir, log)

    assert result.returncode == 1, result.stderr
    assert "REFUSING" in result.stderr and "vllm 0.24.0" in result.stderr
    assert "sync" not in log.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_venv_reuses_a_matching_venv_and_honors_vllm_skip(tmp_path):
    """Nothing the lock carries is installed at another version, so `=0` is simply obeyed."""
    bin_dir, log = _fake_host(
        tmp_path, version_info=RUNNING_VERSION, packages={"vllm": "0.24.0", "numpy": "2.4.6"}
    )

    result = _run_setup_venv(tmp_path, bin_dir, log)

    assert result.returncode == 0, result.stderr
    assert "reusing" in result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "sync --inexact --extra dev" in calls and "build_vllm" not in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_venv_repairs_a_reuse_that_re_pinned_the_stack(tmp_path):
    """The case the host actually hit: no rebuild, torch moved anyway, `=0` overridden."""
    bin_dir, log = _fake_host(
        tmp_path, version_info=RUNNING_VERSION, packages={"vllm": "0.24.0", "torch": "2.13.0"}
    )

    result = _run_setup_venv(tmp_path, bin_dir, log)

    assert result.returncode == 0, result.stderr
    assert "re-pins torch 2.13.0 -> 2.12.1 from the lock" in result.stderr
    assert "build_vllm" in log.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_venv_forces_the_vllm_reinstall_after_an_accepted_rebuild(tmp_path):
    """`VENV_INSTALL_VLLM=0` is honored on a reuse and overridden on a rebuild that dropped vLLM.

    Skipping it here is exactly the silent breakage: the sync leaves the lock's torch installed
    with no vLLM to match it, and the operator was told the venv was reused.
    """
    bin_dir, log = _fake_host(tmp_path, version_info=PATCHED_AWAY)

    result = _run_setup_venv(tmp_path, bin_dir, log, RECREATE_VENV="1")

    assert result.returncode == 0, result.stderr
    assert "REBUILDING" in result.stderr
    calls = log.read_text(encoding="utf-8")
    assert "sync --inexact --extra dev" in calls and "build_vllm" in calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_venv_rejects_an_unusable_vllm_mode_before_the_sync(tmp_path):
    bin_dir, log = _fake_host(tmp_path, version_info=RUNNING_VERSION)

    result = _run_setup_venv(tmp_path, bin_dir, log, VENV_INSTALL_VLLM="maybe")

    assert result.returncode == 2 and "VENV_INSTALL_VLLM must be" in result.stderr
    # Rejected while the venv is still whole: the mode is checked before anything is synced.
    assert "sync" not in log.read_text(encoding="utf-8")
