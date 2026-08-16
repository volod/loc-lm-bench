"""Say which of the two `make venv` is doing -- reuse or rebuild -- and hand the plan to the shell.

`llb.build.venv_plan` decides; this module is what `scripts/setup_venv.sh` runs, so it owns the
account an operator reads and the two values the target's shell acts on. Three things follow from
one plan: the message names the action honestly (never a `reusing` that was a replace), a rebuild
nobody asked for is refused while the venv is still whole, and a rebuild that discarded vLLM forces
the reinstall afterwards -- `VENV_INSTALL_VLLM=0` included, because that flag was set for a venv
that no longer exists.

A refusal has to be actionable, so it names the cheapest way out first: within one `major.minor`
the venv is already RUNNING the patched interpreter, so `make venv-restamp`
(`llb.build.venv_interpreter`) records the new version and the hardware-matched stack stands.
"""

import argparse
import logging
from pathlib import Path

from llb.build import extras, lock_guard, venv_interpreter, venv_plan
from llb.build.venv_plan import Loss, Plan

_LOG = logging.getLogger(__name__)

# A replaced venv loses every CUDA wheel it holds; naming the hardware-matched ones and counting
# the rest keeps the message readable without under-stating the size of the reinstall.
LISTED_LOSSES = 6

ACTION_KEY = "LLB_VENV_ACTION"
FORCE_VLLM_KEY = "LLB_VENV_FORCE_VLLM"
REFUSED_EXIT = 3


def describe(loss: Loss) -> str:
    if not loss.locked:
        return f"{loss.name} {loss.installed} (not in the lock)"
    return f"{loss.name} {loss.installed} -> {'/'.join(loss.locked)} from the lock"


def loss_summary(plan: Plan) -> str:
    """The hardware-matched losses by name, plus a count of everything else that goes with them."""
    listed = list(plan.hardware_matched)[:LISTED_LOSSES]
    remaining = len(plan.losses) - len(listed)
    summary = ", ".join(describe(loss) for loss in listed) or "no installed package"
    return f"{summary} (+{remaining} more)" if remaining > 0 else summary


def at_risk_line(plan: Plan) -> str:
    """The second bucket: on the lock, but only an unsynced extra declares it -- plus the remedy."""
    listed = plan.at_risk[:LISTED_LOSSES]
    names = ", ".join(f"{loss.name} {loss.installed} ({','.join(loss.owners)})" for loss in listed)
    remaining = len(plan.at_risk) - len(listed)
    tail = f" (+{remaining} more)" if remaining > 0 else ""
    needed = ",".join(sorted({group for loss in plan.at_risk for group in loss.owners}))
    return (
        f"also at risk: {names}{tail} -- on the lock, but only an extra this sync does not "
        f"install declares them; put them back with `make install-extras EXTRAS={needed}`"
    )


def refusal_lines(plan: Plan) -> list[str]:
    """The refusal and the two-or-three ways forward, cheapest first."""
    lines = ["REFUSING: that stack is hardware-matched and this run did not ask to replace it."]
    if plan.restampable:
        # A patch-level move kept the ABI, so the rebuild buys nothing the restamp does not.
        lines.append(
            "the venv already RUNS the patched interpreter, so `make venv-restamp` records the "
            "new version and keeps the stack -- prefer that."
        )
    variable = venv_plan.VENV_GUARD.variable
    lines.append(
        "accept the rebuild with `make venv RECREATE_VENV=1`, or set "
        f"{variable}=report to sync anyway ({variable}=off skips this check entirely)."
    )
    return lines


def report_lines(plan: Plan, venv_dir: Path, mode: str) -> list[str]:
    """The operator-facing account: which of the two is happening, and what it costs."""
    if plan.action == venv_plan.CREATE:
        return [f"creating {venv_dir} -- {plan.reason}"]
    if plan.action == venv_plan.UNCHECKED:
        return [f"not checking {venv_dir} -- {plan.reason}; uv decides reuse or replace"]
    if plan.action == venv_plan.REUSE:
        lines = [f"reusing {venv_dir} -- {plan.reason} (RECREATE_VENV=1 to rebuild)"]
        if plan.force_vllm:
            # `--inexact` keeps what the lock does not name; it does not keep a package the lock
            # DOES name at a version vLLM chose, so a plain reuse still moves the stack.
            moved = ", ".join(describe(loss) for loss in plan.repinned[:LISTED_LOSSES])
            lines.append(f"the sync re-pins {moved}, so the vLLM reinstall is forced afterwards")
        return lines
    lines = [
        f"REBUILDING {venv_dir} -- {plan.reason}",
        "uv replaces an environment whose recorded version moved, so the sync discards: "
        f"{loss_summary(plan)}",
    ]
    if plan.at_risk:
        lines.append(at_risk_line(plan))
    if plan.force_vllm:
        lines.append("the vLLM reinstall is forced afterwards, VENV_INSTALL_VLLM notwithstanding")
    if venv_plan.refuses(plan, mode):
        lines.extend(refusal_lines(plan))
    return lines


def report(plan: Plan, venv_dir: Path, mode: str) -> None:
    level = logging.ERROR if venv_plan.refuses(plan, mode) else logging.INFO
    for line in report_lines(plan, venv_dir, mode):
        _LOG.log(level, "[venv] %s", line)


def shell_assignments(plan: Plan) -> list[str]:
    """The plan as `KEY=value` lines for the make target's shell to eval."""
    return [f"{ACTION_KEY}={plan.action}", f"{FORCE_VLLM_KEY}={int(plan.force_vllm)}"]


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venv", required=True, help="path to the project virtual environment")
    parser.add_argument(
        "--root", default=str(venv_plan.PROJECT_ROOT), help="project root holding uv.lock"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="the run explicitly asked for a rebuild"
    )
    parser.add_argument(
        "--extras",
        default="",
        help="comma-separated extras this sync installs (decides what a rebuild restores)",
    )
    parser.add_argument(
        "--restamp",
        action="store_true",
        help="record the interpreter's current version in pyvenv.cfg (patch moves only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    venv_dir = Path(arguments.venv)
    if arguments.restamp:
        return venv_interpreter.restamp(venv_dir)
    mode = lock_guard.guard_mode(venv_plan.VENV_GUARD)
    if mode == lock_guard.GUARD_OFF:
        plan = Plan(action=venv_plan.UNCHECKED, reason=f"{venv_plan.VENV_GUARD.variable}=off")
    else:
        plan = venv_plan.plan_venv(
            venv_dir,
            root=Path(arguments.root),
            requested=arguments.recreate,
            synced_extras=set(extras.parse_groups(arguments.extras)),
        )
    report(plan, venv_dir, mode)
    if venv_plan.refuses(plan, mode):
        return REFUSED_EXIT
    print("\n".join(shell_assignments(plan)))
    return 0


if __name__ == "__main__":
    from llb.core.runtime import run

    raise SystemExit(run(main))
