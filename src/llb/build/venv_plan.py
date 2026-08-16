"""What `make venv` is about to do to `.venv`, and what a rebuild costs.

`make venv` printed `reusing .venv -- updating deps` whenever `.venv/bin/python` existed, and
`uv sync --inexact` is documented as leaving vLLM/torch and every separately installed package in
place. Neither holds once the SYSTEM interpreter the venv points at is patched underneath it:
`pyvenv.cfg` records `version_info` at creation time, so an OS python upgrade leaves the recorded
version behind the real one, uv calls the environment stale, and the sync REPLACES it -- `Would
replace project environment at: .venv` where a venv whose recorded version still matches reports
`Would use`. The replacement installs the lock's torch over the CUDA-matched one the serving stack
needs. On a CUDA host the `VENV_INSTALL_VLLM=auto` step afterwards puts vLLM (and its torch pin)
back, so the damage is a silent full reinstall; with `VENV_INSTALL_VLLM=0`, or on any host that
skips that step, the venv is left holding a torch its vLLM cannot use while the target said
`reusing`.

The decision therefore happens BEFORE the sync, from `llb.build.venv_interpreter`'s reading of
`pyvenv.cfg` against the interpreter it points at -- no uv call, no network, no import of the venv
being judged. `llb.build.venv_state` renders what this module decides and is what the make target
runs. The guard vocabulary is `llb.build.lock_guard`'s, so relaxing it reads the same as relaxing
the other two: refuse (default) | report | off.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.build import extras, lock_guard, lock_reader, venv_interpreter
from llb.core import env
from llb.core.paths import PROJECT_ROOT

CREATE = "create"
REUSE = "reuse"
REBUILD = "rebuild"
UNCHECKED = "unchecked"  # the guard is off; uv decides between reuse and replace on its own

# Packages whose version is chosen for the GPU rather than by `uv.lock` -- installed by
# `scripts/build_vllm.sh` after the sync, deliberately trailing the lock (see AGENTS.md). Losing
# these is what makes a rebuild expensive, so their presence is what the refusal keys on.
HARDWARE_MATCHED = (
    "vllm",
    "vllm-flash-attn",
    "flash-attn",
    "flashinfer-python",
    "torch",
    "torchaudio",
    "torchvision",
    "xformers",
    "bitsandbytes",
)
VLLM_PACKAGE = "vllm"

VENV_GUARD = lock_guard.Guard(
    subject="this venv rebuild",
    variable=env.VENV_STALE_GUARD,
    remedy="re-run with RECREATE_VENV=1 to accept the rebuild",
)


@dataclass(frozen=True)
class Loss:
    """One installed package a rebuilt environment does not reproduce.

    `locked` empty means the lock does not carry the package at all, so the sync simply drops it;
    a non-empty `locked` means it comes back at the lock's version instead of the installed one.
    """

    name: str
    installed: str
    locked: tuple[str, ...]
    # Which optional extras declare it, when that is what decides whether the sync restores it.
    owners: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """The action, its reason, and the two kinds of package a rebuild puts at stake."""

    action: str
    reason: str
    losses: tuple[Loss, ...] = ()
    at_risk: tuple[Loss, ...] = ()
    requested: bool = False
    restampable: bool = False
    vllm_installed: bool = False

    @property
    def hardware_matched(self) -> tuple[Loss, ...]:
        return tuple(loss for loss in self.losses if loss.name in HARDWARE_MATCHED)

    @property
    def repinned(self) -> tuple[Loss, ...]:
        """Hardware-matched packages the lock ALSO carries, so any sync moves them to its version.

        These are the ones a reuse touches. `--inexact` only promises not to REMOVE what the lock
        does not name; a package inside the resolution (torch arrives through
        sentence-transformers) is still installed at the locked version.
        """
        return tuple(loss for loss in self.hardware_matched if loss.locked)

    @property
    def force_vllm(self) -> bool:
        """When the sync moves the stack, reinstalling vLLM is not optional -- `=0` included.

        Two ways that happens, and both end with a torch the installed vLLM cannot use while the
        target reported success: a REBUILD takes the whole stack, and a REUSE re-pins whatever the
        lock also carries. Skipping the reinstall in either case is the failure this module exists
        for, so `VENV_INSTALL_VLLM=0` is overridden rather than obeyed -- it was set for an
        environment the sync no longer leaves in place.
        """
        if not self.vllm_installed:
            return False
        return self.action == REBUILD or bool(self.repinned)


def decide(venv_dir: Path) -> tuple[str, str, bool]:
    """The action `make venv` is about to take, why, and whether a restamp could avoid it."""
    config = venv_interpreter.read_config(venv_dir)
    if not config:
        return CREATE, f"no {venv_interpreter.PYVENV_CFG} under {venv_dir}", False
    interpreter = venv_interpreter.read_interpreter(config, venv_dir)
    if interpreter.recorded is None:
        # Nothing to compare against, so claiming staleness would be a guess: uv still decides.
        return REUSE, f"{venv_interpreter.PYVENV_CFG} records no readable version", False
    if interpreter.path is None:
        home = config.get(venv_interpreter.HOME_KEY, "?")
        return REBUILD, f"the interpreter it was built from is gone ({home})", False
    if interpreter.current is None:
        return REBUILD, f"{interpreter.path} no longer runs", False
    recorded = venv_interpreter.format_version(interpreter.recorded)
    if not interpreter.moved:
        return REUSE, f"records python {recorded}, matching {interpreter.path}", False
    current = venv_interpreter.format_version(interpreter.current)
    return (
        REBUILD,
        f"records python {recorded}; {interpreter.path} is now {current}",
        interpreter.patch_move,
    )


def unreproduced(installed: dict[str, str], locked: dict[str, set[str]]) -> tuple[Loss, ...]:
    """Installed packages a lock-only rebuild does not put back exactly as they are.

    Not the same question as `lock_guard.find_drift`, which asks whether a DECLARED package sits
    off the lock and is a drift to repair. Here anything installed counts -- vLLM and the CUDA
    wheels are not declared at all, and they are precisely what a replace throws away.
    """
    losses = [
        Loss(name, version, tuple(lock_reader.sorted_versions(locked.get(name, ()))))
        for name, version in installed.items()
        if version not in locked.get(name, ())
    ]
    return tuple(sorted(losses, key=lambda loss: (loss.name not in HARDWARE_MATCHED, loss.name)))


def unsynced_extra_risk(
    installed: dict[str, str],
    locked: dict[str, set[str]],
    groups: dict[str, set[str]],
    synced: set[str],
) -> tuple[Loss, ...]:
    """Installed packages the lock carries whose only declaring extras this sync leaves out.

    Matching the lock is not enough to survive a replace: `uv sync` installs the extras it was
    ASKED for. Measured here, `bitsandbytes` 0.49.2 matched the lock exactly and still vanished --
    only `[finetune]` declares it, and `make venv`'s default extras do not include that group. So
    these are named separately from the definite losses, with the install that puts them back.
    Transitive-only packages are out of reach without resolving the lock's graph, so this names
    what an operator installed ON PURPOSE rather than claiming to enumerate every casualty.
    """
    at_risk = []
    for name, version in sorted(installed.items()):
        if version not in locked.get(name, ()):
            continue  # already counted as a definite loss
        owners = extras.declaring_groups(name, groups)
        if owners and not synced.intersection(owners):
            locked_versions = tuple(lock_reader.sorted_versions(locked[name]))
            at_risk.append(Loss(name, version, locked_versions, owners))
    return tuple(at_risk)


def plan_venv(
    venv_dir: Path,
    *,
    root: Path = PROJECT_ROOT,
    requested: bool = False,
    synced_extras: set[str] | None = None,
) -> Plan:
    """Decide the action and price what the sync moves -- everything on a rebuild, the lock's own
    packages on a reuse."""
    if requested and venv_dir.exists():
        action, reason, restampable = REBUILD, "RECREATE_VENV set", False
    else:
        action, reason, restampable = decide(venv_dir)
    if action == CREATE:
        return Plan(action=action, reason=reason, requested=requested)
    inputs = lock_guard.lock_inputs(root)
    locked = inputs[0] if inputs is not None else {}
    installed = lock_reader.venv_versions(venv_dir)
    groups = (
        lock_reader.declared_groups(root / lock_reader.PYPROJECT_FILENAME)
        if inputs is not None
        else {}
    )
    # A reuse REMOVES nothing -- `--inexact` keeps vLLM and the CUDA wheels exactly where they are
    # -- so the only packages it puts at stake are the ones the lock also carries, and the extras
    # question does not arise at all.
    losses = unreproduced(installed, locked)
    at_risk: tuple[Loss, ...] = ()
    if action == REBUILD:
        at_risk = unsynced_extra_risk(installed, locked, groups, synced_extras or set())
    else:
        losses = tuple(loss for loss in losses if loss.locked)
    return Plan(
        action=action,
        reason=reason,
        losses=losses,
        at_risk=at_risk,
        requested=requested,
        restampable=restampable,
        vllm_installed=VLLM_PACKAGE in installed,
    )


def refuses(plan: Plan, mode: str) -> bool:
    """Refuse only the case nobody asked for: a rebuild that silently drops the GPU stack."""
    return (
        plan.action == REBUILD
        and not plan.requested
        and bool(plan.hardware_matched)
        and mode == lock_guard.GUARD_REFUSE
    )
