"""How many steps the REAL controller spent, read back out of the run bundles it already wrote.

The worst-case probe prices what the step budget ALLOWS. This module reads what a served model
actually did: every compact-versus-cap bundle persists one case row per episode with its `n_steps`,
so the extra steps a real controller took beyond perfect play are already on disk and need no run
to recover. The two numbers answer different halves of the same question -- the probe says how much
head-room a guard must carry to be safe whatever the model does, and this says how much of that
head-room the model that produced the published evidence actually used.

The reading is deliberately tolerant of a missing bundle. A published cell's manifest path is
absolute and belongs to the host that ran it, so a fresh clone, a moved `DATA_DIR`, or a CI box
holds the analysis without the bundles behind it. That is a fact about the host, never a reason to
fail an audit whose whole point is running with no run root, so an unreadable bundle reports itself
as unread and the margin the probe measured stands on its own.
"""

from pathlib import Path
from statistics import fmean
from typing import cast

from llb.artifacts.runs.bundle import read_score_rows

# Why a bundle could not be read, so an absent number is never mistaken for a measured zero.
UNREAD_NO_PATH = "the cell recorded no manifest for this arm"
UNREAD_MISSING = "the bundle is not on this host"
UNREAD_NO_STEPS = "the bundle records no per-episode step counts"


def bundle_step_counts(manifest: Path | str) -> list[int]:
    """Every episode's `n_steps` from one run bundle, or an empty list when it cannot be read.

    The scores file sits beside the manifest in the same published directory, which is what makes
    a cell's recorded manifest path enough to recover its per-episode steps.
    """
    scores = Path(manifest).parent / "scores.jsonl"
    if not scores.is_file():
        return []
    return [
        int(cast(int, row["n_steps"]))
        for row in read_score_rows(scores)
        if isinstance(row.get("n_steps"), int)
    ]


def observed_extra_steps(
    manifest: Path | str | None, *, perfect_play_steps: int
) -> dict[str, object]:
    """One arm's per-episode steps, priced against the steps perfect play needs.

    `extra_steps` can be NEGATIVE: an episode the guard refused, or one that answered early, ends
    before the oracle walk does. Those are kept as measured rather than floored at zero, because
    the maximum is the number the margin is read against and clamping would hide a run whose
    episodes never reached the geometry at all.
    """
    if manifest is None:
        return _unread(perfect_play_steps, UNREAD_NO_PATH, manifest=None)
    steps = bundle_step_counts(manifest)
    if not steps:
        reason = UNREAD_MISSING if not Path(manifest).is_file() else UNREAD_NO_STEPS
        return _unread(perfect_play_steps, reason, manifest=str(manifest))
    extra = [count - perfect_play_steps for count in steps]
    return {
        "manifest": str(manifest),
        "read": True,
        "unread_reason": None,
        "perfect_play_steps": perfect_play_steps,
        "n_episodes": len(steps),
        "n_steps": steps,
        "extra_steps": extra,
        "max_extra_steps": max(extra),
        "mean_extra_steps": fmean(extra),
        "n_episodes_beyond_perfect_play": sum(1 for value in extra if value > 0),
    }


def cell_observed_extra_steps(
    manifests: dict[str, object], *, perfect_play_steps: int
) -> dict[str, object]:
    """Both arms of one measured cell, keyed the way the cell records its manifests."""
    return {
        policy: observed_extra_steps(
            cast(str | None, manifest), perfect_play_steps=perfect_play_steps
        )
        for policy, manifest in sorted(manifests.items())
    }


def margin_is_covered(arms: dict[str, object], *, budgeted_extra_steps: int) -> bool | None:
    """Whether every arm that WAS read stayed inside the extra steps the budget priced.

    None when no arm could be read -- the honest answer on a host without the bundles, and not the
    same answer as "the observed steps fit", which is what a bare False or True would imply.
    """
    read = [
        cast(dict[str, object], arm)
        for arm in arms.values()
        if cast(dict[str, object], arm)["read"]
    ]
    if not read:
        return None
    return all(int(cast(int, arm["max_extra_steps"])) <= budgeted_extra_steps for arm in read)


def _unread(perfect_play_steps: int, reason: str, *, manifest: str | None) -> dict[str, object]:
    return {
        "manifest": manifest,
        "read": False,
        "unread_reason": reason,
        "perfect_play_steps": perfect_play_steps,
        "n_episodes": 0,
        "n_steps": [],
        "extra_steps": [],
        "max_extra_steps": None,
        "mean_extra_steps": None,
        "n_episodes_beyond_perfect_play": 0,
    }
