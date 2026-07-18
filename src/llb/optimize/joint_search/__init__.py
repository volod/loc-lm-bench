"""Joint model + RAG-config search (successive-halving + per-finalist multi-obj tune)."""

from llb.optimize.joint_search.constants import JOINT_SEARCH_METHOD
from llb.optimize.joint_search.halving import (
    HalvingLedger,
    HalvingRound,
    ScreenScore,
    build_halving_round,
    finalize_ledger,
    keep_count,
    partition_survivors,
    rank_scores,
    screen_limit_for_round,
)
from llb.optimize.joint_search.models import FinalistTuneResult, JointSearchResult
from llb.optimize.joint_search.report import (
    assert_final_split,
    joint_run_dir,
    write_ledger,
    write_manifest,
    write_scoreboard,
)
from llb.optimize.joint_search.resume import (
    remaining_optuna_trials,
    study_name_for,
)
from llb.optimize.joint_search.schedule import run_joint_search

__all__ = [
    "JOINT_SEARCH_METHOD",
    "FinalistTuneResult",
    "HalvingLedger",
    "HalvingRound",
    "JointSearchResult",
    "ScreenScore",
    "assert_final_split",
    "build_halving_round",
    "finalize_ledger",
    "joint_run_dir",
    "keep_count",
    "partition_survivors",
    "rank_scores",
    "remaining_optuna_trials",
    "run_joint_search",
    "screen_limit_for_round",
    "study_name_for",
    "write_ledger",
    "write_manifest",
    "write_scoreboard",
]
