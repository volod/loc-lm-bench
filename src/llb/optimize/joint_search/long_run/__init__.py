"""Research-scale roster confirmation: predeclared effect, sequential search, adoption verdict."""

from llb.optimize.joint_search.long_run.plan import (
    DEFAULT_MINIMUM_DETECTABLE_GAIN,
    DEFAULT_STABILITY_AGREEMENT,
    DEFAULT_STABILITY_BLOCKS,
    DEFAULT_TRIAL_BLOCK,
    DEFAULT_TRIAL_BUDGET,
    LONG_RUN_METHOD,
    LongRunPlan,
    ScreenSizing,
    declare_plan,
    screen_sizing,
)
from llb.optimize.joint_search.long_run.public_tracks import (
    screen_finalists,
    summarize,
)
from llb.optimize.joint_search.long_run.reference import paired_reference_deltas
from llb.optimize.joint_search.long_run.report import (
    LONG_RUN_JSON,
    LONG_RUN_MD,
    build_payload,
    render_markdown,
    write_long_run,
)
from llb.optimize.joint_search.long_run.run import LongRunResult, run_long_run
from llb.optimize.joint_search.long_run.sequential import (
    STOPPED_BY_BUDGET,
    STOPPED_BY_STABILITY,
    SearchTrail,
    run_trial_blocks,
)
from llb.optimize.joint_search.long_run.stability import (
    BlockSnapshot,
    build_snapshot,
    rank_agreement,
    ranking_from,
)
from llb.optimize.joint_search.long_run.stage import LongRunStage
from llb.optimize.joint_search.long_run.uncertainty import (
    BoardRow,
    BoardUncertainty,
    pareto_frontier,
    read_board_rows,
    read_uncertainty,
)
from llb.optimize.joint_search.long_run.verdict import (
    DECISION_ADOPT,
    DECISION_RETAIN,
    DECISION_UNDECIDED,
    AdoptionVerdict,
    decide,
)

__all__ = [
    "DECISION_ADOPT",
    "DECISION_RETAIN",
    "DECISION_UNDECIDED",
    "DEFAULT_MINIMUM_DETECTABLE_GAIN",
    "DEFAULT_STABILITY_AGREEMENT",
    "DEFAULT_STABILITY_BLOCKS",
    "DEFAULT_TRIAL_BLOCK",
    "DEFAULT_TRIAL_BUDGET",
    "LONG_RUN_JSON",
    "LONG_RUN_MD",
    "LONG_RUN_METHOD",
    "STOPPED_BY_BUDGET",
    "STOPPED_BY_STABILITY",
    "AdoptionVerdict",
    "BlockSnapshot",
    "BoardRow",
    "BoardUncertainty",
    "LongRunPlan",
    "LongRunResult",
    "LongRunStage",
    "ScreenSizing",
    "SearchTrail",
    "build_payload",
    "build_snapshot",
    "decide",
    "declare_plan",
    "paired_reference_deltas",
    "pareto_frontier",
    "rank_agreement",
    "ranking_from",
    "read_board_rows",
    "read_uncertainty",
    "render_markdown",
    "run_long_run",
    "run_trial_blocks",
    "screen_finalists",
    "screen_sizing",
    "summarize",
    "write_long_run",
]
