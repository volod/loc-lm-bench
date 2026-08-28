"""Benchmark commands, grouped by the lane each one drives.

Importing each submodule registers its @app.command handlers on the shared Typer app, so this
file is the registration list: `categories` for the non-agentic benchmarks, and one package per
agentic lane (`context`, `loop`, `memory`, `knowledge_cutoff`) mirroring `llb.bench`.
"""

from llb.cli.bench import misc  # noqa: F401
from llb.cli.bench.categories import (  # noqa: F401
    agentic,
    agentic_compare,
    agentic_context,
    analysis,
    structured,
    tasks,
    tooling,
)
from llb.cli.bench.context import (  # noqa: F401
    compact_vs_cap,
    context_sweep,
    policy_change_audit,
    summary_trim_adoption,
)
from llb.cli.bench.knowledge_cutoff import run, ua  # noqa: F401
from llb.cli.bench.loop import (  # noqa: F401
    controller_authority,
    feedback_adaptation,
    feedback_authority,
    feedback_generalization,
    feedback_transfer,
    policy,
)
from llb.cli.bench.memory import (  # noqa: F401
    boundary_surface,
    crossover_restatement,
    fold_step,
    repeated_fold,
    replication,
    summary_cap,
    transfer,
    trigger_collapse,
    window_elision,
    window_elision_transfer,
)
