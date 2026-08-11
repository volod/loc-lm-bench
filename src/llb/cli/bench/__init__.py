"""Benchmark category commands, each rendered under its own Tier.

Importing each submodule registers its @app.command handlers on the shared Typer app.
"""

from llb.cli.bench import (  # noqa: F401
    category_agentic,
    category_agentic_compare,
    category_agentic_controller_authority,
    category_agentic_loop_feedback_adaptation,
    category_agentic_loop_feedback_authority,
    category_agentic_loop_feedback_generalization,
    category_agentic_loop_feedback_transfer,
    category_agentic_loop_policy,
    category_agentic_memory_boundary_surface,
    category_agentic_memory_crossover_restatement,
    category_agentic_memory_fold_step,
    category_agentic_memory_replication,
    category_agentic_memory_repeated_fold,
    category_agentic_memory_summary_cap,
    category_agentic_memory_transfer,
    category_agentic_memory_trigger_collapse,
    category_agentic_memory_window_elision,
    category_agentic_policy_change_audit,
    category_agentic_compact_vs_cap,
    category_agentic_context_sweep,
    category_analysis,
    category_structured,
    category_tasks,
    category_tooling,
    knowledge_cutoff,
    knowledge_cutoff_ua,
    misc,
)
