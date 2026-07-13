"""Interactive human verifier for the human verification gate sample worksheet.

A terminal session that walks a stratified sample item by item and writes the HUMAN columns
(the four checks, the accept/reject decision, a note, a status) in place. Interactive I/O lives
here, OUT of the pure `verify.py`; the two share the worksheet schema + atomic load/save. This
mirrors how `judge/rate.py` pairs with `judge/calibration.py`.

The card rendering and command parsing (the pure presentation half) live in `verify_card.py`; this
package owns the session loop, split into `report` (pure worksheet summaries + throughput stats),
`commands` (terminal I/O + navigation), `decision` (accept/reject + edit handlers + the shared
`SessionContext`), and `loop` (the `run_session` driver). The public API is re-exported here so
callers keep importing `llb.goldset.verify_session`.

Design notes that matter for trust:
- The second-frontier `cc_*` verdict is HIDDEN by default. The human must verify INDEPENDENTLY;
  seeing the cross-check first anchors them and defeats the point of the gate. `--show-crosscheck`
  reveals it for post-hoc review only.
- The CSV IS the session state: every edit rewrites the whole file atomically, so resume and
  crash-safety are free. Samples are small by design (a few dozen across strata).
- The DECISION (accept/reject) is what advances; marking the individual checks does not, because
  an item has several checks and you set them before deciding.

The loop is driven by an injected input iterator + output sink, so it is fully unit-testable
without a terminal, model, endpoint, or GPU (it operates only on the CSV).
"""

from llb.goldset.verify_card import (
    ACCEPT_CMD,
    CHECK,
    HELP,
    JUMP,
    NEXT,
    PREV,
    QUIT,
    REJECT_CMD,
    Command,
    format_card,
    help_text,
    parse_command,
)
from llb.goldset.verify_session.commands import _go_forward, _go_undecided
from llb.goldset.verify_session.loop import run_session
from llb.goldset.verify_session.report import (
    SESSION_STATS_FILENAME,
    SessionStats,
    append_session_stats,
    clear_human_columns,
    completion_panel,
    decided_count,
    decision_tally,
    first_undecided_index,
    save_human_columns,
    summary_lines,
    throughput_line,
)

__all__ = [
    "ACCEPT_CMD",
    "CHECK",
    "HELP",
    "JUMP",
    "NEXT",
    "PREV",
    "QUIT",
    "REJECT_CMD",
    "SESSION_STATS_FILENAME",
    "Command",
    "SessionStats",
    "_go_forward",
    "_go_undecided",
    "append_session_stats",
    "clear_human_columns",
    "completion_panel",
    "decided_count",
    "decision_tally",
    "first_undecided_index",
    "format_card",
    "help_text",
    "parse_command",
    "run_session",
    "save_human_columns",
    "summary_lines",
    "throughput_line",
]
