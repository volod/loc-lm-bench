"""Interactive human verifier for the human verification gate sample worksheet.

A terminal session that walks a stratified sample item by item and writes the HUMAN columns
(the four checks, the accept/reject decision, a note, a status) in place. Interactive I/O lives
here, OUT of the pure `verify.py`; the two share the worksheet schema + atomic load/save. This
mirrors how the `judge/rate/` package pairs with `judge/calibration.py`.

The card rendering and command parsing (the pure presentation half) live in `verify_card.py`; this
package owns the session loop, split into `report` (pure worksheet summaries + throughput stats),
`commands` (terminal I/O + navigation), `decision` (accept/reject + edit handlers + the shared
`SessionContext`), and `loop` (the `run_session` driver). Import from the specific submodule you
need -- there is no re-export surface.

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
