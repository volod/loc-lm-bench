# Review Workbench

The shared review core and Textual workbench give every human gate the same record, navigation,
progress, and color language while retaining each flow's existing ledger format and verdicts.

## Architecture

`src/llb/review/core.py` defines the backend-neutral `ReviewRecord`, three section roles, keyed
`ReviewAction` values, dataset/record/stratum `ReviewProgress`, the ledger adapter API, and bounded
resume-aware navigation. `src/llb/review/workbench.py` renders that contract with Textual. Record
content, evidence, and metadata use separate cool-color panes; verdict buttons, status, navigation,
and progress use the action color role.

The adapters under `src/llb/review/adapters/` delegate every mutation to the established flow
helpers:

| Adapter | Accepted path | Existing persistence behavior retained |
| --- | --- | --- |
| Goldset verify | verification CSV | human-column merge and atomic canonical CSV writer |
| Judge calibration | calibration CSV | human-column merge and canonical calibration header |
| External RAG | answered JSONL | human-field initializer and canonical JSONL writer |
| Draft compare | comparison run directory or `comparison.json` | both lane worksheets through the goldset writer |
| Knowledge cutoff UA | translation bundle or worksheet | translation profile checks through the goldset writer |
| Prompt system | run directory or `candidates.json` | indented candidate JSON writer and existing statuses |

`src/llb/review/registry.py` detects these signatures conservatively. Opening a ledger does not
create a new artifact root: writes remain in its existing CSV, JSONL, or JSON path.

## Commands and compatibility

Install the optional terminal dependency from the loaded project environment, then open any
supported path:

```bash
source scripts/shared/common.sh
llb_load_env
uv pip install -e ".[review]"
make review-workbench REVIEW_PATH=<ledger-or-run-dir>
# Low-level equivalent:
llb review <ledger-or-run-dir>
```

The existing goldset verify, judge-rate, external-RAG score, draft-compare, and cutoff-review
commands prefer the workbench for their normal review mode. Prompt-system review exposes it as
`--action workbench`. If Textual is absent, established commands with a legacy terminal loop keep
using that loop. Advanced legacy modes such as clear-all confirmation, confidence ordering,
cross-check/judge reveal, or custom external field selection also keep their specialized loop.

Navigation is uniform: `n`/right moves forward, `b`/left moves back, `u` selects the next pending
record, and `q` saves and exits. Action buttons show their key. Goldset-family ledgers retain the
four lower-case pass and upper-case fail keys plus `y` accept, `x` reject, and `c` clear. Judge
calibration retains ratings `1` through `5`; external RAG retains accept/partial/reject; prompt
systems retain approve/pin/reject. Every action writes through immediately, and a fresh session
resumes at the first pending record.

## Verification

`tests/llb/review/` covers navigation and progress, path detection for all six adapters,
byte-for-byte compatibility against the legacy CSV/JSON/JSONL writers, composite draft lanes, the
knowledge-cutoff profile, and Textual pilot interactions for resume, navigation, verdict entry,
progress rendering, and distinct data/evidence/action colors. The tests are part of `make ci`;
Textual is consequently included in the development extra as well as the optional `review` extra.
