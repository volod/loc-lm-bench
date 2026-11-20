# Task Record Template

Copy this file to `NNNN-<group>-<task-id>.md` (planned work; `<group>` is the capability id the
task serves) or `NNNN-adhoc-<slug>.md` (a direct request), where `NNNN` is the next unused sequence
in the [records index](README.md); add its row there, and replace each section's guidance with the
record.

## Work contract

Paste the original plan task verbatim in a fenced block, or state the scope of the direct request.

## Implementation and decisions

Summarize what changed and where it lives, and any material tradeoff or authorized scope change.

## Acceptance evidence

State the status as `open` or `accepted`. List each acceptance gate with the command or artifact
that answered it and its result: what ran on what, the date and host, and every load-bearing
number with its reading. Name any gate that still needs evidence. For research, explain how the
negative result meets the declared gate and link its future-research section.

## Handoff

Name remaining work and the task id that owns it. For planned work, record the `make plan-status`
counts before and after, the capabilities whose groups or registry rows changed, and any
`Requires` edge this task's removal released.
