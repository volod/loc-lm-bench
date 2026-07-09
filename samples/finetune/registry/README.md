# Adapter registry fixture

A committed `registry.jsonl` (the same append-only event log `$DATA_DIR/adapters/registry.jsonl`
uses) holding the two lifecycle cases the unit tests need:

| adapter | entry | what it exercises |
| --- | --- | --- |
| [`stale-adapter`](../stale-adapter/) | `register` + `merge` | staleness detection and merge-event folding |
| [`laundered-adapter`](../laundered-adapter/) | `register` | the contamination guard reading recorded digests |

The `stale-adapter` entry records goldset and corpus digests that do not match
`samples/goldsets/ip_regulation_uk/`, so `llb list-adapters` stamps it `stale` and the board
appends `[stale]` to its row label. The `laundered-adapter` entry records the `final`-split
training ids its own on-disk manifest hides, so the guard refuses it.

Adapter directories are recorded project-relative here; a real registry records absolute paths.
`llb gc-adapters` refuses to delete either one: both live outside `$DATA_DIR`.
