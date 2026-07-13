"""Interactive human review session for external RAG answer logs.

The answered JSONL is the session state. Each human edit rewrites the file atomically, so a
reviewer can quit or interrupt and later resume at the first unscored row. CSV/report artifacts are
written only when every row has a human score and decision.

Submodules (import from the specific one you need -- there is no re-export surface): `commands`
(prompt-command vocabulary + `parse_command` + `help_text`), `cards` (pure card rendering +
`format_card`), `records` (human-field state mutations + index math), and `session` (the
`run_external_rag_session` driver loop).
"""
