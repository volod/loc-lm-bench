# exact_terms_uk -- exact-term lexical-win regression fixture

A committed micro-goldset for the hybrid-retrieval regression (hybrid-retrieval-uk): a
Ukrainian orders registry of 40 near-identical entries that differ ONLY in exact terms --
order numbers, DSTU standard codes, surnames, and funding amounts. Dense-only cosine
retrieval confuses the near-duplicate paragraphs, while lexical BM25 pins the exact number
or code, so hybrid fusion must retrieve strictly better here (`tests/test_hybrid_store.py`
asserts it with a deterministic fake dense index; the real-embedder evidence lives in
`docs/impl/current/rag-core.md`).

Layout:

- `corpus/orders_registry_uk.md` -- the generated registry (one doc, ~41 recursive chunks
  at the default 800/120 chunking).
- `goldset.jsonl` -- 8 human-authored items (6 `final`, 2 `tuning`), each question naming an
  exact term (order number, DSTU code, or amount) with an offset-exact source span.

Regenerating: the fixture is deterministic; edit and re-run the generation snippet recorded
in the git history of this directory if the layout must change (span offsets are validated by
`load_goldset`, which rejects any drift between `text` and the char offsets).
