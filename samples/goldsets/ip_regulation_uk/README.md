# IP regulation Ukrainian gold set

This small fixture is for the M7.3 RAG prompt-system example. The canonical corpus file is
`samples/corpus/ip_regulation_uk.md`; `corpus/ip_regulation_uk.md` is a relative symlink kept so
gold-set-local tooling can still use a `corpus/` root without duplicating the document.

Properties:

- 8 verified `human-authored` items.
- 4 `tuning` items for prompt-system selection.
- 4 `final` items for the held-out comparison.
- Source spans are exact character offsets into `samples/corpus/ip_regulation_uk.md`.

The split is intentionally tiny and instructional. Use it to verify tooling and to teach the
workflow; do not treat the scores as a statistically powered benchmark.
