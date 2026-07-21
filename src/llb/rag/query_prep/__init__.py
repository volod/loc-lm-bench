"""Opt-in query-side processing lane between the user question and retrieval (uk-query-processing).

A pure, unit-testable pipeline of NAMED steps that transforms a query BEFORE it reaches the
retrieval store. It never touches the stored corpus text -- original word forms stay untouched;
this is the query-side twin of the index-side lexical normalization already shipped in
`llb.rag.lexical`. Every step is honest: it reports what it changed so an A/B report can attribute
a per-step retrieval delta before anyone turns the lane on by default.

Steps (applied in the configured order), each living in its own submodule:
  - `normalize`: matching-side casefold, apostrophe-variant unification, and a small
    transliteration table that maps Latin-typed Ukrainian tokens back to Cyrillic (`zakon` ->
    `закон`).
  - `typos`: deterministic corpus-vocabulary typo tolerance. A query token ABSENT from the indexed
    corpus vocabulary is corrected to its nearest in-vocabulary token within Damerau-Levenshtein
    (OSA) distance 1 (2 for tokens over 8 chars); a token the corpus already contains is NEVER
    altered.
  - `glossary`: alias/glossary expansion. When the query mentions a known term (or one of its
    surzhyk / transliterated aliases) the entry's other surface forms are appended.
  - `rewrite`: an optional local-LLM query rewrite through an injected endpoint callable; OFF by
    default and only present when explicitly requested.
  - `hyde`: a local-LLM hypothetical answer used only for dense retrieval.
  - `decompose`: bounded local-LLM subqueries retrieved and fused with a raw-query stabilizer.

`base` holds the shared step ids and the recorded-edit records; `pipeline` composes the steps;
`retrieval` executes structured dense/lexical plans; `report` renders the cumulative A/B stage
table. Import from the specific submodule you need -- the package intentionally keeps no re-export
surface.
"""
