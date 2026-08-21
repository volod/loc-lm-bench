# UA-SQuAD drafted Russian and mixed-query overlay

This query-only overlay pairs the 81 Ukrainian-dominant `final` questions in
`../ua_squad_postedited_v1/goldset.jsonl` with two drafted forms:

- `language_ru`: a natural Russian translation drafted by the local
  `MamayLM-Gemma-3-12B-IT-v2.0` model at temperature zero;
- `language_mixed`: a deterministic UA/RU code-switch composed from the paired Ukrainian and
  Russian tokens by `compose_mixed_question` in
  `src/llb/eval/query_robustness/languages.py`, so it differs from both endpoints.

The source document, reference answer, source spans, and split are byte-for-byte identical to the
Ukrainian item. Variant ids append `--language_ru` or `--language_mixed`. Every row is
`provenance: frontier-drafted` and `verified: false`: the variants are suitable for diagnostic
retrieval and answer-quality evidence, but remain outside leaderboard evidence until a reviewer
accepts their linguistic and semantic fidelity. The language-lane loader enforces that drafted
status and rejects any change beyond id, language, question, provenance, or verification state.
After review, all rows move together to `provenance: human-verified` and `verified: true`; mixed
review states are rejected so a partially reviewed fixture cannot be reported as accepted.

The source fixture's English question `570d4e6cb3d812140066d66d` is labeled `lang: uk`; the lane
excludes it with a Cyrillic-dominance gate instead of treating English as the Ukrainian baseline.

Validate the unchanged source spans against the original committed corpus:

```bash
make validate-goldset \
  GOLDSET=samples/goldsets/ua_squad_postedited_v1_ru/goldset.jsonl \
  CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus
```
