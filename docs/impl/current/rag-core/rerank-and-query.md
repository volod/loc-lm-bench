# Reranking, Context Order, And Query-Side Processing

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Reranking And Context Order (rerank-context-order)

Shipped: the stage between retrieval and generation is tunable -- an optional local
cross-encoder reranker (retrieve `rerank_candidates`, rerank, keep `top_k`), a context-order
policy for how the kept chunks are laid into the prompt, and a lost-in-the-middle position
probe that names the per-model ordering recommendation with measured evidence.

Modules:

- `src/llb/rag/rerank.py` -- the reranker seam. `RerankingRetriever` wraps ANY retrieval
  backend exposing `.retrieve(question, k)` (flat / parent_child / hybrid stores and the
  GraphRAG store alike): it pulls `max(rerank_candidates, k)` candidates, scores every
  (question, chunk text) pair through an injectable `RerankScorer`, and keeps the `top_k`
  best -- each kept chunk carrying `rerank_score`, its original `pre_rerank_rank`, and a fresh
  contiguous `rank`; chunk text and offsets are never altered, so source-span recall@k / MRR
  score the reranked ranking on unchanged rules. The real scorer is `CrossEncoderReranker`
  (lazy sentence-transformers CrossEncoder, the `[rag]` extra; pinned default candidate
  `BAAI/bge-reranker-v2-m3`, multilingual). `maybe_wrap_reranker` applies the config knobs in
  `_load_store` (run-eval, every backend) and the tuner's `_build_store`, so reranking rides
  every existing seam. The wrapper records per-call retrieve/rerank wall-clock
  (`stage_latency`) plus cumulative means (`mean_stage_latency`).
- `src/llb/eval/common.py` -- `order_chunks` / `format_context(chunks, order=...)`: the
  context-order policy (`rank` = best-first, the default; `reverse_rank` = best-last) applied
  ONLY when chunks are laid into the prompt; `retrieved` state stays in rank order so
  retrieval metrics are unaffected. The `[i]` labels number PROMPT positions.
- `src/llb/eval/graph.py` -- the retrieve node applies the policy and records
  `retrieve_latency_s` / `rerank_latency_s` into the case state (journaled by the
  durable-eval-runner, carried into `scores.jsonl` rows); the manifest's `metrics` gains a
  `stage_latency` object (mean retrieve / rerank / generate seconds per case), so the
  reranker's precision gain is always weighed against its measured latency cost.
- `src/llb/eval/position_probe.py` -- `llb probe-context-position` (see
  [evaluation rigor](../rigor-board-judge.md) for the probe contract and artifacts).

Knobs (all `RunConfig` fields, hence in the manifest and the sweep cell fingerprint):
`reranker` (HF cross-encoder id; `None` == off, the default), `rerank_candidates` (pool depth,
default 30), `context_order` (`rank` | `reverse_rank`, applies with or without a reranker).

Commands:

```bash
make run-eval MODEL=<m> RERANKER=BAAI/bge-reranker-v2-m3 RERANK_CANDIDATES=30 CONTEXT_ORDER=rank
make compare-retrieval RERANKER=BAAI/bge-reranker-v2-m3 GOLDSET=<goldset.jsonl> [HYBRID=1]
make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5
make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"    # 0 == reranker-off cell
llb tune --reranker BAAI/bge-reranker-v2-m3 ...       # adds on/off + candidate-depth axes
```

WHICH cross-encoder to run is a measured choice, not a pin: `compare-rerankers` ranks candidates on
one shared candidate pool with the paired verdict plus the latency and VRAM columns a reranker is
actually chosen on -- see [Reranker bake-off](reranker-bakeoff.md), which also records why
`BAAI/bge-reranker-v2-m3` remains the default.

`compare-retrieval --reranker <id>` adds a `<row>+rerank` twin per compared row (the oracle-doc
headroom row excepted), so pre/post-rerank recall@k / MRR compare through the one
`evaluate_retrieval` metric, with mean per-query retrieve/rerank latency echoed per rerank row.
In the sweep grid a `rerank_candidates=0` point is the reranker-off cell; positive depths enable
the sweep-level `--reranker` model (default `BAAI/bge-reranker-v2-m3`). The tuner samples
`use_reranker` on/off and, only when on, the candidate depth (15..60) -- dead parameters are
never sampled.

Tests: `tests/llb/rag/test_rerank.py` (fake cross-encoder: candidate flow, kept set, rank bookkeeping,
stable ties, wrapper delegation, exact context ordering per policy, stage-latency capture and
manifest aggregation, config knob validation), `tests/llb/rag/test_compare_retrieval.py` (rerank twin
rows lift MRR through the shared metric; oracle row excluded), plus grid/tuner coverage in
`tests/llb/cli/test_cli_models.py` / `tests/llb/optimize/test_tuner.py`.

Durable evidence (2026-07-08, real `BAAI/bge-reranker-v2-m3` on the CUDA host RTX 4060 Ti,
outside quick CI), via `compare-retrieval --hybrid --reranker BAAI/bge-reranker-v2-m3`, k=10,
`rerank_candidates=30`:

- `samples/goldsets/exact_terms_uk` (8 exact-term items): the reranker lifts every base row to
  MRR 1.000 -- `dense+rerank` 1.000 vs dense 0.713, `hybrid+rerank` 1.000 vs hybrid 0.938
  (recall already saturated at 1.000). The cross-encoder recovers the exact-term precision that
  dense-only loses, even without the lexical side.
- `samples/goldsets/ip_regulation_uk` (8 items, saturated fixture): every row holds
  1.000/1.000 -- post-rerank uplift-or-tie holds (a tie; the fixture cannot discriminate).
- Measured latency cost: ~150 ms/query steady-state rerank wall-clock at pool depth 30 on the
  16 GB host (~300 ms on the first store while CUDA warms; the first-row mean absorbs the one-off
  model load). Retrieval itself stays ~13 ms/query, so the reranker multiplies retrieval-stage
  cost ~12x while staying far below generation cost.

Durable evidence, full corpus (2026-07-10, rerank-order-full-cohort on the CUDA host, outside
quick CI): rerank twin rows over the verified 44-item quickstart-PDF accepted goldset (1139
chunks, k=10, non-saturated), `BAAI/bge-reranker-v2-m3`:

| row | recall@10 | MRR | rerank ms/query |
| --- | ---: | ---: | ---: |
| dense | **0.955** | 0.740 | -- |
| dense+rerank (pool 30) | 0.909 | 0.859 | 783 |
| dense+rerank (pool 60) | 0.886 | 0.845 | 1434 |
| hybrid | 0.932 | 0.742 | -- |
| hybrid+rerank (pool 30) | 0.932 | **0.871** | 684 |
| hybrid+lemmas+rerank (pool 30) | 0.909 | 0.867 | 673 |

The full-corpus answer to "does the cross-encoder recover the dense recall shortfall?" is NO --
reranking is an MRR tool, not a recall tool, here: at pool 30 it lifts MRR by +0.119..+0.129
(0.740 -> 0.859 dense, 0.742 -> 0.871 hybrid) but DEMOTES gold chunks out of the top-10 on the
dense row (recall 0.955 -> 0.909), and deepening the pool to 60 makes recall worse still (0.886)
while doubling latency -- more candidates just give the cross-encoder more distractors to
promote. Steady-state rerank cost on the full corpus is ~700-780 ms/query at pool 30 (the tiny
fixture's ~150 ms was short-chunk-flattered), a real budget item beside ~5 ms retrieval. Verdict:
keep the reranker OFF by default on this corpus; switch it on (pool 30, ideally over hybrid,
where recall is not paid) only when first-hit rank dominates the harness, e.g. small `top_k`
generation prompts.

End-to-end cross-check (2026-07-10, `make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"
RERANKER=BAAI/bge-reranker-v2-m3`, `llama3.2:3b` on ollama, accepted-goldset final split n=14,
k=5): the reranker DID lift in-run retrieval at this small k (recall@5 0.857 -> 0.929, MRR
0.685 -> 0.893 -- exactly the small-`top_k` regime the retrieval-side verdict carved out for
it), yet the end-to-end objective moved the other way: 0.378 [0.194, 0.584] rerank-off vs
0.312 [0.129, 0.515] rerank-on, overlapping CIs, at +0.96 s/query rerank latency (generation
itself is ~0.56 s/query, so reranking roughly doubles per-question cost). Retrieval uplift did
not translate into answer quality for this model at n=14 -- the off-by-default verdict stands
even in the reranker's best-case retrieval regime, and flipping it on should be justified with
end-to-end (not retrieval-only) evidence on the operator's own model + corpus. Run bundles:
`$DATA_DIR/run-eval/20260710T074826*` (off) / `20260710T074854*` (on) under the
`quickstart-pdf-corpus-rag` data dir, sweep id `rerank-crosscheck`.

## Query-Side Processing (uk-query-processing)

Shipped: an opt-in query lane between the user question and retrieval that measurably helps
Ukrainian queries while NEVER touching the stored corpus text (the query-side twin of the index-side
lexical normalization in [hybrid retrieval](hybrid-retrieval.md)). The raw question is always
preserved -- only the retrieval query is transformed -- and every step is honest: an A/B report
attributes each step's recall@k / MRR delta before anyone turns the lane on by default. Off by
default (`query_prep` empty is an exact no-op).

The `src/llb/rag/query_prep/` package is a pure, unit-testable pipeline of NAMED steps (no store, model,
or `[rag]` extra needed -- it reuses the pure tokenizer in `llb.rag.lexical`):

- `normalize` -- matching-side casefold; apostrophe-variant unification (U+2018 / U+2019 /
  U+02BC / grave / ASCII); Latin-typed Ukrainian back to Cyrillic; and safe Latin-look-alike
  repair inside mixed Cyrillic tokens. Canonical romanization preserves existing uppercase Latin
  acronyms and inserts a minimal ASCII apostrophe separator only where greedy digraph decoding
  would otherwise collide.

  An opt-in **language gate** (normalize-step-language-gate; `RunConfig.query_prep_language_gate`,
  refused at config validation unless the `normalize` step is present) decides transliteration for
  the QUERY AS A WHOLE rather than per token. Per-token transliteration is unconditional, so a
  foreign-language question is rewritten into Cyrillic nonsense the later restoration constraints
  correctly refuse to repair (`What does the Premier of Victoria...` -> `wгат доес тге...`), and it
  then retrieves on garbage. The gate romanizes each of the query's Latin word tokens (short
  uppercase acronyms excluded) and asks whether the decoded form is plausible Ukrainian -- present
  in the corpus vocabulary OR recognized by the pymorphy3 word probe (`_plausibility_probe`,
  reusing the typo guard's probe when both are on). Romanized Ukrainian decodes (near-)entirely to
  plausible forms; foreign text decodes to none. Below `LANGUAGE_GATE_MIN_PLAUSIBLE_SHARE` (0.5)
  the whole query is left untouched; a query with no Latin word tokens transliterates vacuously, so
  homoglyph repair and Cyrillic passthrough are unaffected. A refusal is recorded per query as
  `query_normalize_gate` provenance (only when it fired) and surfaces in the A/B report. On the
  committed `ua_squad_postedited_v1` goldset the gate leaves every untranslated English SQuAD
  question untouched while a romanized-Ukrainian query with a dropped soft sign (`yakist rishennya
  sudu`, 2/3 plausible) still clears the threshold and transliterates. Off by default so per-token
  transliteration stays the explicit baseline.

  An opt-in **dense-lane casing** option (normalize-casefold-dense-lane-cost;
  `RunConfig.query_prep_dense_case` / `--query-prep-dense-case`, refused at config validation
  unless the `normalize` step is present) stops the casefold at the lexical lane. Casefolding is a
  MATCHING-side convention the BM25 index asked for (`llb.rag.lexical.normalize_token` folds both
  sides), but the dense encoder is case-sensitive and never asked for it, so on an otherwise clean
  query the fold is a pure cost. With the option on, `query_prep/casing.py` transfers the raw
  question's capitalization back onto the processed text and `retrieve_prepared` routes that
  re-cased string to the dense lane while the lexical lane keeps the folded `processed` text. The
  transfer is CASE ONLY -- a token-sequence diff (`difflib`) aligns raw and processed tokens, and
  each aligned token is re-cased without replacing a single character, so apostrophe unification,
  transliteration, typo repair, and appended glossary forms all survive. Equal-length substitutions
  align too, which is what carries `Kyiv` -> `київ` -> `Київ` and restores a short Latin acronym
  (`NP`) the step folds to `np`; insertions with no raw counterpart keep the case the step
  produced. The divergence is recorded per case as `query_dense` (only when the two lanes differ)
  in `scores.jsonl` and the durability journal. Off by default so the folded dense query stays the
  explicit baseline.
- `typos` -- deterministic corpus-vocabulary typo tolerance. The token vocabulary is built from
  the indexed corpus (`VocabularyContext.build` over `store.chunks`, whose `.tokens` is the same
  set `build_vocabulary` produces); a query token ABSENT from it is corrected to a nearby
  in-vocabulary token within Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars).
  Tokens shorter than three characters are protected; candidate matching cannot cross
  alphabetic/numeric kinds; a token the corpus already contains is NEVER altered; and a numeric
  token is never "corrected" into a different one. Every correction is logged. An
  opt-in morphology guard (morphology-aware-typo-guard; `RunConfig.query_prep_typo_guard`,
  `--query-prep-typo-guard`, `QUERY_PREP_TYPO_GUARD=1`) additionally skips any OOV token pymorphy3
  recognizes as a valid Ukrainian word form (`llb.rag.lexical.load_uk_word_probe`): a grammatically
  valid inflection (`настанові`, `документами`) is not a misspelling and is left for the index+query
  lemmatization lane to match, while genuine misspellings stay unknown to the probe and are still
  corrected. Off by default so the pure edit-distance behavior remains explicitly selectable.
- **Ambiguity-aware restoration** (`query_prep/restore.py`) decides WHICH near candidate the
  `typos` step may take, and whether taking one is safe at all. Normalization is lossy -- Latin
  typing drops the soft sign and apostrophes, so `sut` inverts to the out-of-vocabulary `сут`,
  one edit from both `суть` and `суд`. Four constraints apply, in this order:
  1. **Surface compatibility (hard filter).** `normalization_provenance` maps every normalized
     token back to the single noisy token that produced it plus the edit `kind`; a candidate
     survives only when re-applying that transform reproduces the typed form
     (`surface_distance <= SURFACE_MAX_DISTANCE`, i.e. exactly). `суть` romanizes back to the
     typed `sut` and is kept; `суд` romanizes to `sud` and is refused. A token whose noise
     normalization already fully explains therefore cannot be rewritten by vocabulary correction
     at all. A replacement two different noisy tokens collapsed onto carries no constraint.
  2. **Short-token length lock.** At or below `AMBIGUOUS_TOKEN_MAX_CHARS` (4) an insertion or
     deletion candidate is refused, because at that length it is a different short word rather
     than a repair (`якв` -> `кв`, `зто` -> `то`). A transliteration provenance licenses the
     length change, since a dropped soft sign is exactly what romanization is known to lose.
  3. **Morphology, then local query context (ranking).** Candidates tied on edit distance are
     ordered by whether the morphology probe knows them as real word forms, then by whether they
     preserve the token's inflectional ending (`MORPH_SUFFIX_CHARS`), then by how often they share
     a corpus chunk with the query's rarest other in-vocabulary tokens
     (`VocabularyContext.cooccurrence` over up to `CONTEXT_MAX_ANCHORS` anchors), then
     alphabetically. Context is what separates `накат` from `наказ` for a query about waves.
  4. **Refusal on an unresolved tie.** When two candidates for a short token are equal on every
     signal above, the token is left unchanged instead of being resolved alphabetically.

  The constraints are always on inside the `typos` step (they only ever refuse or reorder a
  correction, never add one) and need no new knob; the morphology signal rides on the same opt-in
  probe as the guard, and the context index is built in the same pass as the vocabulary.
- `glossary` -- alias/glossary expansion. When the query mentions a known term (or a surzhyk /
  transliterated alias) the entry's other surface forms are APPENDED (the raw query is preserved),
  so retrieval catches the spelling the corpus actually uses. Sourced from a `query_glossary.json`
  built from a draft bundle's `prompt_dictionary_candidates.jsonl` (see
  [data prep](../data-prep.md) query glossary).
- `rewrite` -- an optional local-LLM query rewrite through the run's backend endpoint seam
  (`eval.rag.query_rewrite` prompt). OFF by default and NEVER present unless explicitly requested;
  records both the original and rewritten query per case.
- `hyde` -- generates a short hypothetical answer through the same local endpoint and embeds it
  on the dense lane while retaining the processed user question for BM25 and graph linking. It
  does not alter the question sent to answer generation.
- `decompose` -- parses a bounded JSON or line-list response into at most five subqueries,
  retrieves every subquery, and deduplicates exact source spans with weighted RRF. A 2x
  original-query lane stabilizes ranking when the model over-decomposes a simple question.

Wiring: `src/llb/eval/graph.py` processes the question before retrieval and hands the structured
result to `query_prep/retrieval.py`. `RagStore.retrieve_queries` accepts separate dense and
lexical text; graph, fused, and reranking wrappers preserve that contract. The raw question stays
in state for generation. `scores.jsonl` and the durability journal carry `query_processed`,
`query_corrections`, `query_hypothetical_answer`, `query_decomposition`, and
`query_subqueries`, so normal and resumed runs preserve generated-query provenance. Journal
inclusion also fixes the earlier loss of deterministic query-prep provenance on resume.

Knobs (all `RunConfig` fields, hence in the manifest fingerprint): `query_prep` (ordered list of
`normalize` | `typos` | `glossary` | `rewrite` | `hyde` | `decompose`;
unknown/duplicated steps rejected at config validation), `query_glossary_path`,
`query_prep_typo_guard` (refused at config validation unless the `typos` step is present), and
`query_prep_language_gate` and `query_prep_dense_case` (both refused at config validation unless
the `normalize` step is present).

Commands:

```bash
make build-query-glossary BUNDLE=<draft dir>            # -> <bundle>/query_glossary.json
make run-eval MODEL=<m> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json>
make validate-retrieval GOLDSET=<gs> QUERY_PREP=normalize,typos,glossary QUERY_GLOSSARY=<json> QUERY_PREP_AB=1
make validate-retrieval GOLDSET=<gs> QUERY_PREP=normalize QUERY_PREP_AB=1 QUERY_PREP_DENSE_CASE=1
make validate-retrieval CONFIG=<yaml> GOLDSET=<gs> QUERY_PREP=hyde,decompose \
  QUERY_PREP_MODEL=<m> QUERY_PREP_BACKEND=ollama QUERY_PREP_AB=1 \
  QUERY_PREP_OUT=<report.json>
make bench-query-robustness MODEL=<m> BACKEND=<b> GOLDSET=<gs> [QUERY_PREP_DENSE_CASE=1]
```

The `validate-retrieval --query-prep-ab` A/B report scores `baseline` then each cumulative step
with per-step recall@k / MRR deltas. Model steps use `--query-prep-model` and
`--query-prep-backend`; their completions and parsed subqueries are embedded per case in the JSON
report. Endpoint generators cache a question within one cumulative run, avoiding duplicate model
calls while preserving fixed-temperature results.

Tests: `tests/llb/rag/test_query_prep.py` (apostrophe and mixed-script repair, collision-safe
romanization, Latin acronym preservation, Damerau-Levenshtein transposition, typo correction that
never touches in-vocabulary, short, or cross-kind tokens + long-token distance 2 + deterministic
tie-break, deterministic alias expansion + glossary
build/round-trip, rewrite off-by-default, exact no-op when the lane is off, pipeline ordering +
dependency validation, A/B per-step delta over a fake store, retrieve-node raw-preservation and
processed-query wiring, HyDE dense/lexical separation, decomposition parsing/bounds/RRF span
deduplication, runner resolver dependency wiring, provenance mapping including the ambiguous
same-replacement case, per-kind surface distance, refusal of an incompatible nearest neighbor,
restoration of the romanization-compatible form, the short-token length lock and the
transliteration exemption from it, unresolved-tie refusal, context-driven candidate choice, and
both morphology preferences), `tests/llb/rag/test_query_prep_dense_case.py` (case-pattern
transfer without character replacement, the dense/lexical split reaching the store seam, the
recovered acronym, capitalization carried onto a corrected token, the exact no-op on an
already-lowercase query, glossary insertions left folded, subquery lanes untouched, and the
normalize-step dependency), `tests/llb/rag/test_store.py` (split hybrid
queries), and `tests/llb/executor/test_durable_resume.py` (generated-query journal round trip),
plus config validation in `tests/llb/core/test_config.py`.

The end-to-end noise benchmark, model evidence, and model-specific default recommendation live in
[evaluation
rigor](../rigor-board-judge/robustness-benchmarks.md#ukrainian-query-robustness-benchmark). Its
noise classes are one mechanism each (`transliteration`, `apostrophe_variant`, `mixed_script`,
`keyboard_typos`), so what a mitigation lane recovers is attributable to the noise it inverts rather
than blended across two mechanisms at once.

### Cross-language query processing evidence

The same benchmark now measures Russian and UA/RU code-switched questions against unchanged
Ukrainian corpus evidence. The committed overlay, strict unchanged-gold loader, per-language
recall/MRR/objective report, and CUDA result are documented in
[evaluation rigor](../rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane).
`normalize` remains a character repair mechanism; it is not a language translator. The
`translate_to_uk` robustness lane is an exact fixture-pair retrieval upper bound and is deliberately
absent from `QUERY_PREP_STEPS`, so it cannot silently become a production query transformation.

The measured result does not support adding translation to the shipped query path. Raw Russian
retrieval already matches the Ukrainian baseline, and exact Ukrainian retrieval does not remove the
answer-quality loss. Mixed-query translation restores a small retrieval loss but makes objective
worse than raw mixed. Query-language mitigation therefore remains off by default; the next useful
work is answer-language behavior, not a retrieval translator.

Durable evidence (2026-07-09, `intfloat/multilingual-e5-base`, flat FAISS over
`samples/goldsets/ip_regulation_uk/corpus`, k=5):

- Clean UA goldset queries (`samples/goldsets/ip_regulation_uk`, 8 items): baseline recall@5
  1.000 / MRR 1.000; `+normalize`, `+typos`, `+glossary` all hold 1.000/1.000 (+0.000 each). The
  fixture saturates (as the base-model comparisons here do), so the deltas are honestly zero --
  the typo step also "corrects" a few valid inflected query forms to the nearest corpus form
  (crude inflection matching, not a misspelling; the shipped lemmatization is the right tool for
  inflection), which the A/B would surface as a negative delta on a non-saturated corpus.
- Latin-typed variant of the same 8 queries (each Cyrillic word romanized -- e.g.
  `na yaki dvi velyki hrupy podilyayut pravo intelektualnoyi vlasnosti?`): baseline recall@5
  0.875 / MRR 0.812; `+normalize` (transliteration) RECOVERS to 1.000 / 1.000 -- a +0.125 recall /
  +0.188 MRR uplift. This is the mechanism's honest positive-delta demonstration.

Morphology-guard A/B (2026-07-10, morphology-aware-typo-guard on the CUDA host): over the
verified 44-item quickstart-PDF accepted goldset against the full-corpus 1139-chunk e5-base
store (k=10, non-saturated) the predicted regression is real and the guard removes it:

| stage | recall@10 | MRR | d(MRR) |
| --- | ---: | ---: | ---: |
| baseline | 0.955 | 0.740 | -- |
| +normalize | 0.955 | 0.748 | +0.009 |
| +typos (unguarded) | 0.955 | 0.736 | **-0.012** |
| +typos (guarded) | 0.955 | 0.748 | +0.000 |

Unguarded, the edit-distance step "corrected" valid inflections to the corpus surface form
(`настанові` -> `настанова`) and paid -0.012 MRR; guarded, those known word forms pass through
untouched (the lemmatization lane is the right tool for them) while genuine out-of-vocabulary
typos -- including the mixed-script `wеб` (Latin `w`) -> `веб` -- are still corrected, and the
step becomes MRR-neutral. Verdict: turn the guard on whenever the `typos` step is in use.

### HyDE and decomposition evidence

CUDA-host evidence (2026-07-21): `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` through Ollama,
`intfloat/multilingual-e5-base`, hybrid FAISS, k=10, and the held-out final split (n=13) of the
available verified 40-item accepted set against its full 1124-chunk store:

| stage | recall@10 | MRR | d(MRR) |
| --- | ---: | ---: | ---: |
| baseline | 0.923 | 0.814 | -- |
| +hyde | 0.923 | 0.833 | +0.019 |
| +hyde +decompose | 0.923 | 0.833 | +0.000 |
| baseline (isolated decomposition run) | 0.923 | 0.814 | -- |
| +decompose | 0.923 | 0.827 | +0.013 |

An initial equal-weight decomposition run regressed MRR to 0.699. Replaying its recorded
subqueries showed that adding the original question at 2x weight changed the result from harmful
to useful; the final independent run confirms +0.013 MRR, and the cumulative lane preserves the
larger HyDE gain. Recall is unchanged. Both steps remain opt-in. Reports, including endpoint and
per-case generated text, are under
`$DATA_DIR/query-prep-hyde-decompose/<run>/query_prep_ab_improved.json` and
`decompose_ab_improved.json`.
