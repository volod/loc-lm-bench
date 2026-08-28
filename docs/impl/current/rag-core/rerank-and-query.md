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
  [evaluation rigor](../rigor-board-judge.md) for the probe contract and artifacts). Beside
  `report.md` and `cases.jsonl` it writes `probe.json`: the model, the per-position means and CIs,
  the recommendation, a `separated`/`flat` reading of the head-versus-tail comparison, and the
  retrieval fingerprint of the store the probe queried. A consumer -- the [composed agent operating
  profile](../extended-workflows/agent-operating-profile.md) reads it -- therefore takes the
  decision from a field rather than by parsing the report's prose, and can tell a recommendation
  taken against a re-chunked store apart from one taken against its own. `--corpus-root`
  (`CORPUS_ROOT=`) points the probe at the corpus whose persisted index it should query, matching
  `run-eval`.

Ordering is one of two things this stage does to the kept chunks; the other is what each chunk's
TEXT looks like in the prompt, which is
[prompt-side context assembly](context-assembly.md) -- same boundary (prompt copies only, stored
chunks and their offsets untouched), separate page.

Knobs (all `RunConfig` fields, hence in the manifest and the sweep cell fingerprint):
`reranker` (HF cross-encoder id; `None` == off, the default), `rerank_candidates` (pool depth,
default 30), `context_order` (`rank` | `reverse_rank`, applies with or without a reranker).

Commands:

```bash
make run-eval MODEL=<m> RERANKER=BAAI/bge-reranker-v2-m3 RERANK_CANDIDATES=30 CONTEXT_ORDER=rank
make compare-retrieval RERANKER=BAAI/bge-reranker-v2-m3 GOLDSET=<goldset.jsonl> [HYBRID=1]
make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5 CORPUS_ROOT=<corpus-dir>
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
end-to-end (not retrieval-only) evidence on the operator's own model + corpus. Boundaries: n=14
with overlapping intervals cannot separate the two arms, so this is a "no measured gain" reading,
not a measured loss; and `llama3.2:3b` is below the project's >=7B evidence floor, which bounds the
answer-side half further. What would overturn it: the same cross-check at n large enough to
separate 0.378 from 0.312, on a >=7B model. The two `rerank-crosscheck` run bundles (sweep id
`rerank-crosscheck`, under the `quickstart-pdf-corpus-rag` data dir) are not retained on either GPU
host; the numbers above are the record.

## Query-Side Processing (uk-query-processing)

Shipped: an opt-in query lane between the user question and retrieval that measurably helps
Ukrainian queries while NEVER touching the stored corpus text (the query-side twin of the index-side
lexical normalization in [hybrid retrieval](hybrid-retrieval.md)). The raw question is always
preserved -- only the retrieval query is transformed -- and every step is honest: an A/B report
attributes each step's recall@k / MRR delta before anyone turns the lane on by default. Off by
default (`query_prep` empty is an exact no-op).

The `src/llb/rag/query_prep/` package is a pure, unit-testable pipeline of NAMED steps (no store, model,
or `[rag]` extra needed -- it reuses the pure tokenizer in `llb.rag.vector_store.lexical`):

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
  `RunConfig.query_prep_dense_case` / `--query-prep-dense-case`, refused at config validation unless
  the `normalize` step is present) stops the casefold at the lexical lane. Casefolding is a
  MATCHING-side convention the BM25 index asked for (`llb.rag.vector_store.lexical.normalize_token`
  folds both sides), but the dense encoder is case-sensitive and never asked for it, so on an
  otherwise clean query the fold is a pure cost. With the option on, `query_prep/casing.py`
  transfers the raw question's capitalization back onto the processed text and `retrieve_prepared`
  routes that re-cased string to the dense lane while the lexical lane keeps the folded `processed`
  text. The transfer is CASE ONLY -- a token-sequence diff (`difflib`) aligns raw and processed
  tokens, and each aligned token is re-cased without replacing a single character, so apostrophe
  unification, transliteration, typo repair, and appended glossary forms all survive. Equal-length
  substitutions align too, which is what carries `Kyiv` -> `київ` -> `Київ` and restores a short
  Latin acronym (`NP`) the step folds to `np`; insertions with no raw counterpart keep the case the
  step produced. The divergence is recorded per case as `query_dense` (only when the two lanes
  differ) in `scores.jsonl` and the durability journal. Off by default so the folded dense query
  stays the explicit baseline -- but the measured verdict is to TURN IT ON with the step: cased
  dense text puts the `normalize` lane at or above the unmitigated lane on every noise class and at
  the clean ceiling on three of four, against a 0.0062 MRR give-back on one question ([evaluation
  rigor](../rigor-board-judge/robustness-benchmarks.md#dense-lane-casing-evidence)).
- `typos` -- deterministic corpus-vocabulary typo tolerance. The token vocabulary is built from
  the indexed corpus (`VocabularyContext.build` over `store.chunks`, whose `.tokens` is the same
  set `build_vocabulary` produces); a query token ABSENT from it is corrected to a nearby
  in-vocabulary token within Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars).
  Tokens shorter than three characters are protected; candidate matching cannot cross
  alphabetic/numeric kinds; a token the corpus already contains is NEVER altered; and a numeric
  token is never "corrected" into a different one. Every correction is logged. An
  opt-in morphology guard (morphology-aware-typo-guard; `RunConfig.query_prep_typo_guard`,
  `--query-prep-typo-guard`, `QUERY_PREP_TYPO_GUARD=1`) additionally skips any OOV token pymorphy3
  recognizes as a valid Ukrainian word form (`llb.rag.vector_store.lexical.load_uk_word_probe`): a grammatically
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
  correction, never add one); the morphology signal rides on the same opt-in probe as the guard,
  and the context index is built in the same pass as the vocabulary. How conservative each of them
  is -- the surface budget, the short-token cutoff, and which ranking signal goes first -- is a
  `RestorationPolicy` (`query_prep/restore_policy.py`) rather than three literals, so the constants
  can be swept, pinned with evidence, and overridden per corpus. The shipped values are the ones
  measured in [the sweep below](#restoration-constraint-sweep-restoration-constraint-threshold-sweep).
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
the `normalize` step is present), and the three restoration constants
`query_prep_surface_max_distance` (0), `query_prep_ambiguous_max_chars` (4) and
`query_prep_restore_rank` (`morphology` | `context`), all three refused unless the `typos` step is
present. A lane config that DROPS the `typos` step resets the three to their shipped values
(`RESTORATION_DEFAULTS` in `src/llb/core/config.py`), which is what lets one
`bench-query-robustness` run carry a swept setting into its `normalize,typos` lane while its clean
baseline and its `normalize` lane stay at the default; the run's report header states the setting
it ran.

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
make bench-query-robustness CONFIG=<yaml with the restoration constants> MODEL=<m> BACKEND=<b>
make sweep-restoration-constraints [GOLDSET=<gs> CORPUS=<dir> QUERY_PREP_DENSE_CASE=1 SWEEP_FULL_GRID=1]
```

The `validate-retrieval --query-prep-ab` A/B report scores `baseline` then each cumulative step
with per-step recall@k / MRR deltas. Every stage also records, per case, the query text it produced
plus `retrieval_hit` and `first_hit_rank` (`None` on a miss), so a pooled delta is attributable to
the items that actually moved -- a stage that recovers one item while pushing another down reads as
one averaged number without it. Model steps use `--query-prep-model` and
`--query-prep-backend`; their completions and parsed subqueries are embedded per case in the JSON
report. Endpoint generators cache a question within one cumulative run, avoiding duplicate model
calls while preserving fixed-temperature results.

Tests: `tests/llb/rag/query_prep/test_query_prep.py` (apostrophe and mixed-script repair, collision-safe
romanization, Latin acronym preservation, Damerau-Levenshtein transposition, typo correction that
never touches in-vocabulary, short, or cross-kind tokens + long-token distance 2 + deterministic
tie-break, deterministic alias expansion + glossary
build/round-trip, rewrite off-by-default, exact no-op when the lane is off, pipeline ordering +
dependency validation, A/B per-step delta and per-case hit/rank attribution over a fake store,
retrieve-node raw-preservation and
processed-query wiring, HyDE dense/lexical separation, decomposition parsing/bounds/RRF span
deduplication, runner resolver dependency wiring, provenance mapping including the ambiguous
same-replacement case, per-kind surface distance, refusal of an incompatible nearest neighbor,
restoration of the romanization-compatible form, the short-token length lock and the
transliteration exemption from it, unresolved-tie refusal, context-driven candidate choice, and
both morphology preferences),
`tests/llb/rag/query_prep/test_query_prep_restore_policy.py` (every swept setting's selection over the
committed candidate fixture `tests/fixtures/restoration_candidates.json`, that each fixture case is
decided by ONE constant, the policy reaching the typos step through the pipeline, the step
dependency, and the refused out-of-range value),
`tests/llb/eval/test_restoration_sweep_audit.py` (the one-factor and full grids, noisy/clean token
alignment and its refusal to judge a mismatched sequence, the correct/wrong/unaligned labels, a
refusal counted as a missed opportunity rather than a wrong edit, and summable counts),
`tests/llb/eval/test_restoration_sweep_run.py` (every setting measured on every class against
shared reference lanes over an injected store, determinism, the published bundle, and the
adopt/expose/pin decision on synthetic readings),
`tests/llb/rag/query_prep/test_query_prep_dense_case.py` (case-pattern
transfer without character replacement, the dense/lexical split reaching the store seam, the
recovered acronym, capitalization carried onto a corrected token, the exact no-op on an
already-lowercase query, glossary insertions left folded, subquery lanes untouched, and the
normalize-step dependency), `tests/llb/rag/vector_store/test_store.py` (split hybrid
queries), and `tests/llb/executor/test_durable_resume.py` (generated-query journal round trip),
plus config validation in `tests/llb/core/test_config.py`.

The end-to-end noise benchmark, model evidence, and model-specific default recommendation live in
[evaluation
rigor](../rigor-board-judge/robustness-benchmarks.md#ukrainian-query-robustness-benchmark). Its
noise classes are one mechanism each (`transliteration`, `apostrophe_variant`, `mixed_script`,
`keyboard_typos`), so what a mitigation lane recovers is attributable to the noise it inverts rather
than blended across two mechanisms at once.

### Restoration constraint sweep (restoration-constraint-threshold-sweep)

`llb sweep-restoration-constraints` / `make sweep-restoration-constraints` measures what the three
conservative constants above cost. For each setting it runs the `normalize,typos` lane (morphology
guard on) over the same seeded noise classes the robustness benchmark uses and reports RETRIEVAL
plus a per-edit precision audit; `$DATA_DIR/restoration-sweep/<run>/` holds `report.md`,
`settings.jsonl`, `edit_audit.jsonl`, and `metadata.json`.

It is retrieval-only and one factor at a time. Retrieval-only because the constants decide which
corpus surface a query token is rewritten to, which is a retrieval move -- and because the sweep's
reference lanes reproduce the full benchmark's retrieval cells exactly (below), a setting costs a
store pass instead of a model run. One factor at a time because a per-constant verdict has to be
attributable; `SWEEP_FULL_GRID=1` measures the product instead, and only one-factor settings carry
a verdict. Three lanes that never consult the constraints -- `clean`, `off`, `normalize` -- are
measured once and bound every setting, and retrieval is memoized per (dense, lexical) query pair:
two settings differ on a handful of tokens across a whole split, so the 1,312 lane-item retrievals
of a five-setting two-class run over 82 items collapse to 520 distinct store calls.

The audit is what separates "recovered the user's word" from "rewrote the question into something
the corpus happens to contain". Because every noisy query is generated from a clean one, aligning
the normalized noisy tokens with the normalized clean ones gives each correction a REFERENCE: it is
`correct` when it restored that token, `wrong` when it produced another, and `unaligned` when the
two token sequences do not correspond (the audit then refuses to judge rather than guessing). The
same alignment supplies the denominator retrieval cannot: an OPPORTUNITY is a token the noise made
out-of-vocabulary whose clean form the corpus does contain, so `restoration recall` is the share of
recoverable tokens the constraints actually recovered. Implementation:
`src/llb/eval/restoration_sweep/grid.py` (the grid and the per-setting pass),
`restoration_sweep/lanes.py` (paired readings +
retrieval cache), `restoration_sweep/audit.py` (alignment + labels), `restoration_sweep/verdict.py`
(the pin/adopt/expose rule), `restoration_sweep/report.py`, `restoration_sweep/run.py`, and
`src/llb/cli/eval/restoration_sweep.py`.

The verdict rule is stated once and applied to all three constants: **pin** when no alternative
retrieved more (the conservative default costs no recoverable recall), **adopt** when an
alternative separates on paired recall without raising the wrong-correction share (the default IS
costing recall), **expose** when it gains recall but does not separate at this item count or buys
the gain with more wrong corrections (a real but corpus-dependent trade, so it stays a knob).

CUDA-host evidence (2026-08-19): RTX 4060 Ti 16 GiB, `intfloat/multilingual-e5-base`, flat FAISS
over the committed `ua_squad_postedited_v1` corpus, the full final split (n=82), k=10, seed 13, 8
percent character noise, classes `transliteration` and `keyboard_typos` (the two the typo lane
exists to repair; the apostrophe and homoglyph classes are already fully inverted by `normalize` on
this encoder). Two runs: the folded lane and the recommended dense-cased lane.

The control first: the sweep's reference lanes and default setting reproduce the benchmark's
retrieval table cell for cell -- folded `off` 0.7195 / 0.9268, `normalize` 0.9634 / 0.9024,
`normalize,typos` 0.9634 / 0.9634; dense-cased `normalize` 0.9756 / 0.9268, `normalize,typos`
0.9756 / 0.9512 ([evaluation
rigor](../rigor-board-judge/robustness-benchmarks.md#dense-lane-casing-evidence)). Nothing about
the lane changed when generation was dropped.

Folded lane, pooled over both classes (164 paired readings):

| Setting | Recall@10 | Paired delta vs default | Corrections | Wrong | Wrong share | Restoration recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `surface=0,short=4,rank=morphology` (default) | 0.9634 | - | 289 | 29 | 0.1003 | 0.8442 |
| `surface=1,short=4,rank=morphology` | 0.9756 | +0.0122 `flat` (2/0/162) | 299 | 39 | 0.1304 | 0.8442 |
| `surface=0,short=3,rank=morphology` | 0.9634 | +0.0000 (0/0/164) | 290 | 30 | 0.1034 | 0.8442 |
| `surface=0,short=5,rank=morphology` | 0.9634 | +0.0000 (0/0/164) | 289 | 29 | 0.1003 | 0.8442 |
| `surface=0,short=4,rank=context` | 0.9634 | +0.0000 (0/0/164) | 289 | 29 | 0.1003 | 0.8442 |

Under the dense-cased lane every alternative is 0/0/164. The relaxed surface budget's entire folded
gain is ONE question counted once per class -- `570d4e6cb3d812140066d66d`, the untranslated ENGLISH
item the fold plus per-token transliteration mangles out of the top 10 -- and at budget 1 the typo
step "corrects" that mangled text into Ukrainian words that happen to retrieve the right chunk. The
dense-lane casing recovers the same question by not breaking it in the first place, which is why
the gain disappears there. The extra wrong corrections do not disappear with it.

The one setting the folded lane favored was then re-read end to end through
`bench-query-robustness` with the same model, seed, and lane: it retrieves the extra item and the
answers do not follow ([evaluation
rigor](../rigor-board-judge/robustness-benchmarks.md#relaxed-restoration-budget-end-to-end)).

**Verdict: pin all three; the knobs stay for a corpus that wants the trade.**

- `surface_max_distance` = 0 (`expose` folded, `pin` dense-cased). Relaxing to 1 buys one question
  in the folded lane and nothing at all in the recommended one, and both readings are `flat` -- while
  it makes ten more wrong corrections, eight of them in the `transliteration` class where the exact
  budget makes ZERO (`тге` -> `те` rewrites the English "the" into a Ukrainian word, `хугеноти` ->
  `гугеноти`, `правител` -> `правителі` where the user typed `правитель`). Exact surface
  compatibility is what buys that class's perfect precision, so it is pinned.
- `ambiguous_token_max_chars` = 4 (`pin`). A cutoff of 3 adds exactly one correction on this split
  and it is wrong (`типц` -> `тип` where the user typed `типу`); a cutoff of 5 changes nothing at
  all here, because no five-character token in this split has a length-changing candidate or an
  unresolved tie. The default is measured, and the 5 side of it is untested rather than confirmed.
- `rank_order` = `morphology` (`pin`). Context-first changes exactly two picks out of 289: one
  win (`чаму` -> `часу`, the typed word) and one loss (`сає` -> `сан` instead of `має`). Recall and
  the wrong count are identical; pooled MRR moves +0.0030, which is exactly those two picks. There
  is no evidence for reordering the signals.

Measured 2026-08-19, one run per casing lane (folded and dense-cased). Each recorded a
per-setting-per-class row plus a pooled row, and an edit audit carrying every correction with its
reference and label.

Reading the audit, all 29 of the default setting's wrong corrections are genuinely not the token
the user typed, so the automated label and a human reading agree on 29 of 29. The share splits in a
way the constants cannot fix: 19 of the 29 have a reference the corpus vocabulary does not contain,
so no correct choice existed and the honest alternative is refusal, not a better pick. Only 10 of
289 corrections (3.5 percent) picked a wrong surface when the right one was available.

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
