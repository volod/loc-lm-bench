# Hybrid Retrieval (Dense + BM25 + RRF)

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

Shipped (hybrid-retrieval-uk): retrieval has the full hybrid shape Ukrainian enterprise corpora
need -- dense E5 plus lexical BM25 fused with weighted reciprocal-rank fusion, plus a
chunk-metadata filter seam -- so exact surnames, article/law numbers, codes, and abbreviations
stop losing to semantic-only search.

Modules:

- `src/llb/rag/vector_store/lexical.py` -- pure-Python BM25 (`LexicalIndex`, in-repo)
  over the SAME offset-exact chunks the vector index holds; Ukrainian-aware token normalization
  on the LEXICAL side only (casefold, apostrophe-variant unification U+2019/U+02BC/`` ` ``/`'`,
  punctuation strip). Every apostrophe variant is an IN-WORD character for the token regex, which
  is derived from the same variant list (`_APOSTROPHES`): a converted PDF writes `зобов’язання`
  with U+2019, and a regex that admitted only U+0027 split that into `зобов` + `язання` -- two
  half-words that no later unification can rejoin. Edge apostrophes of any variant are stripped
  after unification, so `'слово'`, `` `слово` ``, and `‘слово’` all index as the bare term;
  opt-in lemmatization via the base dependencies `pymorphy3` +
  `pymorphy3-dicts-uk`, collapsing cases/inflection to lemmas at index AND query time -- the stored
  chunk text stays byte-identical (unit-tested); `rrf_fuse` implements the weighted RRF
  (`score = w/(60+dense_rank) + (1-w)/(60+lexical_rank)`) with deterministic tie-breaks.
  Its generalized `weighted_rrf_fuse` accepts n ranked lists and non-negative weights. A
  zero-weight lane contributes neither score nor candidate membership, fixing endpoint weights
  that previously appended disabled-lane candidates when the active lane returned fewer than k.
- `src/llb/rag/filters.py` -- the chunk-metadata filter seam: `metadata_filter(doc_ids,
  heading_contains, page_range, acl_label)` builds a predicate over `doc_id` plus the
  page-metadata join's `metadata.headers` breadcrumb, `metadata.pages` range, and governance
  `metadata.acl_label`; `RagStore.retrieve(question, k, chunk_filter=...)` applies it BEFORE
  fusion/ranking (with a filter the whole index is scanned, so the cut is exact).
- `src/llb/rag/vector_store/store.py` -- `mode="hybrid"` builds the lexical index beside the vector index;
  fusion runs inside `RagStore.retrieve`, so every dense `VectorIndex` backend
  (FAISS/Chroma/Qdrant/LanceDB) gains hybrid identically. The lexical index persists as
  `lexical_index.json` beside the FAISS artifacts and joins `store_meta.json`
  (`meta.lexical = {lemmatize, n_terms}`). Loading a hybrid store whose lexical file is missing
  refuses with a rebuild message, and `run-eval --retrieval-mode hybrid` over a dense-only store
  refuses too (`_load_store`); a non-hybrid config over a hybrid store serves dense-only.
  The persisted file carries `LEXICAL_INDEX_VERSION`, which is the TOKENIZER generation as much as
  the format: postings are tokenizer output, and the query is tokenized by whichever build reads
  them, so `LexicalIndex.load` refuses a version it did not write and names the rebuild command
  (current: `bm25-uk-v2`; `v1` predates apostrophe variants becoming in-word). Bump it whenever
  tokenization changes.

Knobs (all `RunConfig` fields, hence in the manifest and the sweep cell fingerprint):
`retrieval_mode=hybrid`, `fusion_weight` (dense share of the RRF, default 0.5; 1.0 == dense
order, 0.0 == lexical order), `fusion_candidates` (per-side candidate depth, default 50), and
`lexical_lemmas` (index-time lemmatization, recorded in the store meta).

Commands:

```bash
make build-index RETRIEVAL_MODE=hybrid LEMMATIZE=1    # build-index --retrieval-mode hybrid --lemmatize
make run-eval MODEL=<m> RETRIEVAL_MODE=hybrid FUSION_WEIGHT=0.5
make compare-retrieval HYBRID=1 GOLDSET=<goldset.jsonl>
make sweep SWEEP_RAG_GRID="top_k=3,5;fusion_weight=0.4,0.6"
llb tune ...    # the Optuna space samples retrieval_mode=hybrid + both fusion knobs
```

`compare-retrieval --hybrid` embeds the corpus ONCE and scores five rows sharing that dense
index: `dense`, `lexical` (BM25 alone), `hybrid` (BM25 + weighted RRF), `hybrid+lemmas` (a second,
lemmatized lexical index), and `dense+oracle-doc` -- a diagnostic row restricting candidates to
each gold item's `source_doc_id` through the filter seam, quantifying the recall headroom a
PERFECT document router would buy (never a scoring config).

The `lexical` row is the same hybrid store queried at fusion weight 0, which `weighted_rrf_fuse`
resolves to an exact lexical passthrough (a zero-weight lane is dropped from candidate membership
too). It exists because a LEXICAL-side change -- tokenizer, lemmatization, normalization -- is
invisible in the fused row whenever the dense lane already retrieves the item: the mixed-variant
evidence below measured a lexical lane at half recall while `dense` and the fused row read 1.000
and 0.650, and only the isolated row said which lane was broken. It is a diagnostic like the
oracle row, not a shipping configuration.

The lemma normalizer is reused by the miss analysis: `topic_of` in
`src/llb/board/miss_analysis/classify.py` lemmatizes its heuristic topic key, so Ukrainian case
forms of one topic collapse into a single cluster instead of splitting across inflections.

Fixture: `samples/goldsets/exact_terms_uk/` -- a 40-entry near-identical Ukrainian orders registry
(order numbers, DSTU codes, surnames, amounts; ~41 recursive chunks) whose 8 items ask for exact
terms; the CI regression (`tests/llb/rag/vector_store/test_hybrid_store.py`) proves hybrid strictly
beats a signal-free dense ranking there. Tests: `tests/llb/rag/vector_store/test_lexical.py`
(normalization, BM25 determinism and tie-breaks, lemma matching, save/load),
`tests/llb/rag/test_filters.py` (doc/heading/page/ACL predicates),
`tests/llb/rag/vector_store/test_hybrid_store.py` (fusion order, weight extremes,
filter-before-fusion, refusal paths, config-knob application, byte-identical text), plus grid/tuner
coverage in `tests/llb/cli/test_cli_models.py` / `tests/llb/optimize/test_tuner.py`.

Durable evidence (2026-07-08, real e5-base stores on the dev host, outside quick CI), via
`compare-retrieval --hybrid`; the `lexical` column was added by the 2026-07-24 re-read below,
which reproduced every other cell to four decimals:

- `samples/goldsets/ip_regulation_uk` (8 items, saturated fixture), k=10 and k=3: all five rows
  hold recall 1.000 / MRR 1.000 -- hybrid is equal-or-better than dense on the committed goldset
  (the gate), and the fixture is too small to discriminate further. The `lexical` row now shows
  the saturation is not a fusion artifact: BM25 alone also scores 1.000 / 1.000.
- `samples/goldsets/exact_terms_uk` (8 exact-term items), k=10: recall ties at 1.000 but hybrid
  MRR 0.938 vs dense 0.713 and `lexical` 0.781; at k=3 hybrid holds recall 1.000 / MRR 0.938
  while dense is 0.875 / 0.688 and `lexical` 0.875 / 0.750 -- the strict exact-term win the
  lexical side exists for. The isolated row sharpens what that win IS: the fused row beats BOTH
  lanes alone at both cutoffs, and at k=3 it retrieves an item NEITHER lane ranked in its own top
  3. Hybrid is buying complementarity here, not standing in for a weak dense lane.
  `hybrid+lemmas` matched plain `hybrid` on both fixtures (exact numbers do not inflect). The
  oracle-doc row equals dense on these single-document corpora by construction (a doc filter is a
  no-op with one doc).

Durable evidence, full corpus (2026-07-10, hybrid-comparison-full-corpus on the CUDA host,
outside quick CI): dense vs hybrid over the verified 44-item quickstart-PDF accepted goldset
(5 documents, 1139 chunks, inflection-rich Ukrainian questions; k=10), with the `fusion_weight`
gridded across three runs:

| row | recall@10 | MRR |
| --- | ---: | ---: |
| dense | **0.955** | 0.740 |
| dense+oracle-doc (headroom) | 0.977 | 0.753 |
| hybrid w=0.5 (default) | 0.932 | 0.742 |
| hybrid w=0.6 | 0.932 | 0.750 |
| hybrid w=0.7 | **0.955** | 0.748 |
| hybrid+lemmas w=0.5 | 0.932 | **0.762** |
| hybrid+lemmas w=0.6 | 0.932 | 0.759 |
| hybrid+lemmas w=0.7 | 0.932 | 0.753 |

Fusion-knob verdict as recorded for this corpus: dense-only STAYS the default -- at the 0.5
default the BM25 side actively costs recall (-0.023), and only a dense-heavy `fusion_weight=0.7`
climbs back to the dense recall while adding a small MRR gain (+0.008). The measured lemmatization
delta on an inflection-rich corpus is a real but MRR-only effect: +0.020 MRR at w=0.5 with
recall unchanged (the tiny-fixture zero was a corpus artifact, as predicted). The oracle-doc
router headroom row is finally non-degenerate on this multi-document corpus: perfect document
routing would buy +0.022 recall / +0.013 MRR -- modest, so a learned router stays unattractive
here.

**The `FUSION_WEIGHT=0.7` pin from that verdict is WITHDRAWN** -- see the re-read below.

## Lexical-row re-read of the fusion-weight verdict

Re-read (2026-07-24, CUDA host, `compare-retrieval --hybrid --noise-floor`, reports under
`$DATA_DIR/lexical-row-reread/`) once the `lexical` row existed. Both committed fixtures
reproduced every recorded cell exactly, so the harness is unchanged; the full-corpus table above
could NOT be reproduced, because its verified 44-item quickstart-PDF accepted goldset is no
longer on disk. Two item sets that ARE on disk were run instead: the SAME 5-document goods corpus
at the same `recursive` 800/120 chunking with its 95-item drafted goldset, and a human-accepted
40-item PDF goldset over a single-document corpus.

| corpus / item set | `dense` | `lexical` | `hybrid` w=0.5 | `hybrid+lemmas` w=0.5 | floor (r / MRR) |
| --- | --- | --- | --- | --- | --- |
| goods, 95 drafted (n=95) | 0.674 / 0.409 | 0.621 / 0.435 | 0.695 / 0.449 | **0.726** / **0.463** | 0.000 / 0.008 |
| PDF, 40 accepted (n=40) | 0.925 / 0.852 | 0.875 / 0.790 | 0.925 / 0.869 | 0.925 / **0.906** | 0.000 / 0.006 |

On the goods corpus the weight sweep runs `hybrid` 0.695 / 0.695 / 0.705 recall and
`hybrid+lemmas` 0.726 / 0.726 / 0.716 at w=0.5 / 0.6 / 0.7 -- the dense-heavy weight the recorded
verdict recommended is the WORST of the three for the best row.

What the re-read changes:

- **The premise of the recorded verdict does not hold on either available item set.** "The BM25
  side actively costs recall at w=0.5" was the reason to pin 0.7; here fusion ADDS recall on the
  goods corpus (+0.021 hybrid, +0.053 with lemmas, against a +/-0.000 recall floor) and ties it on
  the accepted set while adding +0.054 MRR (floor +/-0.006). The pin is withdrawn: on both sets
  `w=0.5` is at least as good as `w=0.7`, and `hybrid+lemmas` is the best row everywhere.
- **The `lexical` row is why that premise was risky to state.** BM25 alone retrieves 0.621 on the
  goods corpus against dense's 0.674 -- with a HIGHER MRR (0.435 vs 0.409). The lexical lane is
  not the weak lane; whether fusing it pays is a property of the ITEM SET, not of lane strength,
  and a fused number alone cannot tell those apart. That is the reading the row exists for.
- **What does NOT change:** the shipped defaults. Hybrid stays opt-in (`retrieval_mode=flat`) and
  `fusion_weight` keeps its 0.5 default, so no configuration ships differently -- only the advice
  to pin 0.7 is retracted.
- **What stays unsettled:** the recorded table's item set was human-accepted and the goods item
  set is drafted, so this re-read cannot say the recorded numbers were wrong -- only that nothing
  on hand reproduces them. Settling it needs an accepted ledger over the goods corpus; that is the
  forward `goods-fusion-weight-accepted-ledger` task.

## Paired re-read of the fusion-weight verdict

CUDA-host re-read (2026-07-28), pinned e5-base, k=10, `recursive` 800/120, 2000 paired
resamples, 95% confidence, seed 13, and `NOISE_FLOOR=1`. The goods drafted set (n=95) and PDF
accepted set (n=40) are the same two available item sets used by the lexical-row re-read above.
Every point estimate reproduced exactly. Reports, configs, stores, and per-item vectors are under
`$DATA_DIR/retrieval-comparison-paired-uncertainty/{goods,pdf}-hybrid*/`.

The table gives the point-estimate winning deployable row at each weight against the named
`dense` baseline. "Reading" is the calibrated raw paired reading; the final verdict also applies
the deployable lane x metric family adjustment.

| corpus | weight | winning row | recall delta vs dense (95% interval; w/l/t) | recall reading | MRR delta vs dense (95% interval; w/l/t) | MRR reading | verdict |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| goods drafted | 0.5 | `hybrid+lemmas` | +0.053 [-0.011, +0.116]; 8/3/84 | flat | +0.054 [-0.001, +0.111]; 28/11/56 | separated raw; adjusted p=0.089 | retain `dense` |
| goods drafted | 0.6 | `hybrid+lemmas` | +0.053 [+0.000, +0.116]; 7/2/86 | flat | +0.046 [-0.005, +0.098]; 28/11/56 | flat | retain `dense` |
| goods drafted | 0.7 | `hybrid+lemmas` | +0.042 [-0.011, +0.105]; 6/2/87 | flat | +0.026 [-0.013, +0.067]; 23/10/62 | flat | retain `dense` |
| PDF accepted | 0.5 | `hybrid+lemmas` | +0.000 [+0.000, +0.000]; 0/0/40 | flat | +0.054 [-0.019, +0.137]; 4/1/35 | flat | retain `dense` |
| PDF accepted | 0.7 | `hybrid+lemmas` | +0.000 [+0.000, +0.000]; 0/0/40 | flat | +0.044 [-0.014, +0.115]; 4/1/35 | flat | retain `dense` |

The direct weight comparison reconstructed from those persisted item vectors is flat everywhere.
On goods, w=0.5 minus w=0.7 for `hybrid+lemmas` is +0.011 recall
[-0.021, +0.053] (2/1/92) and +0.028 MRR [-0.002, +0.059] (15/7/73). On the accepted PDF set
recall is itemwise identical and the MRR delta is +0.010 [-0.019, +0.050] (1/1/38). Therefore no
available item set supports a weight preference under paired sampling: `fusion_weight=0.5`
remains the retained default, while hybrid remains opt-in. The earlier measurement-floor claim
that fusion "adds" recall on goods is still the exact point estimate, but it is not a
sampling-separated gain.

## Apostrophe-variant tokenization evidence

Durable evidence (2026-07-24, CUDA host, pinned e5-base, k=10; stores and reports under
`$DATA_DIR/apostrophe-normalizer/`) for making every apostrophe variant an in-word character.
Two corpora, each scored with the v1 tokenizer (unification after tokenizing) and the shipped v2
one over the SAME chunks and the SAME dense index, so only the lexical side differs:

| corpus | items | apostrophe questions | lexical terms v1 -> v2 | recall@10 (dense+BM25) v1 -> v2 | lexical-only recall@10 v1 -> v2 |
| --- | ---: | ---: | --- | --- | --- |
| goods PDFs (U+2019 corpus) | 95 | 2 | 4274 -> 4268 | 0.6842 -> 0.6842 | 0.5053 -> 0.5053 |
| UA SQuAD fixture (U+0027 corpus) | 250 | 14 | 8276 -> 8276 | 0.9640 -> 0.9640 | 0.8600 -> 0.8600 |

**No retrieval metric moved on either corpus**, with or without the dense lane, and the same holds
when every question is re-typed with a different apostrophe variant (the committed
`apostrophe_variant` noise generator, which rewrites apostrophes and nothing else). That is the
honest headline: this is a correctness fix, not a recall win, and no recorded retrieval verdict
changes because of it.

What DID move, measured:

- **Vocabulary.** On the U+2019 corpus, 62 fragment terms (`зв`, `зобов`, `комп`, `обов`, `пам`,
  `дистриб`, ...) disappeared and 56 whole-word terms took their place; apostrophe-bearing index
  terms went from 1 to 57. The SQuAD corpus writes only U+0027, so its index is unchanged -- which
  is the check that the change is a no-op where a corpus never used another variant.
- **Variant invariance on the query side.** Across the SQuAD questions the lexical lane sees 16
  apostrophe-bearing query terms under v2 no matter which variant was typed; under v1 that count
  collapsed from 14 (as typed) to 2 (re-typed with another variant), i.e. 12 words silently
  shattered into fragments. Of the 8 that the corpus can actually match, a variant-mismatched
  query kept 1 under v1 and keeps all 8 under v2.
- **A dead query term became live.** `з'явиться` typed with the keyboard apostrophe returned ZERO
  BM25 candidates against the U+2019 goods corpus under v1 (its postings held `з` + `явиться`) and
  returns hits under v2.
- **Duplicate-collapse `normalized` tier.** It now merges an apostrophe-variant copy of a repeated
  block (the committed near-duplicate fixture's `Застереження` group: 2 -> 3 copies). On the goods
  corpus the measured residue is unchanged (26 / 311 collapsible for `recursive`, 55 / 357 for
  `sentence`) -- its repeated blocks use one variant consistently.
- **Corpus-conflict `hash` tier.** Document-level duplicate yield is unchanged on all three
  measured corpora (goods 0, near-duplicate fixture 0, `conflicts_uk_v1` 1 group): no whole-document
  near-duplicate in them hinges on an apostrophe.

Why the metrics cannot see it: only 2 of 95 goods questions and 14 of 250 SQuAD questions contain
an apostrophe at all, both corpora are internally consistent about which variant they use, and at
k=10 the other query terms already retrieve those items.

The end-to-end query side of the same question is now measured as its own noise class rather than
inferred: `apostrophe_variant` re-types every apostrophe in the question and touches nothing else
([evaluation
rigor](../rigor-board-judge/robustness-benchmarks.md#ukrainian-query-robustness-benchmark)). On the
SQuAD final split all 6 apostrophe-bearing questions still retrieve their gold evidence at k=10 with
an unmitigated re-typed apostrophe, so the dense e5 lane is variant-insensitive here and there is
nothing for normalization to recover. Post-v2 the lexical lane cannot see the class either --
`llb.rag.vector_store.lexical.tokenize` unifies every variant, so a re-typed query and the index
produce the same terms by construction, which IS the fix.

## What the fix is worth when the corpus MIXES variants

The case the two real corpora cannot pose is the MISMATCH: index and query disagreeing about which
apostrophe was typed, which is what a re-ingested edition, a pasted appendix, or two converters
produce. `samples/goldsets/apostrophe_variants_uk/` plants exactly that -- 60 near-identical
Ukrainian registry entries across four documents "converted" with four different apostrophes, one
apostrophe-bearing subject noun as the ONLY discriminating token per entry, and every question
typed with the keyboard apostrophe (45 of 60 items therefore face a mismatch; 15 are the
same-variant control). Its README states the plant.

Durable evidence (2026-07-24, CUDA host, pinned e5-base, `recursive` 800/120, k=10, exact
duplicate collapse 180 chunks -> 80 indexed, `--noise-floor`; reports under
`$DATA_DIR/apostrophe-normalizer/mixed-{before,after}/`). Both arms ran the SAME command over the
SAME corpus, goldset, and seed; the `before` arm ran from a worktree whose ONLY difference is the
pre-fix `src/llb/rag/vector_store/lexical.py`:

| row | v1 recall@10 / MRR | v2 recall@10 / MRR | delta recall@10 |
| --- | --- | --- | ---: |
| `dense` | 1.000 / 0.958 | 1.000 / 0.958 | +0.000 |
| `lexical` | 0.500 / 0.167 | 1.000 / 0.200 | **+0.500** |
| `hybrid` | 0.650 / 0.520 | 1.000 / 0.983 | **+0.350** |
| `hybrid+lemmas` | 0.650 / 0.520 | 1.000 / 0.975 | +0.350 |
| `dense+oracle-doc` | 1.000 / 0.967 | 1.000 / 0.967 | +0.000 |

The measurement floor is +/-0.000 recall@10 in both arms, so every delta above clears it by two
orders of magnitude. Three readings:

- **The lexical lane loses exactly the entries written with a punctuation-class variant.** Per
  variant (measured by `tests/llb/rag/vector_store/test_apostrophe_variant_fixture.py`, which
  reproduces the heavy run's 0.500 exactly): U+0027 15/15 and U+02BC 15/15 under BOTH tokenizers,
  U+2019 0/23 and grave 0/7 under v1 and 23/23 + 7/7 under v2. The v1 misses are TOTAL -- the
  subject term returns zero BM25 candidates, and boilerplate ties recover none of them.
- **U+02BC never needed the fix.** It is a Unicode modifier LETTER, so `\w` already kept
  `памʼятка` whole; only U+2019 and the grave accent are punctuation to the regex and split the
  word. The fix's value is variant-specific, which a single "apostrophe support" claim would hide.
- **A broken lexical lane is worse than no lexical lane.** Under v1 `hybrid` retrieves 0.650 --
  0.350 BELOW the `dense` row it fuses with -- because RRF gives half the rank budget to a lane
  whose candidates are unrelated. The fix does not merely restore the lexical lane, it turns
  hybrid from harmful (-0.350 versus dense) into the best row (+0.025 MRR over dense).

Scope of the claim: the effect size is a property of the PLANT (one discriminating token per
entry, 75 percent of items mismatched). It is a lower bound on nothing and an estimate of nothing;
what it demonstrates is the mechanism and its direction, which the flat real-corpus numbers above
cannot. On the real corpora the dense lane also retrieves every planted item at k=10, so an
operator whose corpus mixes variants would see the breakage ONLY in the `lexical` row -- the
reason that row is now published.
