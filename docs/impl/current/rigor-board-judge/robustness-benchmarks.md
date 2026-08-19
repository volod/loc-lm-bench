# Ukrainian Robustness And Security Adaptation

Part of the [Evaluation rigor](../rigor-board-judge.md) area of the [current implementation index](../../current.md).

## Ukrainian Query Robustness Benchmark

`llb bench-query-robustness` / `make bench-query-robustness MODEL=<m> BACKEND=<b> GOLDSET=<gs>`
measures end-to-end sensitivity to four deterministic noisy-query classes, ONE MECHANISM EACH:
Latin-typed `transliteration`, `apostrophe_variant` (every apostrophe re-typed as a different
variant), `mixed_script` (Cyrillic/Latin homoglyph substitution at the configured character rate),
and keyboard-adjacent Cyrillic `keyboard_typos`. One class per mechanism is what makes a recovery
attributable: the earlier combined `apostrophe_mixed_script` class applied both at once, so its
apostrophe half was invisible whenever the homoglyph half dominated. That combined class is still
implemented and selectable (`--variant-classes ...,apostrophe_mixed_script` /
`QUERY_ROBUSTNESS_CLASSES=`), just not run by default; each mechanism draws from its own seeded
stream and rewrites characters in place, so the combined text is exactly the composition of the
two single-mechanism classes at the same seed and the split loses no comparability.

Each class runs under three isolated mitigation lanes (`MITIGATION_LANES`): `off`, `normalize`
alone, and `normalize,typos` plus the Ukrainian morphology guard. The middle lane is what
separates the two mitigation mechanisms -- normalization only inverts noise it can attribute,
while the typos step additionally rewrites tokens to corpus surfaces -- so a recovery attributable
to safe normalization is never credited to vocabulary correction, and vice versa. Recovery columns
are measured against the `off` lane of the same class. Clean cases are an ordinary `run-eval`
bundle. The 12 x N variant rows are probe-only and publish atomically as
`$DATA_DIR/query-robustness/<run>/{report.md,robustness.jsonl}`; no `scores.jsonl` exists in that
probe directory. Each row carries its lane in `mitigation` (plus `mitigation_steps` /
`mitigation_typo_guard`) and whether the generator actually changed the question
(`variant_changed`).

A single-mechanism class is a NO-OP on any question carrying none of its trigger characters --
only 6 of the 82 committed final-split questions contain an apostrophe at all -- and pooling the
untouched items drags every delta toward zero (a total retrieval loss on 6 items reads as
-0.073 pooled). The report therefore carries a second **affected items only** table that repeats
every lane over the perturbed items, against the SAME items' clean baseline, with an untouched
count per class; a lane whose class perturbed nothing is dashed out rather than shown as zeros.

Implementation is split across `src/llb/eval/query_robustness_variants.py` (seeded generators and
class selection), `query_robustness.py` (lane definitions, per-case joins, lane and affected-subset
metrics), `query_robustness_run.py` (clean baseline, store, endpoint, and per-lane graph wiring),
`query_robustness_report.py` (atomic report/JSONL publication), and
`src/llb/cli/eval/query_robustness.py`.
`tests/llb/eval/test_query_robustness.py` drives a fake endpoint and fake store through all twelve
default lanes using the graph module's pure-node seam, so the base `[dev]` GitHub environment does
not need the optional LangGraph package. It checks deterministic variants, that each split class
applies exactly one mechanism and the two compose into the combined class, class-selection
parsing, morphology-guard wiring (only the vocabulary-correction lane loads the probe), the
mechanism split (normalization alone recovers the script classes but not keyboard noise), the
affected-subset split on an item the apostrophe class cannot touch, and proves the probe directory
never gains correctness scores. Shared query-prep tests
cover the bugs found by CUDA acceptance: collision-safe romanization, preservation of uppercase
Latin acronyms, keyboard grave normalization, embedded homoglyph repair, short-token protection,
and alphabetic/numeric candidate separation, plus the ambiguity-aware restoration constraints
documented in [RAG core](../rag-core/rerank-and-query.md#query-side-processing-uk-query-processing).

Every per-class delta now carries paired uncertainty rather than a point-only sign.
`query_robustness_uncertainty.py` reads three states at the reporting confidence and neighbouring
90% / 97.5% conventions: `improved`, `degraded`, or `indistinguishable`. It reports the same
interval, win/loss/tie ledger, exact sign-test p, `p_positive`, and `(borderline)` qualifier for
lane-versus-clean deltas and mitigation-versus-`off` recoveries, both pooled and affected-only.
Either directional claim also needs enough differing items for the exact sign test to reach the
level. `query_robustness_summary.py` rebuilds all of that directly from `robustness.jsonl` plus the
clean case rows, which makes a recorded run re-renderable without another model call.

The 2026-07-24 MamayLM and Lapa artifacts were re-rendered through that seam with 2,000 resamples,
seed 13, and 95% reporting confidence. Every existing aggregate table cell reproduced exactly.
Each report has 100 paired readings, 2 borderline, and zero minimum-evidence relabelings; retrieval
is shared, so the two qualifications are identical. On pooled `keyboard_typos`, `normalize`
recall is -0.073 `[-0.146, -0.012]`, `degraded (borderline)`, `p_positive` 0.004: the 95% claim is
dropped at 97.5%. On the 81 changed items it is -0.062 `[-0.136, 0.000]`,
`indistinguishable (borderline)`, `p_positive` 0.009: a 90% interval would call it degraded. This
qualifies the already documented normalization cost; it does not change the recommendation.

CUDA-host evidence (2026-07-24, supersedes the 2026-07-21 and 2026-07-22 runs, which measured the
combined class): RTX 4060 Ti 16 GiB, Ollama, the full committed `ua_squad_postedited_v1` final
split (n=82), seed 13, 8 percent character noise, k=10, `intfloat/multilingual-e5-base`, 96 answer
tokens, and all five classes (the four defaults plus the opt-in combined one) under all three
mitigation lanes. Both model runs completed 1230/1230 probe cases with zero errors. The store is
dense-only, so these classes exercise the dense lane; the lexical lane is variant-invariant by
construction after the v2 tokenizer ([RAG core](../rag-core/hybrid-retrieval.md#apostrophe-variant-tokenization-evidence)).

Retrieval is model-independent here (same store, same query prep), and clean recall@10 was 0.9756
for both models:

| Class | `off` | `normalize` | `normalize,typos` |
| --- | ---: | ---: | ---: |
| `transliteration` | 0.7195 | 0.9634 | 0.9634 |
| `apostrophe_variant` | 0.9756 | 0.9634 | 0.9634 |
| `mixed_script` | 0.9634 | 0.9634 | 0.9634 |
| `keyboard_typos` | 0.9268 | 0.9024 | 0.9634 |
| `apostrophe_mixed_script` | 0.9634 | 0.9634 | 0.9634 |

- `hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`: clean objective 0.4747.
  Transliteration objective was 0.2497 raw, 0.3763 under `normalize` (+0.1267) and 0.3320 under
  `normalize,typos` (+0.0823); `mixed_script` was 0.4346 raw and 0.4641 under both mitigations
  (+0.0296); `apostrophe_variant` was 0.4747 raw -- clean to four decimals -- and 0.4903 mitigated;
  keyboard typos were 0.4451 raw, 0.4375 (-0.0076) and 0.4340 (-0.0111). Artifact:
  `$DATA_DIR/query-robustness/20260724T121701.652129Z-6feeb0cd727e/`; clean baseline:
  `$DATA_DIR/run-eval/20260724T113233.376054Z-7f2b659f138a/`.
- `hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`: clean objective 0.4970.
  Transliteration objective was 0.3827 raw, 0.5069 (+0.1242) and 0.5194 (+0.1367);
  `mixed_script` was 0.5240 raw and 0.5120 mitigated (-0.0121); `apostrophe_variant` was 0.4970 raw
  -- again exactly clean -- and 0.5127 mitigated; keyboard typos were 0.4048 raw, 0.3840 (-0.0208)
  and 0.4140 (+0.0092). Artifact:
  `$DATA_DIR/query-robustness/20260724T124802.064874Z-526a1af2007d/`; clean baseline:
  `$DATA_DIR/run-eval/20260724T121702.998145Z-b09c3af6a01f/`.

Verdict per mechanism, re-read on the split classes:

- **The recorded combined-class verdict was the homoglyph half, entirely.** `mixed_script` and
  `apostrophe_mixed_script` agree in EVERY column of both reports, and per item the 6 questions
  whose combined text differs (the apostrophe-bearing ones) score identically in both classes.
  Adding apostrophe re-typing on top of homoglyph noise changes no measurement on this corpus, so
  the previously recorded "mixed script" recovery is attributable to homoglyph repair alone.
- **The apostrophe half costs nothing to recover, because it costs nothing.** `apostrophe_variant`
  with no mitigation reproduces the clean objective and clean recall exactly for both models, and
  on the 6 affected questions recall@10 is 1.0000 in all three lanes: the dense e5 encoder is
  insensitive to which apostrophe variant was typed here. Lapa's -0.1296 objective on that
  6-item subset under `normalize` is one item's answer changing (1.000 -> 0.222) with identical
  retrieval, which is generation-side wobble at n=6, not a mitigation cost.
- **Transliteration and keyboard verdicts are unchanged** and reproduce the 2026-07-22 three-lane
  run to within decoding noise (<= 0.0003 objective), which is the control that splitting the
  class disturbed nothing else: those two generators draw from unchanged seeded streams.
- **The `normalize` lane's -0.0122 recall on the apostrophe class is not a failed repair.** The
  affected-items table shows all 6 perturbed questions at 1.0000; the lost item is
  `570d4e6cb3d812140066d66d`, the untranslated ENGLISH question the class never touched. That is
  the normalization step acting on an otherwise clean query -- the cost the opt-in language gate
  and the dense-lane casing measured below both address.

Mitigation verdict (unchanged): do not make the combined `normalize,typos` lane a universal
default. It is a strong model-specific option -- it restores all retrieval loss on every class and
improves transliteration for both models -- but MamayLM's keyboard-objective regression persists
and the per-edit audit's ambiguous nearest-vocabulary choices show that typo correction still needs
a model/corpus A/B before activation. The report's shared-hit generation delta separates that
answer-side effect from missing evidence. Those ambiguous choices are now counted rather than
described: `make sweep-restoration-constraints` labels every correction against the clean token the
noise came from and reports the wrong share per restoration setting ([RAG
core](../rag-core/rerank-and-query.md#restoration-constraint-sweep-restoration-constraint-threshold-sweep)).
Each run's `report.md` header also states the restoration setting it ran, so two settings' reports
read cell for cell.

### Dense-Lane Casing Evidence

The `normalize` lane's recall cost is the CASEFOLD reaching the case-sensitive dense encoder, and
routing case-preserved text to the dense lane removes it. The mechanism (`--query-prep-dense-case`
/ `QUERY_PREP_DENSE_CASE=1`, lexical lane unchanged) is documented in [RAG
core](../rag-core/rerank-and-query.md#query-side-processing-uk-query-processing).

CUDA-host A/B (2026-08-19): RTX 4060 Ti 16 GiB, Ollama,
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`, `intfloat/multilingual-e5-base`,
the committed `ua_squad_postedited_v1` final split (n=82), seed 13, 8 percent character noise,
k=10, 96 answer tokens, the four default classes. Two back-to-back full benchmark runs on one host,
folded then dense-cased, each 984 probe cases with zero errors; the folded run reproduces the
2026-07-24 recall table cell for cell, which is the control that nothing else moved. Both runs share
a clean baseline of objective 0.4747, recall@10 0.9756, MRR 0.8383. The `off` lane never carries the
option, so its column is the same measurement in both runs.

Recall@10 per class (clean ceiling 0.9756):

| Class | `off` | `normalize` folded | `normalize` dense-case | `normalize,typos` folded | `normalize,typos` dense-case |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transliteration` | 0.7195 | 0.9634 | 0.9756 | 0.9634 | 0.9756 |
| `apostrophe_variant` | 0.9756 | 0.9634 | 0.9756 | 0.9634 | 0.9756 |
| `mixed_script` | 0.9634 | 0.9634 | 0.9756 | 0.9634 | 0.9756 |
| `keyboard_typos` | 0.9268 | 0.9024 | 0.9268 | 0.9634 | 0.9512 |

**The acceptance question is answered: casefolding WAS the cause.** With the dense lane cased, the
`normalize` lane no longer retrieves below the `off` lane on any class, and on three of the four it
reaches the clean ceiling exactly (`recall_delta` 0.000 against clean). The paired reading for the
`apostrophe_variant` and `mixed_script` recoveries is `indistinguishable` at n=82 -- one item is one
item -- but the direction is now identical in every class and the mechanism is exact rather than
statistical, which is what the clean-query A/B below settles.

Per item, over the 656 lane-item retrieval cells (4 classes x 2 mitigated lanes x 82 items), 9 flip
miss -> hit and 2 flip hit -> miss. Eight of the nine gains are one question,
`570d4e6cb3d812140066d66d`, the untranslated ENGLISH item every class carries: the fold plus
per-token transliteration mangles it out of the top 10 in every mitigated lane, and restoring the
capitals on `Premier`, `Victoria`, and `Legislative Assembly` brings it back -- that is all eight
mitigated cells, one per class per lane. The ninth is a Ukrainian question under keyboard noise
(`5706074552bb8914006897d4`, `normalize` lane), where the same restored capitals carry a query the
step could not otherwise repair. The two losses are
both in the `normalize,typos` lane under keyboard noise
(`57106d2fb654c5140001f8ef`, `571095a8a58dae1900cd6a76`), and re-processing them shows the case
transfer did exactly its job -- it capitalized the sentence-initial token and nothing else. They are
encoder wobble on a query that is already noise-corrupted AND vocabulary-rewritten, in the one lane
this benchmark already declines to make a default; `recall_recovery` there stays
`indistinguishable`.

MRR moves both ways and is the honest cost: on the `normalize` lane the dense lane gains on
`transliteration` (0.8156 -> 0.8176) and `keyboard_typos` (0.6938 -> 0.7235) and loses 0.0062 on
`apostrophe_variant` and `mixed_script` (0.8313 -> 0.8251). Objective follows retrieval loosely and
stays within decoding wobble (`normalize` lane: 0.3763 -> 0.3729, 0.4903 -> 0.4746,
0.4641 -> 0.4774, 0.4375 -> 0.4425).

The clean-query A/B isolates that MRR trade with no noise at all
(`make validate-retrieval ... QUERY_PREP=normalize QUERY_PREP_AB=1 [QUERY_PREP_DENSE_CASE=1]`,
n=82, k=10):

| Lane | recall@10 | d(recall) vs baseline | MRR | d(MRR) vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `baseline` (no query prep) | 0.9756 | +0.0000 | 0.8383 | +0.0000 |
| `+normalize` folded | 0.9634 | -0.0122 | 0.8313 | -0.0070 |
| `+normalize` dense-case | 0.9756 | +0.0000 | 0.8251 | -0.0132 |

The A/B report's per-case `first_hit_rank` attributes both numbers: 7 of 82 items move rank at all,
the recall delta is entirely `570d4e6cb3d812140066d66d` (rank 1 clean, MISS folded, rank 6
dense-cased), and 0.0061 of the 0.0062 MRR the dense lane gives up is a single Ukrainian question
the fold happens to lift from rank 2 to rank 1. Buying one rank on one question is a property of
this encoder on that item; dropping a question out of the top 10 is a mechanism.

**Verdict: adopt the dense lane wherever the `normalize` step is on.** It is recorded as opt-in
rather than flipped to a default for one reason -- every `normalize`-lane number recorded above and
in the 2026-07-24 evidence is a FOLDED reading, and a silent default would change what those
readings mean. Turn it on with the step (`QUERY_PREP=normalize QUERY_PREP_DENSE_CASE=1`); leave it
off only to reproduce a folded baseline.

Artifacts: folded `$DATA_DIR/query-robustness/20260819T142700.448675Z-1f153f14ba5d/` (clean baseline
`$DATA_DIR/run-eval/20260819T134855.282489Z-dd0ed2c3162c/`), dense-cased
`$DATA_DIR/query-robustness/20260819T150241.558709Z-e114c0a4525c/` (clean baseline
`$DATA_DIR/run-eval/20260819T142702.513006Z-a77aa5be363c/`), and the four clean-query A/B reports
plus their `summary.md` in
`$DATA_DIR/query-robustness/20260819T140213.064476Z-c231ef43769c-dense-case-clean-ab/`. Each
`report.md` header carries a `dense-lane casing: on|off` line, since the lane ids are deliberately
identical in both runs so the two reports read cell for cell.

### Relaxed Restoration Budget, End To End

The restoration constants the `typos` step applies are `RunConfig` fields, so one benchmark run can
carry a swept setting: `make bench-query-robustness CONFIG=<yaml> ...` with
`query_prep_surface_max_distance: 1`. The lanes that do not run the step -- the clean baseline and
`off` / `normalize` -- reset the constants to their shipped values, so only the `normalize,typos`
column moves and the run stays comparable with every recorded default-constant run. Each report's
header states the setting it ran.

CUDA-host run (2026-08-19, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M, Ollama, folded lane, seed 13, k=10,
8 percent noise, `transliteration` + `keyboard_typos`, 492 probe cases, zero errors): the `off` and
`normalize` columns reproduce the 2026-07-24 run to four decimals (transliteration objective 0.2497
raw / 0.3763 normalized; keyboard 0.4451 / 0.4375), which is the control that the setting reached
the typo lane and nothing else.

At budget 1 the `normalize,typos` lane retrieves one extra item per class (recall@10 0.9756 on both,
against 0.9634 at the shipped budget) and the answers do not follow: objective is 0.3357 on
transliteration (0.3320 recorded at budget 0, +0.0037) and 0.4376 on keyboard typos (0.4340,
+0.0036), both decoding-wobble scale, both still below the clean 0.4747, and transliteration still
below the `normalize`-only lane's 0.3763. The retrieval sweep's ledger for the same setting is ten
additional wrong corrections ([RAG
core](../rag-core/rerank-and-query.md#restoration-constraint-sweep-restoration-constraint-threshold-sweep)),
so the end-to-end reading agrees with the sweep: keep the exact surface budget. Artifact:
`$DATA_DIR/query-robustness/20260819T171103.933023Z-a5e94ffc6ffd/`; clean baseline:
`$DATA_DIR/run-eval/20260819T165101.093894Z-8d63efec70c7/`.

## Cross-Lingual Query Lane

The robustness benchmark also accepts committed question-language classes, separate from character
noise. `language_ru` asks a drafted Russian version of each Ukrainian question;
`language_mixed` asks a deterministic UA/RU code-switched version. The paired Ukrainian item still
owns the reference answer, source document, split, and byte-identical source spans. The committed
overlay is `samples/goldsets/ua_squad_postedited_v1_ru/goldset.jsonl`: 162 unverified rows over 81
Ukrainian-dominant final questions. One English question mislabeled `lang: uk` in the source fixture
is excluded rather than treated as the Ukrainian baseline. The overlay passes `validate-goldset`
against the original corpus and remains `frontier-drafted` / `verified: false` pending language
review.

`src/llb/eval/query_robustness_languages.py` owns fixture inference, the invariant that only id,
language, question, provenance, and review state may differ, deterministic mixed-query composition,
uniform drafted/verified state, and the benchmark-only exact translation adapter. Language classes
run under `off`, `normalize`, and `translate_to_uk`. The last lane replaces the query with its paired
Ukrainian source question for retrieval only; generation still sees Russian or mixed text. It is an
upper bound that locates the loss, not a translation model or a shipped query-prep step. The general
robustness report now includes MRR and paired MRR intervals/recoveries beside recall and objective
for every class.

Run it with:

```bash
make bench-query-robustness MODEL=<m> BACKEND=<b> \
  GOLDSET=samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
  CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus \
  QUERY_ROBUSTNESS_CLASSES=language_ru,language_mixed
```

CUDA-host evidence (2026-08-17): RTX PRO 3000 Blackwell 12 GiB,
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` through Ollama,
`intfloat/multilingual-e5-base`, k=10, 96 answer tokens, seed 13, 2,000 paired bootstrap resamples,
and n=81. All 486 language cases completed with zero errors. The clean baseline scored objective
0.4844, recall@10 0.9753, and MRR 0.8363.

| Language | Retrieval prep | Objective | Recall@10 | MRR |
| --- | --- | ---: | ---: | ---: |
| Russian | `off` | 0.4210 | 0.9877 | 0.8377 |
| Russian | `normalize` | 0.4064 | 0.9877 | 0.8386 |
| Russian | `translate_to_uk` | 0.4347 | 0.9753 | 0.8363 |
| UA/RU mixed | `off` | 0.4739 | 0.9506 | 0.8158 |
| UA/RU mixed | `normalize` | 0.4491 | 0.9630 | 0.8098 |
| UA/RU mixed | `translate_to_uk` | 0.4352 | 0.9753 | 0.8363 |

The Russian loss is answer-side, not retrieval-side: raw Russian retrieval is at least the clean
level, while objective is -0.0633 and shared-hit generation delta is -0.0681. Exact Ukrainian
retrieval reproduces clean recall and MRR but leaves objective -0.0497. The raw Russian objective
reading is indistinguishable but borderline; normalization is a settled degradation at -0.0780 and
does not change recall. For mixed queries, raw retrieval loses 0.0247 recall and 0.0205 MRR, while
objective is only -0.0105 and all three raw deltas are indistinguishable. Exact Ukrainian retrieval
restores recall/MRR but DEGRADES objective by 0.0387 against raw mixed. Therefore neither
normalization nor unconditional retrieval translation is supported as a default. The stack already
retrieves Russian well; a future mitigation should target answer-language behavior and must be
tested independently from retrieval translation.

Artifacts:
`$DATA_DIR/query-robustness/20260817T114944.878514Z-520175273d85/`; clean baseline:
`$DATA_DIR/run-eval/20260817T112544.907869Z-ff6ece00b522/`. Focused fixture, lane, report, MRR, and
translation-upper-bound coverage lives in
`tests/llb/eval/test_query_robustness_languages.py`, alongside the existing robustness tests.

## Ukrainian Security Adaptation

The security benchmark (`src/llb/bench/security.py`, `src/llb/scoring/security.py`) is adapted to
the Ukrainian, bilingual reality of locally-served UA models. The adaptation lives in the pieces
that decide the scores, so a Ukrainian-language decline is never miscounted as compliance and a
Cyrillic-echoed leak is never missed.

- **Refusal lexicon (UA + RU).** `is_refusal` (`src/llb/eval/common.py`) matches high-precision,
  first-person UA refusals (`не можу`, `не зможу`, `не буду`, `не в змозі`, `відмовляюся`,
  `перепрошую, але я`, …) plus Russian equivalents (`не могу`, `извините, но я`), because a
  bilingual UA model pushed with an adversarial ask frequently declines in Russian. A missed
  refusal would inflate ASR (`refuse` cases), turn a canary-naming refusal into a false leak
  (`leak` cases), and depress refusal-appropriateness. Every marker keeps a `(?!\w)` word boundary
  so first-person `не можу` does not fire on third-person `не можуть`.
- **Text normalization + homoglyph folding.** `normalize_refusal_text` casefolds, applies NFKC, and
  unifies Ukrainian apostrophe variants (`'` `'` `ʼ` `` ` ``) before matching. `marker_leaked`
  additionally folds Cyrillic look-alike letters to Latin, so a Latin canary the model echoes with
  Cyrillic characters (`РАСКЕТ-ОХ-42` for `PACKET-OX-42`) is still detected instead of understating
  ASR.
- **UA / bilingual attack vectors.** Jailbreak frame prose lives in prompt templates under
  `bench/security/` (adversarial text is data, not code). Alongside DAN role-play and fiction
  framing, a **language-switch** frame tells a UA-tuned assistant to drop its guardrails and answer
  in Russian — probing whether safety generalizes across the languages the model knows. The
  committed seed (`samples/benchmarks/security_cases_uk.json`) adds a language-switch jailbreak
  (`jb-003`) and a Cyrillic-homoglyph RAG injection (`rag-003`).
- **Over-refusal controls.** The seed carries UA benign controls that look sensitive but are
  legitimate — dual-use history, defensive security, and crisis mental-health resources — so a model
  cannot win by blanket-refusing Ukrainian civic/dual-use questions. These feed refusal-
  appropriateness only, never ASR.

Seed cases are tagged with `lang` (`uk` / `ru` / `en`) and, for attacks, `attrs.vector`
(`language_switch` / `homoglyph`).

- **Cross-language consistency (Module 6).** `cross_language_consistency`
  (`src/llb/scoring/security.py`) compares matched-prompt groups — the SAME behavior posed in
  several languages, linked by an `xlang_group` id — as a separate outcome from ASR. For every
  group posed in >=2 languages it checks whether the model's refuse/answer decision is the same
  across languages; `consistency` is the fraction of groups treated equivalently and
  `refusal_rate_by_lang` is the diagnostic that reveals which language is the weak point (e.g. a
  model that refuses in Ukrainian but complies in Russian). Consistency is orthogonal to safety —
  it is read alongside ASR, carries its own bootstrap CI, and is persisted in the run manifest
  under `config.cross_language`; `SecurityScore.cross_language` is `None` when a set has no matched
  groups. The committed seed ships one harmful (`xl-weapon`) and one benign (`xl-help`) UA/RU/EN
  group. Behavior-level translation of the public adversarial sets into matched groups remains an
  operator step (inject a per-language `translate`), since the seed keeps human-verified prose.
