# Miss Analysis And Context Probes

Part of the [Evaluation rigor](../rigor-board-judge.md) area of the [current implementation index](../../current.md).

## Miss Analysis (analyze-misses)

`llb analyze-misses --run-dir <run>` (`make analyze-misses RUN_DIR=<run>`) explains a finalized
run's wrong answers. Classification, clustering, and recommendations live in
`src/llb/board/miss_analysis/`; probe orchestration lives in `src/llb/board/miss_probe.py`;
tests in `tests/llb/board/test_miss_analysis.py` (a synthetic scored bundle with one case per miss class
proves zero cross-class leakage and that every recommendation line names numeric evidence).

Every miss lands in exactly ONE class, decided in precedence order: `refusal` (typed status),
`format_artifact` (empty / malformed / timeout / backend_error -- output or transport, not
knowledge), `retrieval_miss` (typed status, or the gold span never overlaps a retrieved span),
`judge_disagreement` (objective below the miss threshold while the trusted per-case judge rated >=
0.7 -- a scoring conflict for a human to look at), else `generation_miss` (evidence present, answer
wrong). A scoreable case is a miss when `objective_score < 0.5` (`--miss-threshold` /
`MISS_THRESHOLD=` overrides). Span overlap reads the additive per-case `retrieval.jsonl` records
persist beside `scores.jsonl` (`batch_retrieval_records` in `src/llb/executor/cases.py`; doc id +
char offsets + rank + score + bounded 160-char text preview + the gold spans, plus the other places
of a duplicate-collapsed chunk -- see [the persisted retrieval
record](../rag-core/persistence-and-execution.md#the-persisted-retrieval-record)). When detailed
retrieval evidence is absent, classification uses the scored `retrieval_hit` flag and logs a
warning. Each miss row also lists `retrieved_docs`, the distinct documents its scored context
carried, so a retrieval miss can be read against the document the operator expected.

Misses are clustered by document (`source_doc_id`), topic, and question type, with per-key miss
rates computed over ALL scored cases of that key. Labels come from the goldset's
`item_provenance.jsonl` sidecar when the draft pipeline emitted one (`question_type` / `topic`);
otherwise a deterministic UA/EN interrogative heuristic types the question and the longest
content token stands in for the topic -- lemmatized through the hybrid-retrieval lemma normalizer
(`llb.rag.lexical.ukrainian_lemma`), so Ukrainian case forms of one topic land in a single cluster
instead of splitting across inflections. Recommendations are ranked by the miss count they
address and rendered from `board.miss.*` prompt templates: raise/lower `top_k`, change
chunking, add prompt-system dictionary terms for a dominant generation-miss cluster, try the
named alternative model (cited with its measured objective from comparable sibling bundles --
same split and case count), review refusals / artifacts / judge disagreements.

Probe mode (`--probe-top-k 3,8` / `PROBE_TOP_K=3,8`) re-runs ONLY the miss subset at each
alternative retrieval depth through the normal durable `run_eval` (same recorded config; only
`top_k` and `run_name` change, judge and telemetry off), so the retrieval hypothesis is
confirmed or rejected with measured recovery numbers, and a shallower depth that beats the miss
subset's baseline objective by >= 0.05 earns a "lower top_k" line. Probe bundles are ordinary
run bundles named `miss-probe-<run_id>-k<k>`: a finalized probe is reused (never re-run), an
interrupted probe's staging is found by its pinned config + goldset digests and resumed via the
durable-eval-runner journal, and only then does a fresh probe start. Off-cohort probe bundles
never pollute the board headline (tiny `n_cases` -> cohort exclusion).

Artifacts land at `$DATA_DIR/miss-analysis/<timestamp>/{report.md,misses.jsonl,analysis.json}`;
`llb recommend` appends a `## Miss analysis` section (intro + top 5 ranked lines) from the
latest `analysis.json` when one exists (`format_miss_section_md` in
`src/llb/board/recommend/sections.py`). Run bundles are never mutated. Automatic re-tuning stays
out of scope -- the Optuna tuner owns search.

## Context-Position Probe (probe-context-position)

`llb probe-context-position --model <m> --backend <b> --k <k>`
(`make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5`) measures a model's
lost-in-the-middle sensitivity and names its `context_order` recommendation with evidence
(rerank-context-order). Core in `src/llb/eval/position_probe.py`; CLI in
`src/llb/cli/eval/analysis.py`;
tests in `tests/llb/eval/test_position_probe.py` (a fake store + a fake chat that answers correctly only
when the gold chunk leads the prompt prove case construction, exact gold placement, per-position
scoring, the recommendation rule, and the artifacts -- no backend, no GPU).

Per verified gold item, ONE retrieval at `--candidate-depth` (default 50) supplies both the gold
chunk (the first candidate overlapping a gold span) and the k-1 best-ranked non-gold distractors
-- real retrieved distractors, never synthetic filler. Items whose gold chunk is not retrievable
or that lack k-1 distractors are counted per skip reason (`gold_not_retrieved` /
`too_few_distractors`), never invented. The gold chunk is then laid at the head, middle, and
tail of the fixed-k context (`k >= 3` enforced -- below that the slots collapse) and the same
question is asked three times through the standard RAG chat prompt. Each answer is
status-classified and scored by the objective correctness scorer against the reference answer.

The report gives per-position n / mean objective / bootstrap 95% CI and recommends `rank`
(best-first) when the head mean is at least the tail mean, else `reverse_rank` (best-last);
overlapping head/tail CIs are flagged as unresolved at that n (the recommendation still names
the higher mean, honestly qualified). Artifacts land at
`$DATA_DIR/context-position/<timestamp>/{report.md,cases.jsonl}`; probe cases never enter run
bundles, the board, or correctness aggregates.

Durable evidence (2026-07-10, rerank-order-full-cohort on the CUDA host, outside quick CI):
full-final-split probes (`ua_squad_postedited_v1`, 82 final items, k=5, no LIMIT cap) per
roster model on Ollama:

- `llama3.2:3b`: head 0.448 [0.360, 0.526], middle 0.419 [0.331, 0.498],
  tail 0.433 [0.351, 0.511] -- the mild best-first slope survives at n=82 but the head/tail CIs
  still overlap. Explicit verdict: NOT measurably position-sensitive (head-tail delta 0.015 well
  inside the CIs); the default `rank` ordering stands and no more n will plausibly resolve a
  gap this small into a knob worth setting.
- `gemma4:e4b`: head 0.414 [0.337, 0.493], middle 0.362 [0.291, 0.434],
  tail 0.407 [0.333, 0.482] -- the classic lost-in-the-middle U-shape (the middle slot pays
  ~-0.05 against both edges) but every pairwise CI overlaps at n=82. Explicit verdict: NOT
  measurably head/tail position-sensitive (`rank` stands); the middle dip suggests keeping
  `top_k` small enough that gold evidence never sits deep mid-context, which the shipped
  per-model `top_k` sweep already optimizes.
- `hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`: head 0.517 [0.427, 0.592],
  middle 0.507 [0.422, 0.584], tail 0.505 [0.423, 0.581] -- the flattest profile in the cohort
  (head-tail delta 0.012, all CIs overlap). Explicit verdict: NOT position-sensitive; `rank`
  stands, and the Ukrainian-specialized 12B is the most ordering-robust model probed.
- `hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`: head 0.528 [0.442, 0.606],
  middle 0.481 [0.401, 0.566], tail 0.485 [0.404, 0.566] -- the largest head advantage in the
  cohort (+0.043 over tail) with a mild middle dip, but the CIs still overlap at n=82. Explicit
  verdict: NOT measurably position-sensitive; `rank` (already best-first) captures whatever
  head preference exists, so no knob change is warranted.

Cohort verdict: no probed roster model resolves head/tail position sensitivity at the full
final-split n=82 -- the honest cohort-wide recommendation is that the default `rank` ordering
stands everywhere and `context_order` is not a knob worth per-model tuning on this goldset.
The rerank half of the cohort is recorded in [RAG core](../rag-core.md) Reranking And Context
Order.

### Roster-wide probe cohort (2026-07-24)

A second full-cohort pass on the same host, index, and item set (`ua_squad_postedited_v1`, 82
final items, k=5, Ollama, no LIMIT cap) extends the probe to the Gemma 4, MamayLM v2.0, and
Qwen3.6 rosters. All seven models probed 82/82 items with 0 skips and reliability 1.0.
Artifacts: `$DATA_DIR/context-position/20260724T0{63341,63726,64807,65031,70950,71850,73314}Z/`
(lapa, gemma4:e4b, gemma4:26b, gemma4:e2b, MamayLM-12B, Qwen3.6-35B-A3B, MamayLM-27B).

| model | head | middle | tail | overall | head-tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| `batiai/qwen3.6-35b:iq3` | 0.558 | 0.602 | 0.552 | 0.571 [0.52, 0.62] | +0.005 |
| MamayLM-Gemma-3-27B-IT v2.0 GGUF Q4_K_M | 0.583 | 0.549 | 0.561 | 0.565 [0.52, 0.61] | +0.022 |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.517 | 0.507 | 0.505 | 0.510 [0.46, 0.56] | +0.012 |
| Lapa v0.1.2-instruct GGUF Q4_K_M | 0.528 | 0.481 | 0.485 | 0.498 [0.45, 0.55] | +0.044 |
| `gemma4:e2b` | 0.469 | 0.425 | 0.441 | 0.445 [0.40, 0.49] | +0.028 |
| `gemma4:e4b` | 0.390 | 0.369 | 0.372 | 0.377 [0.34, 0.42] | +0.018 |
| `gemma4:26b` | 0.315 | 0.290 | 0.268 | 0.291 [0.27, 0.32] | +0.047 |

The 2026-07-10 cohort verdict holds unchanged: every head/tail CI still overlaps, so `rank`
stands for all seven and `context_order` remains a knob not worth per-model tuning.

Reproducibility, measured: Lapa and MamayLM-12B reproduced their 2026-07-10 numbers to three
decimals on every position. `gemma4:e4b` did NOT (head 0.414 -> 0.390, tail 0.407 -> 0.372) on
the same index and item set, so ~0.035 is that model's run-to-run noise floor and any e4b delta
below it is unresolvable. Gemma-3-derived GGUFs are bit-stable here; the Gemma 3n/E4B kernel
path is not.

**Position score is not needle-reading skill.** `objective_score` is token F1, which is
precision-sensitive, so it conflates whether the model FOUND the needle with how tersely it
stated it. Scoring the same rows with `contains` (all reference tokens present) separates them
and reverses the ranking:

| model | overall F1 | needle located | F1 given located | answer length vs reference |
| --- | ---: | ---: | ---: | ---: |
| `batiai/qwen3.6-35b:iq3` | 0.571 | 162/246 (0.659) | 0.787 | 1.9x |
| MamayLM-27B v2.0 | 0.565 | 155/246 (0.630) | 0.790 | 1.6x |
| MamayLM-12B v2.0 | 0.510 | 166/246 (0.675) | 0.685 | 2.3x |
| Lapa v0.1.2 | 0.498 | 154/246 (0.626) | 0.688 | 2.7x |
| `gemma4:e2b` | 0.445 | 184/246 (0.748) | 0.542 | 3.9x |
| `gemma4:e4b` | 0.377 | 193/246 (0.785) | 0.444 | 4.6x |
| `gemma4:26b` | 0.291 | 184/246 (0.748) | 0.332 | 5.2x |

The Gemma 4 collection ranks LAST on F1 and FIRST on found-rate: `gemma4:e4b` locates the
needle on 0.785 of probes against Lapa's 0.626 (paired, +0.159 [+0.061, +0.268]), yet loses on
F1 by 0.121 [0.052, 0.194] because it answers at 4.6x reference length. Within Gemma 4,
verbosity rises monotonically with size while F1-given-located falls in lockstep. Among the
Ukrainian-tuned and Qwen models the found-rate spread (0.626-0.675) is entirely inside the CIs,
so their F1 ordering is answer style, not comprehension. Read both columns before calling one
model a better context reader; reference answers on this fixture average 18 characters, which
is what makes token F1 this sensitive to padding.

Caveat on the Qwen row: the 16 GiB roster fallback is an IQ3 (3.5 bpw) artifact while both
MamayLM tags are Q4_K_M (~4.5 bpw), so Qwen ties the 27B from a weaker quantization. The
comparison is conservative in Qwen's favor, not like-for-like.

## Insufficient-Context Abstention Probe (run-eval --insufficient-context-probes)

`llb run-eval --insufficient-context-probes <n>` re-runs a seeded sample of gold items with their
gold evidence excluded from retrieval and scores abstention accuracy -- the share on which the
model correctly declines instead of fabricating an answer. Like the position probe, these probe
cases are scored on their OWN axis (`probes.jsonl` + `insufficient_context_report.md` in the run
bundle) and NEVER enter the correctness aggregates. It is part of the answer-side
groundedness/citation metrics; the mechanism, the deterministic groundedness + citation-validity
scorers (`--score-groundedness` / `--cited-answers`), and durable per-model evidence live in
[RAG core](../rag-core.md) groundedness and citation metrics.
