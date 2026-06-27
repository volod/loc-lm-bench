# data prep Current State

## data prep -- modules + how to run

### Canonical gold-item schema -- `llb.goldset.schema`
Pydantic `GoldItem` + `SourceSpan`. Fields: `id, lang, question, reference_answer,
source_doc_id, source_spans[{doc_id, char_start, char_end, text}], provenance, verified,
split`. Labels are SOURCE-SPAN (char offsets, not chunk ids), so they survive `chunk_size`
tuning. `provenance` and `split` are enforced literals. Only `verified: true` items score
models. Verification may be a local review or acceptance of a pinned upstream post-edited
fixture; `provenance` and fixture metadata preserve the distinction. `load_goldset` /
`dump_goldset` handle JSONL (UTF-8).

### Splits -- `llb.goldset.splits`
`assign_splits(ids, ratios, seed)` -> deterministic, disjoint `calibration / tuning / final`.

### Validator (data bootstrap acceptance) -- `llb.goldset.validate`
Checks every span resolves to its labeled text on disk, ids unique, splits disjoint.

    make validate-goldset          # PASS on the committed public fixture

### Sample generator -- `llb.prep.gen_rag_items`
Reads `samples/rag_items_uk.json`, computes spans, writes + validates a six-item synthetic
format fixture. It remains useful for parser and tiny smoke checks but is no longer the
default demo gold set.

    make gen-rag-items # -> .data/llb/goldset/sample_rag_items.jsonl (6 items)

### SQuAD ingestion (Ukrainian SQuAD ingest) -- `llb.prep.ingest_squad`
Maps SQuAD-format UA QA (flattened, nested, or HF rows where `answers` is a dict-string) ->
canonical items, with spans from the answer offset and a `find()` fallback. Drafts start with
`provenance: public-reused`, `verified: false`. The default ID-keyed verification ledger then
adopts matching canonical items from `ua_squad_postedited_v1`, including their reviewed corpus
files; unmatched drafts remain false. Local file or HF dataset (streams when `--max-items` set).
The HF loader accepts an explicit revision and normalizes both flattened rows and the pinned
source's nested SQuAD article rows.

    make ingest-uk-squad GOLDSET_MODE=development # reproduce the pinned
    reviewed set
    make ingest-uk-squad GOLDSET_MODE=skeleton # editable SQuAD template +
    instructions
    make ingest-uk-squad GOLDSET_MODE=draft # ontology-assisted draft over
    CORPUS
    (verified=false)
    make ingest-squad                          # the bundled fixture (4 items)
    make ingest-squad SQUAD_JSON=path.json     # a local SQuAD-uk export
    python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train # needs
    HF_TOKEN (goldset
    extra via make venv)

The stable public development fixture is
`samples/goldsets/ua_squad_postedited_v1/goldset.jsonl`: 250 verified items and 250 distinct
documents, split cal=86/tun=82/final=82. It is a deterministic subset of the pinned
`FIdo-AI/ua-squad` validation export. The upstream card states that Ukrainian translations
were post-edited and answer spans aligned; `source.json` and the fixture README record the
revision, source SHA-256, selection rule, verification basis, attribution, and data license.
The pinned selection was reviewed by a human and all 250 items are `verified: true`.

`--verified-goldset <path>` replaces the default ledger and may be repeated to combine reviewed
sets. This is the review handoff for frontier drafting `prepare-goldset` and planted-label outputs after a
human flips accepted entries to true; each ledger JSONL has a sibling `corpus/`. Canonical item
replacement, rather than a boolean-only flip, prevents a reused ID from certifying changed
content. `--no-verification-ledger` explicitly disables adoption. A zero-match import warns and
stays unverified.

The default development target uses one code-owned profile, `--pinned-development-source`, so
the Makefile cannot drift from the fixture metadata. It loads `FIdo-AI/ua-squad` revision
`943ef27daea65e400350ef1875d07c7e97288177`, split `validation`, then applies the exact fixture
selection: first grounded QA per distinct context, in source order. Live acceptance generated
250/250 verified items with 86/82/82 calibration/tuning/final splits; all canonical items and
all 250 corpus files exactly matched the committed fixture. This closes verified gold-set ledger and provides a
stable regenerated bundle for initial model tests. Normal initial tests should still use the
committed fixture through `make demo-eval`, which is offline and avoids unnecessary downloads.

`llb.prep.goldset_skeleton` writes an editable SQuAD example and instructions under
`$DATA_DIR/goldset-skeleton/<timestamp>/`. The complete manual is
[`docs/guides/goldset-from-scratch.md`](../../guides/goldset-from-scratch.md).

### RAG chunking / store builder -- `llb.rag.chunking`
Five strategies, every chunk anchored to `doc_id` + char offsets so retrieval scores against
source-span gold labels. We reuse the langchain family where it preserves offsets, and roll
our own where it does not (the span metric is the hard constraint):
- `fixed`, `sentence` -- pure-Python (zero deps), the always-available fallbacks.
- `recursive` -- langchain `RecursiveCharacterTextSplitter` (`add_start_index` -> exact
  offsets); falls back to the pure paragraph->sentence->char split when `[rag]` is absent.
- `markdown` -- structure-aware: headers parsed from the SOURCE (offset-exact), header
  breadcrumbs recorded in chunk `metadata`, long sections sub-split recursively.
- `semantic` -- native: embed sentences with the PINNED embedder, break at distance spikes
  (offset-exact -- langchain's `SemanticChunker` rejoins text and loses offsets, so we do
  not use it). Needs the embedder (`[rag]`).

All langchain use is lazy; `fixed` / `sentence` / `markdown` work without `[rag]`. `--embed`
(with `[rag]`) also builds a per-strategy FAISS index.

    make build-rag-store # chunk samples/corpus, all strategies
    python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
        --strategy markdown --size 800 --overlap 120 [--embed]

On the bundled IP doc: recursive 10 / markdown 8 chunks (markdown carries h1/h2 breadcrumbs).

### Judge calibration (judge calibration statistics stats + judge calibration gate tooling) -- `llb.judge.calibration` + `llb.judge.rate`
Spearman rho (no scipy), bootstrap CI, and the trust decision (`rho >= 0.6` else demote). The
worksheet is a single CSV (`CAL_WS`) kept in one of two roots auto-routed by `CAL_NAME`: PERMANENT
sets (in `CAL_PERMANENT`, the committed goldset by default) live in the TRACKED root `calibration/`
dir so they survive a clone; every other name routes to gitignored `$DATA_DIR/llb/calibration/`
(generated sets, persisted by copying into `calibration/`). It is the session's only state -- each
edit re-reads the
file and writes back ONLY the human columns, merged by `item_id` and rewritten atomically
(`fsutil.atomic_write_text`), so resume + crash-safety are free AND a concurrent `calibration-run`
filling `judge_rating` is never clobbered. Its columns are `item_id, split, provenance, question,
reference_answer,
model_answer, human_answer, human_rating, human_note, human_status, judge_rating`: `provenance` is
copied from the `GoldItem` so a card shows the item's source; the human authors both `human_answer`
and `human_rating` (`human_status` is a pending/rated refinement); `judge_rating` is the judge's
[0,1] score.

Two worksheet emitters: a blank one (`calibration-worksheet`) and a pre-filled one driven from a run
(`run-eval --worksheet`, the `calibration-run` target). The pre-filled path fills `model_answer` and
(when a judge is configured) `judge_rating`, running the judge **ungated** -- calibration measures
whether the judge agrees with humans, so the `rho >= 0.6` threshold is irrelevant at this step; the
human columns are left blank. When the judge backend is unavailable that column is left blank and
the run logs a warning rather than failing. Re-running the pre-fill MERGES existing human columns by
`item_id` (never clobbers them); a row whose regenerated `model_answer` changed has its now-stale
`human_rating` cleared with a warning, while the human's own `human_answer` is kept.

`calibration-rate` (`llb.judge.rate`; also `python -m llb.judge.calibration rate`) is the
interactive rater -- a terminal session that walks the worksheet item by item and fills the human
columns in place. Interactive I/O lives here, OUT of the pure-stats module, and the session loop is
driven by an injectable input iterator + output sink, so it is unit-tested with no model / endpoint
/ GPU. `judge_rating` is HIDDEN by default (an independence control: seeing it first anchors the
rater) and `--show-judge` reveals it for post-hoc review only. Commands: `1`-`5` rate + advance,
`a` author `human_answer`, `note` edit `human_note`, `n`/Enter next, `p`/`b` previous, `j <N>` jump,
`u` next unrated, `c` clear the rating, `?`/`h` help, `q` save + quit. With no `--start` it resumes
at the first unrated item; `--clear` wipes all human columns (confirmation-gated); Ctrl-C and EOF
are caught and treated as save + quit.

The Make targets drive the loop over the verified committed gold set (`GOLDSET` defaults to
`samples/goldsets/ua_squad_postedited_v1` -- all 86 calibration items are `verified: true`, so verified gold-set ledger
is already satisfied for it; its worksheet defaults to the tracked `calibration/ua_squad_postedited_v1.csv`).
Defaults target a local Ollama judge (`gemma3:27b` on :11434) with the embedder pinned to CPU
(`LLB_EMBED_DEVICE=cpu`, so the GPU stays free for the judge), so on the committed goldset it is:

    make calibration-run                  # Ollama gemma3:27b judge (default); vLLM via JUDGE_*
    make calibration-rate                 # interactive: fill the human columns (judge hidden)
    make calibration-score                # rho + bootstrap CI + trust decision (RATINGS=CAL_WS)
    make run-eval JUDGE_RHO=0.628         # carry the trusted decision into a scored run

(`make calibration-worksheet` emits a blank worksheet when you want the rows without a run; a new
goldset / text-corpus draft uses `CAL_NAME=<label>`.) The operator walkthrough is the
[calibration-tooling guide](../../guides/calibration-tooling.md).

**Calibration result (judge calibration gate DONE, 2026-06-24):** 86 independent human ratings scored to
**rho=0.628** (95% bootstrap CI [0.428, 0.772], n=86, judge `gemma3:27b` on Ollama) -> clears the
0.6 gate, `trusted=True`. It is a BORDERLINE pass: the CI lower bound is below 0.6 and the human
ratings skew high (68 of 86 are 5s, the judge mean is ~0.86) because the committed SQuAD-uk
calibration split is easy factual QA with little disagreement to measure -- so the rho is fragile.
The decision is not auto-persisted by `calibration-score`; carry it into a scored run with
`make run-eval JUDGE_RHO=0.628 JUDGE_MODEL=gemma3:27b JUDGE_BASE_URL=http://localhost:11434/v1`,
which records `calibration_rho` + `trusted` in that run's manifest and admits the gated judge.

The committed worksheet IS the canonical calibration: `tests/test_published_calibration.py`
re-derives rho from it on every run (no model/endpoint/GPU), asserting it still clears the 0.6 gate
and matches the pinned 0.628 -- so a fresh clone reproduces the calibration decision and CI catches
any drift. The stats, worksheet I/O, the interactive rater, and the scoring are likewise tested
(`tests/test_calibration.py` + `tests/test_rate.py`).

The runtime judge scorer is failure-tolerant for local-model diagnostics: an empty candidate answer
gets zero faithfulness/relevancy without calling DeepEval, and a malformed local-judge JSON response
zeros only the affected metric with a warning. The benchmark continues, objective scores remain the
headline, and the diagnostic zero records the judge-quality failure instead of aborting a run.

Judged composite smoke run (2026-06-26): `make calibration-score` re-confirmed the tracked
calibration worksheet at rho=0.628, CI [0.428, 0.772], n=86, threshold=0.6, trusted=true. Running
`make composite-headline MODEL=gemma4:26b BACKEND=ollama JUDGE_RHO=0.628 JUDGE_MODEL=gemma3:27b`
over the committed sample suite completed with a guarded composite score of 0.321, CI [0.22, 0.42],
avg_reliability=0.500, n_cases=39, unresolved=no. Category diagnostics: text-analysis objective
0.000 and no judged records in the sample bundle; summarization objective 0.000 with faithfulness
0.000; structured field-accuracy 0.333 and conformance 0.333; security defense 0.500, ASR 0.500,
and refusal_quality 0.666667; agentic completion 0.750 with trajectory_quality 0.250; tooling
call-accuracy 1.000. The run exposed two local-judge hazards that the scorer now handles: empty
candidate summaries used to make DeepEval reject `actual_output`, and `gemma3:27b` sometimes emitted
malformed JSON for agentic faithfulness. Both now become zero-valued diagnostics with warnings, not
pipeline failures.

#### Judge model (OQ2 decided) + bias disclosure

The v1 judge is a **local Gemma-4 model**, chosen over a frontier API for **no corpus
data-egress and reproducibility**. The id is configured through `judge_model` /
`--judge-model` / `JUDGE_MODEL` and must match the id exposed by the local OpenAI-compatible
endpoint. `judge_base_url` / `--judge-base-url` / `JUDGE_BASE_URL` keeps that endpoint separate
from the candidate backend. Existing `hosted_vllm/` and `ollama_chat/` prefixes remain accepted
and are stripped before requests.

- - **12 GB** (`ollama_chat/gemma-4-e4b-it`): smallest Gemma 4 via GGUF/CPU offload; the 12B will
- not fit
- - **16 GB (this box)** (`hosted_vllm/google/gemma-4-12B-it-qat-w4a16-ct`): biggest Gemma 4 that
- fits; the configured default
- - **32 GB** (`hosted_vllm/google/gemma-4-12B-it`): bf16 12B (higher fidelity) + headroom to
- co-host judge + a candidate

On 16 GB a 12B judge normally cannot co-reside with a vLLM candidate. Use Ollama GGUF/CPU
offload, a smaller test judge, or another local host while generating the calibration worksheet.

**Bias (disclosed, not eliminated).** This judge is **not independent of the candidate pool**:
Gemma-4 (E4B/12B) are candidates, and MamayLM v2 + Lapa are Gemma-3 fine-tunes -- so the judge
shares architecture, tokenizer, and pretraining lineage with most of the pool and may
**self-prefer Gemma-family answers** over the non-Gemma ones (Qwen3.6, Llama 3.2). The bias can
move the *ranking*, not just absolute scores. It is accepted because: (1) the judge is **gated**
(Premise 2) -- it enters ranking only when calibration rho >= 0.6 against the human-verified set,
else it is demoted to a diagnostic and objective correctness ranks alone; (2) the headline blend
keeps objective reference-correctness weighted; (3) the disclosure (`JUDGE_BIAS_NOTE` in
`scoring/judge.py`) travels with the run; and (4) a **non-Gemma cross-check judge** (e.g.
Qwen3.6 or a frontier model) can re-score the same calibration split to quantify the family
delta, with the board's judge-cohort guard preventing mixed cohorts in one board. The spec also
cautions a small local judge may not clear the gate for Ukrainian -- a 12B is borderline; if rho
< 0.6 the judge stays demoted, which is the gate working as designed.

## data prep status

| Step | What | State |
|------|------|-------|
| data bootstrap.1 schema | Pydantic `GoldItem` / `SourceSpan` | DONE |
| data bootstrap.2 sample generator | `gen_rag_items` + sample spec | DONE |
| Ukrainian SQuAD ingest stable public gold set | pinned post-edited UA-SQuAD fixture (250 items/docs) | DONE |
| data bootstrap.4 splits | deterministic disjoint partition | DONE |
| judge calibration statistics calibration stats | rho + CI + blank/pre-filled worksheet | DONE (code) |
| chunking | fixed/sentence/recursive RAG-store builder | DONE |
| acceptance | validator PASS (sample + fixture + 250-item set), suite green | DONE |

Remaining (blocked on a judge choice or human input; scoped forward in [`plan.md`](../plan.md)):
- **Judge-calibration close-out (plan judge calibration gate):** the stats, the gate, the executor judge
  wiring, the chosen judge (OQ2 -- a local Gemma-4 model, bias disclosed above), and the full
  pre-filled-worksheet scaffolding (model answers + ungated `judge_rating` via
  `make calibration-run` / `run-eval --worksheet --judge-model`, scored by
  `make calibration-score`) all exist and are unit-tested. The only required residual is external:
  collecting the human `human_rating` column over the verified calibration split.
