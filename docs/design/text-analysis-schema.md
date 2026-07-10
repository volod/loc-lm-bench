# Text-Analysis Scoring Schema (text analysis proposal -- text-analysis sign-off sign-off has been provided).

Signed off 2026-06-23 by volod; thresholds accepted.

Status: **APPROVED**

This is proposal text analysis; a human sign-off (human decision, item text-analysis sign-off) is provided.

- Executable form: [`src/llb/scoring/text_analysis.py`](../../src/llb/scoring/text_analysis.py)
  (taxonomy, planted-label model, matching engine) + the `PlantedLabelRecord` / `SubtaskScore`
  contracts in [`src/llb/contracts.py`](../../src/llb/contracts.py).
- Tests: [`tests/llb/scoring/test_text_analysis.py`](../../tests/llb/scoring/test_text_analysis.py).
- Spec basis: `docs/design/spec.md` Appendix D ("Text Analysis"), the
  `prepare-synthetic-corpus` planter constraints, and the open question flagged in the
  2026-06-19 eng-review ("Text-analysis scoring SCHEMA ... is an explicit open question to
  define before that benchmark is trusted").

This schema covers the OBJECTIVE, checkable part. Free-form quality is scored by the gated
judge (judge calibration gate) exactly as on the RAG board, and only when the judge is trusted.

---

## 1. Sub-tasks (the unit of credit)

Each text-analysis document carries planted labels under one or more sub-tasks. A sub-task is
the unit of credit: a candidate's output is scored per sub-task and the per-sub-task scores
are reported separately (and averaged for the objective headline). The sub-tasks mirror spec
Appendix D's "Text Analysis" decomposition:

| Sub-task (`kind`) | What the candidate must recover | Scoring |
|---|---|---|
| `key_fact` | planted atomic facts | OBJECTIVE |
| `entity` | named entities present in the doc | OBJECTIVE |
| `topic` | planted topics / themes | OBJECTIVE |
| `trend` | planted directional trends (`attrs`: subject, direction) | OBJECTIVE |
| `risk` | planted risks / problems | OBJECTIVE |
| `decision` | planted decisions / action items | OBJECTIVE |
| `contradiction` | planted internal contradictions (`attrs`: paired span ids) | OBJECTIVE |
| `narrative` | the doc's overarching narrative | JUDGED (objective floor) |
| `insight` | a non-stated inference | JUDGED (objective floor) |
| `long_doc` | long-document comprehension answer | JUDGED (objective floor) |

The OBJECTIVE sub-tasks are scored by planted-label matching alone (no LLM). The JUDGED
sub-tasks still receive an objective floor score (the same matcher, against any planted
anchor labels), but their HEADLINE quality is the gated judge -- and per the project's gate
(Premise 2), the judge contributes only when calibration rho clears the threshold; otherwise
those sub-tasks fall back to the objective floor and are reported as judge-demoted.

`long_doc` is evaluated through the map-reduce template
([`src/llb/eval/map_reduce.py`](../../src/llb/eval/map_reduce.py)); its synthesized answer is
scored by reference answer-correctness + the gated judge, like a RAG answer.

## 2. Planted-label taxonomy (what `prepare-synthetic-corpus` must emit)

Every planted ground-truth label is a STRUCTURED record (`PlantedLabelRecord`) so recovery is
checkable without hand-labeling. The planter emits one JSON object per label:

```json
{
  "label_id": "synth-007-trend-1",
  "kind": "trend",
  "value": "Зростання частки відновлюваної енергії",
  "aliases": ["частка ВДЕ зросла", "більше відновлюваної енергетики"],
  "doc_id": "synth-007",
  "char_start": 412,
  "char_end": 468,
  "scoring": "objective",
  "attrs": {"subject": "відновлювана енергія", "direction": "up"}
}
```

- `label_id` -- stable, unique; this is the identity the matcher credits (the "label-ID
  matching" basis). It survives chunking and re-runs.
- `kind` -- one of the sub-tasks above (validated; unknown kinds are rejected).
- `value` -- the canonical surface string the candidate must produce.
- `aliases` -- other ACCEPTED surface forms (synonyms, morphological variants the author
  considers equivalent). Matching tries all of `value` + `aliases`.
- `doc_id` / `char_start` / `char_end` -- ground the label in the synthetic doc (the
  exact-substring grounding already enforced by the planter), so a label can never point at
  text that is not there, and the source-span metric span metric still applies.
- `scoring` -- `"objective"` or `"judged"` (defaults to the kind's column above).
- `attrs` -- kind-specific structure for the author / judge: a trend's `subject` + `direction`,
  a contradiction's paired span ids, etc. Not used by the surface matcher today (documented as
  a residual in `plan.md` -- direction-aware trend credit is a natural next step).

Wiring note: text analysis DEFINES this taxonomy + the engine. Extending the `prepare-synthetic-corpus`
prompt to emit the richer per-kind labels (today it emits QA-style `key_fact` labels) lands
with the first scored text-analysis category (category expansion chat-period), behind the
verified-data gate.

## 3. Matching engine (the text-analysis sign-off matching basis)

Decided engine (text-analysis sign-off): **planted-label-ID matching + pinned-embedder cosine. NOT lemmatization,
NOT LLM-entailment.** A candidate emits free-text items per sub-task; each is matched to at most
one planted label:

1. **Exact / normalized surface match -> full credit (1.0).** `normalize_surface` casefolds,
   collapses whitespace, and strips surrounding punctuation (deliberately not lemmatization). A
   normalized prediction equal to a label's `value` or any alias is a full match.
2. **Otherwise, pinned-embedder cosine** over the prediction vs each surface, taking the best:
   - cosine `>= TAU_FULL` -> full credit (1.0) -- paraphrase / morphology;
   - `TAU_PARTIAL <= cosine < TAU_FULL` -> `PARTIAL_CREDIT` -- near miss;
   - `cosine < TAU_PARTIAL` -> no match.

The embedder is the project's PINNED embedder (the same one used for retrieval and the category expansion
summarization-coverage metric), so the cosine basis is consistent across the framework.

**Proposed thresholds (sign-off knobs):**

| Constant | Value | Meaning |
|---|---|---|
| `TAU_FULL` | `0.85` | cosine at/above this is a full match |
| `TAU_PARTIAL` | `0.70` | cosine in `[0.70, 0.85)` earns partial credit |
| `PARTIAL_CREDIT` | `0.5` | weight of a partial match |

These are the explicit numbers to confirm or adjust at sign-off. They live as named constants
in `text_analysis.py` (no magic numbers), and the human verification gate stratified human
sample is the empirical check that they neither over-credit paraphrases nor reject valid
morphology.

## 4. Per-sub-task credit and aggregation

Matching is a greedy one-to-one assignment (each prediction and each label used at most once,
highest-credit pairs first). Then, per sub-task:

- `recall = sum(matched credit) / number of planted labels` -- did the candidate recover the
  planted ground truth.
- `precision = sum(matched credit) / number of predictions` -- UNMATCHED predictions are false
  positives, so hallucinated extractions lower precision (the "no-hallucinated-extraction"
  pressure).
- `f1 = harmonic mean(precision, recall)` -- the per-sub-task headline.

Document objective headline = mean F1 over the OBJECTIVE sub-tasks present. Judged sub-tasks
are scored (floor) but kept OUT of the objective headline, which the gated judge owns.

Reporting rules (consistent with the rest of the framework):

- Every headline carries a bootstrap CI (the ranking rigor machinery), over the per-document scores.
- Real-corpus and synthetic results are reported SEPARATELY, never merged (spec; the planter
  already tags `synthetic: true` in provenance).
- The category renders under its own tier (proposed `TIER_TEXT_ANALYSIS`) and is never
  cross-ranked with the RAG board -- wired when category expansion builds the runner, mirroring the existing
  `TIER_SCREEN` / `TIER_PRIVATE` guard.

## 5. Worked example

Planted `topic` labels `T1="економіка"`, `T2="енергетика"`; candidate emits
`["економіка", "погода"]`.

- `економіка` exact-matches `T1` -> credit 1.0.
- `погода` matches no label (cosine below `TAU_PARTIAL`) -> false positive.
- `T2` unrecovered.
- recall = 1.0/2 = 0.5; precision = 1.0/2 = 0.5; F1 = 0.5.

---

## Sign-off (text-analysis sign-off) -- how to approve this schema

This is the human gate. Nothing about it requires running a GPU.

1. **Read** this document plus `src/llb/scoring/text_analysis.py` (the engine is short and the
   tests in `tests/llb/scoring/test_text_analysis.py` show the exact credit behavior). Run the
   tests if you want to see the numbers move:

   ```
   make test                                  # full suite, or:
   .venv/bin/python -m pytest tests/llb/scoring/test_text_analysis.py -q
   ```

2. **Confirm or adjust the four decisions** that are genuinely yours to make:
   - the **sub-task set** (Section 1) -- is anything missing or out of scope for your corpus?
   - the **objective-vs-judged split** -- e.g. should `narrative` be objective-anchored?
   - the **thresholds** `TAU_FULL=0.85`, `TAU_PARTIAL=0.70`, `PARTIAL_CREDIT=0.5` (Section 3);
   - the **matching basis** is already decided (label-ID + cosine, not lemmatization /
     entailment) -- confirm you still agree.

   To change a number, edit the constant in `text_analysis.py` (the tests assert against the
   constants, not hardcoded values, so they follow your change).

3. **Confirm the OQ4 corpus fact** this depends on (text-analysis sign-off item 3): whether your text-analysis
   reference answers already EXIST or must be authored, and which corpus is real vs synthetic.

4. **Record the sign-off.** Add a dated line to the top of this file, e.g.
   `Signed off 2026-06-__ by <name>; thresholds accepted as proposed (or: TAU_FULL -> 0.9).`
   That line is the trust signal the downstream category expansion runner reads before it lets
   a text-analysis `verified=true` item score models. (The human verification gate stratified
   human sample-verify of drafted items remains a separate gate, per Spec Premise 3.)

Until step 4 is recorded, the schema is a committed proposal only -- the engine and templates
exist and are tested, but the benchmark stays un-trusted for headline use.
