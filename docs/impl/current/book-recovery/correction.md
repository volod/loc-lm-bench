# Evidence-constrained correction

An optional `correct` stage lets a local vision model propose a small substitution in one
block. A mechanical guard admits the proposal or refuses it. An admitted edit is applied by
`resolve` and shown in both renderings. A refusal stays out of the adopted text and becomes
an uncertainty marker. The block ledger is not rewritten. The stage is not in the default
stage list, and it stays off until a release reading uses the adoption report below. The
contract is [Scanned book recovery](../../../design/spec.md#scanned-book-recovery). The
overlay and the markers are [uncertainty and review](uncertainty-and-review.md).

## What a proposal is

The model sees the block's primary reading, a crop of that block padded by 8 px, the
previous and next blocks clipped to 240 characters, and the facing block of the neighbour
page when this block is first or last on the page. It returns a JSON list of edits. Each
edit names a character span, a replacement, a class, evidence, and whether the crop shows
the replacement. Prose is not mined for a correction. A reply that is not that list
proposes nothing. A transport failure is retried and is not stored as an empty edit list.

Ornament, figure, and neighbour-page-bleed blocks are skipped, and so is an empty reading.
Offsets are into the primary reading. The primary is the same shape `resolve` uses
(`stage`, `engine`, `variant`, optional `model`).

## The guard

The admitted classes and the budget are frozen. Changing them changes the stage tool
identity, so a page corrected under the old guard is not reused.

| Limit | Value |
| --- | --- |
| Classes | `digit`, `sign`, `decimal`, `unit`, `index`, `latin-greek`, `bracket`, `equation-number` |
| Edits admitted per block | 3 |
| Character operations inside one edit | 4 |
| Visual agreement | required for every class |

A class the guard does not list is a rewrite, including a proposal whose class is `prose`.
A wider diff is a rewrite. A digit edit may only substitute. Each change is a digit
replaced by a digit, or one of `0/O`, `1/l/I`, and `5/S`. An insertion such as `5` to `15`
is a rewrite.
Sign, decimal, unit, bracket, and Latin/Greek edits are admitted only when every changed
character is that class, which is the only case where an insertion or deletion is allowed.
An index edit may change only script characters. An equation-number edit replaces one
parenthesized number, such as `(3-4)` with `(3-5)`. Inside it, a dash may only be
substituted, and an insertion or deletion has to be a digit. Empty evidence is
`unsupported`. Proposals past the budget are `budget`, in start order. A span that
overlaps an admitted edit is `unsupported`. A span outside the block, or a replacement
equal to the slice, is dropped.

Visual agreement means the crop was produced and the proposal sets `visual` to true. No
class is admitted on language alone. A digit change with `visual` false, or with an empty
crop, is `no-visual`.

## What resolve does with the file

Admitted edits are `author: model` rows. `source` is the model identity (`ollama:<tag>` or
`fake:<name>`). `confusion_class` is the class the guard admitted. The edit id is
`model:<block-id>:<start>:<end>:<class>`. Refusals are a separate list on the same file,
with a reason of `rewrite`, `budget`, `no-visual`, or `unsupported`. Resolve does not apply
a refusal. It leaves that span unadopted unless an admitted edit covers it. The marker
class is `digit` for a digit or equation number, `sign` for a sign, `decimal` for a
decimal, and `symbol` otherwise. A reviewer edit still wins on the same span
([the overlay](uncertainty-and-review.md#the-overlay)).

The file is `pages/<pNNNN>/corrections.json` (`llb.book-ocr-model-corrections` 1.0.0). The
stage also keeps it under `pages/<pNNNN>/correct/`. Per-block replies are
`pages/<pNNNN>/correct/cache/<block-id>.json`. The cache key includes the model, the guard
identity, the prompt version, the block anchor, the crop digest, the block text, and the
neighbour text. Scripted `fake:` proposals are part of the page parameters, not that cache.
The page key also digests the facing block on each neighbour page, so an interior change
on a neighbour does not re-run this page.

## How to run

Name the stage. The default run does not.

```yaml
stages:
  - render
  - name: ocr
    params:
      engine: <engine>
      variant: original
  - blocks
  - name: correct
    model: ollama:<vision-model-tag>
    params:
      primary:
        stage: ocr
        engine: <engine>
        variant: original
  - name: resolve
    params:
      primary:
        stage: ocr
        engine: <engine>
        variant: original
```

The model is checked before a page runs. `ollama:<tag>` must be a tag the local daemon
serves. `fake:<name>` is always present, and it is the only backend that reads `proposals`.
Any other backend, including a frontier model, is refused. The stage has no frontier
client, so a non-redistributable book is not sent off the host. The survey's region-level
candidates are in [tool selection](tool-selection.md#shortlist-per-stage).

A `fake:` proposal matches a block whose primary text equals `text`, and `page_id` when
that field is set:

```yaml
proposals:
  - text: "The value is 0.05."
    page_id: p0001
    start: 16
    end: 17
    replacement: "S"
    class: digit
    evidence: "crop shows S"
    visual: true
```

The request asks the local daemon with thinking off and temperature 0. Braces in the OCR
text are sent as braces. When the reply's content is empty and a thinking field is
present, that field is parsed instead. The model is unloaded at the end of the batch.

## The adoption report

`adoption_report` in `src/llb/prep/book/correct/report.py` compares a reference with the
text before correction and the text after. `errors_fixed` counts reference characters that
were wrong before and are not wrong after, plus insertions the correction removed.
`tokens_harmed` counts reference characters the correction newly gets wrong, plus
insertions it adds. A token here is a character, so a digit inside a numeral is visible.
The class maps use the quality harness and name every confusion class, including zeros
([dangerous substitutions](quality-metrics.md#dangerous-substitutions)). The pinned
fixture, `The value is 0.05.`:

| Before | After | Fixed | Harmed | Class that moved |
| --- | --- | --- | --- | --- |
| `The value is O.05.` | the reference | 1 | 0 | `0/O` fixed |
| the reference | `The value is 0,05.` | 0 | 1 | `decimal/comma` harmed |
| `The value is O.05.` | `The value is 0.0S.` | 1 | 1 | `0/O` fixed, `5/S` harmed |

This is a fixture, not a reading of a book. The stage stays off until a release reading
reports both columns on a reference transcription.

## Where it lives

| Piece | Module |
| --- | --- |
| Stage | `src/llb/prep/book/stages/correct.py` |
| Page orchestration | `src/llb/prep/book/correct/page.py` |
| Guard and frozen limits | `src/llb/prep/book/correct/guard.py`, `constants.py` |
| Ollama proposer | `src/llb/prep/book/correct/ollama.py` |
| Neighbours and the edge digest | `context.py`, `edges.py` |
| Per-block cache | `src/llb/prep/book/correct/cache.py` |
| Parameters | `src/llb/prep/book/correct/params.py` |
| Adoption report | `src/llb/prep/book/correct/report.py` |
| Corrections contract | `src/llb/core/contracts/data_prep/book_resolve.py` |
| Fixtures | `tests/llb/prep/book/test_correct_guard.py`, `test_correct_stage.py` |

## Evidence

On 2026-11-27, on host vola-ASUS-V16-V3607VH1 (ASUS V16), this pass did not call a vision
model. No Ollama daemon is installed, and the stage is off by default. The acceptance is
the fake-model fixtures.

- `test_the_guard_classes_and_budget_stay_frozen` pins the classes, the budget of 3, the
  per-edit cap of 4, the tool identity, and the stage's absence from the default list.
- `test_a_rewrite_is_refused_and_a_digit_without_the_crop_is_not_admitted` refuses a
  whole-block rewrite and a digit change with no crop or with `visual` false.
- `test_admitted_edits_reach_both_renderings_and_refusals_become_markers` admits `5` to
  `S` into the vault passage and the corpus passage, marks the digit that had no visual
  agreement, refuses a rewrite so the replacement is in neither rendering, and leaves the
  ledger bytes unchanged. A resume runs no stage again.
- `test_a_frontier_model_is_refused_before_a_page_runs` refuses `openai:gpt-4o`.
- `test_a_block_is_not_sent_to_the_model_twice` reads the second pass from the per-block
  cache.
- `test_the_adoption_report_separates_fixes_from_new_errors` pins the table above.

What would overturn this: a class admitted without a crop; a rewrite present in either
rendering; a ledger byte changed by the stage; the budget or the class list changing
without the tool identity changing.
