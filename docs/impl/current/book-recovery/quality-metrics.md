# Quality metrics

`llb book-ocr score` compares a recovery with a reference transcription. Each page, and each page
class, gets character and word error, missing and duplicated words, reading-order agreement, a
count per dangerous-substitution class, formula fidelity, table-structure similarity, figure-box
overlap, caption association, and printed-page accuracy. The same report scores the existing PDF
lane and, when the run has resolved pages, the routed pipeline. The substitution counts stay
beside the error rates. The report has no field that folds them into one number.

The harness does not read a scan and does not produce a reference. The 1,067-page target handbook
is the book a later pilot scores; transcribing it is
`book-pilot-reference-transcription`, and choosing a pipeline
comes after that. Operator usage is the
[book-ocr guide](../../../guides/data-prep/book-ocr.md#score).
The contract is [Scanned book recovery](../../../design/spec.md#scanned-book-recovery).

## Reference transcription

A reference is a directory of `pages/*.json`, one file per PDF page. `load_reference` sorts by
`pdf_page` and refuses a directory with no pages, a repeated `pdf_page`, or a page that breaks
the rules below.

| Field | Meaning |
| --- | --- |
| `pdf_page` | 1-based page in the PDF |
| `page_class` | Any non-empty string. The class stage's ids and precedence are [page classes](tool-selection.md#page-classes). The score still accepts any other non-empty string |
| `printed_page` | The printed number as text, or explicit `null` for an unnumbered page |
| `width`, `height` | The rendered page, in pixels |
| `blocks` | Blocks in the transcription; use an empty list for a verified blank page |

A block has `block_id`, `kind` (a ledger kind), `region` (`x0`, `y0`, `x1`, `y1`, origin top
left; integers are accepted and stored as floats), `reading_order`, and `text`. A formula also
needs non-empty `latex`. A table needs `table.rows`, each row a list of strings or of
`{text, colspan, rowspan}`. A captioned figure has a `caption_id` naming a caption block on the
same page. Leave it null or omit it for an uncaptioned figure.
`equation_number` is optional. Ids and reading orders are unique. A region has area and lies
inside the page. `caption_id` on anything but a figure is refused. Score also refuses a reference
formula whose LaTeX does not parse.

The synthetic page the pins use is
`tests/llb/prep/book/fixtures/metrics/reference/pages/p0001.json`: class `normal`, printed page
`57`, one body line, one formula, one 2-by-2 table, one figure, its caption, and the page number.

## What a lane submits

A book-ocr lane is named `book-ocr/<engine>`. With no `--engine`, each engine in the ledger is
its own lane. A block's reading is the `original` variant when that variant exists, otherwise
the lexicographically first variant. A table grid is parsed from markdown pipe rows or from HTML;
a separator row is skipped. Text that does not parse is an empty grid, and the table metric is
then zero, not absent. A figure's caption link is the nearest caption by center, one caption to
one figure. Formula text that contains `\`, `_`, or `^` is read as LaTeX.

The existing lane is `existing-pdf`, from a `llb.pdf-citations` sidecar beside its markdown. Each
PDF page is one body block whose text is `markdown[text_start:text_end]`. The block has no
region. That lane is text-only: figure overlap, table similarity, and caption association are
`absent` (the value is null), never zero and never dropped. A formula it wrote as plain text is
still scored against the reference LaTeX. `formula_parse_valid` on a text-only page is null,
because the lane did not claim to emit LaTeX.

A metric is `scored` when a number was computed, `absent` when the lane cannot express the
structure, and `not_applicable` when the page has nothing for that metric to measure (no formula,
no table, no figure, or fewer than two blocks whose order can be compared). A reference page a
lane did not submit is an empty submission of that lane. It stays in the page count and in the
character-error mean, and `missing_submissions` counts it. A page the reference does not list is
not scored.

## Resolved pipeline

A run that contains `pages/<page-id>/resolved.json` adds a `book-ocr/resolved` lane beside the
raw engine lanes. `--engine` still limits the raw lanes. The resolved lane stays. A run with no
resolved pages has no such lane.

The lane reads the artifacts the vault and the corpus projection render from. Adopted text comes
from `resolved.json`. Kind, region, and reading order come from the block ledger. A table is the
grid `table_markdown` renders from that text. A formula is the display math `formula_math`
renders, and the equation number is the one the structure inventory recorded for that block, or
else the first parenthesized number in the text. A caption link is the figure and caption that
sit next to each other in reading order. The printed page is the structure file's mapping for
that PDF page.

An unresolved span counts as one abstention for each uncertainty class on the span: `digit`,
`sign`, `decimal`, and `symbol`. Every class is present on every page, zeros included. The count
is not subtracted from character error. Characters the span withheld are absent from the adopted
text, so they raise character error when the reference still contains them.

A structural metric is `absent` when the resolved page has no block of that kind, or when the
structure file has no row for the page. Absence is not a zero. A raw engine lane still scores a
missing block as zero, so the two lanes can be compared. `not_applicable` still wins when the
reference has nothing to measure. A reference page with no `resolved.json` is a missing
submission of this lane.

## Character and word error

Both rates use the reading string: blocks in reading order, whitespace collapsed to single
spaces. A figure, an ornament, and neighbour-page bleed contribute nothing. A table contributes
its cell labels. An equation number is appended when the block text does not already contain it.
The rate is the Levenshtein distance divided by the reference length. An empty reference side is
`not_applicable`.

Missing words are reference words absent from the hypothesis multiset. Duplicated words are
hypothesis words beyond that multiset, repeats and novel words together. A one-word substitution
is word-error 1 and also one missing word and one duplicated word. Neither total adds a further
substitution penalty.

## Dangerous substitutions

Every class is present on every page, zeros included. A count is never added into the character
error. The classes, in the order a substitution is claimed, are `0/O`, `1/l/I` (also `|`),
`5/S` (also `s`), `minus/hyphen/dash` (hyphen, minus, and the en and em dashes, including
substitutions between these signs and plus), `unit`
(degree against `o` or `0`, micro against `u` or Greek mu, ohm against `O`, angstrom against
`A`, per-mille against `%`), `latin/greek` (the lookalike pairs), `bracket`, and
`decimal/comma`. A deletion or insertion counts only for a minus, a bracket, a unit symbol, a
Greek letter, or a dot or comma between digits. A deleted digit is not `5/S`.

An equation number is a parenthesized run of numbers separated by a hyphen or a dash. Two
numbers with the same digits stay in the character alignment, so a hyphen against a minus inside
an otherwise equal number counts as `minus/hyphen/dash`. A changed, missing, or extra number is
removed from that alignment and counts as one `equation-number` event, paired by order.

Subscripts and superscripts are stripped before the character pass, so they are not also digits
or letters. `script` counts atoms `(base, subscript, superscript)`. The same triple matches. The
same base with a different script is one change. A leftover atom on either side is one change.
Unicode scripts and LaTeX scripts both count; a text-only page counts both forms of its plain
text.

The class rollup sums the counts. Its rate is that sum divided by the pages' reference
characters, and null when those pages have none.

## Formula fidelity

Canonical form ignores whitespace and spacing commands, drops `\left` / `\right` and the size
commands, unwraps `\mathrm`, `\text`, `\operatorname`, `\mathbf`, `\mathit`, and `\textrm`, and
rewrites `\le`/`\leq`, `\ge`/`\geq`, `\ne`/`\neq`, `\to`/`\rightarrow`, `\cdotp`/`\cdot`,
`\dfrac`/`\tfrac`/`\frac`, and `\ldots`/`\dots`. A one-token script may omit its braces:
`v_E`, `v_{E}`, and `v_{{E}}` are one spelling. `I_sp` is not `I_{sp}`, because TeX takes one
token. Unicode dashes become `-`. `\tag`, `\begin`, `\end`, `&`, and `\\` are removed.

Terms are the top-level sides of `=`, then each top-level `+` or binary `-`. Fidelity is the
share of reference terms the hypothesis still has. Each subtractive term retains its sign.
Each matched occurrence in the hypothesis counts once per reference formula. An unparseable LaTeX
hypothesis has null fidelity, fails the parse check and is left out of the page ratio.
A structured formula
with no match contributes every term as missed. Parse status covers every formula the structured
lane submits, including extra formulas on pages with no reference equation. It is null when
the lane submits no formula. Plain text, including the existing lane, is
scored by substring: `\frac{a}{b}` becomes `a/b`, scripts flatten, and the Greek commands become
letters. Character error still compares the reading strings, so two equivalent spellings can
move the character error and still score fidelity 1.

## Tables, figures, order, and the printed page

Table similarity is one minus the ordered tree-edit distance over the two sizes. The tree is the
table node, then each row, then each cell. Inserting or deleting a node costs one. Relabelling a
cell (text, colspan, or rowspan) costs one. Cell text is compared after whitespace collapse. An
empty side costs the other side's full size, so a missed table scores zero. A text-only lane is
`absent` instead.

Figure overlap is the mean intersection-over-union of the reference figures after a one-to-one
match. A match needs overlap of at least 0.5. An unmatched figure contributes zero. A text-only
lane is `absent`.

Caption association uses that figure match and the same match on caption boxes. A figure is
correct when its `caption_id` is the block id of the hypothesis caption that matched the
reference caption. A matched uncaptioned figure earns credit when it has no caption link.
Unmatched figures stay in the denominator. A text-only lane is `absent`. The resolved lane links
a figure to the caption the vault places beside it. A nearer caption that is not that neighbor
does not count.

Reading-order agreement is the share of matched pairs whose relative order agrees. On a
structured lane the match is by overlap within a kind, and the ranks are `reading_order`. On a
text-only lane each reference reading is sought in the plain text and then blanked, so a repeat
can match again, and the rank is the character offset. Fewer than two comparable pairs is
`not_applicable`.

A structured lane needs exactly one page-number block matching the reference token after
stripping whitespace and surrounding punctuation. An unnumbered reference needs a submitted
page with no page-number block. For text-only output, a known folio is sought as a whole word.
On an unnumbered page this lane reports the metric as `absent`. The resolved lane scores the
printed-page mapping in the structure file instead. A matching token in the body, or on a
page-number block, does not stand in for a wrong mapping. When the structure file has no row
for the page, the metric is `absent`.

## Per-class rollup

`by_page_class` means the scored values and skips `absent` and `not_applicable`. When nothing was
scored and something was absent, the class metric is `absent`; otherwise it is `not_applicable`.
Lanes are not averaged together.

## Where it lives

| Piece | Path |
| --- | --- |
| Reference | `src/llb/prep/book/metrics/reference.py` |
| Ledger and citation loaders | `src/llb/prep/book/metrics/submission.py` |
| Character and word error | `src/llb/prep/book/metrics/align.py` |
| Confusion classes | `src/llb/prep/book/metrics/confusion.py` |
| LaTeX spelling, the parse check, term fidelity | `src/llb/prep/book/metrics/latex_norm.py`, `latex_parse.py`, `latex.py` |
| Table tree | `src/llb/prep/book/metrics/tables.py` |
| Overlap, matching, rank agreement | `src/llb/prep/book/metrics/geometry.py`, `layout.py` |
| Resolved lane | `src/llb/prep/book/metrics/resolved.py` |
| One page, the report | `src/llb/prep/book/metrics/page_score.py`, `score.py`, `report.py` |
| Command | `src/llb/cli/book/score.py`, `make book-ocr-score` |

The survey smoke scorer (`src/llb/prep/book/survey/score.py`) is a separate check that a candidate
tool ran. It is not this harness. The ledger the score reads is the
[block ledger](block-ledger.md). The citation sidecar is the existing PDF lane's page spans.

Tests: `tests/llb/prep/book/test_quality_text.py` (error rates and each confusion class),
`test_quality_structure.py` (LaTeX, tables, boxes, order), `test_quality_score.py` (the
fixture page, the existing lane, and the command), and `test_quality_resolved.py` (the resolved
lane beside the raw engines). The synthetic runs live in `resolved_score_fixture.py` and
`resolved_score_cases.py`.

## How to run

```sh
llb book-ocr score --reference <reference-dir> <run-dir>
llb book-ocr score --reference <reference-dir> --citations <doc>.citations.json
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_RUN=<run-dir>
make book-ocr-score BOOK_REFERENCE=<reference-dir> BOOK_CITATIONS=<doc>.citations.json
```

`BOOK_ENGINE` limits the raw ledger to one engine and still scores `book-ocr/resolved` when the
run has resolved pages. `BOOK_SCORE_OUT` writes the JSON report, and `BOOK_JSON=1` prints it.
Without `--json` the command prints one line per lane, per page, and per class. A lane line names
its missing submissions. A page line names character error, any substitution class whose count is
not zero, and any non-zero abstention. A class line adds missing pages and structural fields the
submitted pages left absent. `--out` still prints that summary. Exit code 2 is a missing input or
a reference the loader refused. The command writes nothing unless `--out` is set.

## Evidence

Checked on 2026-11-22 by the CPU tests named above. No model was loaded and no GPU memory was
taken. The numbers are the hand counts those tests assert. A change to the edit distance, the
class priority, the term split, or the tree-edit costs that moves one of these fractions
overturns the pin.

- A table cell of 200 characters and 96 words, the last character `5` read as `S`: character
  error 1/200 (below 0.01), word error 1/96, one missing word, one duplicated word, class `5/S`
  equal to 1 and every other class 0. The one-cell tree is size 3+3 and the relabel costs 1, so
  table similarity is 5/6.
- `alpha beta` (10 characters) against `alpha bet`: character error 1/10, one missing word, one
  duplicated word, every class 0.
- `alpha beta gamma` (16 characters, 3 words) against `alpha gamma gamma`: the word alignment
  substitutes `beta` with `gamma`, so word error is 1/3. The multiset still has one missing word
  and one duplicated word. Character error is 4/16.
- Each confusion class, including `(3-4)` against `(3-5)` as `equation-number` and `(3-4)` with a
  minus sign against a hyphen as `minus/hyphen/dash`, counts 1 and the other classes stay 0.
  `v` with a subscript `e` against `v` is one `script` event.
- `v_E` and `v_{E}` canonicalize equal and score fidelity 1. `\frac{a}{` does not parse, and the
  page's `formula_parse_valid` is false. The four-term equation
  `v_{spc}(t_{1}) = v_{E}(t_{1}) + \omega_{E}(t_{1} - t_{0}) = v_{E}(t_{0})` scores 1; dropping
  the last term scores 3/4. The same 3/4 holds when the remaining terms are plain text.
- A 2-by-2 table with one cell relabelled scores 13/14. Identical boxes score overlap 1. A box
  `(0,0,10,10)` against `(0,0,10,5)` overlaps by 1/2. Reading orders `(0,1,2)` against
  `(0,2,1)` agree on 2/3 of the pairs. A reversed plain-text page agrees on 0.
- The fixture page mirrored as a ledger scores 0 character error, fidelity 1, table and figure
  and caption and printed page and reading order all 1, and no substitution counts. The same
  reading submitted as an existing-lane citation scores character error 0, reading order 1,
  printed page 1, and formula fidelity 1, with table, figure, and caption `absent`. The class
  rollup for `normal` has `5/S` rate 0. `ScoreReport` and `PageScore` have no `score` or
  `headline` field. A ledger of that page through `llb book-ocr score --json` exits 0 on lane
  `book-ocr/fake` and the JSON has no `score`. A figure whose `caption_id` names nothing exits 2.

## Resolved-lane evidence

Checked on 2026-11-24. The synthetic counts are what `tests/llb/prep/book/test_quality_resolved.py`
asserts. No model was loaded.

- Prose class, two pages. Page 1 adopts `cat` against `cat` and withholds one symbol. Page 2 has
  a ledger and no resolved file, so its three characters are missing. Mean character error and
  word error are 1/2, reference characters 6, missing words 1, duplicated words 0, `5/S` rate 0,
  symbol abstentions 1, digit abstentions 0, printed-page accuracy 1/2. Glyph class, one page,
  reference `5` adopted as `S`: character error 1, word error 1, one missing word, one duplicated
  word, `5/S` count 1 and rate 1, every other substitution class 0. The lane's missing submissions
  are 1. An extra resolved page outside the reference is not scored. The page class on the report
  is the reference class.
- Replacing page 1's adopted text with `dog` from the `deskew` variant changes only
  `book-ocr/resolved` page 1, from character error 0 to 1. The raw `book-ocr/alpha` page stays at
  0. Restoring page 1 and then adopting `5` on page 3 changes only the resolved page: character
  error goes from 1 to 0 and `5/S` from 1 to 0. The raw page stays at `5/S` 1.
- A figure's neighbor caption is wrong and a nearer caption is right. The raw lane's
  nearest-caption link scores caption association 1. The resolved lane links the neighbor and
  scores 0. The structure mapping says `99` and the reference folio is `57`, which also appears
  in the body and in a page-number block. The raw lane scores printed-page accuracy 1. The
  resolved lane scores 0. A resolved page of only body text, against a reference that has a
  formula, a table, a figure, and a caption, marks those four fields `absent`. The raw lane
  scores the formula and the table. The class records one unscorable page for each of the four.
  With no structure file, printed-page accuracy is `absent` even when the body contains the
  reference folio.
- A formula `v_{E}` whose inventory number is `(3-4)`, and a pipe table of two rows, score
  character error 0, formula fidelity 1, `formula_parse_valid` true, table similarity 1, and
  printed-page accuracy 1.
- `llb book-ocr score --json` on that synthetic run exits 0 with lanes `book-ocr/alpha`,
  `book-ocr/beta`, and `book-ocr/resolved`, and the JSON has no `score` field. The prose rollup
  is character error 0.5, missing submissions 1, symbol abstentions 1. `--engine alpha` drops
  `book-ocr/beta` and keeps `book-ocr/resolved`. The text summary contains `missing 1` and
  `abstain symbol 1`.

**What ran.** On 2026-11-24, on the Intel Core Ultra 7 265HX host, CPU only. The resolved lane
read the 50-page slice (PDF pages 1-40 and 290-299) of the 1,067-page Space Mission Engineering
scan, SHA-256 `89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`, 163,709,872
bytes. The slice is one RapidOCR reading: 5,278 body blocks, no unresolved spans, and a
printed-page mapping on every page. Scored against a reference built from that adopted text and
those mapped folios, both lanes had character error 0 on all 50 pages and 0 missing submissions.
The resolved lane had printed-page accuracy 1 on all 50 pages and 0 abstentions. Pages 1-5 are
unnumbered, and both lanes score printed-page accuracy 1 there. On the other 45 pages the raw
lane scores 0, because the ledger has no page-number block. The mapped folio occurs as a whole
word in the body on 39 of those 45 pages, and that occurrence does not give the raw lane credit.
This round trip checks that the lane reads the artifacts the renderers use. It is not a
transcription of the scan. A later score of this slice against the same adopted text and the same
mapped folios that moves the resolved character error off 0, or the resolved printed-page
accuracy off 1, overturns it.

The harness record is
[0008](../../records/0008-scanned-book-recovery-book-recovery-quality-metrics.md).
The resolved-lane record is
[0018](../../records/0018-scanned-book-recovery-book-resolved-pipeline-scoring.md).
