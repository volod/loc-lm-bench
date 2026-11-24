# Scanned book recovery implementation audit

## Work contract

Audit the implementation and plan against the scanned-book spec and the target engineering
handbook. Prioritize restoration quality, fix demonstrated defects, and add intermediate audits
with dependencies that keep later work behind the relevant quality decisions.

## Implementation and decisions

The unit of acceptance is one source PDF recovered into two traceable renderings. The run
skeleton has full-length resume evidence. Recognition quality still needs a verified pilot,
and the corpus projection, real full-book recovery and reader acceptance remain open.
See the [spec](../../design/spec.md#scanned-book-recovery) and [run evidence](../current/book-recovery/pipeline-and-resume.md).

| Property | Implementation and evidence inspected | Finding and disposition |
| --- | --- | --- |
| Input fidelity and cache identity | Inventory, variant guards, original-coordinate maps, resume and class-routing tests | The frozen selection refers to this PDF. Exercise thin strokes, rotation, bleed and coordinate round trips together before pilot selection decisions. |
| Technical content | Reference validation, term matching, dangerous substitutions and figure association | Subtraction lost its sign at term boundaries. Repeated terms could reuse a single plain-text match. Figures required a caption and references required a folio. Corrected these cases and the parse check for extra formulas. |
| Accuracy of the selection decision | Score CLI, submission loader and pilot comparison contract | The scorer selects a raw engine reading and infers caption links by proximity. Add a resolved-output lane before choosing a production pipeline. |
| Visible omissions | Queue builder, vault loader, page accounting and marker validator | The queue builder refuses the entire queue when a page lacks resolution. A missing queue or queue-to-ledger drift can produce a warning. Per-block uncertainty alone cannot account for a column that yielded no block. Add a page and region accounting task. |
| Source evidence and publication | Vault load, asset construction, staged publication and renderer tests | The vault held all decoded pages, accumulated encoded fallbacks and discarded colour in crops. Stream image assets, preserve source channels and verify the PDF checksum before publication. |
| Structure, projection and reader acceptance | Structure and cross-reference fixtures, vault validation and the remaining corpus contract | Mechanical vault checks cover anchors and navigation. Revision identity, failed-page markers in both outputs, gold-span refusal and actual Obsidian behaviour still need their integration and operator gates. |

The executable checks are under `tests/llb/prep/book/`:

| Contract | Fixtures to rerun | Boundary |
| --- | --- | --- |
| Deterministic input and routing | `test_book_resume.py`, `test_ledger_reuse.py`, `test_page_class_stage.py` | Cache invalidation, duplicate-page identities and class routing each have a pin. Real engine ranking needs transcription. |
| Original image evidence | `test_page_variants.py`, `test_engine_adapters.py` | The fixtures plant an erased minus, a lost decimal, a cut edge and a rotated box. Partial recognition on intact pixels is a separate failure. |
| Technical fidelity and abstention | `test_quality_edge_cases.py`, `test_quality_score.py`, `test_review_queue.py`, `test_resolve_stage.py` | The fixtures distinguish recognition, alignment, edits, orphaned decisions and review. A scorer of the adopted pipeline is still required. |
| Navigable output | `test_structure.py`, `test_reference_links.py`, `test_render.py`, `test_vault_validate.py` | Gaps, unnumbered plates, unsupported hierarchy, ambiguous references, stable output and mechanical validation have fixtures. Obsidian and projected chunk provenance have separate acceptance gates. |

The [quality metrics](../current/book-recovery/quality-metrics.md) document what scores can
establish. The [reading vault](../current/book-recovery/structure-and-rendering.md#reading-vault)
documents the publication change. The parser checks its supported expression grammar. The
human reference and rendered-formula checks remain necessary for real notation.

## Acceptance evidence

Status: accepted for this interim implementation audit. On 2026-11-23 the 215 book tests
passed with `.venv/bin/python -m pytest -q tests/llb/prep/book`. The full `make ci` passed
with 5,184 tests passing, seven skipped and 50 slow or opt-in tests deselected. The first
attempt used the base environment and encountered missing optional RAG, ontology, eval and
linkage dependencies and restricted local sockets. The successful run used the same source
tree with a temporary virtual environment containing the locked test dependencies and local
sockets available. `make lint-md`, `make lint-spec-plan` and `git diff --check` passed.

CPU inspection on 2026-11-23 on the Intel Core Ultra 7 265HX host used `pdfinfo`, `sha256sum`,
and the production inventory functions on the committed pilot selection. The target PDF has
1,067 pages and matches the selection's source digest. All 25 pilot pages are greyscale at
150 dpi with zero embedded text characters. Four have a rotated image placement (three at
270 degrees and one at 90 degrees). This establishes the need for OCR and rotation coverage.
A different digest requires a new input check. These observations measure no transcription errors.

Synthetic regressions exercise both directions of a sign flip, repeated terms, an extra malformed
formula, a verified blank versus a missing submission, false folios in body text, and a figure
with no caption. Renderer regressions inspect actual colour channels and fallback bytes, and
substitute a different source PDF before republishing.

On the same CPU host and date, two fresh Python processes ran `build_vault` with page fallback
and the copy policy. To reproduce the input, use the config, structure and page writers in
`test_render.py` with a single book node spanning all pages, one body line and one figure per
page, and a hard link to the same solid 800 by 1,200 three-channel PNG at each render path.
Read Linux `getrusage(RUSAGE_SELF).ru_maxrss` immediately before and after publication:

| Pages | Process peak before / after (KiB) | Output verified |
| ---: | ---: | --- |
| 32 | 113,272 / 120,272 | 32 crops and 32 byte-identical fallback PNGs |
| 1,067 | 114,892 / 132,456 | 1,067 crops and 1,067 byte-identical fallback PNGs |

The longer fixture represents nearly 3 GiB of decoded page pixels. Its process peak during
publication was about 129 MiB. This exercises image memory and page accounting only. A real
book with many more blocks can require substantially more memory for text and layout. Missing
assets, changed fallback bytes, or memory growth proportional to decoded page pixels would
overturn this result.

## Handoff

The plan grows from 66 to 71 tasks, all five additions in scanned-book recovery. The group goes
from 13 to 18 open tasks across both lanes. Twelve earlier implementation records cover the
foundation. The capability remains planned. The pilot plan also retains the existing PDF lane for a
class with inconclusive evidence. An overall win on other pages cannot establish a benefit
for that class.

- `book-resolved-pipeline-scoring` and `book-missing-region-accounting` own the two implementation gaps.
- `book-input-and-pilot-audit` gates the comparison with the existing PDF lane.
- `book-technical-content-audit` gates the synthetic end-to-end run after specialist recovery and correction.
- `book-publication-and-resume-audit` gates the full scan run after integration.
- The existing closing audit retains all required tasks, including reader and retrieval evidence.
