# Data Prep

Data prep turns source documents into benchmarkable, verified records. Keep the distinction clear:
automated drafting and cross-checking can prepare evidence, but only reviewed `verified: true`
items score models.

## Gold Item Contract

`src/llb/goldset/schema.py` defines `GoldItem` and `SourceSpan`.

```text
id
lang
question
reference_answer
source_doc_id
source_spans[{doc_id, char_start, char_end, text}]
provenance
verified
split
```

The important design choice is span identity. Labels use document-relative character offsets rather
than chunk ids, because chunking strategy is an experiment variable. `load_goldset` and
`dump_goldset` read and write UTF-8 JSONL.

## Splits And Validation

`src/llb/goldset/splits.py` assigns deterministic disjoint
`calibration`, `tuning`, and `final` splits. `src/llb/goldset/validate.py` checks ids, split
labels, corpus references, and exact span text.

```bash
make validate-goldset
```

The default target validates `samples/goldsets/ua_squad_postedited_v1/goldset.jsonl` against its
sibling `corpus/`.

## Committed Fixture

`samples/goldsets/ua_squad_postedited_v1/` is the default development fixture:

- `goldset.jsonl`: 250 verified Ukrainian QA items;
- `corpus/`: one exact source document per item;
- `source.json`: upstream identity, revision, source digest, selection rule, and license context.

The fixture is intentionally small and stable. It is suitable for smoke checks, retrieval
comparisons, and development regressions. It is not a substitute for evaluating a private target
corpus.

## Ingestion

`src/llb/prep/ingest_squad.py` maps SQuAD-like rows to `GoldItem` records. It accepts local JSON,
Hugging Face rows, flattened rows, nested article rows, and rows whose `answers` value is encoded
as a dict string.

```bash
make ingest-uk-squad GOLDSET_MODE=development
make ingest-uk-squad GOLDSET_MODE=skeleton
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>
make ingest-squad SQUAD_JSON=path.json
python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train
```

Draft imports start with `verified=false`. A verification ledger can adopt matching canonical rows
by id. Adoption replaces the whole canonical item and corpus file, which prevents a reused id from
certifying changed content.

`src/llb/prep/goldset_skeleton.py` writes an editable from-scratch SQuAD template under
`$DATA_DIR/goldset-skeleton/<timestamp>/`.

`make pdf-to-markdown`, `llb pdf-to-markdown`, and `llb ingest-pdf-corpus` extract local PDF
directories into the canonical `.md` corpus shape used by RAG, ontology drafting, prompt-system
packages, and GraphRAG. The default `PDF_PARSER=auto` path uses PyMuPDF4LLM with OCR disabled for
born-digital PDFs, and Docling with Tesseract CLI OCR (`ukr+eng`) for image-only PDFs when the
`pdf-quality` extra and OCR apt packages are installed. Marker, Unstructured, and MarkItDown remain
available as explicit `PDF_PARSER=<tool>` probes, but they are not default full-corpus candidates.
The converter writes stable ASCII `pdf-<digest>.md` ids, preserves the source PDF path in a manifest,
and skips PDFs only when the selected parser output stays below `--min-chars`.

Conversion is incremental: each manifest item records `source_sha256`, and a rerun reuses the
existing `.md` plus citation sidecar when the source fingerprint, requested parser, and min-chars
still match and the outputs exist (`reused: true` in the manifest; `[pdf-corpus] reuse ...` in the
log). `--refresh` (make: `PDF_REFRESH=1`) forces a full reconversion. This makes quickstart reruns
skip the docling/OCR pass entirely for an unchanged corpus.

```bash
make pdf-to-markdown
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir> PDF_MIN_CHARS=500 PDF_PARSER=auto
make pdf-to-markdown PDF_REFRESH=1
llb ingest-pdf-corpus --pdf-root <pdf-dir> --out-dir <out-dir> --min-chars 500 --parser auto
```

The make alias defaults `PDF_DIR` to `$DATA_DIR/quickstart-pdf-corpus`. When `out-dir` is omitted,
the default is `<pdf-dir>/_md`, for example `.data/quickstart-pdf-corpus/_md`. Each successful
document gets a `pdf-<digest>.citations.json` sidecar with source PDF, parser, PDF diagnostics, page
numbers, generated-corpus character spans, and page-local block spans when the parser exposes them.
The same directory also contains `pdf_corpus_manifest.json` and `pdf_corpus_quality.json`; the
quality report records parser attempts, diagnostics, page coverage, citation coverage, structure
markers, and the selection score.

Ontology draft bundles preserve that PDF evidence. When a source document has a matching
`*.citations.json` sidecar, `prepare-goldset-draft` copies it into the bundle `corpus/` directory
and writes these review artifacts beside `goldset.jsonl`:

- `pdf_ontology_report.json`: parse rate, elapsed seconds, grounded entity/event/claim/fact counts,
  page-span citation coverage, citation-valid needle count, dictionary-term yield, and quality gates
  with a `passed` roll-up (grounded extractions of any kind + a non-empty gold set, plus a
  citation-valid needle for PDF corpora).
- `prompt_dictionary_candidates.jsonl`: source-backed entity and relation terms with supporting
  spans and PDF page references when sidecars exist.
- `needle_items.jsonl`: drafted gold items whose source spans map back to PDF page sidecars.

The artifacts are diagnostics for review and construction. Drafted rows still remain
`verified=false` until the human verification gate emits an accepted ledger.

The ontology-assisted seed sampler uses entities, subject-relation-object facts, grounded claims,
and grounded events as draft targets. Seeds carry document, section, difficulty, and semantic-type
coverage strata, so a full-corpus draft can spread questions across manuals, dictionaries, and
after-action-style documents even when a document has few SRO facts.

The local `$DATA_DIR/quickstart-pdf-corpus` corpus run produced 19 markdown files, 19 citation
sidecars, and zero skips under `.data/quickstart-pdf-corpus-md`. Sixteen born-digital PDFs used
PyMuPDF4LLM. The three PDFs that had zero embedded text were recovered by Docling OCR:

| Doc id | Pages | OCR chars | Citation pages |
| --- | ---: | ---: | ---: |
| `pdf-3c3a452a8e9c.md` | 24 | 4,641 | 24 |
| `pdf-3bc34dd5f5c2.md` | 61 | 14,670 | 55 |
| `pdf-3db280e14095.md` | 59 | 11,296 | 58 |

The PDF quickstart validation flow is documented in
[`docs/guides/quickstart-pdf-corpus.md`](../../guides/quickstart-pdf-corpus.md). The source PDFs are
under `.data/quickstart-pdf-corpus/`, the full converted markdown corpus is under
`.data/quickstart-pdf-corpus-md/`, and the reviewable draft bundle is under
`.data/quickstart-pdf-corpus-draft/`. The grouped quickstart wrapper is
`make quickstart-pdf-corpus`; it logs conversion, indexing, drafting, graph build, and validation
steps under `$DATA_DIR/llb/logs/quickstart/`. The PDF wrapper passes `QUICKSTART_SKIP_APT` through to
the `pdf-quality` venv step, so hosts that cannot use apt can run with the default
`QUICKSTART_SKIP_APT=1` when the required OCR binaries are already available or the corpus is mostly
born-digital.

`quickstart-pdf-corpus-draft` is the full-quality path, not a small subset. It defaults to
`QUICKSTART_PDF_DRAFT_DOCS=all`, `QUICKSTART_DRAFT_MODEL=auto`,
`QUICKSTART_DRAFT_MAX_ITEMS=180`, `QUICKSTART_DRAFT_VERIFY_N=40`, and
`QUICKSTART_DRAFT_NUM_CTX=16384`. With the auto model setting it prints ranked local candidates
from `llb recommend` JSON when benchmark artifacts exist; otherwise it prompts the operator to run
the local committed-goldset benchmark, choose an Ollama model manually, or opt into a frontier
`litellm` model. Auto-selection is backend-aware: drafting always talks to the local Ollama native
endpoint (the only layer honoring `think=false` and `num_ctx`), so
`llb.quickstart.model_choice drafter` picks `recommended_for_host` only when it is Ollama-served
and otherwise falls back to the highest-ranked Ollama candidate -- a vLLM-only HF id (for example
`google/gemma-4-E4B-it-qat-w4a16-ct`) is never handed to Ollama. The draft step prints an estimated
hour count (character-based, `wc -m`, since Cyrillic UTF-8 bytes would double it) and requires
confirmation before the full ontology/goldset generation starts. Model scoring remains gated on
`verify-review` and `verify-accept`.

The accepted ledger emitted by `verify-accept` contains only the rows a human explicitly accepted
in the worksheet; the complete drafted set (all `goldset.jsonl` rows and the citation-valid
`needle_items.jsonl` subset) stays in the draft bundle at `verified=false`. To enlarge the
verified ledger later, re-draw a bigger worksheet with `make verify-sample VERIFY_N=<n>` and review
it -- no re-draft needed.

Measured on 2026-07-02 (16 GB RTX 4060 Ti host, drafter `batiai/qwen3.6-35b:iq3` via Ollama with
`num_ctx=16384`): a bounded 4-document quick run
(`QUICKSTART_PDF_DRAFT_DOCS="pdf-2ff96d2db393 pdf-3c3a452a8e9c pdf-b117ebb25eb7 pdf-d2e2499d3d06"
QUICKSTART_DRAFT_MAX_ITEMS=80 QUICKSTART_DRAFT_VERIFY_N=20`) drafted 274k chars in 24 minutes:
26 extraction windows at ~48 s each, 80 draft calls at ~3.2 s each, 100 percent extraction parse
rate, 132 entities / 159 facts / 86 claims / 75 events grounded, a 452-seed pool, 70 of 80 drafts
kept (2 circular, 3 duplicate, 5 ungroundable), all 70 citation-valid needles, gates passed. The
full 19-document corpus is 8.0M chars (668 windows), so a `QUICKSTART_DRAFT_MAX_ITEMS=400` full
draft projects to roughly 9-10 hours on this host and about 350 kept items from a roughly
2,000-seed pool.

## Verification Gate

The verification path has a mechanical half and a human half.

```bash
make cross-check-goldset BUNDLE=<draft> CROSS_CHECK_MODEL=<model>
make verify-sample BUNDLE=<draft> VERIFY_N=<n>
make verify-review VERIFY_WS=<worksheet>
make verify-accept VERIFY_WS=<worksheet> BUNDLE=<draft>
```

`src/llb/prep/cross_check.py` checks grounding and non-circularity before calling an injectable
second verifier for support and answerability. A pass means the item is reviewable, not verified.

`src/llb/goldset/verify.py` handles stratified sampling, worksheet IO, acceptance arithmetic, and
accepted-ledger emission. `src/llb/goldset/verify_session.py` owns the interactive terminal loop.
The review session keeps command parsing, navigation, row edits, clear confirmation, and
persistence in small helpers so the loop reads as worksheet orchestration.
The accepted ledger writes copied corpus files plus canonical `verified=true` rows.
`prepare-goldset-draft` can also write the first worksheet in the same run with
`--verification-sample-size <n>`; the make wrapper exposes this as `DRAFT_VERIFY_N=<n>`.

The rationale is anti-anchoring and auditability: automated cross-check context can be shown to a
reviewer, but it is hidden by default; the accepted ledger is a new reviewed artifact rather than an
in-place mutation of the draft.

## Judge Calibration

Judge calibration is a separate human-rating problem. The code measures whether a local judge
tracks human ratings on the calibration split. The trust gate is Spearman rho `>= 0.6`; below that,
the judge remains diagnostic.

Modules:

- `src/llb/judge/calibration.py`: worksheet IO, Spearman rho, bootstrap CI, trust decision;
- `src/llb/judge/rate.py`: interactive human rater;
- `src/llb/scoring/judge.py`: runtime gated judge scoring.

```bash
make calibration-run
make calibration-rate
make calibration-score
make run-eval JUDGE_RHO=<rho> JUDGE_MODEL=<model> JUDGE_BASE_URL=<url>
```

`calibration-run` pre-fills model answers and optional ungated judge ratings.
`calibration-rate` hides judge ratings by default so the human rating is independent.
It stores only human-owned worksheet columns, supports resume/review navigation, and exits without
editing when the start-fresh clear prompt is not confirmed. The rating session uses the same
parser/navigation/edit-helper shape as verification review.
`calibration-score` computes rho and confidence interval from the filled worksheet.

Tracked calibration worksheets live in `calibration/`. Generated worksheets for temporary corpora
live under `$DATA_DIR/llb/calibration/` unless deliberately promoted.

## Chunking

`src/llb/rag/chunking.py` keeps every chunk offset-exact. Strategies:

- `fixed`: dependency-free fixed windows;
- `sentence`: dependency-free sentence-aware chunks;
- `recursive`: LangChain recursive splitter when available, pure fallback otherwise;
- `markdown`: heading-aware chunks with breadcrumb metadata;
- `semantic`: pinned-embedder breakpoints while preserving source offsets.

```bash
make build-rag-store
python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
  --strategy markdown --size 800 --overlap 120 --embed
```

Production RAG indexes are built through `llb build-index` or `make build-index`.
