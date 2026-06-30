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

```bash
make pdf-to-markdown
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir> PDF_MIN_CHARS=500 PDF_PARSER=auto
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

- `pdf_ontology_report.json`: parse rate, elapsed seconds, grounded entity/fact/claim counts,
  page-span citation coverage, citation-valid needle count, and dictionary-term yield.
- `prompt_dictionary_candidates.jsonl`: source-backed entity and relation terms with supporting
  spans and PDF page references when sidecars exist.
- `needle_items.jsonl`: drafted gold items whose source spans map back to PDF page sidecars.

The artifacts are diagnostics for review and construction. Drafted rows still remain
`verified=false` until the human verification gate emits an accepted ledger.

The local `$DATA_DIR/quickstart-pdf-corpus` corpus run produced 19 markdown files, 19 citation
sidecars, and zero skips under `.data/quickstart-pdf-corpus-md`. Sixteen born-digital PDFs used
PyMuPDF4LLM. The three PDFs that had zero embedded text were recovered by Docling OCR:

| Source PDF | Pages | OCR chars | Citation pages |
| --- | ---: | ---: | ---: |
| `Doktryna_MPZ_OS.pdf` | 24 | 50,548 | 24 |
| `Доктрина БПЛА.pdf` | 61 | 120,556 | 60 |
| `Настанова з бойової підготовки Mastanova_z_b_pidotovky.PDF` | 59 | 136,351 | 59 |

The PDF quickstart validation flow is documented in
[`docs/guides/quickstart-pdf-corpus.md`](../../guides/quickstart-pdf-corpus.md). The source PDFs are
under `.data/quickstart-pdf-corpus/`, the full converted markdown corpus is under
`.data/quickstart-pdf-corpus-md/`, and the reviewable Gemma 4 draft bundle is under
`.data/quickstart-pdf-corpus-draft/`. The draft bundle contains 7 `verified=false` items and a
4-row `verify_sample.csv`; model scoring is intentionally gated on `verify-review` and
`verify-accept`. The grouped quickstart wrapper is `make quickstart-pdf-corpus`; it logs conversion,
indexing, drafting, graph build, and validation steps under `$DATA_DIR/llb/logs/quickstart/`.

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
