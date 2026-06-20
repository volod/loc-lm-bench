# loc-lm-bench — Implemented (current state)

A snapshot of what exists and runs **today**. Forward work lives in
[`plan.md`](plan.md); the full spec is [`design.md`](../design.md).

**Status:** Milestone 0 complete: schema, validator, disjoint splits, SQuAD ingestion, a
real 250-item Ukrainian gold set (HPLT/ua-squad), judge-calibration stats, and a chunking
RAG-store builder, plus dev tooling. 27 tests passing, `ruff` clean.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable, and seeds `.env` from `.env.example`.

    make            # list targets
    make venv       # .venv (py3.11) + base deps + .env
    make test       # pytest (27 tests)

Heavy/eval deps are opt-in extras, kept out of the base install so `make venv` is fast:
`rag, track, board, prep, telemetry, goldset, dev`
(e.g. `uv pip install -e ".[dev,goldset]"`). vLLM/torch/flash-attn are hardware-matched
and installed via a separate path per AGENTS.md, never declared here.

Gitignored: `.data/` (runtime output), `.env` (secrets), `.venv/`.

## Repo layout (current)

    pyproject.toml                 # package "llb": deps + extras, pytest/ruff config
    Makefile                       # venv, test, gen-rag-items, validate-goldset,
                                   #   ingest-squad, ingest-uk-squad, build-rag-store, calibration-worksheet
    .env.example                   # DATA_DIR + frontier-API key placeholders
    samples/                       # COMMITTED DATA (kept separate from code)
      rag_items_uk.json            #   sample RAG spec: source docs + item defs
      squad_uk_fixture.json        #   SQuAD-format UA fixture (tests/demo)
      corpus/ip_regulation_uk.md   #   substantial UA domain doc (IP regulation) for chunking
    scripts/
      gen_rag_items.sh             # thin entrypoint -> llb.prep.gen_rag_items
    src/llb/
      goldset/schema.py            # GoldItem + SourceSpan (Pydantic), load/dump
      goldset/splits.py            # deterministic disjoint split assignment
      goldset/validate.py          # corpus-grounded validator + CLI
      prep/gen_rag_items.py        # spec -> seed gold set
      prep/ingest_squad.py         # SQuAD-format (local or HF) -> canonical gold items
      judge/calibration.py         # Spearman rho + CI + trust decision + worksheet
      rag/chunking.py              # fixed/sentence/recursive chunking -> RAG store
    tests/                         # 27 tests across the above

Runtime output (gitignored) under `$DATA_DIR/llb/` (default `.data/llb/`):
`corpus/`, `goldset/*.jsonl`, `rag/chunks/<strategy>.jsonl`, `calibration_worksheet.csv`.

## Implemented modules + how to run

### Canonical gold-item schema — `llb.goldset.schema`
Pydantic `GoldItem` + `SourceSpan`. Fields: `id, lang, question, reference_answer,
source_doc_id, source_spans[{doc_id, char_start, char_end, text}], provenance, verified,
split`. Labels are SOURCE-SPAN (char offsets, not chunk ids), so they survive `chunk_size`
tuning. `provenance` and `split` are enforced literals. Only `verified: true` items score
models. `load_goldset` / `dump_goldset` handle JSONL (UTF-8).

### Splits — `llb.goldset.splits`
`assign_splits(ids, ratios, seed)` -> deterministic, disjoint `calibration / tuning / final`.

### Validator (M0 acceptance) — `llb.goldset.validate`
Checks every span resolves to its labeled text on disk, ids unique, splits disjoint.

    make validate-goldset          # PASS on the sample set

### Sample generator — `llb.prep.gen_rag_items`
Reads `samples/rag_items_uk.json`, computes spans, writes + validates a seed gold set.

    make gen-rag-items             # -> .data/llb/goldset/sample_rag_items.jsonl (6 items)

### SQuAD ingestion (M0.3) — `llb.prep.ingest_squad`
Maps SQuAD-format UA QA (flattened, nested, or HF rows where `answers` is a dict-string) ->
canonical items (`provenance: public-reused`, `verified: false`), span from the answer
offset with a `find()` fallback. Local file or HF dataset (streams when `--max-items` set).

    make ingest-uk-squad                       # HPLT/ua-squad -> 250-item real gold set
    make ingest-squad                          # the bundled fixture (4 items)
    make ingest-squad SQUAD_JSON=path.json     # a local SQuAD-uk export
    python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train   # needs ".[goldset]" + HF_TOKEN

The current real set is `.data/llb/goldset/goldset_uk.jsonl` (250 items, splits
cal=86/tun=82/fin=82, 239 corpus docs). All `verified: false` pending human review.

### RAG chunking / store builder — `llb.rag.chunking`
Chunks a corpus with three strategies (pure-Python, no deps): `fixed` (char window +
overlap), `sentence` (packs whole sentences, never cuts mid-sentence), `recursive`
(paragraph -> sentence -> char). Each chunk carries `doc_id` + char offsets, so retrieval
can be scored against source-span gold labels. `--embed` (with the `[rag]` extra) also
builds a per-strategy FAISS index.

    make build-rag-store                       # chunk samples/corpus with all strategies
    make build-rag-store CORPUS_DIR=path/      # your own corpus
    python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
        --strategy recursive --size 800 --overlap 120 [--embed]

On the bundled IP doc: fixed 9 / sentence 9 / recursive 25 chunks -> `.data/llb/rag/chunks/`.

### Judge calibration (M0.5 stats) — `llb.judge.calibration`
Spearman rho (no scipy), bootstrap CI, trust decision (`rho >= 0.6` else demote).
Worksheet emitter for the human to fill once M1 produces answers.

    make calibration-worksheet                 # blank worksheet from the calibration split
    python -m llb.judge.calibration score --ratings <file>   # rho + CI + decision

## Milestone 0 status

| Step | What | State |
|------|------|-------|
| M0.1 schema | Pydantic `GoldItem` / `SourceSpan` | DONE |
| M0.2 sample generator | `gen_rag_items` + sample spec | DONE |
| M0.3 real gold set | `ingest_squad` + 250 items from HPLT/ua-squad | DONE |
| M0.4 splits | deterministic disjoint partition | DONE |
| M0.5 calibration stats | rho + CI + worksheet | DONE (code) |
| chunking | fixed/sentence/recursive RAG-store builder | DONE |
| acceptance | validator PASS (sample + fixture + 250-item set), 27 tests | DONE |

Remaining (does not block Milestone 1):
- **Judge-calibration ratings:** the math + worksheet exist; producing model answers and
  judge ratings needs Milestone 1 (a running backend). Then `calibration score` gates at
  rho >= 0.6.
- **Human verification + more sources:** the 250 public-reused items are `verified: false`;
  a reviewer flips `verified: true` (and may add own-corpus items) before they score
  models. Belebele-uk / other sources plug into the same ingestion engine.
