# Chain-of-questions artifacts

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

`src/llb/goldset/chains.py` defines canonical `ChainItem` / `ChainStep` rows for ordered
2-4-step chain-of-questions fixtures. Each step carries a question, reference answer,
dependency note, and exact `SourceSpan` list; `validate_chains` checks duplicate ids, step order,
span offsets, span reuse within a chain, and final-answer leakage from the first step's passage.

`make prepare-goldset-draft DRAFT_CHAINS=1` passes `--chains` to the ontology pipeline. The
pipeline walks the same 2-hop knowledge-graph paths as multi-hop drafting, builds ordered chain
rows in `src/llb/prep/ontology/chains.py`, records `stages.chains` in `provenance.json`, and writes
`<bundle>/chains.jsonl` beside `goldset.jsonl`.

Chain generation keeps strict directed `A -> B -> C` paths first. If that topology does not fill
the requested path budget, it adds exact-grounded pairs of facts incident on the same topic node.
This gives chain review enough candidates on sparse directed graphs without weakening the strict
directed semantics used by flat multi-hop questions. Generated questions and dependency notes are
Ukrainian, matching the chain artifact's `lang=uk` contract.

The five-document PDF chain bundle contains 214 grounded facts. Its strict directed topology yields
9 paths; the shared-topic fallback fills the configured 80-path budget, producing 80 unverified
chains with no dropped rows. `make validate-goldset` passes all 80 chains, calibration gates pass,
and every generated question uses Ukrainian wording. Human verification remains the authority for
whether a shared-topic sequence provides useful progressive context.

The public single-PDF literature corpus under `$DATA_DIR/quickstart-pdf-corpus` converts to one
626,093-character Markdown document with a page-citation sidecar. A local Ollama `gemma4:e4b`
chain draft with a 16,384-token context extracted 301 entities and 213 grounded facts, then wrote
20 flat drafts and 32 unverified chains under
`$DATA_DIR/prepare-goldset/chain-goldset-public-literature`. The extraction parse rate and PDF page
citation coverage are both 1.0, every calibration gate passes, and `make validate-goldset` passes
all 32 chains. The deterministic chain worksheet samples 20 of the 32 candidates at
`$DATA_DIR/prepare-goldset/chain-goldset-public-literature/verify_chains.csv`; its manifest records
`kind=chains`, seed 13, and a single source-document stratum. The converted source and bundle do
not contain the prior restricted corpus markers.

Human review accepted all 20 sampled chains. That run used an explicit 10-chain promotion
override. `make chain-goldset-finalize` required every row to carry `verified=true`, validated the
accepted ledger, and promoted `samples/goldsets/chain_context_uk_v1`. The committed fixture
contains 20 chains and a compact 36-span corpus rather than the complete copyrighted source
publication; promotion remaps every span offset and validates the result before making the
destination visible.

## Complete chain-goldset workflow

Use shell variables once so every later command is short and paste-safe. Select a new bundle path
for each draft run and a destination that does not already exist:

```bash
export DATA_DIR="${DATA_DIR:-$PWD/.data}"
export CHAIN_CORPUS="$DATA_DIR/quickstart-pdf-corpus-md"
export CHAIN_BUNDLE="$DATA_DIR/prepare-goldset/<run-name>"
export CHAIN_WS="$CHAIN_BUNDLE/verify_chains.csv"
export CHAIN_FIXTURE="$PWD/samples/goldsets/<fixture-name>"
```

1. Convert source PDFs to a normalized Markdown corpus. Skip this operation when the input is
   already `.md` or `.txt`:

   ```bash
   make pdf-to-markdown \
     PDF_DIR="$DATA_DIR/quickstart-pdf-corpus" \
     PDF_OUT_DIR="$CHAIN_CORPUS" \
     PDF_PARSER=auto
   ```

2. Run the non-human pipeline shortcut. It drafts chains, requires calibration to pass, validates
   every generated chain against the copied corpus, derives the review size from the resulting
   population, and writes a deterministic worksheet. A failed stage stops the target immediately:

   ```bash
   make chain-goldset-pipeline \
     CHAIN_CORPUS="$CHAIN_CORPUS" \
     CHAIN_BUNDLE="$CHAIN_BUNDLE" \
     CHAIN_WS="$CHAIN_WS" \
     CHAIN_MAX_PATHS=80 \
     DRAFT_MODEL=gemma4:e4b \
     DRAFT_BACKEND=ollama \
     DRAFT_MAX_ITEMS=20 \
     DRAFT_NO_THINK=1 \
     DRAFT_NUM_CTX=16384 \
     DRAFT_TIMEOUT=900
   ```

3. Review the worksheet interactively:

   ```bash
   make verify-review VERIFY_WS="$CHAIN_WS"
   ```

   For every step, compare `A` with `SOURCE`, confirm that `Q` is answered, and confirm that later
   steps use useful context from earlier steps. Reject a chain when its final answer is already
   available from the first step, a cited span does not support its answer, or the dependency is
   artificial. Press `y` to accept or `x` to reject; use `x <code>` for an explicit rejection code,
   `o` for a note, `b`/`u`/`j<N>` to navigate, and `q` to save and quit. Re-running the same command
   resumes at the first undecided row.

4. Emit the accepted ledger after every worksheet row has a decision:

   ```bash
   make verify-accept \
     BUNDLE="$CHAIN_BUNDLE" \
     VERIFY_WS="$CHAIN_WS"
   ```

5. Run the final pipeline shortcut. This replaces any inline Python count check and manual copy:

   ```bash
   make chain-goldset-finalize \
     CHAIN_BUNDLE="$CHAIN_BUNDLE" \
     CHAIN_FIXTURE="$CHAIN_FIXTURE"
   ```

   Finalization derives the required count as `ceil(reviewed sample size *
   CHAIN_MIN_RETENTION_FRACTION)` (default retention 0.50). `CHAIN_MIN_ACCEPTED=<n>` remains an
   explicit override. The fixture manifest records the reviewed count, retention assumption,
   derived target, override, and selected target. Finalization also refuses a missing accepted
   ledger, any `verified=false` row, a structural or span validation error, or an existing
   destination.

For an interrupted extraction, resume only the draft stage, then run validation and sampling again:

```bash
make prepare-goldset-draft \
  DRAFT_RESUME="$CHAIN_BUNDLE" \
  DRAFT_NO_THINK=1 \
  DRAFT_NUM_CTX=16384
make validate-goldset \
  CHAINS="$CHAIN_BUNDLE/chains.jsonl" \
  CORPUS="$CHAIN_BUNDLE/corpus"
make verify-sample \
  BUNDLE="$CHAIN_BUNDLE" \
  VERIFY_KIND=chains \
  VERIFY_WS="$CHAIN_WS"
```

The standard human verification target handles chains without a new command. `VERIFY_KIND=auto`
selects `chains.jsonl` when present; use `VERIFY_KIND=goldset` or `VERIFY_KIND=chains` to force a
mode. Each chain review card starts with 64 `+` characters and renders each step densely as
single-line `Q`, `A`, `SOURCE`, optional `DEPENDENCY`, and truncated `CONTEXT` fields. Questions,
answers, sources, and dependencies use distinct ANSI colors on an interactive TTY; redirected and
test output stays uncolored, and `NO_COLOR` disables color explicitly. The reviewer compares `A`
with `SOURCE`, then checks that `Q` is answered and later steps add context. The same navigation and
note shortcuts remain (`Enter`/`n`, `b`, `u`, `j<N>`, `o`, `?`, `q`). Chain answer edits are
blocked; reject and note the chain when a step needs a different span. `make verify-accept` writes
accepted chain ledgers under `<bundle>/accepted/chains.jsonl` with copied corpus files and
`verified=true`. `src/llb/goldset/promote_chains.py` implements the final acceptance-count,
verification, compaction, offset-remapping, and atomic-promotion gate exposed by
`make chain-goldset-finalize`.

```bash
make verify-sample BUNDLE=<bundle> VERIFY_KIND=chains
make verify-review VERIFY_WS=<bundle>/verify_sample.csv VERIFY_ORDER=confidence
make verify-accept VERIFY_WS=<bundle>/verify_sample.csv BUNDLE=<bundle>
```

Unit coverage: `tests/llb/goldset/test_goldset_verify.py` (schema validation, chain worksheet
cards, edit blocking, accepted chain ledger) and `tests/llb/prep/ontology/test_ontology_yield.py`
(graph-path chain construction and draft-bundle emission). Promotion failure modes and compact
corpus offset remapping are covered by `tests/llb/goldset/test_promote_chains.py`.

The local `$DATA_DIR/quickstart-pdf-corpus` corpus run produced 19 markdown files, 19 citation
sidecars, and zero skips under `.data/quickstart-pdf-corpus-md`. Sixteen born-digital PDFs used
PyMuPDF4LLM. The three PDFs that had zero embedded text were recovered by Docling OCR:

| Doc id | Pages | OCR chars | Citation pages |
| --- | ---: | ---: | ---: |
| `pdf-3c3a452a8e9c.md` | 24 | 4,641 | 24 |
| `pdf-3bc34dd5f5c2.md` | 61 | 14,670 | 55 |
| `pdf-3db280e14095.md` | 59 | 11,296 | 58 |

The PDF quickstart validation flow is documented in
[`docs/guides/quickstart/quickstart-pdf-corpus.md`](../../../guides/quickstart/quickstart-pdf-corpus.md).
The source PDFs are
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
`QUICKSTART_DRAFT_MAX_ITEMS=180`, a verification target derived at 95% confidence and 0.10
precision, and `QUICKSTART_DRAFT_NUM_CTX=16384`. The default
`QUICKSTART_MODEL_SELECTION=auto` resolves the
most capable Gemma 4 target from the CUDA serving-tier manifest, filtering out vLLM rows whose
configured `max_model_len` is below `QUICKSTART_DRAFT_NUM_CTX`. On 12 GB hosts the PDF drafter
uses the offloaded vLLM target `google/gemma-4-12B-it-qat-w4a16-ct` with `max_model_len=16384`,
`gpu_memory_utilization=0.90`, `cpu_offload_gb=16`, and `kv_offloading_size_gb=32`. On 16 GB hosts
the same 12B target uses `gpu_memory_utilization=0.85` plus the same context, CPU-offload, and
KV-offload settings. `benchmark`, `choose`, and `frontier` are explicit operator modes when the
host-fit Gemma 4 default is not appropriate. A vLLM pick sets
`QUICKSTART_DRAFT_BACKEND=vllm`; `prepare-goldset-draft` starts `VllmLauncher`, points the local
draft endpoint at `http://localhost:<port>/v1`, and records `endpoint.backend` plus
`endpoint.base_url` in provenance. `--no-think` still works for reasoning models: Ollama uses
native `/api/chat` `think=false`, while vLLM uses OpenAI-compatible `extra_body`
(`chat_template_kwargs.enable_thinking=false`, `include_reasoning=false`,
`reasoning_effort=none`). Fresh non-resume draft runs clear prior extraction journal state in the
output directory before the first model call; only `--resume` reuses journaled windows. The draft
step prints an estimated hour count (character-based, `wc -m`, since Cyrillic UTF-8 bytes would
double it) and requires confirmation before the full ontology/goldset generation starts. The logged
make wrapper cannot prompt inside the tee'd child
process, so unattended full-draft runs require `QUICKSTART_ASSUME_YES=1`; the non-interactive error
prints the exact rerun command. The PDF and mixed-corpus
quickstart wrappers pass `DRAFT_REQUIRE_PASSED_GATES=1`, so a zero-item or ungrounded draft writes
its inspection bundle and then exits non-zero instead of continuing to graph/validation. The wrapper
passes the full PDF RAG store at
`$QUICKSTART_PDF_RAG_DATA/llb/rag` into the needle retrieval-rank annotator. Model scoring remains
gated on `verify-review` and `verify-accept`.

The host selector reads the curated serving manifest for the detected 12/16/24/32 GiB tier. It
considers `gemma-4` and `gemma-4-*` entries, ranks CUDA/vLLM rows ahead of Ollama/offload rows, then
chooses the largest parameter count in that backend class. Long-context callers pass a minimum
context requirement so short-context vLLM cells cannot be selected for PDF drafting. The 12/16 GiB
tiers therefore use the extra `gemma-4-12b-vllm` target with CPU weight/KV offload, while 24/32 GiB
tiers use the primary 31B vLLM target. The complete selection contract and override precedence are
in [the inference configuration guide](../../../inference/config-example.md#automatic-cuda-host-draft-model-selection).

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
