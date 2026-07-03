# Creating test artifacts with AI provider services (open data)

This manual is a step-by-step procedure for producing loc-lm-bench test artifacts -- goldsets,
prompts, chain scenarios, security-test scenarios, and local run results -- with the help of
external AI provider services:

- **Claude Projects** -- <https://claude.ai/projects>
- **Google NotebookLM** -- <https://gemini.google.com/notebook> (a.k.a. notebooklm.google.com)
- **ChatGPT Projects** -- <https://chatgpt.com/projects>

These services support Ukrainian well and can lift the quality and completeness of draft test
sets. This manual explains **when** you may use them, **exactly what** to produce, and **how** to
bring the results back into the local benchmark.

> **The benchmark's purpose is to evaluate LOCAL RAG and LOCAL LLM inference.** External services
> are used here only to *draft candidate test data*. They never retrieve, answer, judge, or
> score. The scored run always happens locally against local models.

## 0. The one rule that gates everything: open data only

External AI services are permitted **only for open data** -- documents that are public or have
been explicitly cleared for third-party processing. Uploading a document to Claude, NotebookLM,
or ChatGPT publishes its contents to that provider and it may be retained.

**If any part of the corpus is restricted or private, the local flow is the main approach** --
not a fallback. Draft locally with the ontology-assisted pipeline running an open-weights model
on your own GPU:

```bash
make quickstart-pdf-corpus            # full local PDF -> draft bundle pipeline
make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir> DRAFT_MODEL=<local-tag>
```

The local lane produces the same artifact shapes this manual covers, with **zero data egress**.
Use it whenever classification is uncertain. This restates the settled
[data-egress policy](../impl/current/scope-boundaries.md#data-egress).

Decision:

```
Is EVERY document in the corpus public / cleared for third-party upload?
├─ No / unsure ─▶ LOCAL flow only (make quickstart-pdf-corpus). Stop; do not read further.
└─ Yes ─────────▶ External-service flow below is allowed as a drafting aid.
```

## 1. What you produce and where it goes

Every artifact below is defined precisely by the
[external-service draft contract](../design/external-draft-contract.md). The copy-paste prompts
are in [`external-service-prompts/`](external-service-prompts/README.md).

| Artifact | Prompt | Import / run command | Status |
| --- | --- | --- | --- |
| Ontology / topic inventory | `01` | steers `02`-`04` (not scored) | works today |
| Goldset draft (SQuAD JSON) | `02` | `make ingest-squad SQUAD_JSON=` | works today |
| Chain-of-questions draft | `03` | blocked on `chain-goldset-generation` | review-only |
| Security cases | `04` | `make bench-security SECURITY_CASES=` | works today |
| Local run results | -- | `make run-eval` / `make bench-security` | works today |

Session outputs live under `$DATA_DIR/external-drafts/<service>-<YYYYMMDD>/`, each with an
`external_provenance.json` sidecar (contract section 6). Nothing external ever sets
`verified: true`; every drafted item is reviewed locally before it can score a model.

## 2. Stage the corpus for upload

Upload the **staged corpus text**, not the original PDFs -- the drafting model must quote the same
text the local RAG index will contain, so quotes re-ground cleanly on import.

```bash
# PDFs -> canonical .md corpus (born-digital via PyMuPDF4LLM, image-only via Docling OCR)
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir>
# .md / .txt corpora are already staged; use the files as-is.
```

Note the produced `.md` file ids (`pdf-<digest>.md`) and keep `pdf_corpus_manifest.json`; you
record the doc ids and the manifest sha256 in the sidecar. Upload the `.md`/`.txt` files to the
service. For a large corpus, upload in themed groups so the model keeps every document in context.

## 3. Configure the project and run the prompts

1. **Create a project** (Claude Projects / ChatGPT Projects) or a **notebook** (NotebookLM).
2. Paste [`00-project-instructions.md`](external-service-prompts/00-project-instructions.md) into
   the project custom instructions (Claude/ChatGPT) or as the first message (NotebookLM), then
   attach the staged corpus files.
3. Run [`01-ontology-inventory.md`](external-service-prompts/01-ontology-inventory.md) once. Save
   the JSON as `inventory.json`. This is your coverage plan -- the external analogue of the local
   ontology extraction.
4. Run [`02`](external-service-prompts/02-goldset-draft.md),
   [`03`](external-service-prompts/03-chain-questions.md), and
   [`04`](external-service-prompts/04-security-cases.md), feeding the inventory in as the coverage
   plan. Ask for **batches of 10-20 items** and say "continue" until the plan is covered;
   large single replies truncate and produce invalid JSON.
5. Export each reply's code block to a file. Merge same-type batches locally (concatenate JSONL;
   merge the `data` arrays for SQuAD JSON).

Service notes:

- **NotebookLM** is grounded in the uploaded sources by design, which suits inventory and QA
  drafting; ask it to output the same JSON shapes. It is more conversational -- restate the
  "raw JSON only, one code block" instruction if it adds prose.
- **Claude / ChatGPT Projects** keep the instructions and files across chats in the project, so
  you can run `01`-`04` as separate chats without re-uploading.

## 4. Write the provenance sidecar

Beside the exported files, write `external_provenance.json` (contract section 6):

```bash
nano "$DATA_DIR/external-drafts/claude-20260703/external_provenance.json"
```

Record service, model, date, operator, the doc ids you uploaded, the manifest sha256, and
`"data_classification": "open"`. **An artifact bundle without this sidecar must not be imported.**

## 5. Validate and import each artifact locally

### 5a. Goldset (SQuAD JSON, Artifact A)

```bash
python -m json.tool "$DATA_DIR/external-drafts/claude-20260703/goldset_draft.json" >/dev/null
make ingest-squad SQUAD_JSON="$DATA_DIR/external-drafts/claude-20260703/goldset_draft.json"
# then structurally validate the canonical output against its corpus:
make validate-goldset GOLDSET=<canonical.jsonl> CORPUS=<corpus-dir>
```

Import re-grounds each answer by substring search and **skips any answer that is not a verbatim
substring** of its context -- your first quality signal. A high skip count means the model
paraphrased; tighten the verbatim rule and redraft.

### 5b. Security cases (Artifact C)

```bash
python -m json.tool "$DATA_DIR/external-drafts/claude-20260703/security_cases.json" >/dev/null
make bench-security \
  SECURITY_CASES="$DATA_DIR/external-drafts/claude-20260703/security_cases.json" \
  SECURITY_MODEL=<local-model> SECURITY_BACKEND=ollama
```

The loader rejects unknown families and malformed detector kinds at load. Benign controls feed
the over-refusal metric; matched bias pairs feed decision-consistency; denial-guard asks feed
attack-success rate. Run only against **local** models -- this is a local-inference safety probe.

### 5c. Chains (Artifact D) -- review-only

Keep chain drafts as review material until the `chain-goldset-generation` forward task lands;
do not import them into scoring paths.

## 6. Human verification gate (required before any headline use)

External drafting does not shortcut review. Route imported goldset drafts through cross-check and
the human verification gate exactly as any draft bundle (see
[goldset-from-scratch](goldset-from-scratch.md) and [verification tooling](verification-tooling.md)):

```bash
make cross-check-goldset BUNDLE=<bundle> CROSS_CHECK_MODEL=<second-model>
make verify-sample  BUNDLE=<bundle> VERIFY_N=30
make verify-review  VERIFY_WS=<bundle>/verify_sample.csv
make verify-accept  BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv
```

Only accepted `verified: true` items score models. Security cases go through the same `verify-*`
review before any headline metric.

## 7. Produce local run results

With verified data in hand, the scored run is entirely local:

```bash
make build-index GOLDSET=<verified.jsonl> CORPUS=<corpus>
make run-eval    MODEL=<local-tag> BACKEND=<ollama|vllm|llama.cpp> GOLDSET=<verified.jsonl>
make bench-security SECURITY_CASES=<verified-cases.json> SECURITY_MODEL=<local-tag>
```

Run artifacts land under `$DATA_DIR/run-eval/<timestamp>-<run-id>/`. These local results -- not
anything produced inside the external service -- are the benchmark's output.

## 8. What you may and may not send to a service

| Allowed to upload (open data only) | Never upload |
| --- | --- |
| Public/cleared corpus text for drafting | Restricted or private corpus documents |
| A topic or entity list to steer coverage | Model outputs to be judged or scored |
| Draft QA for a second-opinion rewrite | Any material for retrieval, answering, or ranking |

Scoring, judging, and ranking are local-only by design (see
[product decisions](../impl/current/scope-boundaries.md)). The external service is a drafting
assistant at the front of the pipeline and nothing more.

## See also

- [External-service draft contract](../design/external-draft-contract.md) -- exact artifact shapes.
- [Prompt pack](external-service-prompts/README.md) -- the copy-paste prompts.
- [Create a gold set (end-to-end)](goldset-from-scratch.md) -- the local drafting spine.
- [Data prep](../impl/current/data-prep.md) -- the local ontology pipeline and PDF conversion.
- [Product decisions: data egress](../impl/current/scope-boundaries.md#data-egress) -- the policy.
