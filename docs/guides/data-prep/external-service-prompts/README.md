# External-service prompt pack

Copy-paste prompts for drafting benchmark artifacts with AI provider services (Claude Projects,
Google Gemini / NotebookLM, ChatGPT Projects, or any assistant that grounds a chat in uploaded
files). The workflow manual is
[`../external-ai-service-artifacts.md`](../external-ai-service-artifacts.md); the output shapes
these prompts request are defined by the
[external-service draft contract](../../../design/external-draft-contract.md).

**Open data only.** These prompts are used with corpora cleared for third-party processing.
Restricted data goes through the local ontology pipeline with open-weights models instead
(`make quickstart-corpus`, `make ingest-uk-squad GOLDSET_MODE=draft`).

## Before you open any service (prerequisites)

The per-service setup below assumes two local artifacts already exist. Do NOT create a project
or upload anything until you have both -- this is the most common first-time mistake, and it
surfaces only later as dropped rows at import:

1. **The staged corpus files** (`.md`/`.txt`): produced by
   `make pdf-to-markdown PDF_DIR=<dir>` or `make ingest-corpus CORPUS_ROOT=<dir>`. You upload
   these STAGED files, never the original PDFs, so drafted quotes match the exact text the local
   RAG index will contain.
2. **The doc-id list with sizes** from the produced `corpus_manifest.json` /
   `pdf_corpus_manifest.json`: pasted into the first chat so the model can name documents
   exactly, and recorded (with the manifest sha256) in the `external_provenance.json` sidecar.

If you are here without them, start from the workflow manual's
[workflow at a glance](../external-ai-service-artifacts.md#workflow-at-a-glance) and complete
its steps 1-2 first.

| File | Purpose | Output |
| --- | --- | --- |
| [`00-project-instructions.md`](00-project-instructions.md) | Session/system instructions pasted once per project | -- |
| [`01-ontology-inventory.md`](01-ontology-inventory.md) | Entity / relation / topic inventory to steer coverage | `inventory.json` |
| [`02-goldset-draft.md`](02-goldset-draft.md) | Needle-in-haystack QA drafting | SQuAD JSON (contract Artifact A) |
| [`03-chain-questions.md`](03-chain-questions.md) | Chain-of-questions drafting | provisional chains JSONL (Artifact D) |
| [`04-security-cases.md`](04-security-cases.md) | Corpus-specific security scenarios | security cases JSON (Artifact C) |

## Per-service setup (one-time per corpus project)

This table assumes the [prerequisites above](#before-you-open-any-service-prerequisites) --
staged `.md`/`.txt` files and the doc-id list -- already exist. Any service qualifies when it can
(a) hold your custom instructions, (b) ground its answers in uploaded files, and (c) return raw
JSON in code blocks. Record the service slug you actually used in the
`external_provenance.json` sidecar.

| Service | Where `00-project-instructions.md` goes | Corpus upload | Session notes |
| --- | --- | --- | --- |
| Claude Projects | Project -> "Set project instructions" | Project knowledge: add the staged `.md`/`.txt` files | Instructions and files persist across chats; run `01`-`04` as separate chats in one project |
| ChatGPT Projects | Project -> "Instructions" | Project files | Same persistence model as Claude Projects |
| NotebookLM | No instructions field: paste `00` as the FIRST message of every notebook chat | "Sources": add the staged files | Grounded-by-design, which suits inventory and QA drafting; restate "raw JSON only, one code block" whenever it adds prose |
| Gemini (Gems) | Gem editor -> "Instructions" (create a custom Gem per corpus) | Attach the staged files to the Gem (or per chat) | Re-attach files if a new chat loses them; verify it cites the uploaded text, not general knowledge |

Setup to-dos that apply to every service:

1. Stage the corpus first (`make ingest-corpus CORPUS_ROOT=<dir>` or
   `make pdf-to-markdown PDF_DIR=<dir>`) and upload the STAGED `.md`/`.txt` files -- never the
   original PDFs -- so quotes match the text the local RAG index will contain.
2. Paste the doc-id list (and sizes) from `corpus_manifest.json` / `pdf_corpus_manifest.json`
   into the first chat message so the model can name documents exactly and size its coverage.
3. Use the service name in every id the prompts generate (`ext-<service>-<yyyymmdd>-<seq>`), so
   multi-service merges never collide.
4. For a large corpus, upload in themed groups so the model keeps every document in context.

## Usage discipline

1. Configure the project per the table above, then upload the staged corpus files.
2. Run `01` once per corpus per service; save each service's output as `inventory.json`.
   When several services produced inventories, merge them into one wider coverage plan:
   `make curate-drafts CURATE_KIND=inventory CURATE_INPUTS="<inv1> <inv2>" CURATE_OUT=<merged>`.
3. Run `02`-`04` feeding the (merged) inventory in as the coverage plan. Ask for output in
   **batches** (10-20 items per reply) to avoid truncated JSON; say "continue with the next
   batch" until the coverage plan is exhausted.
4. Save every exported reply plus the required `external_provenance.json` sidecar under
   `$DATA_DIR/external-drafts/<service>-<YYYYMMDD>/`. Raw reply text with fenced code blocks is
   acceptable input for curation -- no manual JSON surgery needed.
5. Curate before importing: merge all batches and services, re-ground quotes, drop invalid and
   flabby items, and deduplicate (see the next section).
6. Record the prompt-pack git commit in the sidecar; edit prompts here, not ad hoc in the chat,
   so sessions stay reproducible.

## Maximizing yield with several services (best-of-N drafting)

Different services surface different entities, facts, and question angles from the same corpus;
the union after curation is consistently larger and better-covering than any single service's
output. The intended pattern:

1. Run the same prompt pack over the same staged corpus in two or more services (for example
   Claude Projects and Gemini/NotebookLM).
2. Keep each service's exports in its own `$DATA_DIR/external-drafts/<service>-<date>/` bundle
   with its own sidecar.
3. Merge + deduplicate + filter with one command per artifact kind:

```bash
make curate-drafts CURATE_KIND=squad \
  CURATE_INPUTS="$DATA_DIR/external-drafts/claude-<date>/goldset.json \
                 $DATA_DIR/external-drafts/gemini-<date>/goldset.json" \
  CURATE_OUT=$DATA_DIR/external-drafts/merged-<date>/goldset.json \
  CURATE_CORPUS=<staged-corpus-dir>
```

Curation repairs near-verbatim quotes back to exact corpus text, drops answers that are not
substrings of their context, rejects flabby questions (answer leaks, document-structure
references, vague stubs), removes exact and near-duplicate questions across services (pinned-E5
cosine, bias pairs and cross-language groups protected), and writes a
`*.curation_report.json` with per-source and per-reason counts. `CURATE_DEDUP_AGAINST=<bundle>`
also suppresses re-drafts of questions an earlier accepted bundle already covers. Kinds:
`squad`, `grounded`, `security`, `chains`, `inventory`.

## Using the generated artifacts

After curation, each artifact flows through validation, human verification, and the local
benchmark exactly like locally drafted data:

| Artifact | Import / run | Structural gate | Human gate | Then benchmark with |
| --- | --- | --- | --- | --- |
| Goldset (SQuAD) | `make ingest-squad SQUAD_JSON=<merged>` | `make validate-goldset` | cross-check + `verify-*` | `make build-index` -> `make run-eval` / `make sweep` -> `make recommend` |
| Security cases | `make bench-security SECURITY_CASES=<merged>` | schema check at load | `verify-*` before headline | per-family ASR + over-refusal + bias consistency |
| Chains | review-only until `chain-goldset-generation` lands | -- | -- | -- |
| Inventory | steers prompts `02`-`04` (not scored) | quote grounding via curation | -- | -- |

Step-by-step references:

- [Creating test artifacts with AI provider services](../external-ai-service-artifacts.md) --
  the full workflow this pack belongs to, including corpus staging, sidecars, and import.
- [External-service draft contract](../../../design/external-draft-contract.md) -- exact shapes,
  field rules, and per-artifact validation gates.
- [Create a gold set from scratch](../goldset-from-scratch.md) -- the local drafting spine the
  imported goldset joins (validation, cross-check, splits).
- [Verification tooling](../../human-tooling/verification-tooling.md) and
  [human-in-the-loop evaluation](../../human-tooling/human-in-the-loop-evaluation.md) -- the
  review gate every imported item must pass before it can score a model.
- [Run the RAG core](../../benchmarking/run-rag-core.md) -- index build, retrieval validation, and `run-eval`
  over the verified goldset.
- [Quickstart: any corpus](../../quickstart/quickstart-any-corpus.md) -- the end-to-end corpus-to-benchmark
  pipeline the external lane plugs into.
- [Security learning path](../../learning-path/learning-path-security.md) -- how security cases
  are scored and reported.
