# External-service prompt pack

Copy-paste prompts for drafting benchmark artifacts with AI provider services
(claude.ai Projects, NotebookLM, ChatGPT Projects). The workflow manual is
[`../external-ai-service-artifacts.md`](../external-ai-service-artifacts.md); the output shapes
these prompts request are defined by the
[external-service draft contract](../../design/external-draft-contract.md).

**Open data only.** These prompts are used with corpora cleared for third-party processing.
Restricted data goes through the local ontology pipeline with open-weights models instead
(`make quickstart-pdf-corpus`, `make ingest-uk-squad GOLDSET_MODE=draft`).

| File | Purpose | Output |
| --- | --- | --- |
| [`00-project-instructions.md`](00-project-instructions.md) | Session/system instructions pasted once per project | -- |
| [`01-ontology-inventory.md`](01-ontology-inventory.md) | Entity / relation / topic inventory to steer coverage | `inventory.json` |
| [`02-goldset-draft.md`](02-goldset-draft.md) | Needle-in-haystack QA drafting | SQuAD JSON (contract Artifact A) |
| [`03-chain-questions.md`](03-chain-questions.md) | Chain-of-questions drafting | provisional chains JSONL (Artifact D) |
| [`04-security-cases.md`](04-security-cases.md) | Corpus-specific security scenarios | security cases JSON (Artifact C) |

Usage discipline:

1. Paste `00-project-instructions.md` into the project's custom instructions (Claude / ChatGPT
   Projects) or as the first chat message (NotebookLM), then upload the staged corpus files.
2. Run `01` once per corpus; feed its output back into `02`-`04` as the coverage plan.
3. Ask for output in **batches** (10-20 items per reply) to avoid truncated JSON; say
   "continue with the next batch" until the coverage plan is exhausted.
4. Save every exported file plus the required `external_provenance.json` sidecar under
   `$DATA_DIR/external-drafts/<service>-<YYYYMMDD>/`.
5. Record the prompt-pack git commit in the sidecar; edit prompts here, not ad hoc in the chat,
   so sessions stay reproducible.
