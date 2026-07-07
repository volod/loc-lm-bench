# 02 -- Goldset draft (SQuAD-format JSON, contract Artifact A)

Run after `01-ontology-inventory.md`. Treat the curated inventory as a coverage map, not as text
that must be pasted whole. For each drafting reply, paste only the slice for the exact document
being drafted, or a compact section/topic slice of that document when the document inventory is
large. For NotebookLM, do not paste the JSON slice into the prompt: convert it to a text source,
upload that source, and reference the source file name in `COVERAGE PLAN`. The output is curated
(`make curate-drafts CURATE_KIND=squad`) and then imported with
`make ingest-squad SQUAD_JSON=<merged>`; it flows through validation, cross-check, and the human
verification gate, and every item arrives `verified=false`.

Ask for one document per reply; say "continue" between batches. Keep every reply as its own
exported file -- curation merges batches and services, so there is no need to hand-edit JSON.
NotebookLM should produce at most 15 items per reply.

Operator notes -- sizing `<N>` from the corpus statistics:

- A practical needle density before questions start repeating facts is about ONE item per
  2,000-4,000 characters of staged text, with a floor for small documents. Working defaults:

  | Staged document size | Items to request |
  | --- | --- |
  | under 2k chars (a how-to note, a short dialog) | 3-5 |
  | 2k-30k chars (an order, a short regulation) | 8-15 |
  | 30k-150k chars (a manual chapter, a long act) | 20-40, in 2-3 batches |
  | over 150k chars (a full system manual) | 40-80, batched section by section |

- Over-requesting is safe when you curate afterwards: near-duplicates and weak items are dropped
  by `curate-drafts`, so it is better to ask each service for the full per-document budget and
  let the merge keep the union of the strongest items.
- When drafting with several services, give each the same coverage plan and the same `<N>`;
  their overlap is removed at curation and their disjoint finds add up.
- NotebookLM has a lower practical reply limit: keep `<N> <= 15` and continue with the next
  topic/window when you need more items.

Mapping `inventory.curated.json` to the `COVERAGE PLAN` placeholder:

1. Pick the staged document id you are drafting, for example `pdf-6d8c2128b330.md`.
2. Extract the matching document slice from the curated inventory. This keeps the selected
   `documents[]` entry and only the `cross_document[]` links that mention the same doc:

   ```bash
   INV="$DATA_DIR/quickstart-pdf-corpus-md/api-provider-inventory.curated.json"
   DOC="pdf-6d8c2128b330.md"
   OUT="$DATA_DIR/quickstart-pdf-corpus-md/coverage-$DOC.json"

   jq --arg doc "$DOC" \
     '{documents: [.documents[] | select(.doc == $doc)],
       cross_document: [.cross_document[] | select(.docs | index($doc))]}' \
     "$INV" > "$OUT"
   ```

3. For Claude, ChatGPT, or Gemini, paste `coverage-$DOC.json` into `COVERAGE PLAN` if the service
   can handle it. For NotebookLM, convert the same slice into a text source and upload it:

   ```bash
   TXT="$DATA_DIR/quickstart-pdf-corpus-md/coverage-6d8c2128b330.txt"
   make coverage-plan-text COVERAGE_JSON="$OUT" COVERAGE_TEXT="$TXT"
   ```

   Add `coverage-6d8c2128b330.txt` to NotebookLM sources. In the prompt, write
   `COVERAGE PLAN: coverage-6d8c2128b330.txt` and set `For the document <exact file name>` to the
   same staged doc id. Do not paste large JSON or long text chunks into NotebookLM chat.
4. If a pasted slice is still too large for a non-NotebookLM chat, paste a compact slice and draft
   one section-like batch at a time. Increase or move the slice windows when you say "continue"
   (`[0:40]`, then `[40:80]`, and so on):

   ```bash
   jq --arg doc "$DOC" \
     '{documents: [.documents[] | select(.doc == $doc) | {
         doc,
         topics: .topics[0:40],
         entities: .entities[0:25],
         relations: .relations[0:20],
         numeric_facts: .numeric_facts[0:20],
         sensitive_topics: .sensitive_topics[0:10]
       }],
       cross_document: [.cross_document[] | select(.docs | index($doc))][0:10]}' \
     "$INV" > "${OUT}-0-40.json"
   ```

Do not paste the full corpus-level inventory for prompt 02 when it is large; the model should see
the document it is drafting plus only enough inventory context to spread questions across that
document. For NotebookLM, prefer the uploaded coverage text source even for a single filtered topic.

---

```text
Using the coverage plan below and the attached documents, draft needle-in-haystack QA items.

COVERAGE PLAN:
<paste a compact curated-inventory slice for the exact document, OR for NotebookLM write:
Use source coverage-<doc-id>.txt>

For the document <exact file name>, produce <N> items as ONE JSON object.
For NotebookLM, <N> must be 15 or lower:

{
  "version": "1.0",
  "data": [
    {
      "title": "<exact file name>",
      "paragraphs": [
        {
          "context": "<verbatim passage copied from the document, 200-1500 characters,
                       one coherent topic, complete sentences>",
          "qas": [
            {
              "id": "ext-<service>-<yyyymmdd>-<seq, e.g. 0001>",
              "question": "<Ukrainian question>",
              "answers": [{"text": "<exact substring of context>", "answer_start": 0}]
            }
          ]
        }
      ]
    }
  ]
}

Item requirements:

1. CONTEXT is copied verbatim from the document -- an exact substring, no edits, no ellipses.
2. ANSWER is the minimal exact substring of its context that fully answers the question
   (an entity, number, date, phrase, or short clause -- not a whole sentence unless needed).
   Leave answer_start as 0; it is recomputed locally.
3. COVERAGE: spread items across the plan's topics, entities, relations, and numeric facts for
   this document -- do not cluster on the first pages. At most 3 items per context passage.
4. QUESTION-TYPE MIX per batch, tagged in the id suffix is NOT needed, but follow roughly:
   40% factoid (who/what/where/when), 15% definition, 15% numeric (values, dates, quantities),
   15% procedural (how / in what order), 15% comparative or relational (uses a relation from
   the plan).
5. NEEDLE QUALITY: each question must be answerable ONLY from its context passage -- prefer
   facts unique in the corpus; avoid questions whose answer appears in many documents; avoid
   questions that quote so much of the context that retrieval becomes trivial; never leak the
   answer inside the question.
6. Questions are natural questions a domain user would ask -- no "according to paragraph 3"
   style references to document structure, and no "у цьому документі / in this document"
   phrasing.

After the JSON, on one line outside the code block, report: items produced, items dropped by
self-check and why.
```
