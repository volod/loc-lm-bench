# 02 -- Goldset draft (SQuAD-format JSON, contract Artifact A)

Run after `01-ontology-inventory.md`. Paste the inventory (or the relevant document's slice of
it) into the placeholder. The output imports with `make ingest-squad SQUAD_JSON=<file>` and then
flows through validation, cross-check, and the human verification gate; every item arrives
`verified=false`.

Ask for one document per reply; say "continue" between batches. Concatenate batches locally by
merging the `data` arrays.

---

```text
Using the coverage plan below and the attached documents, draft needle-in-haystack QA items.

COVERAGE PLAN:
<paste inventory.json or its per-document slice here>

For the document <exact file name>, produce <N, e.g. 20> items as ONE JSON object:

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
   style references to document structure.

After the JSON, on one line outside the code block, report: items produced, items dropped by
self-check and why.
```
