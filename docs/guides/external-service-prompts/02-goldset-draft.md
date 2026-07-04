# 02 -- Goldset draft (SQuAD-format JSON, contract Artifact A)

Run after `01-ontology-inventory.md`. Paste the (merged) inventory -- or the relevant document's
slice of it -- into the placeholder. The output is curated
(`make curate-drafts CURATE_KIND=squad`) and then imported with
`make ingest-squad SQUAD_JSON=<merged>`; it flows through validation, cross-check, and the human
verification gate, and every item arrives `verified=false`.

Ask for one document per reply; say "continue" between batches. Keep every reply as its own
exported file -- curation merges batches and services, so there is no need to hand-edit JSON.

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

---

```text
Using the coverage plan below and the attached documents, draft needle-in-haystack QA items.

COVERAGE PLAN:
<paste inventory.json or its per-document slice here>

For the document <exact file name>, produce <N> items as ONE JSON object:

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
