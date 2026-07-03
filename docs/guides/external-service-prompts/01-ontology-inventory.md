# 01 -- Ontology / entity / topic inventory

Run once per corpus, after `00-project-instructions.md`. The output is the coverage plan that
steers the drafting prompts (`02`-`04`): it plays the same role the local ontology extraction
(`src/llb/prep/ontology/`) plays in the local lane -- entities, relations, and topic strata that
question drafting should spread across. Save the output as `inventory.json` in the session
bundle.

The entity `type` vocabulary below is the project's closed 13-type set
(`src/llb/prep/ontology/entity_types.py` / `docs/design/graph-ontology-schema.md`), so the
inventory maps onto the local knowledge-graph tooling without renaming.

---

```text
Build a knowledge inventory of ALL attached documents. Work document by document; do not skip
any. Output one JSON object in a code block:

{
  "documents": [
    {
      "doc": "<exact file name>",
      "topics": ["<3-8 short Ukrainian topic labels covering the whole document>"],
      "entities": [
        {"name": "<canonical Ukrainian surface form>",
         "type": "PERSON|ORG|LOC|GPE|NORP|LAW|WORK|PRODUCT|EVENT|DATE|DURATION|MONEY|QUANTITY",
         "mentions": <approximate count>,
         "quote": "<one exact verbatim sentence mentioning it>"}
      ],
      "relations": [
        {"subject": "<entity>", "relation": "<short Ukrainian verb phrase>",
         "object": "<entity>",
         "quote": "<exact verbatim sentence stating this relation>"}
      ],
      "numeric_facts": [
        {"fact": "<short Ukrainian restatement>",
         "quote": "<exact verbatim sentence containing the number/date>"}
      ],
      "sensitive_topics": ["<topics in this document a safety-guarded model might wrongly
                            refuse to discuss, or that need careful handling: weapons,
                            casualties, medical, legal liability, personal data, etc.>"]
    }
  ],
  "cross_document": [
    {"entity_or_topic": "<name>", "docs": ["<file>", "<file>"],
     "note": "<what connects them -- candidate for multi-hop questions>"}
  ]
}

Rules: entities with fewer than 2 mentions may be omitted unless they anchor a relation or a
numeric fact; every "quote" field is an exact substring of the named document; cap entities at
25 and relations at 20 per document, choosing the most load-bearing ones. If the type does not
fit the closed list, choose the closest one. Emit one document per reply if the corpus is
large; I will say "continue".
```
