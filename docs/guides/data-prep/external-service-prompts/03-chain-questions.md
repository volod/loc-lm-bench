# 03 -- Chain-of-questions draft (provisional, contract Artifact D)

Drafts ordered 2-4 step question sequences in which each step supplies more specific context for
the topic (topic overview -> narrowing detail -> exact fact). This prompt emits the provisional
external quote-based form. Its curator re-grounds and filters rows but does not convert them into
canonical `ChainItem` offsets, so do not feed these exports directly into scoring.

Best seeds: the `cross_document` section and multi-relation entities from `01`'s inventory.

Multi-service exports still merge cleanly today:
`make curate-drafts CURATE_KIND=chains CURATE_INPUTS="<f1> <f2>" CURATE_OUT=<merged.jsonl>
CURATE_CORPUS=<staged-corpus>` validates step structure, re-grounds step quotes to exact corpus
text, drops chains whose final answer is already findable from the step-1 passage, and
deduplicates chains that walk the same question sequence.

---

```text
Using the coverage plan below and the attached documents, draft chain-of-questions test
sequences.

COVERAGE PLAN:
<paste inventory.json (especially cross_document and relations) here>

Produce <N, e.g. 10> chains as JSON Lines -- one JSON object per line, no wrapping array:

{"chain_id": "ext-chain-<yyyymmdd>-<seq>", "lang": "uk",
 "steps": [
   {"order": 1, "question": "<broad topic-overview question>",
    "reference_answer": "<verbatim quote answering it>",
    "source_doc_id": "<exact file name>", "quote": "<same verbatim quote>",
    "dependency_note": ""},
   {"order": 2, "question": "<narrower question that presumes the step-1 answer>",
    "reference_answer": "<verbatim quote>", "source_doc_id": "<exact file name>",
    "quote": "<verbatim quote>",
    "dependency_note": "<one sentence: what step 1 establishes that this step builds on>"},
   {"order": 3, "question": "<the exact-fact question>", "reference_answer": "<verbatim quote>",
    "source_doc_id": "<exact file name>", "quote": "<verbatim quote>",
    "dependency_note": "<what step 2 establishes>"}
 ]}

Chain requirements:

1. 2-4 steps per chain; each step's quote is an exact substring of its named document.
2. Steps cite DISTINCT passages; a chain may cross documents when the coverage plan links them.
3. Specificity must strictly increase: step 1 is answerable from a topic overview; the final
   step's answer must NOT be findable from the step-1 passage alone.
4. Each step must still be a well-formed standalone question when its dependency_note context
   is given -- a reader who knows the previous answers can pose it directly.
5. Answers follow the same minimal-verbatim-span rule as single questions.

After the output, report on one line: chains produced, chains dropped by self-check and why.
```
