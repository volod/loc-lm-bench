# Record-linkage sample

A committed record table with a KNOWN identity structure, plus the specification and reviewer
labels that go with it. It is the default input of `make link-records` and the fixture the seam's
tests assert against, so a change here is a change to what the acceptance gate means.

| File | Contents |
| --- | --- |
| `entity_records_uk.jsonl` | 36 Ukrainian institution records over 12 real entities |
| `entity_spec_uk.json` | The comparison and blocking specification the records are linked under |
| `entity_labels_uk.jsonl` | 19 reviewer decisions sampled across the probability range |

## What the records are

One row per mention of an institution as an extraction pipeline would leave it: a surface name, a
list of abbreviations, a registry code, a postal address, a city, and an effective date. The
variation is the kind a Ukrainian corpus actually produces:

- genitive forms of the same name ("Львівський національний університет" /
  "Львівського національного університету");
- Latin homoglyph `i` substituted for Cyrillic `і`;
- abbreviated address prefixes (`вулиця` -> `вул.`, `проспект` -> `просп.`);
- single-digit typos in the registry code;
- distinct institutions that share a city and most of their name -- the Lviv national university
  and the Lviv national medical university are two entities, not one.

No single feature separates that set: the name similarity of two records of one entity overlaps
the name similarity of two Lviv universities, and so does the address similarity. Combining six
weak signals does separate it, which is the point the fixture exists to make.

`truth_entity_id` and `source_doc` carry the known structure. `truth_entity_id` is deliberately
NOT part of the specification -- neither compared nor retained -- so the model never sees it; the
tests read it only to check what the model recovered.

## Running it

    make link-records
    make link-records LINK_LABELS=samples/linkage/entity_labels_uk.jsonl
    make replay-linkage LINK_BUNDLE=<the run directory the fit printed>

The first fits without labels and proposes 12 identity clusters, one per entity. The second also
fits the match parameters from the reviewer labels, prints the labelled precision/recall curve the
operating threshold is read off, and scores the run's own cut both pairwise and after clustering.
The third re-scores the same records from the saved model and must reproduce the same
probabilities.

## Changing it

Both fits must still recover the 12 entities exactly, and the run must stay byte-reproducible
(`duckdb_threads: 1` in the specification is what makes it so). Regenerate the labels from a fresh
run's `pairs.jsonl` when the specification changes -- a label file sampled from a different model
no longer covers the probability range it claims to.
