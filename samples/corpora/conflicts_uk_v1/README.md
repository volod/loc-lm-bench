# conflicts_uk_v1 -- planted corpus-conflict fixture

A seven-document Ukrainian corpus that plants one instance of every relation
`llb audit-corpus-conflicts` is supposed to find, so each effort tier can be asserted against a
known answer in CI. See
[corpus hygiene](../../../docs/impl/current/data-prep.md#corpus-hygiene-conflict-detection-corpus-conflict-detection)
for the detector itself.

The documents are a plausible institutional set: a 2021 regulation on handling citizens' appeals,
its 2024 revision, and satellite notes.

| document | plants | found at | as |
| --- | --- | --- | --- |
| `regulation-2021.md` | the baseline | -- | -- |
| `regulation-2021-copy.md` | a byte-identical re-upload | `hash` | `duplicate` (raw) |
| `regulation-2021-reformatted.md` | a reissue differing only in case, whitespace, punctuation, and front matter | `hash` | `duplicate` (normalized) + staleness |
| `regulation-2024.md` | a revision that changes one deadline and restates two sections unchanged | `claim` | `superseded_by` **and** `duplicate` |
| `e-appeals-note.md` | a note whose whole content the 2024 revision absorbed | `lexical` | `subsumed_by` (containment) |
| `deadline-note.md` | a vague restatement of the specific 2024 deadline | `claim` | `subsumed_by` |
| `archive-policy.md` | an unrelated control | -- | never reported |

Two properties the fixture exists to pin down:

**Partial supersession.** `regulation-2024.md` versus `regulation-2021.md` is one document pair
that must yield *different relations for different claims* -- `superseded_by` for the deadline that
changed from thirty calendar days to fifteen working days, `duplicate` for the sections restated
unchanged. A detector that labels document pairs rather than claim pairs cannot express this, which
is why the fixture makes it the headline case.

**Containment is the low-Jaccard case.** `e-appeals-note.md` sits entirely inside
`regulation-2024.md`: containment 0.955, Jaccard 0.296. Blocking tuned for high Jaccard (MinHash
LSH) misses it with probability ~0.999, so `test_subsumption_is_the_low_jaccard_case_lsh_blocking_would_miss`
guards the blocking strategy against a regression back to sketch-based candidate generation.

Governance front matter (`version`, `effective_date`, `source_system`, `language`) is what lets the
staleness ordering resolve, and is deliberately excluded from content hashing: two byte-identical
documents carrying different `effective_date` values must still read as duplicates.
