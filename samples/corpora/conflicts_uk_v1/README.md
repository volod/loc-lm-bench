# conflicts_uk_v1 -- planted corpus-conflict fixture

A seven-document Ukrainian corpus that plants one instance of every relation
`llb audit-corpus-conflicts` is supposed to find, so each effort tier can be asserted against a
known answer in CI. See
[corpus hygiene](../../../docs/impl/current/data-prep/conflict-detection.md#corpus-hygiene-conflict-detection-corpus-conflict-detection)
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

**The adjudicator-calibration probe.** `adjudicator_probe.json` beside this file carries two
tiers of frozen actionable/complementary labels, each half and half. Every claim-tier audit
adjudicates them before quoting a precision figure, so this corpus doubles as the calibration set
for the model that judges other corpora ([conflict
detection](../../../docs/impl/current/data-prep/conflict-claim-precision.md#the-frozen-calibration-probe)).
The probe addresses passages by `doc_id` + heading line, never by copied text: editing a section
below fails the probe loudly instead of leaving a frozen label on text that moved.

| tier | corpus | pairs | what it asks |
| --- | --- | --- | --- |
| `base` | `corpus/` | 24 | is this adjudicator broken? -- the only tier that gates |
| `hard` | `probe_hard/` | 16 | which of two working adjudicators is better? -- reported, does not gate |

Two sections that carry byte-identical text (the copy and its original) must not both appear as
probe pairs against the same third section -- that is one prompt counted twice, not two
observations. The base tier does carry two pairs that show the SAME two passages with A and B
exchanged (`general-provisions-2021-reformatted` against `general-provisions-reformatted-2024`, and
the officer pair beside it), because the 2024 revision restates those sections verbatim; that is an
order-symmetry check on `subsumes`/`subsumed_by` rather than a second observation, and the hard tier
deliberately carries none.

**The hard tier's own corpus.** `probe_hard/` is five Ukrainian budget-process documents authored
for one job: pairs whose actionable/complementary split is arguable on a shallow reading and
determinate on a close one. Its actionable half restates one fact under a different heading, in
different units (two working weeks against ten working days), or with only one clause of two
changed; its complementary half puts two numeric claims about different quantities into the same
sentence shape (thirty calendar days to submit against thirty calendar years to retain; an advance
and a balance that share one five-working-day window). Each pair records the shallow reading it
exists to catch in its `trap` field.

It is a SEPARATE corpus rather than more sections here on purpose. The documents above plant one
instance of every relation the detector must find, each with a known answer; an arguable pair is
precisely what that fixture must not contain, and adding one would change what the detector tiers
return over it. `probe_hard/` is never audited -- only adjudicated.

**Repeated metadata is not a claim.** `archive-policy.md` and `deadline-note.md` each carry one
number-heavy `Reiestr vydannia` publication record under the same structural heading. The semantic
filter must exclude both blocks without a vocabulary-specific label, while preserving the ordinary
archive-policy prose as a single-occurrence negative control.
