# editions_uk_v1 -- planted document-edition fixture

A twenty-six-document Ukrainian corpus whose EDITION structure is known: six documents that were
re-issued, re-uploaded, or reformatted over the years, plus two short notes whose whole content a
longer document absorbed, plus eight unrelated documents that share only the formulaic sentences an
institutional corpus is full of. It is the fixture the edition-linkage lane of
`llb audit-corpus-conflicts --linkage` is asserted against
([entity resolution](../../../docs/impl/current/entity-resolution.md#the-document-edition-lane)).

It exists because [`conflicts_uk_v1`](../conflicts_uk_v1/README.md) cannot serve this purpose: seven
documents make twenty-one document pairs, and a Fellegi-Sunter fit over twenty-one pairs cannot
estimate a non-match parameter, let alone the levels no pair of that corpus exhibits. That fixture
pins what each TIER finds; this one pins what a fit over a corpus's document pairs recovers.

## What is planted

| family | documents | relation planted | found by |
| --- | --- | --- | --- |
| `appeals/` | `polozhennia-2019`, `polozhennia-2022`, `polozhennia-2022-copy`, `polozhennia-2022-sharepoint` | a 2022 revision changing one deadline, a byte-identical re-upload of it, and a reformatted re-issue in another system | `hash` (raw + normalized), `lexical` (Jaccard 0.92) |
| `archive/` | `instruktsiia-2018`, `instruktsiia-2021`, `instruktsiia-2021-copy` | a revision changing one storage temperature, plus a byte-identical re-upload | `hash` (raw), `lexical` (Jaccard 0.91) |
| `edoc/` | `rehlament-2020`, `rehlament-2023`, `rehlament-2023-portal` | a revision changing one approval deadline, plus an upper-cased re-issue on a third system | `hash` (normalized), `lexical` (Jaccard 0.87) |
| `travel/` | `poriadok-2021`, `poriadok-2024` | a revision changing one per-diem amount | `lexical` (Jaccard 0.90) |
| `privacy/` | `polityka-2022`, `polityka-2022-copy` | a byte-identical re-upload | `hash` (raw) |
| `hr/` | `polozhennia-2020`, `polozhennia-2023` | a revision changing one retention period | `lexical` (Jaccard 0.89) |

Two notes are absorbed whole rather than revised:

| note | sits inside | containment | why it is not the same document |
| --- | --- | --- | --- |
| `appeals/pamiatka-stroky-rozghliadu.md` | the 2022 appeals editions | 1.00 | a one-section extract of a three-section regulation |
| `privacy/dovidka-zghoda-subiekta.md` | both privacy copies | 1.00 | the consent section of a longer policy |

The appeals note is deliberately an extract of the section the 2022 revision CHANGED, so its
containment in the 2019 edition is 0.735 -- below the tier's 0.9 cutoff. That near miss is the
corpus's negative control for containment, and it is the one unreported pair the fit ranks among
the reported ones.

The eight remaining documents (`procurement/`, `inventory/`, `safety/`, `library/`, `access/`,
`audit/`, `energy/`, `civil/`) are unrelated subjects that share one of two formulaic closing
sentences with each other and with the families above. That is what makes them candidate pairs
without making them relations: the corpus's 140 candidate pairs carry 20 relations and 120
non-relations, so a fit that merges everything fails the fixture as loudly as one that merges
nothing.

## What it is built to pin down

**Two measures, one identity question.** The lexical tier reports a `duplicate` off a Jaccard and a
`subsumed_by` off a containment. The two numbers are not on one scale, so the list a reviewer reads
has no order across relations. The plant carries both relations at high scores -- every subsumption
here has containment 1.0 -- so a fit that ranks them well must be combining evidence rather than
re-reading one measure.

**Subsumption is not identity.** The two notes must be scored and ranked, and must NOT be merged
into the documents that absorbed them. A lane that puts a note in its parent's edition group has
answered a different question from the one linkage asks.

**Editions cross the governance fields.** A re-ingested edition carries a NEW `effective_date` and
often a different `source_system` (`docflow`, `sharepoint`, `portal`, `wiki` all appear, and
`appeals/polozhennia-2022-sharepoint.md` and `edoc/rehlament-2023-portal.md` cross systems). A
model that prices a date gap or a system change as strong non-match evidence loses exactly the
pairs this corpus is about.

**The current edition is sometimes a tie.** `archive/instruktsiia-2021.md` and its byte-identical
copy carry one date, and so do the two privacy copies. A group like that has no single current
edition, and the lane must say so instead of naming an arbitrary member.

Governance front matter (`version`, `effective_date`, `source_system`, `language`) is what lets the
edition ordering resolve, and content hashing excludes it -- two byte-identical documents carrying
different dates still read as duplicates.
