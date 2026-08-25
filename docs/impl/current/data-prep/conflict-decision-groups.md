# Corpus Hygiene: Decision Groups and the Counts They Carry

How many decisions a conflict audit's row count actually is. Detection, the effort tiers, the
relation vocabulary, and the claim-tier precision block live in
[conflict detection](conflict-detection.md); what a resolution policy then does with a group lives
in [conflict resolution](conflict-resolution.md). This page owns the counting: the distinct-unit
census, the two grouping rules and the decision RANGE between them, the `to decide` / `to review`
split, the optional policy projection, and the `groups.json` sidecar every consumer joins on.

## The count and the units behind it

A finding COUNT is the first number an operator reads, and it is the number they size a review
budget against -- so on a corpus whose conflicts concentrate it is the number most able to mislead.
The clustered precision bound already refuses to certify a floor when the conflicts sit on a handful
of chunks, but that refusal lived inside the precision section while the headline count, the
relation table, and the findings list still read as N independent results.

Every count the audit prints now carries the distinct units it rests on:
`report.md`'s headline line, every row of its relation table, the findings section, the CLI summary,
and `summary.json` (`finding_census`, plus `relation_census` beside the unchanged `relations`
counts). `src/llb/conflicts/grouping/census.py` computes the census and the grouping;
`src/llb/conflicts/report/findings.py` renders the grouped findings section.

- **Unit.** The CHUNK each side of a finding rests on, falling back to its document for the `hash`
  and `lexical` tiers, which compare whole documents and carry no chunk.
- **Census.** `documents`, `document_pairs`, `chunk_units`, `groups`, `largest_group` -- rendered as
  one line beside the count: `8 findings on 4 documents / 3 document pairs / 9 chunk units, in 2
  groups (largest 6)`.
- **Group.** Findings joined **transitively** by a shared unit, the same closure the `hash` tier
  applies to duplicate groups. A chunk that conflicts with six neighbours is ONE group, and so are
  three copies of one document: a shared unit is what makes two rows the same piece of evidence.
  The report leads the findings section with a **Decision groups** table (rows, `to decide`,
  relations, shared unit, documents, top score) and prints every row below it under its group label.
  `groups` is what an operator triages; `findings` is what a resolution lane consumes. The closure
  is one of [two measured grouping rules](#how-many-decisions-the-row-count-is), and it is the one
  the audit quotes.
- **Ranked by stake, named by file order.** The two are deliberately separate. A group's ID comes
  from `findings.jsonl` in file order and is the join key `groups.json` and `plan.json` share, so it
  never moves. The decision TABLE is ordered by `stake_key`
  (`src/llb/conflicts/report/findings.py`): `to decide` rows first, then rows, then top score, then
  the id. `G3` leading the table is therefore a ranking, not a renumbering, and the row table below
  it stays in the file's order -- the order a resolution lane consumes.
- **Rendering only.** `findings.jsonl` keeps one line per row, byte-identical and in the same order
  as before -- the resolution lane reads rows, and nothing here suppresses, merges, or deduplicates
  one. `tests/llb/conflicts/test_finding_census.py` asserts that, the one-group collapse, the
  transitive closure, the document-tier fallback, and the census beside each printed count;
  `test_group_ranking.py` asserts the ranking never renumbers a group.

Why the table is not ranked on score: a score is the model's confidence in ONE pair and says nothing
about how much the group holding it is worth. Measured on the goods semantic bundle (100 rows, 6
groups) a 0.002 score difference put the 14-row decision ahead of the 29-row one; on the goods
budget-100 claim bundle four groups tie at 1.000, so three of them were ordered by the
claim-identity tiebreak underneath -- a document id. Ranked by stake, the 29-row decision moves
above the 14-row one and the 3-row decision above the 2-row one, while the group holding the only
actionable row still leads the claim bundle.

### To decide and to review are two counts, never one

A decision group carries TWO counts of its work, and they are not the same number:

| count | what it counts | where it is computed | who prints it |
| --- | --- | --- | --- |
| **to decide** (`decide_rows`) | rows whose RELATION is not `complementary` (`is_actionable`) | detection, from the relation alone | `report.md` headline, decision table, precision block; `plan.json` decisions |
| **to review** (`review_rows`) | rows whose POLICY outcome is `review_required` | resolution, from `(relation, tier, governance, policy)` | `plan.json` decisions, the CLI summary, `resolution_review.jsonl`; the audit only when [asked to project it](#projecting-the-review-count-one-command-earlier) |

Neither can serve both roles, and the reason is structural rather than an implementation gap: the
review count is a property of a resolution POLICY, and the audit runs before an operator has chosen
one. That is why the audit prints only `to decide`, names `to review` in the same breath, and says
where it lives; and why `plan.json` -- the only artifact that holds both -- prints them side by side
per group. The vocabulary is defined once in `src/llb/conflicts/constants.py` (`DECIDE_LABEL`,
`REVIEW_LABEL`, `decide_count`); `FindingGroup.decide_rows` and `group_decisions` are its two
consumers.

They diverge in BOTH directions, measured 2026-08-12 on the RTX PRO 3000 Blackwell 12 GiB CUDA host
over the two goods bundles:

- **Goods budget-100 claim bundle** (5 markdown documents, claim tier at `MAX_CANDIDATE_PAIRS=100`;
  lookup key `corpus-conflicts` run `two-counts-census-goods-budget100`):
  **1 to decide, 0 to review.** The single `subsumed_by` row is work by the relation vocabulary --
  it is what the precision block measures and what leads `findings.jsonl` -- but the conservative
  policy resolves subsumption as `keep_both` plus an annotation, so it costs a human nothing. An
  operator funding one review off the audit report would have funded a review that does not exist.
- **Goods semantic bundle** (the same 5 documents at the semantic tier; lookup key
  `corpus-conflicts` run `two-counts-goods-semantic`):
  **100 to decide, 100 to review**, group for group (51/51, 29/29, 14/14, 3/3, 2/2, 1/1). Every
  semantic-tier duplicate needs review under every policy because that tier has no deletion
  authority, so the two counts coincide exactly -- and this is the bundle an operator meets first,
  which is precisely why one name for both was survivable long enough to become misleading.

**Which count each ranking uses is stated where the ranking is.** The audit's decision table ranks
on `to decide` because it has nothing else; the review ledger ranks on `to review`, the count an
operator actually funds, because by then the policy has run. On the goods semantic bundle that moves
the ledger from id order (51, 14, 29, 3, 1, 2 rows) to review stake (51, 29, 14, 3, 2, 1), so the
reviewer meets the biggest open decision first; a group's records stay one contiguous block either
way. `tests/llb/conflicts/test_two_counts.py` pins the divergence on a mixed fixture (one claim-tier
duplicate the policy accepts plus one semantic-tier duplicate it escalates: 2 to decide, 1 to
review), the naming in both artifacts, and the ledger ranking.

### Projecting the review count one command earlier

The audit can name **to review** but cannot measure it, so an operator sizing a review budget off
`report.md` alone has to run the resolver to learn what their corpus costs. `resolve_finding` is a
pure function of `(relation, tier, governance, policy)` and `findings.jsonl` already carries all
four, so the audit can PROJECT that count under the policies the operator names -- one command
before the resolver runs, with no second model call and no second implementation of the number.

Naming ONE policy answers "what does my corpus cost?". Naming several also answers the question an
operator asks immediately afterwards -- "what would the other policy cost?" -- because each extra
policy is one more pass over rows the audit already holds and no model call at all.

```bash
make audit-corpus-conflicts CORPUS=<dir> EFFORT=semantic STORE=<store> \
  PROJECT_POLICY=conservative,prefer-newer
# or: llb audit-corpus-conflicts --project-policy conservative,prefer-newer
```

- **Opt-in, and off by default.** Without the flag nothing about the report changes. Measured: a
  control run of the goods semantic bundle re-rendered with this code is byte-identical to the
  bundle rendered before it existed, apart from the corpus path and one tier's `seconds` column (the
  `projection-control-goods-semantic` report against the `two-counts-goods-semantic` one, both
  `corpus-conflicts` runs of 2026-08-12 on the 12 GiB Blackwell host). Both of those reports predate
  the [granularity section](#how-many-decisions-the-row-count-is), which every report now carries;
  the claim is about the projection, so re-checking it means comparing two reports rendered by the
  same code with and without the flag.
- **What it adds.** A headline line per policy, one `to review (projected, <policy>)` column per
  policy in the decision-groups table, and `policy_projection` in `summary.json`
  (`schema_version: 3`, `kind: projection`, `basis`, `policies`, `by_policy` with each policy's
  `review_rows` / `review_groups` / per-group counts, and `deltas`). The per-group counts are keyed
  by the same group ids `groups.json` and `plan.json` use. The CLI echoes the same numbers.
  `findings.jsonl` and `groups.json` are untouched: the projection depends on a policy, and those
  two artifacts must stay readable by a consumer that has not chosen one. A delta also brings
  [the governance coverage behind it](#the-precondition-behind-a-zero-delta), which is recorded on
  every run whether or not a policy was named.
- **One document shape, whatever N is.** The FIRST policy named is the baseline and its own counts
  stay at the top level of `policy_projection` (`policy`, `review_rows`, `review_groups`,
  `groups`), exactly where a single-policy consumer already reads them; `policies` / `by_policy` /
  `deltas` are added beside them. `project_review_rows(rows, policy)` is the one-policy case of
  `project_policies(rows, policies)` and returns the same document, so there is never a second
  schema to branch on.
- **The delta is the answer, not the columns.** `deltas` carries one entry per non-baseline policy:
  `review_rows` (signed, negative means the switch removes review work), the same per group, and
  `moved_groups` -- WHICH decisions the choice touches. A corpus-wide total of `-2` hides whether
  that is one group changing or twenty cancelling out, which is why the group list rides with it.
  A delta cell is rendered `+2` / `-2` / `0` through one helper so it can never be misread as a
  count, and `0` is bare rather than `+0`.
- **And the delta reads as a SHARE, because a sign is not a magnitude.** `-2` on a 17-row fixture
  and `-2` on a corpus of thousands are the same sign and different decisions, so each delta also
  carries `moved_rows`, `actionable_rows`, and `moved_share`, rendered as
  `moves 2 of 9 actionable rows (22.2%)` in the headline and the CLI through one helper
  (`moved_share_phrase`). The denominator is the audit's own **to decide** set (`decide_count`,
  the one `is_actionable` definition), so the share is measured against the rows the audit already
  calls work rather than against a second notion of "relevant". `moved_rows` is a GROSS count of
  rows whose resolved status differs, not the net: two rows moving opposite ways inside one group
  would cancel in `review_rows` and still cost an operator two decisions -- which the two shipped
  policies [cannot yet do](#what-the-policy-choice-costs-measured). `moved_share` is `null`, never
  `0.0`, on a corpus with nothing to decide -- no work at all and work the choice never touches
  are opposite readings.
- **A projection, never a measurement**, said at every appearance -- the flag, every headline line,
  the column paragraph, the CLI lines, and the artifact's own `kind`/`basis` fields. Each column is
  only true of the policy it names, the delta inherits that caveat from both columns it subtracts,
  and a reviewer's decisions can still move any of them.
- **One policy renders exactly the column it always did.** With a single `--project-policy` value
  the header is `to review (projected)` with no policy suffix and no delta column, and the report
  is byte-identical to the one the single-policy path produced. Measured: the group-table header of
  the `policy-choice-goods-single` report is identical to the one in the
  `projected-review-goods-semantic` report (both `corpus-conflicts` runs of 2026-08-12 on the 12 GiB
  Blackwell host), and a CI test asserts the two rendering paths agree byte for byte on the same
  rows.
- **Equal to what the resolver measures, column by column.** Each column replays the shipped
  `resolve_finding` over the audit's own rows and counts `review_required` per group, so it must
  equal `plan.json`'s `review_rows` group for group under that policy -- N columns are N READINGS
  of one implementation, not N implementations.
  `tests/llb/conflicts/test_policy_projection.py` pins that equality per column on a fixture that
  separates the policies (a dated supersession plus an undated contradiction as the control), and
  pins the delta as the difference of the two columns rather than a third computation. The ranking
  does not change: the decision table still ranks on `to decide`, the only count that exists
  without a policy.

**The layering decision.** The projection needs the resolution vocabulary and the report must not
have it, so it is composed ABOVE both layers: `src/llb/conflicts/resolution/projection.py` imports
`resolution_policy`, the CLI calls it and puts the plain JSON result on
`AuditResult.policy_projection`, and `report.py` / `report/findings.py` / `report/projection.py`
render data they never derive -- `report/projection.py` owns every way a projected count is allowed
to appear, so the headline, the columns, and the prose cannot drift apart. `conflicts.report*`
therefore still imports nothing from `conflicts.resolution_*`, which `test_policy_projection.py`
asserts by discovering every `report*.py` and reading its import lines rather than by convention --
so a renderer added later is held to the rule without anyone remembering to list it.

The cheaper alternative -- the report simply pointing at the resolver -- was kept as
well rather than replaced: every mention of the projection names `resolve-corpus-conflicts` and
where the measured count lives, because the resolver without `--apply` already plans without
touching a corpus byte and remains the only thing that MEASURES the count.

Measured 2026-08-12 on the 12 GiB Blackwell CUDA host over the two goods bundles, which are the two
ends of the divergence:

- **Goods semantic bundle** (100 rows, 6 groups; lookup key `corpus-conflicts` run
  `projected-review-goods-semantic`): projected `{G1: 51, G2: 14, G3: 29, G4: 3, G5: 1, G6: 2}` =
  100 rows in 6 groups, equal group for group to the `review_rows` the resolver then wrote into
  `plan.json` in the same directory, and equal to `decide_rows` as well -- every semantic-tier
  duplicate is to review under every policy. Here the projection confirms the headline instead of
  correcting it -- and this is the bundle an operator meets first, which is exactly why the
  coincidence is easy to mistake for a rule that the claim bundle below breaks.
- **Goods budget-100 claim bundle**
  (lookup key `corpus-conflicts` run `projected-review-goods-budget100`; RTX PRO 3000
  Blackwell 12 GiB, 954-chunk store at resolved cosine 0.3604, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M
  agreeing with all 24 frozen probe pairs, 8 min 52 s of adjudication): the divergent end. The
  report reads `to decide: 1 of 100 rows` and `to review (PROJECTED under policy conservative): 0
  rows in 0 decision groups` -- the corpus's one actionable row is a `subsumed_by` the conservative
  policy settles as `keep_both`, so it costs a human nothing. The resolver run on the same rows
  then measured `review_rows` 0 in every one of the six groups. An operator funding one review off
  the audit report alone would have funded a review that does not exist; the projection says so one
  command earlier, and says under which policy.

#### What the policy choice costs, measured

The two policies part in exactly one place: a **dated supersession**. `conservative` escalates it
to `review_required`; `prefer-newer` suppresses the older side and accepts it. Every other relation
resolves the same way under both. So a corpus with no dated supersession has a zero delta by
construction, and both goods bundles are that corpus -- which is why the delta column needed a
third bundle to be worth printing at all.

A sign, though, only answers WHETHER the choice is a choice. `-2` on a 17-row fixture and `-2` on a
corpus of thousands are the same sign and completely different decisions, so every delta is also
read as a SHARE of the rows the audit calls work: `moved_rows` of `actionable_rows`, rendered
`moves 2 of 9 actionable rows (22.2%)`. That is the number that transfers between corpora.

Measured 2026-08-13 on the RTX PRO 3000 Blackwell 12 GiB CUDA host, MamayLM-Gemma-3-12B-IT-v2.0
Q4_K_M adjudicating, one claim-tier run per corpus present on this host with
`PROJECT_POLICY=conservative,prefer-newer`. Every run agreed with all 24 frozen probe pairs before
its rows were adjudicated.

| bundle | rows | to decide | `conservative` | `prefer-newer` | delta | groups moved | rows moved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conflicts fixture, claim | 17 | 9 | 2 | 0 | **-2** | G3 | **2 of 9 (22.2%)** |
| goods, claim budget 100 | 100 | 2 | 0 | 0 | 0 | none | 0 of 2 (0.0%) |
| quickstart-PDF, claim budget 100 | 100 | 0 | 0 | 0 | 0 | none | 0 of 0 |
| goods, semantic (re-read) | 100 | 100 | 100 | 100 | 0 | none | 0 of 100 (0.0%) |

The first three rows are the `corpus-conflicts` runs `policy-share-fixture-claim`,
`policy-share-goods-claim`, and `policy-share-quickstart-pdf-claim`. The fourth is the committed
semantic bundle (`policy-choice-goods-semantic`, 2026-08-12) re-read through `project_policies`: the
share is a pure function of the rows, so an audit already on disk gets its share back with no model
call and no store.

- **The corpus where the choice is not free** is the committed fixture at
  `samples/corpora/conflicts_uk_v1/`, which already plants a dated supersession (the 2021 vs 2024
  thirty-versus-fifteen-day deadline, with `effective_date` on both documents) -- so nothing had to
  be planted for this. Run: `make audit-corpus-conflicts
  CORPUS=samples/corpora/conflicts_uk_v1/corpus EFFORT=claim STORE=<heading@600 store>
  MIN_CLAIM_TOKENS=10 PROJECT_POLICY=conservative,prefer-newer` (lookup key `corpus-conflicts` run
  `policy-share-fixture-claim`; 19-chunk `multilingual-e5-base` `heading@600` store, 14 rows
  adjudicated in 49 s). It returns **17 findings in 4 groups, 9 to decide**, including 2
  `superseded_by` rows, and the report reads `policy choice conservative -> prefer-newer: -2 rows to
  review, falling in 1 decision group (G3). The choice moves 2 of 9 actionable rows (22.2%)`. Both
  columns were checked against the thing they project: running `resolve-corpus-conflicts` on the
  same rows under each policy wrote `plan.json` decisions equal to the projection group for group
  (`{G1: 0, G2: 0, G3: 2, G4: 0}` and all zeros), in that same run's `resolve-conservative/` and
  `resolve-prefer-newer/` sub-bundles. The same check passed on the other two bundles, where both
  policies leave every group at zero.
- **The corpora where it is free are free for a reason that is not about their knowledge.**
  `superseded_by` is derived from `compare_editions`, which needs `effective_date` or `version` on
  BOTH sides, and neither quickstart corpus carries either field on any document: 0 of 5 goods
  documents and 0 of 3 quickstart-PDF documents, against 7 of 7 in the fixture. Their zero delta is
  therefore a property of how they were ingested (PDF conversion, no governance front matter), not
  a finding that their revisions agree under both policies. The audit now draws that distinction
  itself, beside the delta: see
  [the precondition behind a zero delta](#the-precondition-behind-a-zero-delta).
- **A zero share and no share are different answers.** goods claim reads `0 of 2 actionable rows
  (0.0%)` -- there IS work to decide (2 `subsumed_by` rows) and the policy choice touches none of
  it. quickstart-PDF reads `0 of 0 actionable rows` with `moved_share: null` -- its 100 rows are
  every one `complementary`, so there is no work for a policy to move in the first place. The
  artifact keeps them apart (`null` is never written as `0.0`) because they lead an operator to
  opposite next steps.
- **The share counts rows that MOVED, not the net -- and today those are the same number.**
  `review_rows` is a signed per-group total, so two rows moving opposite ways inside one group
  would cancel to zero while still costing two decisions; `moved_rows` is the gross count and
  cannot. The two shipped policies cannot actually produce that case: `prefer-newer` differs from
  `conservative` on one relation in one direction (it settles a dated supersession instead of
  escalating it) and never turns an accepted row back into review work, so `|delta| == moved_rows`
  on all four bundles above and on any corpus this pair is run over. The gross count is therefore
  carried against a third policy rather than proven necessary by these corpora, and
  `tests/llb/conflicts/test_policy_projection.py` pins both halves: that the shipped pair only
  settles rows, and that a cancelling delta renders as the rows it moved instead of as **no
  difference**.

Three limits of this measurement, all worth stating because they bound what the 22.2% means:

- **The only non-zero share on this host comes from a planted corpus.** The fixture is 7 small
  documents built to contain one dated supersession, so 22.2% is the share of a corpus designed to
  have one, not an estimate of what a production corpus carries. The 8-document HR corpus is
  operator data and absent from this host; measuring a real dated corpus stays open work in
  [`plan.md`](../../plan.md) (`conflict-policy-delta-on-an-operator-corpus-with-dated-revisions`).
- **Group ids are stable inside a run, not across two.** The fixture was audited twice -- the
  `corpus-conflicts` runs `policy-choice-fixture-claim` (2026-08-12) and
  `policy-share-fixture-claim` (2026-08-13) -- and returned the same 17 rows, the same relations,
  and the same document pairs both times -- but the adjudicator's scores are not bit-reproducible,
  the row order is score-ranked, and so the group holding the supersession was `G4` in the first run
  and `G3` in the second. Nothing joins across runs today (`source_findings_sha256` pins a plan to
  its own rows), so no artifact is wrong; a reader comparing two audit reports by group label would
  be. A row-derived group key is tracked in [`plan.md`](../../plan.md)
  (`conflict-group-ids-that-survive-a-re-run`).
- **And the share itself is not reproducible across runs.** A third audit of the same fixture at
  the same settings (`corpus-conflicts` run `governance-coverage-fixture-claim`, 2026-08-13: same 7
  documents, same 19-chunk store, same 24/24 frozen probe, same 14 adjudicated rows, same 0.4286
  claim-tier precision)
  returned the same 17 rows and 9 actionable rows but a different relation MIX: one row the earlier
  two runs called `superseded_by` came back `subsumed_by`, so the delta read **-1 row, 1 of 9
  (11.1%)** in `G4` instead of -2 and 22.2%. The adjudication endpoint runs at temperature 0.2 with
  no seed, so a relation on a borderline pair is a sample rather than a constant -- and the
  policy-choice share inherits that variance directly, because it is a count of one relation.
  Quoting 22.2% as the fixture's share therefore over-states the precision of a single run; the
  spread is unmeasured, and measuring it is tracked in [`plan.md`](../../plan.md)
  (`conflict-policy-share-across-repeat-audits-of-one-corpus`).

#### The precondition behind a zero delta

A zero delta has two opposite readings and the delta cannot tell them apart: the corpus may carry
dated revisions the two policies agree on, or it may carry no governance dates at all -- in which
case `superseded_by` can never be derived, the zero is a property of the INGESTION rather than of
the knowledge, and it is fixed where the corpus is built rather than where it is reviewed. Every
corpus on this host that reports a zero is the second case, so the audit reports the PRECONDITION
beside the delta rather than leaving an operator to read "the choice is free here" off a run that
could not have said anything else.

`src/llb/conflicts/governance/coverage.py` counts it at the three levels it can be missing at,
using `compare_editions` -- the same orderability test that promotes a dated contradiction to
`superseded_by`, so the precondition cannot drift from the thing it is a precondition for:

| level | field | what it means |
| --- | --- | --- |
| corpus | `dated_documents` of `documents`, plus `documents_by_field` | documents recording `effective_date` or `version`; zero here means no run over this corpus can ever produce a non-zero delta |
| corpus | `orderable_document_pairs` of `document_pairs` | the corpus's OWN document pairs that function can order -- what the corpus could have supplied, with no candidate list and no store involved |
| run | `orderable_pairs` of `returned_pairs`, plus `orderable_share` | returned pairs whose two sides that function actually orders; the stricter count, and the one a policy is replayed over |

The three are different questions. A corpus can date every document and still have nothing to
order, because two documents carrying the SAME date order no better than two undated ones -- which
is why the middle count is not implied by the first. And it can have orderable document pairs and
still return none of them, which is why the middle count is not implied by the last.
`orderable_share` is `null` rather than `0.0` when a run returned no pair at all, the same
distinction `moved_share` draws one level up.

The middle count is what names the STAGE a run lost an orderable pair at, and it is the reason the
level exists: two corpora an operator would fix in opposite ways used to print the same structural
line. `orderable_document_pairs` is derived from the distinct ordering KEYS rather than by
enumerating pairs -- inclusion-exclusion over the date-key and version-key multisets
(`document_pair_orderability`, with `edition_key` in `src/llb/conflicts/governance/editions.py` as
the shared key function) -- so a corpus of thousands of documents stays off a quadratic path. The
unit test pins that count against enumerating `compare_editions` over every pair of every corpus
drawable from a governance pool covering present, absent, blank, shared, and unparseable fields.

**Where it appears.** `governance_coverage` rides in `summary.json` on every run, projection or
not -- it is detection-side and policy-free (`schema_version` 3 since the stage attribution below
joined it). The READING is printed once beside the delta, in `report.md` and in the CLI through the
same helper, and only where a delta exists: with one policy there is no choice to call free, and
with no `--project-policy` the report is unchanged. A non-zero delta carries the counts without a
reading, because the delta already is one. A zero delta gets one of three, and the stage is the
whole content of the difference:

| coverage | reading | where the fix is |
| --- | --- | --- |
| a returned pair orders | about the corpus's **KNOWLEDGE**: dated pairs were reachable and the policies settled them the same way | nowhere -- this is evidence |
| no returned pair orders, but document pairs do | **STRUCTURAL for this RUN**, and the stage that lost the orderable pair is **RETRIEVAL** | the candidate list -- narrowed to one knob by the [stage attribution](#which-stage-lost-the-orderable-pair), or listed as all four when the run recorded none |
| nothing orders at either level | **STRUCTURAL**, and no run over this corpus could have differed | **INGESTION**: record `effective_date` or `version` (or distinct ones -- one edition on every document counts as none) |

`tests/llb/conflicts/test_governance_coverage.py` pins the counts, all three readings, and the gate
-- three corpora audited the same way whose coverage reads at three different stages and whose
delta is zero on all three, so the coverage is never an input to the delta.

Measured over the four bundles the [policy-choice table](#what-the-policy-choice-costs-measured)
above was measured on, recomputed from each `findings.jsonl` plus its corpus with no model, no
store, and no re-adjudication (`pair_orderability` reads rows alone; only the document count needs
the corpus):

| bundle | dated documents | orderable document pairs | orderable returned pairs | delta | reading |
| --- | --- | --- | --- | --- | --- |
| conflicts fixture, claim | 7 of 7 | 20 of 21 | 16 of 17 (0.941) | **-2** | non-zero; the counts ride with it |
| goods, claim budget 100 | 0 of 5 | 0 of 10 | 0 of 100 (0.0) | 0 | STRUCTURAL, INGESTION |
| quickstart-PDF, claim budget 100 | 0 of 3 | 0 of 3 | 0 of 100 (0.0) | 0 | STRUCTURAL, INGESTION |
| goods, semantic | 0 of 5 | 0 of 10 | 0 of 100 (0.0) | 0 | STRUCTURAL, INGESTION |

No committed bundle reads as a retrieval miss: every corpus on this host that reports a zero either
carries no governance date anywhere (the three undated rows, where re-ingesting IS the fix) or is
the planted fixture, whose returned pairs order. The stage split was therefore demonstrated on a
purpose-built dated corpus rather than found in the archive; the two runs below are that
demonstration.

The shipped command was then run end to end on the fixture to check the live path rather than the
recompute (2026-08-13, RTX PRO 3000 Blackwell 12 GiB, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M, 24/24
frozen probe pairs, 14 rows adjudicated in 54 s; lookup key `corpus-conflicts` run
`governance-coverage-fixture-claim`):

```text
make audit-corpus-conflicts CORPUS=samples/corpora/conflicts_uk_v1/corpus EFFORT=claim \
  STORE=<19-chunk heading@600 store> MIN_CLAIM_TOKENS=10 \
  PROJECT_POLICY=conservative,prefer-newer
[conflicts] policy choice conservative -> prefer-newer: -1 rows to review in G4; moves 1 of 9
  actionable rows (11.1%)
[conflicts] governance coverage: 7 of 7 documents with `effective_date` or `version`
  (7 `effective_date`, 7 `version`), 16 of 17 returned pairs orderable by `compare_editions` --
  the rows the choice moves are drawn from those orderable pairs.
```

That run is also where the [third limit](#what-the-policy-choice-costs-measured) above came from:
its delta is -1 rather than the -2 the two earlier audits of the same corpus measured, while its
coverage is identical to theirs. The precondition is a property of the corpus and reproduces; the
delta is a property of one adjudication and does not. Its coverage line predates the document-pair
clause and is quoted as it was printed; the fixture's document-pair count (20 of 21) comes from the
recompute above and from the hash-tier run below.

**The stage split, run end to end.** Three documents, all dated, all versioned: two byte-identical
copies of one edition and a third carrying a later edition of a claim that contradicts them
(`.data/corpus-governance-stage-demo/`). The hash tier returns exactly one pair -- the two copies,
which share an edition and order no better than two undated documents -- so the run has orderable
document pairs and no orderable returned pair, which is the reading the count was added for
(2026-08-13, 12 GiB Blackwell host; lookup key `corpus-conflicts` run
`doc-pair-orderability-retrieval-miss-hash`):

```text
make audit-corpus-conflicts CORPUS=<dated-corpus> EFFORT=hash \
  PROJECT_POLICY=conservative,prefer-newer
[conflicts] policy choice conservative -> prefer-newer: 0 rows to review -- the choice is free on
  this corpus; moves 0 of 1 actionable row (0.0%)
[conflicts] governance coverage: 3 of 3 documents with `effective_date` or `version`
  (3 `effective_date`, 3 `version`), 2 of 3 document pairs and 0 of 1 returned pair orderable by
  `compare_editions` -- so the zero above is STRUCTURAL for this RUN, and the stage that lost the
  orderable pair is RETRIEVAL, not ingestion: this corpus DOES carry document pairs
  `compare_editions` can order, and none of them reached the rows the audit returned. Fixable
  where the candidate list is built (raise `--effort` or `--max-candidate-pairs`, re-chunk, or
  re-embed), not by re-ingesting the corpus.
```

That four-knob tail is quoted as it was printed; the same command now ends in the one knob the
[stage attribution](#which-stage-lost-the-orderable-pair) picked (`corpus-conflicts` run
`stage-attribution-effort-hash`).

Before this count, that run printed the ingestion line -- advice that would have changed nothing on
a corpus already dated end to end. Taking the advice it prints now recovers the pair: the same
corpus at `--effort semantic` against a store of its own returns the cross-edition pair as well,
and the reading moves to KNOWLEDGE with the delta still zero (2026-08-13, 12 GiB Blackwell host;
lookup key `corpus-conflicts` run `doc-pair-orderability-retrieval-recovered-semantic`, against a
store built over the demo corpus itself; the documents are one sentence each, so the run needs
`MIN_CLAIM_TOKENS=8` and `COS_THRESHOLD=0.7` to keep them above the claim floor):

```text
make audit-corpus-conflicts CORPUS=<dated-corpus> EFFORT=semantic STORE=<its-own-store> \
  COS_THRESHOLD=0.7 MIN_CLAIM_TOKENS=8 PROJECT_POLICY=conservative,prefer-newer
[conflicts] governance coverage: 3 of 3 documents with `effective_date` or `version`
  (3 `effective_date`, 3 `version`), 2 of 3 document pairs and 1 of 2 returned pairs orderable by
  `compare_editions` -- so the zero above is about this corpus's KNOWLEDGE: a dated supersession
  was reachable on this run (these pairs carry the fields that promote one), and none of them
  became one the policies part over.
```

The corpus-side counts are identical across the two runs (3 of 3 documents, 2 of 3 document pairs),
which is the point: the corpus did not change between them, only the stage that chose the
candidates -- and only the run-level count moved.

**Orderability is necessary, not sufficient -- by a wide margin.** The fixture's 16 of 17 returned
pairs are orderable and only 2 rows move: a pair must also be adjudicated `contradicts` before the
dates promote it to `superseded_by`, so the orderable share is a loose upper bound on the delta and
must never be read as a prediction of one. What it does rule out is the opposite error, which is
the one an operator actually makes: a zero delta on a corpus with no orderable pair is not evidence
about the corpus at all.

Surfacing the corpus-side counts at ingestion time -- where the third reading says the fix belongs
-- is tracked in [`plan.md`](../../plan.md)
(`corpus-ingestion-reports-the-governance-coverage-the-audit-blames-it-for`).

#### Which stage lost the orderable pair

The reading above names RETRIEVAL, and RETRIEVAL is four knobs at once. The stage attribution picks
between them on ONE pair -- an orderable document pair no returned row joins -- with the stage read
off what the run already recorded. Three modules: `governance/stage_rule.py` (the stage vocabulary
and which stage one pair stopped at), `governance/stage_search.py` (which lost pair is named), and
`governance/stage.py` (the attribution payload and its sentence). Nothing here re-runs detection,
moves a threshold, or looks at a pair the corpus cannot order.

| stage | what places it there | the one knob |
| --- | --- | --- |
| `effort` | the run read no store, so only whole documents were ever compared | raise `--effort` to `semantic` or `claim` |
| `duplicate_collapse` | a side has no chunk in the store because the hash tier proved it a copy of one that does | none -- read the pair through the copy the store kept |
| `chunking` | a side has no chunk in the store, and no copy of it does either | rebuild the store over this corpus, or re-chunk it |
| `claim_token_floor` | a side's chunks are in the store and every one is excluded from comparison | lower `--min-claim-tokens` to the value the record names, or re-chunk ([bundle record](conflict-bundle-record.md#why-a-document-is-not-comparable-and-the-floor-that-returns-it)) |
| `candidates` | both sides are comparable and the pair still never reached a row | raise `--max-candidate-pairs`, or lower the cosine threshold |

The stages are tried in that order, which is the order a pair meets them: a document the store
never held cannot have met a threshold. That is also how a pair whose two documents stopped at
different stages is attributed -- the earlier stop is what explains the pair.

**Which pair is named: the earliest stage, not the first pair in corpus order.** A run can lose
pairs at several stages at once and only one pair is reported, so the choice IS the reading. Corpus
order would name whichever lost pair's documents sort first, which is a file name rather than a
diagnosis: on the 7-document fixture that pair is `archive-policy.md` + `deadline-note.md` at
`candidates`, two unrelated documents that were never going to pair, so a corpus with one genuine
chunking gap and many merely unrelated pairs would report a candidate budget and never mention the
gap. The earliest stage is reported instead, with the corpus-first pair that demonstrates it, and
it is the one an operator has to fix first anyway.

**Except the stage whose knob is `none`, which sorts last.** `duplicate_collapse` is second in
pipeline order and LAST in report order, because it is not a loss anyone acts on -- the claim is in
the store the whole time under the copy that survived -- so it can never displace a stage that names
real work, and it is reported only when a run lost nothing else. Not hypothetical: the demo corpus's
claim-floor run below loses `handbook_2026.md` + `policy_2024.md` at the floor AND
`handbook_2026.md` + `policy_2024_copy.md` at duplicate collapse, so a strict pipeline order would
answer "none -- read this pair through `policy_2024.md`" on a run whose actual knob is
`--min-claim-tokens`. The report order is DERIVED from `STAGE_KNOBS` rather than written out, so a
stage added with a knob takes its pipeline position automatically.

**What the scan costs.** Every stage below the effort dial is a property of one DOCUMENT rather than
of a pair (no stored chunk, no comparable chunk), so the documents that can demonstrate a stage are
found in a single pass over the corpus and only THEIR pairs are ever tested: linear in the corpus
per stage, with the quadratic sweep reached only for `candidates`, which is the cost the
corpus-order scan already paid. Every hit is confirmed against the pair rule, which stays the single
implementation of the stage order. CI pins the bound by counting the document pairs each rule tests
on a 60-document corpus whose only lost pair is the last one in corpus order: **59 against 1,770**.

The attribution rides in `governance_coverage` as
`lost_orderable_pair` (`documents`, `stage`, `reason`, `knob`) and prints as one sentence after the
counts, in `report.md` and the CLI alike. Where it is present the retrieval reading DROPS its
four-knob list; where it is absent -- a corpus that can order nothing, or a run that returned every
orderable pair it could have -- nothing is printed and nothing is invented. The per-document input
is `DocumentChunks`, folded once in `run_semantic_tiers` from the store's chunks, the tier's
comparable ordinals, and the hash tier's settled copies; below the semantic tier it is `None`, and
that absence IS the `effort` reading (with no stage to be earliest, corpus order is the whole rule
there). `DocumentChunks` lives in `src/llb/conflicts/bundle/document_chunks.py` and is written into
`summary.json` so the reading survives the store
([recomputing the stage from a finished bundle](#recomputing-the-stage-from-a-finished-bundle));
`tests/llb/conflicts/test_governance_stage.py` pins each stage, the earliest-stage rule, its cost
bound, and the silence.

**Measured 2026-08-13 on the 12 GiB Blackwell CUDA host, one run per stage** (the semantic runs read
real e5-base store vectors, no model call). Each row names its `corpus-conflicts` run:

| run | stage named | pair |
| --- | --- | --- |
| `stage-attribution-effort-hash` (demo corpus, `--effort hash`) | `effort` | `handbook_2026.md` + `policy_2024.md` |
| `stage-attribution-claim-floor` (demo corpus, `--effort semantic`, default floor) | `claim_token_floor` | the same pair |
| `stage-attribution-recovered` (demo corpus, `MIN_CLAIM_TOKENS=8`) | `duplicate_collapse` | `handbook_2026.md` + `policy_2024_copy.md` |
| `stage-attribution-fixture-semantic` (7-document fixture, 19-chunk store) | `candidates` | `archive-policy.md` + `deadline-note.md` |
| `stage-earliest-chunking-gap` (3-document corpus, store built before its third document) | `chunking` | `a-archive.md` + `z-travel.md` |

The three demo-corpus runs are one corpus read three ways, and they walk an operator through the
fix: at `--effort hash` the knob is the effort dial, and raising it moves the attribution to the
claim-token floor -- which is exactly the knob the recovery run above had to turn
(`MIN_CLAIM_TOKENS=8`, its documents being one sentence each) and which the four-knob list did not
mention at all. Lowering the floor then returns the cross-edition pair, and what is left is the
`duplicate_collapse` case: `policy_2024_copy.md` is byte-identical to `policy_2024.md`, the store
holds one chunk set for both, and the pair through the collapsed copy can never be returned. That is
the stage whose knob is *none*, and it is why "rebuild the store" is not the advice for every
chunkless document -- a rebuild would collapse the duplicate again.

That run re-executed under the earliest-stage rule (`corpus-conflicts` run
`stage-earliest-claim-floor`) names the same stage and the same pair, which is the knobless-stage
rule doing its job -- the corpus loses a second pair at duplicate collapse, and a strict pipeline
order would have answered with it:

```text
make audit-corpus-conflicts CORPUS=<dated-corpus> EFFORT=semantic STORE=<its-own-store> \
  COS_THRESHOLD=0.7 PROJECT_POLICY=conservative,prefer-newer
[conflicts] governance coverage: ... 2 of 3 document pairs and 0 of 1 returned pair orderable by
  `compare_editions` -- so the zero above is STRUCTURAL for this RUN, and the stage that lost the
  orderable pair is RETRIEVAL, not ingestion: ... Fixable where the candidate list is built, not by
  re-ingesting the corpus. Earliest stage an orderable document pair was lost at: the CLAIM-TOKEN
  FLOOR, shown by `handbook_2026.md` + `policy_2024.md` (every chunk of `handbook_2026.md` and
  `policy_2024.md` in the store is excluded from comparison -- front matter, below
  `--min-claim-tokens`, or a repeated metadata block). One knob: lower `--min-claim-tokens`, or
  re-chunk so the claim lands in a longer chunk.
```

The claim-floor sentence above is the reading BEFORE the per-document exclusion record: the
disjunction is what a run that kept one exclusion total could offer. A run under the current build
names the reason per document and the floor value that returns the pair ([bundle
record](conflict-bundle-record.md#why-a-document-is-not-comparable-and-the-floor-that-returns-it));
the stage and the pair are unchanged.

**Do the two rules ever disagree? Not on a single bundle this host had.** Recomputed over every
audit bundle on disk at the time -- each bundle's own rows from its `findings.jsonl`, the
per-document chunk accounting rebuilt from the store that run read, no model and no
re-adjudication. (Rebuilding that accounting from the store is exactly what
[the bundle record](#recomputing-the-stage-from-a-finished-bundle) replaced afterwards; those 23
bundles predate the record and now read as "not recomputable" rather than being re-derived from a
store that has since moved.)

| bundles | attribution | earliest stage vs first pair |
| --- | --- | --- |
| 11 (goods, quickstart-PDF) | none -- nothing orderable at either level | both rules silent |
| 12 (fixture x 6, demo corpus x 6) | `effort` x 4, `candidates` x 5, `claim_token_floor`, `duplicate_collapse` x 2 | **same pair on all 12** |

So the rule change is invisible in the archive, and the reason is corpus size rather than luck: the
two dated corpora on this host are 3 and 7 documents whose stores hold every document they were
built from, so each run loses its pairs at ONE stage with a knob. The change decides only a corpus
that loses pairs at two such stages -- which is the case the fixture in CI pins and the run below
builds, and which is what a real operator corpus looks like.

**The disagreement, run end to end** (2026-08-13, 12 GiB Blackwell CUDA host, real e5-base store
vectors, no model call; lookup key `corpus-conflicts` run `stage-earliest-chunking-gap`). Three
dated documents in a purpose-built demo corpus, two of them unrelated to each other, and a store
built over an earlier state of the corpus that did not yet contain the third -- the ordinary shape
of a store one ingest behind its corpus:

```text
make build-index CORPUS=<corpus-before-the-third-document> CHUNK_STRATEGY=heading CHUNK_SIZE=600
make audit-corpus-conflicts CORPUS=<dated-corpus> EFFORT=semantic STORE=<the-stale-store> \
  COS_THRESHOLD=0.9 MIN_CLAIM_TOKENS=8 PROJECT_POLICY=conservative,prefer-newer
[conflicts] governance coverage: 3 of 3 documents with `effective_date` or `version`
  (3 `effective_date`, 3 `version`), 3 of 3 document pairs and 0 of 0 returned pairs orderable by
  `compare_editions` -- so the zero above is STRUCTURAL for this RUN ... Earliest stage an orderable
  document pair was lost at: CHUNKING, shown by `a-archive.md` + `z-travel.md` (no chunk of
  `z-travel.md` is in the store the audit read). One knob: rebuild the store over this corpus, or
  re-chunk it.
```

The run loses all three of its document pairs: `a-archive.md` + `b-visit.md` at `candidates` (two
comparable documents with nothing in common) and the two pairs through `z-travel.md` at `chunking`.
Corpus order names the first of those -- `a-archive.md` + `b-visit.md`, and "raise
`--max-candidate-pairs`, or lower the cosine threshold", advice that would return unrelated rows
and still never reach the missing document. The earliest stage names the gap.

#### Recomputing the stage from a finished bundle

The stage rule reads three inputs and only one of them used to survive the run: `findings.jsonl`
carries the returned pairs, but the corpus's ordering fields and the per-document chunk accounting
were re-derived by reading the corpus and rebuilding the store -- and both of those move. A store
rebuilt since collapses a different duplicate, chunks a document differently, or is one ingest ahead
of the one the run read; a re-ingested corpus carries a new `effective_date`. Either one answers a
DIFFERENT question while looking like the same recompute, which is what made this reading unlike
the granularity rules `make compare-conflict-granularity` re-scores from rows alone.

So the run writes both of them down beside the coverage they explain, as
`stage_attribution_inputs` in `summary.json` (`src/llb/conflicts/bundle/record.py` builds it,
`stage_replay.py` re-reads it;
`AuditResult.stage_inputs` carries it). The stage was the FIRST reading to get that treatment and is
no longer the only one: which other questions a finished bundle answers alone, which it refuses, and
where the record's size bound draws the line are in
[what a bundle can answer alone](conflict-bundle-record.md). The two keys below are the stage's own
share of that record.

| key | what it carries | why it is recorded |
| --- | --- | --- |
| `documents` | every corpus document in corpus order, with the `effective_date` / `version` it was audited under -- and the id alone where it has neither | corpus order is data, not presentation: it picks between two pairs lost at the same stage |
| `chunks` | `stored` / `comparable` / `copies` per document (`DocumentChunks.payload`) | the store's own answer, which a rebuild changes |

`documents` is also the record's id table: from `schema_version` 4 `chunks` keys on a document's
POSITION in it rather than on its id, and so does every other map in the record
([the id table](conflict-bundle-record.md#the-id-table-every-document-named-once)); from
`schema_version` 5 a document with no ordering field to label is recorded as that id alone
([the label](conflict-bundle-record.md#the-label-a-document-with-nothing-to-label-was-carrying)).
All three forms resolve to the same document ids and every reading replays identically through each,
so the figures below -- measured before either change -- are an upper bound on what the record costs
today.

`chunks` is ABSENT below the semantic tier, never empty: that absence is what the `effort` reading
is read from, and an empty accounting says the opposite (a store that held nothing) -- a run whose
record is edited to carry one names CHUNKING on every pair instead, which CI pins. Chunk text and
chunk ordinals are deliberately not recorded, so the record costs one small entry per DOCUMENT
rather than per chunk: measured at **533 bytes** on the goods budget-100 bundle, whose store holds
954 chunks over 5 documents.

```bash
make recompute-conflict-stage STAGE_RUNS="<audit-run-dir> <audit-run-dir>"
# -> $DATA_DIR/corpus-conflict-stage/<run>/{stage.md,stage.json}
```

Each bundle is re-read from its own `summary.json` and `findings.jsonl` -- no store, no corpus, and
no model call -- and reported beside the attribution its run recorded, so a rule change is scored on
the bundles where the two readings PART. A bundle written before the record answers "not
recomputable" and keeps its recorded attribution: no answer is the correct answer there, since the
only thing left to derive one from is a store that has moved since.

**Measured 2026-08-13 over the whole archive** (31 audit bundles on the 12 GiB Blackwell CUDA host,
no model call; lookup key `corpus-conflict-stage` run `archive-replay`):

| bundles | recomputed | reading |
| --- | --- | --- |
| 5 (the `stage-replay-*` runs, one per stage) | `effort`, `chunking`, `claim_token_floor`, `duplicate_collapse`, `candidates` | **same stage and same pair as the run recorded, on all five** |
| 1 (goods, budget 100, 954 chunks) | nothing lost -- the corpus orders no document pair | recomputable and empty, which is the run's own answer |
| 25 (every bundle written before the record) | none | not recomputable, recorded attribution intact |

The five are the one-run-per-stage table above re-run under the current build (same corpora, same
knobs: `--effort hash`; `COS_THRESHOLD=0.7`; `COS_THRESHOLD=0.7 MIN_CLAIM_TOKENS=8`;
`COS_THRESHOLD=0.9 MIN_CLAIM_TOKENS=8` against the stale store; `MIN_CLAIM_TOKENS=10` against the
fixture's heading store), and each one re-reads to its own answer with the store no longer
consulted. `tests/llb/conflicts/test_stage_replay.py` pins the equality at every stage over a JSON
round-trip, the refusal on a bundle without the record and on one from a newer schema, and the
independence directly -- it deletes the corpus and rebuilds the store to a DIFFERENT stage between
the run and the re-read.

### One actionable set

"Actionable" -- the predicate behind the **to decide** count -- has ONE definition, `is_actionable`
in `src/llb/conflicts/constants.py`: every relation but `complementary`, plus anything the
vocabulary does not recognize. `decide_count` is the same predicate over a relation census, so the
group table, the plan's `decide_rows`, and the CLI all count one set. The claim-tier
precision block counts that set (`AdjudicatedRow.actionable`) and the report's ordering promotes
that set (`finding_sort_key`), so a row the audit counts as work can never sort below a row it
counts as none. Within each of the two buckets the order is descending score, then claim identity.

That ordering writes `findings.jsonl` as well as `report.md`, and both are read top-down, so the
two artifacts lead with the same row. Measured on the goods budget-100 bundle: its single
`subsumed_by` row -- the only row an operator has to decide -- sat at position **23 of 99**, below
22 `complementary` rows the model scored higher, and now leads the file and the report. The
resolution lane is unaffected by the reordering: planning the same 99 rows in either order yields
byte-identical overlays (only `source_findings_sha256`, which pins the file, differs), the same six
decision groups with the same ids, and an apply/rollback that leaves every corpus byte untouched.
`overlay_from_plan` sorts each document's annotations and suppress-spans by their own identity to
guarantee that, so a re-sorted audit cannot republish a store generation that changes nothing.

### The `groups.json` sidecar

The grouping is machine-readable, not only rendered: `write_audit` emits `groups.json` beside
`findings.jsonl` (`src/llb/conflicts/grouping/artifact.py`). Each group carries `group_id`, `rows`,
`finding_ids`, `relations`, `shared_units`, `documents`, `document_pairs`, and `top_score`; the
document also carries the census and `source_findings_sha256`, which pins it to the exact rows on
disk and equals the `source_findings_sha256` the resolution plan records.

The `finding_ids` are the SAME ids `plan.json` uses (`finding_id` in `hashing.py`), so a group id
joins the audit, the plan, and the review ledger without any consumer re-deriving anything. Both
sides nonetheless compute the grouping from `findings.jsonl` ROWS through one function, and the
audit writes the rows and the sidecar from one list, so a consumer that never reads the sidecar --
including an audit run from before it existed -- derives identical groups by grouping the file in
its own order. [Conflict
resolution](conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger) is the first
consumer.

### Measured on the goods corpus

CUDA host (RTX PRO 3000 Blackwell, 12 GiB), goods corpus (5 markdown documents, 954-chunk
`multilingual-e5-base` `recursive@800/120` store), MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M adjudicating,
`MAX_CANDIDATE_PAIRS=100`, resolved cosine 0.3604 over 55,865 exhaustive comparable pairs. The
adjudicator agreed with all 24 frozen probe pairs (Wilson 95% lower bound 0.862) in 1 min 34 s and
adjudicated the 100 rows in 9 min 23 s, with 1 unparsable verdict inside the 5-row allowance.

**99 findings on 5 documents / 6 document pairs / 79 chunk units, in 6 groups (largest 51).** The
row count over-states the independent evidence by a factor of sixteen: a triage list of 99 rows is
six decisions, on a corpus of five documents that can only supply ten document pairs and used six.
Only 1 row was actionable (`subsumed_by`), resting on one left and one right chunk, so no measured
budget clears a precision floor -- the clustered bound and the census agree, and now the headline
count agrees with both instead of reading as 99 results.

The largest group is also the reading's sharp edge: G2 chains 51 rows across three documents through
23 shared chunks, so transitive closure can UNDER-state the work as much as a raw row count
over-states it. Read the pair, not either number alone -- the row count bounds the decisions from
above, the group count from below. That upper bound is now
[measured rather than the row count](#how-many-decisions-the-row-count-is): the same 51-row group
holds 23 distinct pieces of shared evidence, and the bundle's range is 6 to 46, not 6 to 100.

Measured 2026-08-12; lookup key `corpus-conflicts` run `census-goods-budget100`. This is a different
store generation from the
[budget-100 precision runs](conflict-detection.md#measured-both-quickstart-corpora)
(954 chunks at cosine 0.3604 against 1,139 at 0.3648), and it returned 1 actionable row where that
run returned 8; the candidate list at a fixed budget is a rank cutoff into the store's own
similarity ordering, so the two lists are not the same rows.

## How many decisions the row count is

A row count over-states the decisions and a transitive group count under-states them, so the audit
used to quote a range whose top end was the row count -- which on a concentrated corpus is a very
loose bound. Both ends are now GROUP counts, measured under two rules over the same rows, and the
audit states which one it quotes and why.

| rule | what a group is | shape |
| --- | --- | --- |
| `transitive` (quoted) | rows joined by the transitive closure over a shared unit | a PARTITION: every row is in exactly one group, so the sizes sum to the row count |
| `shared_unit` | one unit that more than one row rests on, plus one group per row that shares no unit | a COVER: a row carrying two shared units joins two groups |

`src/llb/conflicts/grouping/granularity.py` computes both (`QUOTED_RULE` names the quoted one, and
every renderer reads it from there); `report/granularity.py` renders them. Two units that join
exactly the same rows are ONE group -- a left and a right chunk that only ever appear together are
one piece of evidence seen from both ends. Every shared-unit group is a subset of one transitive
group by construction, so the cover REFINES the partition and the per-group split adds up:
`quoted_group_split` reports, per quoted group id, how many distinct pieces of shared evidence its
chain runs through, which is what tells a 6-row fan on one chunk apart from a 51-row chain.

**Where it appears.** `summary.json` carries `group_granularity` (both distributions, the
`decision_range`, and the per-group split) on every run; `report.md` renders a
**How many decisions the row count is** section under the decision-groups table. `findings.jsonl`,
`groups.json`, `plan.json`, and the group ids are untouched -- the second rule is a reading, never
a regrouping. `tests/llb/conflicts/test_group_granularity.py` pins the rule, the partition/cover
invariants, the refinement, and the rendered claim.

**Recomputing it over runs already on disk** costs no model call and no store, because the rules
read `findings.jsonl` and nothing else:

```bash
make compare-conflict-granularity GRANULARITY_RUNS="<audit-run-dir> <audit-run-dir>"
# -> $DATA_DIR/corpus-conflict-granularity/<run>/{granularity.md,granularity.json}
```

### Measured, four bundles

Measured 2026-08-12 on the RTX PRO 3000 Blackwell 12 GiB CUDA host. No adjudication and no
encoding: the two goods bundles are the committed budget-100 artifacts re-read, and the two new
bundles are semantic-tier runs over stored vectors. Lookup key: `corpus-conflict-granularity` run
`both-rules-three-corpora`.

| bundle | rows | transitive | shared unit | decision range | rows in 2 groups | memberships |
| --- | --- | --- | --- | --- | --- | --- |
| goods semantic (100 rows) | 100 | 6 | 46 | 6 - 46 | 67 | 167 |
| goods claim budget-100 | 99 | 6 | 45 | 6 - 45 | 65 | 164 |
| quickstart-PDF semantic (3 docs, 1,177-chunk store) | 100 | 5 | 36 | 5 - 36 | 80 | 180 |
| committed fixture semantic (7 docs, 9-chunk store) | 13 | 3 | 8 | 3 - 8 | 8 | 21 |

The largest quoted group's own split, which is where the range comes from: goods semantic G1's 51
rows run through **23** distinct shared units (G3's 29 rows through 12, G2's 14 through 8);
quickstart-PDF G1's 94 rows through **31**; the fixture's G2 10 rows through **6**.

**The audit keeps quoting `transitive`, and the shared-unit count becomes the top of the range.**
The reason is structural rather than corpus-dependent: a count an operator funds has to account for
every row exactly once, and only the partition does. The cover double-counts on every corpus
measured -- 65 to 80 percent of rows carry two shared units, so its group sizes sum to 164-180
memberships over 99-100 rows. Funding one review per shared-unit group would fund 46 reviews on a
100-row bundle and meet 67 of those rows twice.

**What it buys is the other end.** The shared-unit count is 5 to 9 times the quoted group count and
0.36 to 0.62 times the row count, so it tightens the goods reading from "6 to 100 decisions" to
"6 to 46" and the quickstart-PDF reading from "5 to 100" to "5 to 36". That is the number the
earlier reading was missing: the quoted count under-states by a factor of 7.7 on goods and 7.2 on
quickstart-PDF, and now the audit says so instead of leaving the operator to infer it from a
`shared unit` cell that reads `(+21 more)`.

**No corpus shape gets the other rule.** The three real corpora differ in size, document count, and
concentration and all three behave the same way, so the adoption is unconditional. What IS
corpus-dependent is how WIDE the range is, and the audit reports that per run rather than assuming
it: when every quoted group rests on a single shared unit the two rules agree, the range collapses
to a point, and the report says so in those words -- which is exactly the case CI pins on a fixture
whose rows all quote one chunk (6 rows, 1 group under both rules).

**The measurement's own limit.** The 8-document HR corpus that the
[claim-tier precision tables](conflict-detection.md#measured-both-quickstart-corpora) were measured
on is operator data and is not present on this host, so the second and third bundles here are the
3-document quickstart-PDF corpus and the committed fixture instead. Both are strictly narrower than
HR: the quickstart-PDF documents are a subset of the goods corpus, and the fixture's 9-chunk store
is far too small to produce a long chain. A wider corpus could still separate the rules, and
re-running `make compare-conflict-granularity` over an HR bundle is the cheap way to check -- it
reads `findings.jsonl` and needs no model.
