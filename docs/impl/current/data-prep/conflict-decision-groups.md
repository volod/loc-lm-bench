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
counts). `src/llb/conflicts/census.py` computes the census and the grouping;
`src/llb/conflicts/report_findings.py` renders the grouped findings section.

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
  (`src/llb/conflicts/report_findings.py`): `to decide` rows first, then rows, then top score, then
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
budget-100 claim bundle four groups tie at 1.000, so three of them were ordered by the claim-identity
tiebreak underneath -- a document id. Ranked by stake, the 29-row decision moves above the 14-row
one and the 3-row decision above the 2-row one, while the group holding the only actionable row
still leads the claim bundle.

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

They diverge in BOTH directions, measured on the two goods bundles:

- **Goods budget-100 claim bundle** (`.data/corpus-conflicts/20260812T-two-counts-census-goods-budget100/`):
  **1 to decide, 0 to review.** The single `subsumed_by` row is work by the relation vocabulary --
  it is what the precision block measures and what leads `findings.jsonl` -- but the conservative
  policy resolves subsumption as `keep_both` plus an annotation, so it costs a human nothing. An
  operator funding one review off the audit report would have funded a review that does not exist.
- **Goods semantic bundle** (`.data/corpus-conflicts/20260812T-two-counts-goods-semantic/`):
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
  bundle rendered before it existed, apart from the corpus path and one tier's `seconds` column
  (`.data/corpus-conflicts/20260812T-projection-control-goods-semantic/report.md` against
  `20260812T-two-counts-goods-semantic/report.md`). Both of those reports predate the
  [granularity section](#how-many-decisions-the-row-count-is), which every report now carries; the
  claim is about the projection, so re-checking it means comparing two reports rendered by the same
  code with and without the flag.
- **What it adds.** A headline line per policy, one `to review (projected, <policy>)` column per
  policy in the decision-groups table, and `policy_projection` in `summary.json`
  (`schema_version: 2`, `kind: projection`, `basis`, `policies`, `by_policy` with each policy's
  `review_rows` / `review_groups` / per-group counts, and `deltas`). The per-group counts are keyed
  by the same group ids `groups.json` and `plan.json` use. The CLI echoes the same numbers.
  `findings.jsonl` and `groups.json` are untouched: the projection depends on a policy, and those
  two artifacts must stay readable by a consumer that has not chosen one.
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
- **A projection, never a measurement**, said at every appearance -- the flag, every headline line,
  the column paragraph, the CLI lines, and the artifact's own `kind`/`basis` fields. Each column is
  only true of the policy it names, the delta inherits that caveat from both columns it subtracts,
  and a reviewer's decisions can still move any of them.
- **One policy renders exactly the column it always did.** With a single `--project-policy` value
  the header is `to review (projected)` with no policy suffix and no delta column, and the report
  is byte-identical to the one the single-policy path produced. Measured: the group-table header of
  `.data/corpus-conflicts/20260812T-policy-choice-goods-single/report.md` is identical to the one
  in `20260812T-projected-review-goods-semantic/report.md`, and a CI test asserts the two rendering
  paths agree byte for byte on the same rows.
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
have it, so it is composed ABOVE both layers: `src/llb/conflicts/policy_projection.py` imports
`resolution_policy`, the CLI calls it and puts the plain JSON result on
`AuditResult.policy_projection`, and `report.py` / `report_findings.py` / `report_projection.py`
render data they never derive -- `report_projection.py` owns every way a projected count is allowed
to appear, so the headline, the columns, and the prose cannot drift apart. `conflicts.report*`
therefore still imports nothing from `conflicts.resolution_*`, which `test_policy_projection.py`
asserts by discovering every `report*.py` and reading its import lines rather than by convention --
so a renderer added later is held to the rule without anyone remembering to list it.

The cheaper alternative -- the report simply pointing at the resolver -- was kept as
well rather than replaced: every mention of the projection names `resolve-corpus-conflicts` and
where the measured count lives, because the resolver without `--apply` already plans without
touching a corpus byte and remains the only thing that MEASURES the count.

Measured on the two goods bundles, which are the two ends of the divergence:

- **Goods semantic bundle** (`.data/corpus-conflicts/20260812T-projected-review-goods-semantic/`,
  100 rows, 6 groups): projected `{G1: 51, G2: 14, G3: 29, G4: 3, G5: 1, G6: 2}` = 100 rows in 6
  groups, equal group for group to the `review_rows` the resolver then wrote into `plan.json` in
  the same directory, and equal to `decide_rows` as well -- every semantic-tier duplicate is to
  review under every policy. Here the projection confirms the headline instead of correcting it --
  and this is the bundle an operator meets first, which is exactly why the coincidence is easy to
  mistake for a rule that the claim bundle below breaks.
- **Goods budget-100 claim bundle**
  (`.data/corpus-conflicts/20260812T-projected-review-goods-budget100/`, CUDA host, RTX PRO 3000
  Blackwell, 954-chunk store at resolved cosine 0.3604, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M
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

CUDA host, RTX PRO 3000 Blackwell.

| bundle | rows | `conservative` | `prefer-newer` | delta | groups moved |
| --- | --- | --- | --- | --- | --- |
| conflicts fixture, claim tier | 17 | 2 | 0 | **-2** | G4 |
| goods semantic, budget 100 | 100 | 100 | 100 | 0 | none |

- **The corpus where the choice is not free** is the committed fixture at
  `samples/corpora/conflicts_uk_v1/`, which already plants a dated supersession (the 2021 vs 2024
  thirty-versus-fifteen-day deadline, with `effective_date` on both documents) -- so nothing had to
  be planted for this. Run:
  `make audit-corpus-conflicts CORPUS=samples/corpora/conflicts_uk_v1/corpus EFFORT=claim
  STORE=<heading@600 store> MIN_CLAIM_TOKENS=10 PROJECT_POLICY=conservative,prefer-newer`
  (`.data/corpus-conflicts/20260812T-policy-choice-fixture-claim/`, 19-chunk
  `multilingual-e5-base` `heading@600` store, MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M agreeing with all
  24 frozen probe pairs, 14 rows adjudicated in 53 s). It returns **17 findings in 4 groups, 9 to
  decide**, including 2 `superseded_by` rows. The report reads `to review (PROJECTED under policy
  conservative): 2 rows in 1 decision group`, `(PROJECTED under policy prefer-newer): 0 rows in 0
  decision groups`, and `policy choice conservative -> prefer-newer: -2 rows to review, falling in
  1 decision group (G4)`. Both columns were then checked against the thing they project: running
  `resolve-corpus-conflicts` on the same rows under each policy wrote `plan.json` decisions equal
  to the projection group for group (`{G1: 0, G2: 0, G3: 0, G4: 2}` and all zeros), under
  `.data/corpus-conflicts/20260812T-policy-choice-fixture-claim/resolve-{conservative,prefer-newer}/`.
- **The corpus where it is free** is goods
  (`.data/corpus-conflicts/20260812T-policy-choice-goods-semantic/`, 100 rows, 6 groups): every row
  is a semantic-tier duplicate, that tier has no deletion authority under either policy, so both
  columns read 100 and the delta is 0 in every group. The report says so in those words -- **no
  difference on this corpus** -- rather than printing two identical columns and leaving the
  operator to compare them. Reading a zero delta correctly matters: it means the policy question
  can be dropped on this corpus, not that the policies are interchangeable in general.

The measurement's own limit: the fixture is 7 small documents, so its delta shows the mechanism
rather than a realistic magnitude. Whether a production corpus carries enough dated supersessions
for the choice to matter is a per-corpus question, and the delta column is what answers it in one
command.

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
`findings.jsonl` (`src/llb/conflicts/group_artifact.py`). Each group carries `group_id`, `rows`,
`finding_ids`, `relations`, `shared_units`, `documents`, `document_pairs`, and `top_score`; the
document also carries the census and `source_findings_sha256`, which pins it to the exact rows on
disk and equals the `source_findings_sha256` the resolution plan records.

The `finding_ids` are the SAME ids `plan.json` uses (`finding_id` in `hashing.py`), so a group id
joins the audit, the plan, and the review ledger without any consumer re-deriving anything. Both
sides nonetheless compute the grouping from `findings.jsonl` ROWS through one function, and the
audit writes the rows and the sidecar from one list, so a consumer that never reads the sidecar --
including an audit run from before it existed -- derives identical groups by grouping the file in
its own order. [Conflict resolution](conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
is the first consumer.

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

Artifacts: `$DATA_DIR/corpus-conflicts/20260812T-census-goods-budget100/`. This is a different store
generation from the
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

`src/llb/conflicts/granularity.py` computes both (`QUOTED_RULE` names the quoted one, and every
renderer reads it from there); `report_granularity.py` renders them. Two units that join exactly
the same rows are ONE group -- a left and a right chunk that only ever appear together are one
piece of evidence seen from both ends. Every shared-unit group is a subset of one transitive group
by construction, so the cover REFINES the partition and the per-group split adds up:
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

CUDA host (RTX PRO 3000 Blackwell, 12 GiB). No adjudication and no encoding: the two goods bundles
are the committed budget-100 artifacts re-read, and the two new bundles are semantic-tier runs over
stored vectors. Artifacts:
`$DATA_DIR/corpus-conflict-granularity/20260812T-both-rules-three-corpora/`.

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
