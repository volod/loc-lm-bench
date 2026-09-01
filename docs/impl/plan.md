# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

Every task serves a capability registered in
[the spec](../design/spec.md#capability-registry), and the capability groups below appear in the
registry's row order -- the implementation line. That is the whole scheduling rule: **take the first
task of the earliest group that still has one.** Which entry is best written, most recently added,
or most sharply specified does not enter into it.

Work that no registered capability covers is not blocked -- it is a signal the spec has a gap. Close
the gap first: [Extending this specification](../design/spec.md#extending-this-specification) is the
six-step lifecycle for turning a discovery into a registered capability with a declared evaluation,
and only then into tasks here. `make lint-spec-plan` enforces the join in both directions.

## Forward Tasks

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land to `make ci` green with
  committed fixtures, injected fakes, and deterministic harnesses. A few carry a heavy real-model
  run for durable evidence; those runs are deterministic and execute on the CUDA host without any
  human judgment, so they stay in this section.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance without a
  human in the loop: the deliverable *is* human judgment (verification-gate reviews, drafting
  oversight, measured reviewer throughput) or requires a human authorization (egress consent +
  API spend). An agent can still build the supporting code and unit tests; the marked **human
  step** is what gates completion.

Both sections carry the same capability groups in the same order, so the two run in parallel: an
agent works its section's line while a human-gated item waits on its reviewer.

Task ids are stable and never change. Every task carries a `Serves` line naming its capability and a
`Dependencies` line. Within a group the same logic applies one level down:

1. a task another task names as a prerequisite comes first;
2. then work that changes what later tasks measure -- a chunker, an encoder roster, or a detector
   change invalidates every reading taken against the old one;
3. then required before `(optional)`;
4. then cheap before expensive: a `CLEAR` task costs a fixture test, a `RUN NEEDED` task costs a GPU
   run, so a run of CLEAR work lands as one pass rather than interleaved.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md). For tasks that depend on scoring or judging, the calibrated
local-judge baseline and tuning/sweep behavior live in
[evaluation rigor](current/rigor-board-judge.md); the prompt-system package flow and other
extended workflows live in [extended workflows](current/extended-workflows.md).

Every task below carries an explicit `Agent status` line with one of four markers:

- **CLEAR** -- agent-buildable to `make ci` green with fixtures/fakes; no run evidence, no human
  gate.
- **RUN NEEDED** -- agent-buildable, but acceptance requires a heavy deterministic run; every dev
  box is a proper CUDA host, so the agent executes these runs itself on the current machine.
- **BLOCKED BY HUMAN** -- the acceptance gate consumes an artifact only a human step can produce.
- **HUMAN-GATED** -- the deliverable itself is human judgment or authorization; supporting code and
  unit tests are agent-buildable.
- **RESEARCH** -- in addition to Agent status. It marks the task implementation as unknown in
advance, and a negative result is a valid outcome that must be recorded rather than worked around.

**`(optional)` is binding within a group.** It marks work already judged not to be the next thing
worth doing for that capability, so it sorts behind every non-optional task in its own group. It
does NOT reorder the line: an optional task of an earlier capability still precedes a required task
of a later one, because the trust chain is what the order encodes.

## Agent Implementation Tasks

Agent-buildable work, grouped by the capability it serves and ordered by the
implementation line in the [capability registry](../design/spec.md#capability-registry).
Take the first task of the earliest group that still has one; see
[Adding Future Tasks](#adding-future-tasks) before adding one.

### Corpus conflict and governance -- `corpus-conflict-audit`

#### conflict-adjudicator-probe-difficulty (optional)

The frozen calibration probe is passed **24/24** by MamayLM-Gemma-3-12B, which means the gate is
currently proving only that an adjudicator is not badly broken -- a probe nobody fails cannot
separate a good adjudicator from an adequate one, and the audit will happily quote a precision
figure from either
([conflict detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe)). Add a
harder frozen tier: pairs whose actionable/complementary split is genuinely arguable (a restated
fact under a different heading, two numeric claims about different quantities in the same
sentence shape, a partial supersession where only one clause changed), score two host-fit model
families against both tiers, and either raise `MIN_ADJUDICATOR_ACCURACY_LCB` on the evidence or
record that the gate's job is a floor rather than a ranking.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse `src/llb/conflicts/claim/calibration.py`, its heading-addressed probe
  format, and the planted fixture; a new hard tier may need new fixture sections, which must stay
  offset-exact and pass the existing corpus-unchanged assertion.
- User-visible outcome: the calibration gate distinguishes adjudicators an operator would actually
  choose between, instead of only rejecting a broken one.
- Scope boundary: in scope -- the harder probe tier, its frozen labels and rationale, a two-family
  comparison, and a gate-threshold decision. Out of scope -- changing the adjudication prompt, the
  relation vocabulary, and scoring agreement on anything but the actionable binary before the
  comparison supports it.
- Data and artifact paths: `samples/corpora/conflicts_uk_v1/adjudicator_probe.json` and the
  existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts.
- Execution path: one calibration-only run per model family on the CUDA host; CI covers the new
  tier's passage resolution and label balance.
- Acceptance gates: `make ci` green; both families score both tiers; the report states whether the
  hard tier separates them and either raises the gate or records why it stays a floor.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe).

#### conflict-precision-bound-at-document-clustering (optional)

The clustered bound treats a CHUNK as the independent unit, and on a concentrated corpus that is
still too generous. Measured: every one of the goods corpus's 8 actionable rows at budget 100 points
at the same right document, so at the document level the corpus supplies one observation, not eight
-- yet the chunk-level bound resamples 42 left and 51 right chunks
([conflict detection](current/data-prep/conflict-detection.md#measured-both-quickstart-corpora)).
The bound refused a floor there anyway, which is why this is a sharpening rather than a correction,
but nothing establishes which unit the audit should quote when the two disagree. Compute the bound
at both clusterings on the same rows, report them side by side on both quickstart corpora, and state
the rule: quote the document-level bound always, quote the chunk-level bound when the corpus spreads
its conflicts, or quote both.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `two_way_proportion_bound` already takes arbitrary cluster keys, so the
  document-level bound is the same estimator over `doc_id` keys; the rows and their verdicts come
  from the existing budget-100 runs and need no re-adjudication.
- User-visible outcome: the precision floor an operator reads is clustered on the unit their corpus
  actually repeats, instead of on whichever unit the chunker happened to produce.
- Scope boundary: in scope -- the second clustering, the side-by-side report, and a stated rule for
  which bound the audit quotes. Out of scope -- a third clustering level, changing the estimator,
  and changing the calibration gate.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: recompute both bounds over the committed budget-100 per-row ledgers on the CUDA
  host (no model calls needed); CI covers both clusterings over fixture rows.
- Acceptance gates: `make ci` green; both corpora report both bounds at every measured budget; the
  report names which bound the audit quotes and why, and the chunk-level bound never reads lower
  than the document-level one on the same rows.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision).

#### conflict-candidate-record-cap-on-a-natively-dense-corpus (optional)

The candidate record's cap is priced, but on a density no operator would run at: no corpus on this
host saturates the list at any real threshold, so the depth/cost curve was measured by dropping the
cosine to 0.25 on the 250-document quickstart corpus, which forms 3,127 candidate rows against the
38 the same corpus returns at the 0.6 it was actually audited at
([bundle record](current/data-prep/conflict-bundle-record.md#how-deep-the-prefix-reaches-and-what-the-depth-costs)).
Two things that decide the cap are therefore substitutes rather than measurements. The RANKS-PER-PAIR
collapse ratio (1.11 at cap 200, 1.22 over the whole list) is what turns a cap in document pairs into
an answerable depth in ranks, and on a corpus whose density comes from genuine near-duplicates rather
than from a loosened threshold it could be far higher -- the same cap would then buy much less depth.
And the depth a re-read is asked at rests on a structural bound (the question is downward, so the
run's own budget is the ceiling) plus two operating budgets, never on an observed re-read. Re-price
it on a corpus that saturates the list at an operating cosine, and record the budget each
`recompute-conflict-stage` run is actually asked at so the depth side stops being an argument.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `--max-candidate-record-pairs` and the recorded `cap` are current behavior;
  `CandidateRecord.of` in `src/llb/conflicts/bundle/candidate_record.py` is where the collapse
  happens and `covered_to_rank` is the ratio's numerator. The corpus is the missing input: it needs
  many near-duplicate documents, not a lowered threshold.
- User-visible outcome: an operator on a duplicate-heavy corpus learns whether the shipped cap still
  answers the budgets they ask, instead of inheriting a number priced on a synthetic density.
- Scope boundary: in scope -- one natively dense corpus, its ranks-per-pair ratio at several caps,
  the re-read depths observed, and a keep-or-change verdict on the constant. Out of scope --
  recording chunk-level pairs, removing the cap, changing the refusal past it, and re-deciding the
  cap before a corpus that saturates at an operating threshold exists.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` and
  `$DATA_DIR/corpus-conflict-stage/<run>/` artifacts only.
- Execution path: one semantic-tier run per cap on the CUDA host; CI covers the ratio and the
  refusal over fixtures.
- Acceptance gates: `make ci` green; the reading states the ranks-per-pair ratio at each measured cap
  beside the quickstart corpus's, and either keeps `DEFAULT_CANDIDATE_RECORD_PAIRS` with its reason
  or moves it; a stage re-read records the budget it was asked at.
- Documentation target:
  [bundle record](current/data-prep/conflict-bundle-record.md#how-deep-the-prefix-reaches-and-what-the-depth-costs).

#### conflict-claim-yield-across-store-generations (optional)

The claim tier's candidate list at a fixed budget is a RANK cutoff into the store's own similarity
ordering, so it is a property of the store as much as of the corpus -- and the two measured goods
budget-100 runs disagree sharply about how much the corpus contains: 8 actionable rows on the
1,139-chunk store at resolved cosine 0.3648, 1 actionable row on the 954-chunk store at 0.3604
([conflict detection](current/data-prep/conflict-decision-groups.md#measured-on-the-goods-corpus)). The
two runs differ in chunk count, duplicate collapse, and resolved threshold at once, so nothing
establishes which factor moved the yield, and an operator cannot tell whether a low actionable count
means a clean corpus or an unlucky store. Vary one factor at a time (duplicate collapse on/off,
chunk size, budget) on the same corpus and record which one the yield tracks.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse `make build-index` per store variant and `make audit-corpus-conflicts
  EFFORT=claim MAX_CANDIDATE_PAIRS=100`; the calibration probe and precision block are current
  behavior.
- User-visible outcome: an operator reading a low actionable count learns whether it is evidence
  about their corpus or about the store they happened to build.
- Scope boundary: in scope -- the one-factor-at-a-time store sweep, the per-variant actionable
  yield and overlap of the returned rows, and a stated reading. Out of scope -- changing the
  chunker, the threshold calibration, and the candidate budget defaults before the sweep supports it.
- Data and artifact paths: one `$DATA_DIR/corpus-conflicts/<run>/` per variant.
- Execution path: one claim run per store variant on the CUDA host; no new CI coverage beyond the
  existing fixtures.
- Acceptance gates: `make ci` green; every variant reports its actionable yield and its row overlap
  with the baseline variant; the reading names the factor the yield tracks, or records that the
  variants do not separate.
- Documentation target: [conflict
  detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision).

#### conflict-policy-delta-on-an-operator-corpus-with-dated-revisions (optional)

The policy-choice delta and its share of actionable rows are measured, but on exactly one corpus
where the share is non-zero -- the committed 7-document fixture, which was planted to contain one
dated supersession
([decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured)).
Every other corpus on this host carries no governance dates at all, so its zero share is structural
and carries no information about magnitude. The open question is unchanged and now sharper: on a
corpus of genuine dated revisions, what share of the actionable rows does the policy choice move?
Run the shipped `--project-policy conservative,prefer-newer` reading against the 8-document HR
corpus (operator data, absent from this host) or another corpus with governance dates on both sides
of a revision pair, and record the share beside the fixture's.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. The projection, the delta, and the `moved_rows` / `moved_share` reading are
  current behavior; the corpus needs `effective_date` or `version` on both sides of a revision
  pair, which is what promotes `contradicts` to `superseded_by`.
- User-visible outcome: an operator learns whether the resolution-policy choice is a real decision
  on corpora like theirs, or a knob that is free in practice.
- Scope boundary: in scope -- one claim-tier run per dated corpus and the share beside the
  fixture's. Out of scope -- adding a policy, changing how `superseded_by` is derived, and making
  any policy the default.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` layout.
- Execution path: one claim-tier run per corpus on the CUDA host with both policies projected; no
  new CI coverage beyond the existing fixtures.
- Acceptance gates: `make ci` green; every corpus reports its delta and the share of actionable
  rows it moves; each projected column still equals the `plan.json` `review_rows` the same policy
  writes on the same rows.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured).

#### conflict-policy-share-across-repeat-audits-of-one-corpus (optional)

The policy-choice share is quoted from ONE audit and is not reproducible across audits of the same
corpus: three claim-tier runs of the committed fixture at identical settings returned the same 17
rows, the same 9 actionable rows, and the same 0.4286 claim-tier precision, yet the third called one
row `subsumed_by` where the first two called it `superseded_by` -- so the delta read 1 of 9 (11.1%)
instead of 2 of 9 (22.2%)
([decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured)).
The share is a count of the one relation the policies part on, so it inherits the adjudicator's
sampling variance undivided, and the endpoint runs at `temperature 0.2` with no seed
(`EndpointConfig` in `src/llb/prep/ontology/endpoints/config.py`). Audit one corpus N times at the
shipped settings, report the spread of the relation mix and of `moved_share`, and decide between the
two fixes the spread implies: pin the adjudication call (temperature 0 plus a seed where the backend
honors one) so a repeat audit is comparable, or quote the share with a run-to-run band. A negative
result -- the spread is small enough that a point estimate is honest -- is a valid outcome.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `make audit-corpus-conflicts EFFORT=claim PROJECT_POLICY=conservative,prefer-newer`
  and its artifacts are current behavior; the three bundles above are the first three samples and
  need no re-running. Do not change the relation vocabulary or the prompt -- that would measure a
  different adjudicator instead of this one's variance.
- User-visible outcome: an operator comparing this week's policy share to last week's learns which
  part of the difference is their corpus and which part is the model being asked twice.
- Scope boundary: in scope -- the repeat runs, the spread of the relation mix and the share, and the
  pin-or-band decision. Out of scope -- changing the adjudication prompt, the calibration gate, and
  quoting a band before the spread is measured.
- Data and artifact paths: one `$DATA_DIR/corpus-conflicts/<run>/` per repeat.
- Execution path: N claim-tier runs of one corpus on the CUDA host; CI covers whatever pinning the
  decision adopts, over the injected adjudicator.
- Acceptance gates: `make ci` green; every repeat reports its relation mix and share; the reading
  states the spread and either pins the call or records the band the share must be quoted with.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured).

#### conflict-decision-group-partition-refinement (optional)

The audit now quotes a RANGE because neither measured rule is a decision count: the transitive
closure is a partition but too coarse, and the shared-unit rule is finer but a COVER -- 65 to 80
percent of rows join two of its groups on every corpus measured, so its count cannot be funded one
review each
([decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is)).
A single number needs a third rule that is BOTH a partition and finer than the closure. Measure one:
cut the closure's chain at its weakest joins (a shared unit carrying only low-score pairs, or only
`complementary` relations, is a chunk two decisions merely pass through) and check whether the
resulting partition lands inside the measured range rather than collapsing back to one of its ends.
A negative result -- no cut rule lands inside without splitting a genuine decision -- is a valid
outcome and closes the question.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED, RESEARCH
- Dependencies: none. Reuse `granularity.py`'s two rules as the bracket the third must sit inside and
  `compare-conflict-granularity` to read the result over committed runs.
- User-visible outcome: either one decision count an operator can fund directly, or a recorded reason
  the audit will keep quoting a range.
- Scope boundary: in scope -- the cut rule, its partition proof, and the comparison against both
  measured ends. Out of scope -- changing detection, changing `findings.jsonl` or the group ids
  `plan.json` joins on, and adopting a third rule as the quoted one before it lands inside the range.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` and
  `$DATA_DIR/corpus-conflict-granularity/<run>/` layouts.
- Execution path: recompute over committed run artifacts on the CUDA host; the measured bundles are
  all narrower than the 8-document HR corpus (which is operator data and absent from this host), so
  the run needs at least one bundle from a corpus that supplies a genuinely long chain. CI covers the
  cut rule and the partition invariant over fixture rows.
- Acceptance gates: `make ci` green; the third rule is asserted to be a partition on every fixture;
  every measured bundle reports its count under all three rules; the reading states whether the
  partition lands strictly inside the range, or records that it does not and why.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is).

### Documentation integrity -- `documentation-integrity`

#### conflict-bundle-record-page-is-past-the-split-threshold (optional)

[bundle record](current/data-prep/conflict-bundle-record.md) is ~740 lines and its headings describe
two subjects: what SHAPE the record is written in (the id table, the label it dropped, the affix
fold, the count default, the store identity, and the size table that prices them) and what the
record ANSWERS (the verdict table, the exclusion floor, the candidate budget, the depth/cost curve).
That is well past the ~500-line split rule in [AGENTS.md](../../AGENTS.md), and the shape subject is
where every successive size change lands, so the page grows in one place and has now been pushed
past the rule three times. The test side is already split along exactly this seam
(`test_bundle_record.py` against `test_bundle_id_table.py` and `test_store_identity.py`), so the
heading seam is established rather than proposed --
move the shape subsections to a new topic page, add its row to the area page, and repoint the
inbound links.
Anchors keep their fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `conflict-detection.md`, and `conflict-decision-groups.md`.
- User-visible outcome: a reader looking for what a bundle can answer stops scrolling past three
  successive byte-accounting decisions to reach it.
- Scope boundary: in scope -- the split, the area-page row, and the link repointing. Out of scope
  -- rewriting the moved text and changing any measured result.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes.
- Documentation target: [data prep](current/data-prep.md) and the two pages the split produces.

#### conflict-decision-groups-page-is-past-the-split-threshold (optional)

[decision groups](current/data-prep/conflict-decision-groups.md) is ~800 lines and its headings
describe two subjects: how many decisions a row count is (census, grouping rules, ranking, the
`groups.json` sidecar) and what a policy choice costs (the projection, the governance-coverage
precondition, the stage attribution and its bundle re-read). That is past the ~500-line split rule
in [AGENTS.md](../../AGENTS.md), and the second subject is where every recent addition lands, so the
page grows in one place. Split along the heading seam -- move the projection/coverage/stage
subsections to a new topic page, add its row to the area page, and repoint the inbound links.
Anchors keep their fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `conflict-detection.md`, and `conflict-resolution.md`.
- User-visible outcome: a reader looking for what a policy choice costs stops scrolling past the
  counting rules to reach it.
- Scope boundary: in scope -- the split, the area-page row, and the link repointing. Out of scope
  -- rewriting the moved text and changing any measured result.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes.
- Documentation target: [data prep](current/data-prep.md) and the two pages the split produces.

#### entity-resolution-page-is-a-single-page-area-with-four-subjects (optional)

[entity resolution](current/entity-resolution.md) is ~650 lines and its headings describe four
subjects: the shared linkage seam (its vocabulary, artifacts, commands, and committed sample) and
one page's worth of lane apiece for gold items, graph nodes, and document editions. That is past the
~500-line split rule in [AGENTS.md](../../AGENTS.md), and the page is a single-page AREA rather than
a container, so every new lane lands on it. Turn it into an area directory: keep the orientation,
the confidence contract, and the boundary on `entity-resolution.md` with a table of its pages, and
move the seam and the three lanes to `current/entity-resolution/<topic>.md`. Anchors keep their
fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `current.md`, `robustness-ontology-backends.md`, two `graphrag-backend/`
  pages, `data-prep/ingestion-corpora.md`, `data-prep/conflict-detection.md`, and the
  `samples/corpora/editions_uk_v1/README.md`.
- User-visible outcome: a reader who wants one lane stops scrolling past the other three and the
  seam to reach it, and the next lane to move onto the seam gets a page instead of a section.
- Scope boundary: in scope -- the split, the area-page table, and the link repointing. Out of scope
  -- rewriting the moved text, changing any measured result, and merging the per-lane test
  paragraphs.
- Acceptance gates: `make lint-md` green with zero broken links; no page past ~500 lines; no anchor
  text changes.
- Documentation target: [entity resolution](current/entity-resolution.md) and the pages the split
  produces.

#### summary-input-elision-page-is-past-the-split-threshold (optional)

[summary-input bounds and elision](current/extended-workflows/summary-input-elision.md) is ~520
lines and its headings describe two subjects: what the summarize-input BOUND is (the step-aligned
cap, and what eliding under it costs on a typed-memory shape) and what the entry-aware TRIM is (the
middle-critical transfer, the adoption ladder with its arm-order and guard-fit subsections, and the
shipped default). That is past the ~500-line split rule in [AGENTS.md](../../AGENTS.md), and the
trim subject is the one that grows: it arrived as a prototype, became a policy field, and has now
taken a default move, each time landing on the same page. The seam is already a heading boundary
and the code is split along it too (`llb.bench.memory.window_elision` against
`llb.bench.summary_trim`), so move the trim sections to their own topic page, add its row to the
area page, and repoint the inbound links. Anchors keep their fragments, so only paths change and
`make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `extended-workflows.md` and `samples/benchmarks/agentic_context_policy_pins.json`,
  whose `published_in` anchors the pin gate checks by path.
- User-visible outcome: a reader who wants the shipped trim default stops scrolling past the whole
  bound story and two study ladders to reach it.
- Scope boundary: in scope -- the split, the area-page row, the `published_in` repointing, and the
  link repointing. Out of scope -- rewriting the moved text and changing any measured result.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes; the pin gate's doc-anchor test still passes.
- Documentation target: [extended workflows](current/extended-workflows.md) and the two pages the
  split produces.

## Human-Assisted Tasks

Work whose acceptance needs human judgment or authorization, in the same capability
order. An agent can build the supporting code and unit tests; the marked human step is
what gates completion.

### Gold data and ground truth -- `gold-data`

#### multihop-ledger-human-acceptance

Accept (or reject) the drafted multi-hop retrieval slice through the verification gate, then re-run
BOTH draft-grounded lanes on the accepted ledger -- the fusion sweep and the end-to-end
answer-quality comparison -- so the graph-weight verdict rests on human-reviewed questions instead
of drafted ones. The drafted set, its worksheet, the matched vector/graph stores, and the measured
draft-grounded sweep plus answer-quality comparison are current behavior in
[GraphRAG](current/graphrag-backend.md); every
drafted multi-hop item is span-exact and Ukrainian-gated by construction, but only a reviewer can
say whether a shared-bridge question genuinely needs both facts.

- Serves: `gold-data` -- [Gold data and ground truth](../design/spec.md#data-and-ground-truth)
- Agent status: HUMAN-GATED
- Dependencies: the [widened handoff](current/graphrag-backend/fusion-sweep-evidence.md#widened-multi-hop-review-handoff)
  supplies the worksheet, while the [paired-power
  contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes) derives
  the accepted-ledger requirement from a predeclared minimum detectable retrieval gain, expected
  discordance, confidence, and power. Human step that gates completion: a reviewer decides
  `accept`/`reject` for every worksheet row -- specifically whether the question is answerable ONLY
  with both cited spans -- and signs off on the resulting accepted ledger.
- User-visible outcome: a graph-weight recommendation for multi-hop retrieval backed by a
  human-accepted ledger, or a recorded finding that shared-bridge drafting does not produce
  genuine multi-hop questions and the slice must come from another source.
- Scope boundary: in scope -- worksheet review, `verify-accept`, re-running the sweep over BOTH
  span-identity policies on the accepted ledger, and the adopt-or-reject verdict per knob --
  including whether `graph_fusion_span_identity` flips from `exact` to `overlap` as the shipped
  default, which is currently gated only by the drafted ledger
  ([GraphRAG](current/graphrag-backend/span-and-depth-evidence.md#span-identity-evidence)). Out of
  scope -- graph schema changes and fusion mechanics (the candidate-depth and span-identity verdicts
  are current behavior in
  [GraphRAG](current/graphrag-backend/span-and-depth-evidence.md#candidate-depth-evidence)).
- Data and artifact paths: the widened drafted bundle and worksheet named in
  [the handoff](current/graphrag-backend/fusion-sweep-evidence.md#widened-multi-hop-review-handoff),
  plus a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` sweep over `accepted/goldset.jsonl` and
  its `answer-quality/` comparison.
- Execution path: start at `make verify-review VERIFY_WS=<widened-multi-hop-worksheet>`, then
  `make verify-accept VERIFY_WS=<widened-multi-hop-worksheet>
  BUNDLE=<widened-multi-hop-bundle>`, then
  `make compare-graph-fusion GOLDSET=<accepted>/goldset.jsonl
  GRAPH_FUSION_CANDIDATES=k,50 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap`,
  then `make compare-answer-quality GOLDSET=<accepted>/goldset.jsonl FUSION_COMPARISON=<that
  sweep>/comparison.json` -- WITHOUT `INCLUDE_DRAFTED`, since an accepted ledger no longer needs
  the drafted-grounding escape.
- Acceptance gates: every worksheet row has a decision; the accepted ledger satisfies its
  predeclared paired-power requirement; retention is reported by relation pair, document mode, and
  source document with uncertainty; the re-run sweep reports the same rows with paired intervals and
  the human records the adopt-or-reject verdict per graph strategy and per span-identity policy; the
  answer-quality comparison re-runs on the accepted ledger with `grounding: verified`. If power
  remains insufficient or rejection failures are statistically concentrated in a stratum, run the
  relation-stratified widening workflow from [data
  prep](current/data-prep/ingestion-corpora.md#widening-a-multi-hop-review-slice) against the latest
  reviewed ledger.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md).

### Entity resolution -- `entity-resolution`

#### entity-merge-labelled-set

Produce the reviewer-labelled merge set that turns a linkage probability into an operating point. An
unsupervised fit ranks pairs and clusters them, but nothing in it says where to cut: the
confidence contract publishes a match probability only with the labelled set it was scored against,
so until a human has read a sample of proposed merges there is no defensible default threshold for
either the graph overlay or the gold-item suppression policy. Sample the proposed merges across the
probability range -- not only the confident ones, because the threshold lives where the model is
unsure -- and record same-thing / different-thing / cannot-tell per pair through the review
workbench, so the decisions land in a ledger like every other human gate.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: HUMAN-GATED
- Dependencies: whichever supplies the candidate pairs to label -- the gold-item shadow lane's
  disagreement list ([the gold-item lane](current/entity-resolution.md#the-gold-item-lane)) or the
  graph node lane's scored pairs
  ([the graph node lane](current/entity-resolution.md#the-graph-node-lane)). The seam's label-based
  fit and labelled accuracy curve already exist -- see
  [entity resolution](current/entity-resolution.md). Human step that gates
  completion: a Ukrainian-reading reviewer decides each sampled pair. Reuse the adapter pattern in
  [review workbench](current/review-workbench.md) rather than building a second review surface.
- User-visible outcome: both linkage domains gain a threshold with a precision and recall attached
  to it, and the operator can see what a stricter or looser cut would cost.
- Scope boundary: in scope -- the workbench adapter for merge decisions, a stratified sample across
  probability bands, the label ledger, the labelled accuracy curve computed from it, and the
  threshold recommendation that follows. Out of scope -- labelling every candidate pair, labelling
  CONFLICT relations (a different question with a different adjudicator), and adopting a threshold
  that the paired retrieval or drafting evidence does not support.
- Data and artifact paths: `$DATA_DIR/entity-merge-labels/<run>/` -- the sampled pairs, the decision
  ledger, and the accuracy curve; the ledger is the frozen label set later fits are scored against.
- Execution path: generate the sample from a linkage run's `pairs.jsonl`, review it in the
  workbench, then re-score the model against the ledger with the seam's label-based fit and accuracy
  analysis.
- Acceptance gates: every sampled pair carries a decision or an explicit cannot-tell; the sample
  covers each probability band, including the uncertain middle; the curve reports precision and
  recall at several thresholds with the reviewer cost the sample took; the recommended threshold
  is stated with what it would merge and what it would leave apart.
- Documentation target: the labelling section of `docs/impl/current/entity-resolution.md`, with the
  measured reviewer cost going to [review workbench](current/review-workbench.md).

#### gold-item-drop-policy-adoption

Decide whether the gold-item drop policy moves from the shipped cosine constant to the fitted
linkage model, and make the decision on the labelled curve rather than on the shadow report alone.
The shadow lane already publishes, per drafting run, the cut that reproduces today's decisions and
every item where a probability cut and the constant disagree -- see
[the gold-item lane](current/entity-resolution.md#the-gold-item-lane). What it cannot supply is
which side of each disagreement is right: a paraphrase the constant kept and the model would drop
is either a duplicate the constant missed or a distinct question the model would destroy, and only
a reviewer decision separates those. Score the model against the merge ledger, read the threshold
off the labelled accuracy curve, and either flip `NEAR_DUP_COSINE_THRESHOLD` and the multi-hop
answer constant for the model at that threshold, or record why the labelled curve does not support
the move.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the reviewer merge ledger from `entity-merge-labelled-set` under Human-Assisted
  Tasks is the gate -- without it there is no curve to read a threshold off. The lane, its record
  specification, and the shadow report already exist; the seam's label-fitted m estimation and
  labelled accuracy curve already exist.
- User-visible outcome: a drafting run's suppression decision carries a threshold with a precision
  and recall attached to it, and a reviewer who wants a looser or stricter policy can see what it
  would cost before changing it.
- Scope boundary: in scope -- scoring the fitted model against the merge ledger, the threshold
  recommendation, flipping the default (or recording the negative result), and re-running the
  shadow report at the adopted cut so the change is visible in one diff. Out of scope --
  re-deduplicating already-accepted ledgers, changing what a gold item contains, changing the
  drafting prompts, and adopting a cut the labelled curve does not support.
- Data and artifact paths: the drafting bundle's `linkage/` gains the accuracy curve a labelled fit
  writes; the adopted threshold lands beside the constants it replaces.
- Execution path: the drafting entrypoint with the shadow lane enabled and the merge ledger passed
  as the label table, over the same fixtures the lane is tested on.
- Acceptance gates: `make ci` green; the model is scored against the ledger with precision and
  recall at the adopted cut and at the shipped constant's operating point, so the two policies are
  compared on the same labelled pairs; a cut that does not beat the constant on both is recorded as
  a negative result and the constant stays.
- Documentation target: the gold-item section of `docs/impl/current/entity-resolution.md`.

### Retrieval evidence -- `retrieval-evidence`

#### embedder-decision-on-a-resolvable-item-set

The embedder choice is undecidable on the item sets the repo currently has, and the paired lane says
so precisely: on the accepted converted-PDF ledger 36 of 40 questions are TIED between the leader
and the incumbent, so the 95% paired interval spans `[-0.050, +0.150]` and only a consistent
~4-question gap could ever clear zero; on the committed fixture the baseline already retrieves
0.980, leaving 5 questions of headroom for any candidate to win ([RAG
core](current/rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty)). The
sub-base roster addition `intfloat/multilingual-e5-small` is ~3x faster on warm CUDA with flat
quality on n=82 and still RETAIN ([RAG
core](current/rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small)) -- include it when
the enriched ledger re-runs so a cheap CUDA swap can clear an adoption bar if the discordance is
there. Both existing sets are at their ceiling, which is a property of the QUESTIONS, not of the
encoders. Build an item set that can decide it: predeclare a minimum detectable recall gain and the
split size it needs, then assemble a ledger enriched with questions the incumbent currently MISSES
(mine the per-item vectors in `report.json` for baseline zeros, plus domain-term and
morphology-heavy questions the general E5 encoder is expected to fail), accept it through the
verification gate, and re-run the bake-off on it. Record whether any candidate then separates -- a
recorded "still undecidable at n=N" is a valid outcome and is what would justify closing the
question. The size the ENRICHMENT has to buy is already priced: the withdrawn `e5-large` adopt
differs on 5 of 250 items, and at that rate the reporting level needs 300 items, which no committed
goldset reaches -- so plain "more questions" is not the route, raising the discordance rate is
(double the rate, halve the floor) ([the
re-decision](current/rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs)).

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes)
  supplies the item count the predeclared gain needs; the paired bake-off lane, the verdict, and
  `report.json` are current behavior ([RAG
  core](current/rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
  Human step that gates completion: a reviewer accepts the enriched question set through `make
  verify-review` / `make verify-accept`, since an unaccepted ledger cannot settle a default.
- User-visible outcome: either a measured embedder swap an operator can adopt, or a recorded
  statement of how many questions a decision would need -- instead of a permanently open ranking.
- Scope boundary: in scope -- the power target, the enriched ledger, its acceptance, and one
  re-run per corpus with the verdict restated. Out of scope -- new candidates, fine-tuning (that
  is `ua-embedder-domain-finetune`), and widening the verdict bar beyond recall@k.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout.
- Execution path: `make compare-embeddings GOLDSET=<accepted-enriched-goldset> NOISE_FLOOR=1` on
  the CUDA host; no new CI coverage.
- Acceptance gates: the predeclared minimum detectable gain and split size are written down BEFORE
  retrieval; every candidate row reports its paired interval on the accepted ledger; the verdict is
  recorded as adopt, retain, or explicitly undecidable at the reached sample size.
- Documentation target: the embedder bake-off evidence in [RAG core](current/rag-core.md) and the
  recommendation line in [platform matrix](current/platform-vector-matrix.md#embedding-bake-off).

#### goods-fusion-weight-accepted-ledger

Settle the goods-corpus fusion-weight verdict on an item set someone accepted. The recorded verdict
("the BM25 side costs recall at w=0.5, pin `FUSION_WEIGHT=0.7`") was measured on a verified 44-item
quickstart-PDF accepted goldset that is no longer on disk, and the lexical-row re-read could not
reproduce it: on the SAME corpus at the SAME chunking, the 95-item drafted goldset inverts it --
fusion ADDS recall at w=0.5 (+0.021, +0.053 with lemmas, against a +/-0.000 floor) and w=0.7 is the
worst of the three weights for the best row ([RAG
core](current/rag-core/hybrid-retrieval.md#lexical-row-re-read-of-the-fusion-weight-verdict)). The
pin is already withdrawn; what remains is deciding whether the recorded verdict was an artifact of
its item set or of the drafting, which only an accepted ledger over that corpus can answer.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code -- `make compare-retrieval HYBRID=1 NOISE_FLOOR=1` and the
  verification gate are both current behavior. Human step that gates completion: a reviewer
  accepts (or rejects) enough of the drafted goods questions through
  `make verify-review` / `make verify-accept` to produce an accepted goods ledger.
- User-visible outcome: an operator on a converted-PDF Ukrainian corpus gets a fusion-weight
  recommendation backed by human-reviewed questions, instead of two drafted-versus-vanished item
  sets that disagree.
- Scope boundary: in scope -- worksheet review, the accepted ledger, one re-run per recorded
  weight with the `lexical` row, and a keep-or-change verdict on the recorded table. Out of scope
  -- widening the weight grid, new fusion mechanics, and changing the shipped defaults before the
  accepted run supports it.
- Data and artifact paths: the drafted bundle under
  `$DATA_DIR/graph-vector-fusion-multihop/goods-draft/` plus a new
  `$DATA_DIR/lexical-row-reread/goods-accepted-w<weight>/` per weight.
- Execution path: `make verify-review` then `make verify-accept` over the goods draft, then
  `make compare-retrieval CONFIG=<cfg> GOLDSET=<accepted>/goldset.jsonl SPLIT= HYBRID=1
  NOISE_FLOOR=1` at `FUSION_WEIGHT=0.5,0.6,0.7`.
- Acceptance gates: every worksheet row has a decision; the three weights are scored on the
  identical accepted item set with the `lexical` row and the corpus's floor; the recorded table is
  restated as reproduced, corrected, or retired.
- Documentation target: the hybrid-retrieval evidence section of
  [RAG core](current/rag-core/hybrid-retrieval.md#hybrid-retrieval-dense--bm25--rrf).

#### reranker-end-to-end-crosscheck-on-an-evidence-model

The off-by-default reranker verdict asks to be overturned with END-TO-END evidence, and the only
end-to-end run behind it cannot supply that: it scored `llama3.2:3b` -- below the project's >=7B
live-evidence floor -- on n=14, where the two arms' intervals overlap so widely that the reading is
"no measured gain" rather than a measured loss, and its bundles are retained on no host
([rerank](current/rag-core/rerank-and-query.md#reranking-and-context-order-rerank-context-order)).
The retrieval half still stands on its own at that small k. So the open question is narrow -- does
that retrieval lift reach the ANSWER on a model the project accepts for evidence? Re-run the same
`rerank_candidates=0,30` cross-check on a >=7B local model, at an n predeclared to separate the two
recorded objectives, and restate the off-by-default verdict as reproduced, corrected, or retired. A
negative result -- the lift still does not reach the answer -- is the valuable outcome, because it
is what the shipped default rests on.

- Serves: `retrieval-evidence` -- [RAG core](../design/spec.md#capability-registry)
- Agent status: RUN NEEDED
- Dependencies: none in code. `make sweep` already takes the grid, and the paired lane already
  supplies the interval and the minimum-evidence gate
  ([paired verdicts](current/rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)).
- User-visible outcome: an operator deciding whether to pay ~0.96 s/query for reranking gets an
  answer measured on a model they would actually serve, instead of one bounded by a 3B smoke model.
- Scope boundary: in scope -- the predeclared n, the two-arm run on a >=7B model, and the restated
  verdict. Out of scope -- changing the shipped `rerank_candidates` default before the run supports
  it, widening the pool grid, and swapping the reranker.
- Data and artifact paths: one `$DATA_DIR/run-eval/<run>/` per arm under the sweep's own root.
- Execution path: `make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"
  RERANKER=BAAI/bge-reranker-v2-m3` on a CUDA host with a >=7B model resident; CI is unchanged.
- Acceptance gates: `make ci` green; the model is >=7B; the predeclared n is stated BEFORE the run
  and met; both arms report objective with paired intervals and the win/loss/tie ledger; the reading
  states whether the retrieval lift reaches the answer, and whether the minimum-evidence gate is
  cleared.
- Documentation target:
  [rerank](current/rag-core/rerank-and-query.md#reranking-and-context-order-rerank-context-order).

#### fusion-routing-calibration-power (optional)

Increase the sidecar-free routing calibration's statistical power before reconsidering its
production defaults. The first held-out measurement cannot separate its positive retrieval deltas
from zero; see the compact result and frozen-policy diagnostics in
[GraphRAG](current/graphrag-backend/sidecar-free-routing-calibration.md#sidecar-free-heuristic-calibration).
Assemble a larger, independent multi-span tuning/final ledger, declare its minimum detectable gain
and split sizes before retrieval, then repeat the frozen-policy workflow without widening the
threshold grid.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes)
  supplies the predeclared split sizes, and `multihop-ledger-human-acceptance` must provide a
  non-empty accepted multi-span ledger. Human
  step that gates completion: accept enough additional genuinely multi-span questions to meet the
  predeclared split sizes.
- User-visible outcome: operators can distinguish a useful sidecar-free route from a sparse-win
  artifact before changing the fallback defaults.
- Scope boundary: in scope -- a prospective power target, disjoint tuning/final splits, and one
  repeat of the existing deterministic calibration. Out of scope -- widening the threshold grid,
  a learned router, and selecting on final.
- Data and artifact paths: a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` calibration over
  the accepted ledger.
- Execution path: run `make calibrate-fusion-routing` with the predeclared splits, then run the
  masked `make compare-graph-fusion` reproduction for the frozen policy on each split.
- Acceptance gates: route precision/recall and paired retrieval intervals meet the predeclared
  power target; a threshold changes only if the tuning gain clears zero without single-span
  regression and the untouched final split confirms the same direction.
- Documentation target: the sidecar-free calibration subsections of
  [RAG core](current/rag-core.md) and [GraphRAG](current/graphrag-backend.md).

#### embedding-clustered-chunk-merging (optional)

The measured near-duplicate residue is real but not text-reachable: on the goods corpus 20.7% of the
exact-collapsed chunks have a neighbour at cosine >= 0.99, and the `normalized` collapse tier merges
26 of those 13105 pairs ([RAG
core](current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers)). Only an
embedding-side merge can reach the rest, which the collapse lane deliberately does not do because a
false merge silently deletes a distinct passage from the index. Decide it with a measured
false-merge rate instead of by assumption: cluster the survivors by cosine at several thresholds,
sample the merges for human reading (the residue report's pair sampler already renders them), and
score retrieval per threshold against the corpus's own measurement floor. Adopt only if a threshold
lowers the fragile count without a recall regression AND its sampled false-merge rate is acceptable
on a corpus whose facts differ by one number.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none. Reuse `measure_duplicate_residue` in `src/llb/rag/duplicates/residue.py` for
  the clustering and the sampler, and `collapse_duplicate_chunks` for the merge itself. Human step
  that gates completion: a reviewer reads the sampled merges at the candidate threshold and calls
  the false-merge rate acceptable or not.
- User-visible outcome: either an embedding-side merge tier with a measured false-merge rate, or a
  recorded finding that the residue must be left in the index.
- Scope boundary: in scope -- the clustering, the sampled review, the per-threshold retrieval run,
  and the adopt-or-reject verdict. Out of scope -- learned merge policies and corpus rewriting.
- Data and artifact paths: `$DATA_DIR/retrieval-noise-floor/<run>/`.
- Execution path: `make measure-duplicate-residue` per threshold, then `make compare-retrieval
  CHUNK_STRATEGIES=sentence,recursive NOISE_FLOOR=1` per candidate on the CUDA host.
- Acceptance gates: `make ci` green; the report carries the per-threshold recall against the floor,
  the fragile count, and the human's false-merge reading.
- Documentation target:
  [RAG core](current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers).

#### procedural-size-lever-resolution (optional)

The `size` lever against fragmented procedural evidence came ONE item short of a reading: on the
14-item `procedural` slice of the goods ledger, `size400` doubles whole-span delivery (0.357 ->
0.714) on 7 discordant items split 6/1, an exact randomization p of 0.0625 against the 0.025 a
separation needs, where a clean 7/0 would have separated ([two levers against fragmented
evidence](current/rag-core/fragmented-evidence.md)). The slice already clears the minimum-evidence
gate, so this is not an underpowered comparison -- it is a decidable one that came out short, and a
handful of accepted procedural items is what settles it. Enrich the procedural slice through the
verification gate, re-read the same `CHUNK_SIZES` lever on it, and -- only if it then separates --
take the end-to-end answer reading the spec requires of a retrieval configuration change, so the
served-context price is weighed against an answer the model actually produced rather than against
intactness alone.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code. Reuse the `--sizes` lanes and per-slice paired readings in
  `src/llb/rag/comparison/run.py`. Human step that gates completion: a reviewer accepts the added
  procedural items through the verification gate, since a drafted item cannot license a default
  change.
- User-visible outcome: the operator learns whether a wider `size` cap is worth roughly double the
  served context on procedural questions, or that it is not, on an item set that can say so.
- Scope boundary: in scope -- the added items, the re-read of the same three caps, and the
  answer-side reading conditioned on the model that took it. Out of scope -- changing the default
  `size` without the end-to-end reading, and re-tuning the chunker.
- Data and artifact paths: the existing `$DATA_DIR/table-aware-chunking/<run>/` comparison layout.
- Execution path: `make compare-retrieval CONFIG=<goods.yaml> CHUNK_SIZES=200,400,800 STITCH=1` on
  the CUDA host, then `make compare-answer-quality` on the same items if the slice separates.
- Acceptance gates: `make ci` green; the procedural slice reports its paired `span_intact@k` delta
  with the discordant count beside it; a separation is followed by the answer-side reading naming
  its model, and a non-separation is recorded as the negative result it is.
- Documentation target: [two levers against fragmented
  evidence](current/rag-core/fragmented-evidence.md).

#### cross-lingual-query-fixture-review (optional)

Review the Russian and UA/RU questions in the committed cross-language overlay for linguistic
naturalness and semantic equivalence to each paired Ukrainian question, then either accept the
whole overlay or keep it diagnostic-only with corrected draft rows. The diagnostic lane and its
current evidence are described in [evaluation
rigor](current/rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane).

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code. Human step that gates completion: a Ukrainian/Russian reviewer reads
  every UA/RU/RU triplet and confirms that both variants preserve the question's factual intent,
  named entities, numbers, and answer target.
- User-visible outcome: the language overlay either becomes accepted evidence or remains clearly
  drafted with a complete list of rows requiring correction.
- Scope boundary: in scope -- linguistic/semantic review, corrections, and one uniform review-state
  transition. Out of scope -- changing source spans or answers, translating the corpus, adding new
  languages, and changing query-prep defaults.
- Data and artifact paths: `samples/goldsets/ua_squad_postedited_v1_ru/goldset.jsonl`; a review may
  change all rows together to `provenance: human-verified` / `verified: true` only after every row
  passes.
- Execution path: compare each overlay pair with its id-matched Ukrainian row, correct rejected
  questions, run `make validate-goldset` against the original corpus, then re-run
  `make bench-query-robustness QUERY_ROBUSTNESS_CLASSES=language_ru,language_mixed`.
- Acceptance gates: every triplet has a reviewer decision; all accepted rows preserve the exact
  non-question payload; the fixture has one uniform review state; validation and the paired CUDA
  rerun are green; the current evidence page records the reviewed result.
- Documentation target: the cross-lingual section of
  [evaluation rigor](current/rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane).

### Answer scoring -- `answer-scoring`

#### calibrate-headline-format-weight

Calibrate the fact/format tradeoff against adversarial context-copy answers and human pairwise
utility labels, then retain or revise the declared weight without changing the decomposition
contract ([current scoring](current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy)).
Sweep predeclared weights over matched terse, fluent-but-wrong, verbose-supported, and
context-copy cases; measure agreement with the accepted labels and stability across model
families; require a held-out confirmation before changing the default.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: BLOCKED BY HUMAN
- Dependencies: a reviewer-accepted pairwise preference ledger covering the four answer shapes.
- User-visible outcome: the explicit format share is grounded in operator utility and resists a
  model that raises recall by repeating retrieved context.
- Scope boundary: in scope -- fixture and worksheet generation, weight sweep, family-stratified
  agreement, uncertainty, and an adopt-or-retain verdict. Out of scope -- prompt rewriting,
  judge-model substitution, and changing the precision/recall/found-rate columns.
- Data and artifact paths: accepted labels under `$DATA_DIR/review/headline-format/`; sweep
  artifacts under `$DATA_DIR/verbosity-sensitivity/<run>/weight-calibration/`.
- Execution path: agent prepares and validates the blinded worksheet; human reviewers accept the
  pairwise labels; agent runs the deterministic sweep and CUDA-host family confirmation.
- Acceptance gates: `make ci` green; all answer-shape strata are represented; held-out agreement
  and confidence intervals are reported per family; any default-weight change names every roster
  rank it changes.
- Documentation target: [RAG core](current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy)
  and the ranking policy in [evaluation rigor](current/rigor-board-judge.md).

### Judge calibration -- `judge-calibration`

#### frontier-judge-authorization

Authorize the frontier scorer lane against real providers. The report tooling is current behavior
([frontier judge agreement and cost report](current/rigor-board-judge/judging.md#frontier-judge-agreement-and-cost-report));
what remains is entirely the human authorization and the judgment it produces.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
- Agent status: HUMAN-GATED human_decision: panding
- Command: once a real Anthropic key is in `.env` (~$0.40, 86 items):

  ```bash
  make frontier-judge-agreement \
    FRONTIER_JUDGE_MODELS=anthropic/claude-sonnet-4-5 \
    FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=1.00
  ```

- Dependencies: the agreement lane is current behavior; it runs on the 86-row calibration
  worksheet `calibration/ua_squad_postedited_v1.csv` (every row carries both a human and a local
  judge rating). Human step that gates completion: the operator puts a real Anthropic / OpenAI /
  Google key in `.env` (all three are currently blank placeholders, so no live run is possible),
  approves the per-run spend cap, and signs off on the resulting report.
- User-visible outcome: a decision record stating whether each frontier judge is trusted for
  autonomous gates on Ukrainian data, plus default budget caps derived from measured
  cost-per-item rather than from a guess.
- Scope boundary: in scope -- running the existing lane on the committed UA fixture, reviewing
  the rho and cost tables, recording an accept/reject per provider, and landing the resolved caps
  in the sample configs. Out of scope -- sending any private corpus to a provider, changing the
  headline-ranking policy, and any further report tooling.
- Data and artifact paths: `$DATA_DIR/frontier-judge/<run>/`; fixture is
  `samples/goldsets/ua_squad_postedited_v1/`.
- Execution path: `make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<id>[,<id>...]
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`; needs live provider access and spend, so it
  stays outside CI entirely.
- Acceptance gates: the report carries a non-`n/a` rho per provider against both references and a
  priced cost-per-item with cap math; the human replaces `human_decision: pending` with an accept
  or reject per provider; the accepted caps land in the sample configs with the decision recorded.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) judge section and
  [product decisions](current/scope-boundaries.md) for the trust decision per provider.

#### frontier-judge-retrieved-context-agreement

Optional. Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
- Agent status: HUMAN-GATED
- Dependencies: blocked by `frontier-judge-authorization` (needs the same provider keys and
  spend). Reuse the agreement lane in `src/llb/scoring/frontier_agreement/` and the store-loading
  seam used by the context-position probe.
- User-visible outcome: evidence for whether frontier-judge trust measured on oracle context
  transfers to the noisy contexts a real scored run produces.
- Scope boundary: in scope -- the retrieved-context source, the two-mode comparison, and the
  delta. Out of scope -- changing the default grounding of the authorization lane before the
  comparison says it should.
- Data and artifact paths: no new roots; a second grounding mode inside the existing
  `$DATA_DIR/frontier-judge/<run>/` bundle.
- Execution path: `make frontier-judge-agreement` with a grounding-mode knob plus a built store;
  CI covers mode selection over a fake store and fake completers.
- Acceptance gates: `make ci` green; the gold-span mode reproduces the current numbers exactly; a
  live run reports rho under both modes with the delta called out.
- Documentation target: the frontier-judge agreement subsection of
  [evaluation rigor](current/rigor-board-judge.md).

### Graph retrieval and ontology -- `graph-retrieval`

#### ontology-axiom-signoff

Accept or reject each candidate axiom in `samples/ontology/axioms_uk_v1.ttl`, one at a time, before
any of them can gate an answer. An axiom is BUSINESS LOGIC, not a measurement: whether "has owner"
admits one value or many, whether `PERSON` and `ORG` are genuinely disjoint in this domain, and
whether a relation's range is closed are claims about the world that no corpus statistic can settle.
Inducing them from corpus frequency would only restate what the extractor happened to emit -- the
same circularity the conflict tier already hit, where the null and the observed population turned
out to be the same set ([data
prep](current/data-prep/conflict-detection.md#known-limitation-there-is-no-independent-null)) -- and
the cost of a wrong axiom is asymmetric and silent: at the ledger it deletes a true fact from the
report's attention, and at the answer gate it converts correct answers into `ontology_violation`.
The corpus cannot review itself here, which is why this is the one piece of the validation
architecture that sits in this section. The existing signed type-vocabulary review is the precedent
for the form ([graph ontology schema](../design/graph-ontology-schema.md)).

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: HUMAN-GATED
- Dependencies: the candidate axioms, their Turtle rendering, and the per-axiom evidence rows
  (supporting facts, contradicting facts, and the measured base rate per corpus) already ship as
  `axiom_evidence.jsonl` beside each validation run ([robustness and
  ontology](current/robustness-ontology-backends.md#ontology-axiom-layer)). Reuse the
  review-workbench ledger pattern
  ([review workbench](current/review-workbench.md)) so the decisions are recorded the same way every
  other review ledger is. Human step that gates completion: a domain reviewer decides `accept` or
  `reject` for EVERY candidate axiom and signs the resulting axiom file.
- User-visible outcome: a signed, dated constraint set an operator can point the answer gate at,
  where every enabled axiom is a decision someone made rather than a statistic the corpus produced.
- Scope boundary: in scope -- the per-axiom review worksheet (each axiom rendered as Turtle plus a
  Ukrainian-language gloss and its supporting/contradicting facts with exact spans), the review
  pass, the signed axiom file, and a recorded reason per rejection. Out of scope -- authoring new
  axiom CLASSES or changing the checker, changing the 13-type vocabulary or the relation caps, and
  enabling any axiom the reviewer did not accept.
- Data and artifact paths: the worksheet under `$DATA_DIR/ontology-validation/<run>/axiom_review.jsonl`,
  built from the `axiom_evidence.jsonl` the validation run already writes beside it; the signed set
  committed at `samples/ontology/axioms_uk_v1.ttl`, where a signature is `dcterms:creator` +
  `dcterms:date` on that axiom's `owl:Axiom` annotation block, mirroring the dated sign-off
  convention of [graph ontology schema](../design/graph-ontology-schema.md).
- Execution path: `make validate-ontology-axioms` to regenerate the candidates and their evidence,
  then `make review-workbench REVIEW_PATH=<axiom-review-jsonl>`; no GPU is required for the review
  itself.
- Acceptance gates: every candidate axiom has a decision and every rejection has a stated reason;
  the committed axiom file contains only accepted axioms and carries the reviewer name and date;
  re-running the checker over both quickstart corpora with the signed set reports its violation base
  rate per accepted axiom; `ontology-validated-answer-gate` refuses to enable an axiom absent from
  the signed file.
- Documentation target: the ontology-assisted drafting section of
  [robustness and ontology](current/robustness-ontology-backends.md#ontology-assisted-drafting), the
  constraints section of [graph ontology schema](../design/graph-ontology-schema.md), and the
  trust boundary in [product decisions](current/scope-boundaries.md).

### Autonomous orchestration -- `autonomous-orchestration`

#### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Serves: `autonomous-orchestration` -- [Autonomous orchestration](../design/spec.md#autonomous-orchestration)
- Agent status: HUMAN-GATED
- Dependencies: the autonomous lane is current behavior ([Auto-RAG](current/auto-rag.md));
  assisted review uses the [review workbench](current/review-workbench.md). Human step that gates
  completion: the operator
  performs both
  runs, reviews gated records in the workbench, measures their own throughput against the legacy
  per-flow sessions, and accepts or rejects the recommendation bundles.
- User-visible outcome: recorded evidence that the autonomous lane produces an acceptable
  recommendation without human action, and that the assisted lane's unified workbench is at least
  as fast and less error-prone than the legacy TUIs.
- Scope boundary: in scope -- the two runs, reviewer-throughput measurement (records per minute,
  correction rate), a comparison of the two recommendation bundles, and the acceptance decision.
  Out of scope -- fixing findings (each finding becomes a new forward task).
- Data and artifact paths: both run bundles under `$DATA_DIR/auto-rag/<run>/`; throughput notes
  and the acceptance record under `$DATA_DIR/auto-rag/<run>/acceptance/`.
- Execution path: `make auto-rag CORPUS=<dir> SCORER_POLICY=auto`, then
  `make auto-rag CORPUS=<dir> SCORER_POLICY=human` with workbench review at each gate; both on
  the CUDA host with a real corpus the operator owns.
- Acceptance gates: the human signs the acceptance record; both bundles are complete and
  reproducible from their manifests; throughput numbers and any usability findings are captured
  as new forward tasks before this item leaves the file.
- Documentation target: `current/auto-rag.md` acceptance evidence and
  [human evaluation guide](../guides/human-tooling/human-in-the-loop-evaluation.md).

### Robotics RAG and hardware operation -- `robotics-rag-operation`

#### robotics-mhs-preview-conformance

Replace the protocol-neutral fake with a real adapter only when an authorized preview participant
can provide an inspectable, versioned MHS contract and a non-production test endpoint. Map discovery,
reference generation, read, write, enforced limits, errors, and receipts onto the project records,
then run the same conformance suite as the emulator. A missing schema, incompatible license,
unstable identity/state semantics, or transport that bypasses the external gate is a valid negative
result and must leave the project labelled protocol-neutral.

- Serves: `robotics-rag-operation` -- [Robotics RAG and hardware operation](../design/spec.md#robotics-rag-and-hardware-operation)
- Agent status: HUMAN-GATED, RESEARCH
- Dependencies: the [robotics boundary contracts](current/robotics-rag/boundary-contracts.md) and
  the [action gate and emulator](current/robotics-rag/action-gate-and-emulator.md). Human step
  that gates completion: an authorized MHS preview participant accepts the applicable terms,
  identifies the exact contract/package version, and grants time-bounded access to a simulator or
  read-only test device without placing credentials in the repository or run artifacts.
- User-visible outcome: either a version-scoped MHS adapter with reproducible conformance evidence,
  or a precise incompatibility/insufficient-contract decision that prevents an unsupported
  compatibility claim.
- Scope boundary: in scope -- one optional adapter, contract and license pinning, discovery/read
  against a non-production endpoint, fake write conformance with no physical effect, and deviations
  from the project contract. Out of scope -- applying for access without the operator, accepting
  terms on their behalf, live hardware writes, bypassing the project gate, storing credentials, and
  declaring compatibility across untested MHS versions.
- Data and artifact paths: sanitized contract metadata and conformance output under
  `$DATA_DIR/robotics-mhs-conformance/<run>/`; committed tests use redacted recorded responses or an
  injected fake under `samples/robotics/mhs/`, never preview package bytes whose terms forbid it.
- Execution path: add `make robotics-mhs-conformance` behind a `robotics-mhs` optional dependency
  group and explicit endpoint configuration; keep preview access outside quick CI and replay the
  adapter against contract-safe fixtures in `make ci`.
- Acceptance gates: the human records the exact contract and allowed artifact handling; discovery,
  read, limit refusal, error, and receipt cases map without loss; every write attempt still traverses
  the project gate; changed MHS identity or schema makes the adapter fixture stale; the report says
  `MHS-compatible` only for the tested version; a negative result removes this task to future
  research with its reopening condition instead of weakening the adapter contract.
- Documentation target: `docs/impl/current/robotics-rag/mhs-adapter.md` and the dependency/security
  boundary in `docs/impl/current/robotics-rag.md`.

#### robotics-supervised-device-canary

Run the admitted MHS adapter on one isolated workcell under its hardware owner's supervision. Start
with discovery/read-only calls and shadow proposals, reconcile every observed identity, unit, limit,
state transition, and receipt against the emulator, and only then authorize one predeclared low-risk
write whose physical envelope, rollback, interlocks, and stop procedure the owner signs. Any drift
returns the capability to shadow mode; the canary never turns on unattended operation.

- Serves: `robotics-rag-operation` -- [Robotics RAG and hardware operation](../design/spec.md#robotics-rag-and-hardware-operation)
- Agent status: HUMAN-GATED
- Dependencies: the [held-out emulator benchmark](current/robotics-rag/benchmark.md) and
  `robotics-mhs-preview-conformance`. Human step that gates completion: the hardware owner supplies
  the isolated device, approves the risk and rollback worksheet, verifies interlocks and an
  operator-controlled emergency stop, observes the entire run, grants the proposal-bound write
  approval, and signs the canary verdict.
- User-visible outcome: a bounded real-device reading that names where emulator behavior transfers
  and where it does not, without silently promoting a benchmark result into deployment authority.
- Scope boundary: in scope -- read-only and shadow phases, one low-risk bounded write, pre/post-state
  capture, stop and rollback drills, emulator/device mismatch logging, and a human verdict. Out of
  scope -- safety certification, people or uncontained hazards in the work envelope, high-risk
  tools, generated procedure files, model-driven emergency actions, parallel live writes,
  unattended running, and generalizing from one workcell to another.
- Data and artifact paths: run bundle under `$DATA_DIR/robotics-canary/<run>/`; raw telemetry and MCAP
  remain in the authorized HFlow data root and are referenced by digest; the signed risk worksheet,
  approvals, redacted device snapshot, and verdict live in the run bundle with no secrets.
- Execution path: add `make bench-robotics-canary MODE=read,shadow,bounded-write`; require a fresh
  preflight before each phase, and require a new explicit approval digest before the write phase.
  The operator, not the model, advances phases.
- Acceptance gates: the owner signs the worksheet and final verdict; the physical stop prevents
  motion independently of the model and adapter; zero out-of-policy invocations occur; every action
  has matching pre-state, decision, request, receipt, and post-state evidence; an ambiguous result
  triggers read/escalation with no retry; every emulator/device mismatch is reported; any mandatory
  gate failure retains shadow/read-only mode and records the negative result.
- Documentation target: `docs/impl/current/robotics-rag/device-canary.md` and a canary procedure under
  `docs/guides/human-tooling/`.

### Corpus conflict and governance -- `corpus-conflict-audit`

#### conflict-adjudicator-label-slice

Produce frozen human labels for real candidate rows so the audit's measured claim-tier precision can
be trusted off the planted fixture. The shipped calibration gate scores the adjudicator only against
the seven-document planted probe, whose relations are synthetic by construction and which the
current host model passes 24/24 ([conflict
detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe)); nothing measures
whether the model agrees with a human on HR or goods rows.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: HUMAN-GATED
- Dependencies: the audit's precision block and its calibration gate are current behavior
  ([conflict detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision)) and
  consume the resulting bound; candidate ranking and claim adjudication are current behavior too.
  Human step that gates completion: an authorized reviewer assigns one relation from the claim
  vocabulary to every row of the frozen slice without seeing the model's verdict.
- User-visible outcome: a committed frozen slice plus a measured human-versus-adjudicator agreement
  bound -- what lets a precision number transfer to corpora the planted fixture does not represent.
- Scope boundary: in scope -- slice selection stratified by corpus and rank band, blind review,
  agreement measurement, and the resulting calibration bound. Out of scope -- relabelling the planted
  fixture, changing the relation vocabulary, and using the reviewed slice to fit any threshold.
- Data and artifact paths: `$DATA_DIR/corpus-conflicts/<run>/` for the slice and agreement report;
  the frozen labels are committed under `samples/`.
- Execution path: generate the stratified slice from a claim-tier run, review it with
  `make review-workbench REVIEW_PATH=<slice-jsonl>`, then re-measure agreement against it.
- Acceptance gates: every slice row carries a human relation; agreement is reported with a clustered
  interval; and the precision block stays suppressed on any corpus whose agreement bound is unmet.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md) and [review
  workbench](current/review-workbench.md).

#### corpus-conflict-resolution-review

Review the unresolved semantic conflict candidates through the workbench, then feed the accepted
ledger back into the resolver and repeat the retrieval plus verified answer-quality comparison.
The resolver behavior and the reason semantic candidates have no automatic suppression authority
are current behavior in
[data prep](current/data-prep/conflict-resolution.md#corpus-conflict-resolution-corpus-conflict-resolution).

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: HUMAN-GATED
- Dependencies: the resolution lane is current behavior. Human step that gates completion: an
  authorized corpus reviewer chooses `keep_both`, `drop_a`, or `drop_b` for every escalated row
  and signs off on the resulting suppression directives before application.
- User-visible outcome: an accepted or rejected suppression policy backed by reviewed conflict
  labels and a repeatable effect report, instead of adopting semantic similarity candidates as
  deletions.
- Scope boundary: in scope -- workbench review, accepted-ledger application, the same before/after
  metrics, and an adopt-or-revert decision. Out of scope -- changing detector thresholds,
  rewriting source text, or adding the resolver to auto-rag before the reviewed run supports it.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/resolution_review.jsonl`,
  `plan.json`, `conflict_overlay.json`, and `effect.md`; no new artifact root.
- Execution path: `make review-workbench REVIEW_PATH=<resolution-review-jsonl>`, then
  `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<resolution-review-jsonl>
  APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>` and repeat the fixed verified objective run.
- Acceptance gates: every review row has a decision; the regenerated plan has no unresolved
  records; rollback still restores the exact baseline; the human accepts only if retrieval and
  verified objective metrics do not regress.
- Documentation target: the resolution evidence subsection of [data prep](current/data-prep.md)
  and the conflict adapter notes in [review workbench](current/review-workbench.md).

### Operator review tooling -- `operator-review-tooling`

#### conflict-group-review-throughput (optional)

Whole-group review is now possible but unmeasured: a reviewer can settle a decision group with one
`keep_both` row, and on the goods semantic bundle six such rows settle all 100 escalations
([conflict resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger))
-- but nothing establishes that a human reading ONE group record decides as accurately as one
reading its rows, which is the assumption the whole collapse rests on. Measure it: have a reviewer
settle one corpus's escalations row by row and another's group by group, record wall-clock time per
decision and the disagreement rate between the two passes on the same rows, and state whether group
review is safe for `keep_both` at the group sizes this repo actually produces (largest 51 rows).

- Serves: `operator-review-tooling` -- [Operator review tooling](../design/spec.md#operator-review-tooling)
- Agent status: HUMAN-GATED -- the deliverable is the reviewer's measured throughput and agreement;
  the ledger, the timing capture, and the disagreement report are agent-buildable.
- Dependencies: reuse the grouped review ledger and the group-wide keep in
  [conflict resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
  and the review TUI adapter in `src/llb/review/adapters/conflicts.py`.
- User-visible outcome: an operator learns whether reviewing by decision costs accuracy before
  adopting it as the default review mode.
- Scope boundary: in scope -- the paired review passes, per-decision timing, the disagreement
  report, and a recommendation on group size limits. Out of scope -- extending group decisions to
  destructive actions before the measurement supports it, and any change to the grouping rule.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts plus a
  timing ledger beside them.
- Execution path: two review passes over the committed goods bundle; CI covers the timing capture
  and the disagreement report over fixture ledgers.
- Acceptance gates: `make ci` green; both passes cover the identical rows; the report states time
  per decision and the disagreement rate, and either recommends group review or names the group
  size above which it stops being safe.
- Documentation target: [conflict
  resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
  and the [verification gate](current/data-prep/verification-gate.md).

#### conflict-review-ledger-cost-model (optional)

The review ledger now ranks its blocks by `to review`, on the assumption that N open rows cost a
reviewer N times one row ([conflict
resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)).
That is almost certainly wrong in the direction that matters: the goods semantic bundle's 51-row
group is 51 rows against ONE shared chunk, so a reviewer who reads that chunk once decides the rest
by comparison, while 51 single-row groups are 51 unrelated reads. Ranking on raw open rows therefore
over-states the concentrated group exactly where grouping was introduced to stop over-stating it.
Measure it: time a reviewer (or a scripted proxy over the review TUI) on a concentrated block versus
the same number of scattered rows, fit a per-group cost of the form `a + b * open_rows`, and rank on
the fitted cost -- or record that the linear rank is within the measurement's own noise.

- Serves: `operator-review-tooling` -- [Operator review tooling](../design/spec.md#operator-review-tooling)
- Agent status: HUMAN-GATED
- Dependencies: reuse the ledger, its group blocks, and the review TUI's group titles; the two
  goods bundles under `$DATA_DIR/corpus-conflicts/` already supply one concentrated and one
  scattered shape.
- User-visible outcome: the first decision the ledger shows a reviewer is the one that actually
  costs the most of their time, not the one with the most rows.
- Scope boundary: in scope -- the timing protocol, the fitted cost, and the ranking key it implies.
  Out of scope -- changing what a review record is, group identity, and the audit-side ranking,
  which has no policy to rank on.
- Human step: a reviewer works both shapes under measurement; the fit cannot be produced from
  artifacts alone.
- Acceptance gates: `make ci` green; the report states the fitted per-group and per-row costs with
  their uncertainty, and either changes `_stake_order` or records that the linear rank survives.
- Documentation target: [conflict
  resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger).

## Adding Future Tasks

**First decide whether the work is a task at all.** Three kinds of finding look like forward work
and are not:

- A **chore**: a source file over the soft line limit, or a run artifact already inside its size
  bound that could still be smaller. Do it inline while editing the file, or leave it. Written up
  with acceptance gates, a chore competes with the product on equal footing and each round produces
  the next round.
- An **audit of our own output**: an artifact's field layout, an artifact's byte count. The measure
  of an audit is a decision it changes, not a byte it saves.
- A **new domain capability** the spec does not describe. This one IS worth doing, and it is not a
  plan task yet: run the six steps of
  [Extending this specification](../design/spec.md#extending-this-specification) first -- state the
  problem in operator terms, amend the spec, declare the evaluation, register the capability, and
  only then write tasks under it. That order is what makes a discovery durable rather than a private
  detour.

**Then write the task.** It needs a stable descriptive id such as `platform-matrix-power` or
`prompt-system-tuning`, kept only while work remains under it, and enough detail for an engineer or
an agent to execute without guessing. File it under the `###` group of the capability it serves, in
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; mark any cross-section block explicitly.

Each task entry must include:

- Serves: the capability id from the
  [capability registry](../design/spec.md#capability-registry) this work advances, linked to the
  spec section that owns it. Checked by `make lint-spec-plan`.
- Dependencies: prerequisite tasks (by number/id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, route it by the three kinds above: handle a chore inline or
dropped, a self-audit gets dropped, and a genuine capability goes through the spec lifecycle before
it becomes tasks. Extension is welcome and finishing a task while adding nothing is equally normal --
what is not normal is a task quietly appearing under a capability nobody amended the spec for. Put
current behavior and durable decisions in current docs, never in this plan.

A RESEARCH task whose answer comes back negative leaves this file too, but it is not simply deleted:
move it to [future research](future-research.md) with what closed it and the conditions that would
make it worth reopening. Its measurements belong in the current docs like every other finished piece
of work; what future-research.md adds is the reasoning a later reader needs before spending the same
effort again.
