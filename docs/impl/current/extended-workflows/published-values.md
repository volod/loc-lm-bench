# Published crossovers under the shipped cap

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

A study's committed design and the analysis it publishes beside a run bundle are both registered
records (`llb.study-design`, `llb.study-analysis`), validated before publication and written in the
study's own local form so the bytes a citation resolves against never move; see
[run, study, and board contracts](../artifact-contracts/run-and-evaluation-contracts.md).

`make bench-agentic-context-compact-crossover-restatement` answers the question the step-aligned
bound leaves behind: every compact routing number an operator applies was measured under the retired
trigger bound, which discounted compact's own cost wherever it actually trimmed the folded
transcript. Re-running four studies to find out where that mattered would be the expensive answer.
The cheap one is exact.

The bound reaches a run through exactly ONE prompt -- the summarize call -- so a cell whose
summarize input is identical under both bounds sends bit-identical prompts under both, and its
published cost cannot have moved. `compact_fold_input_probe` decides that per cell with no model, so
the study opens with a model-free AUDIT of every published cell in the boundary surface, the trigger
collapse, and the fold-step crossover, and re-measures only the cells the audit calls
bound-sensitive. The committed design is
`samples/benchmarks/agentic_compact_crossover_restatement_design.json`; it names each audited study
by its in-repo design path and every crossover that study published, and validation refuses a design
whose path is missing, declares a different study kind, publishes a crossover at a depth the study
does not test, omits the fold step a crossover lands in, publishes a malformed statement, or
publishes a derived value that does not declare what it is computed out of, how it is computed, and
how the published statement is read.

Every verdict is stated over TWO walks of that world, because the oracle walk is the shortest
transcript that finishes and a longer real one can reach a summarize-input cap the oracle never
touched. Each audit row therefore carries `worst_case_verdict` beside `verdict`, read on the longest
transcript the cell's own step budget allows, and the summary rolls up
`worst_case_only_sensitive` -- the cells the oracle calls invariant and imperfect play does not. On
the committed studies the two agree cell for cell, and the mechanism is geometric rather than lucky:
see [the imperfect-play safety margin](imperfect-play-margin.md#the-bound-invariance-verdict-stated-for-the-worst-case).

The fold-step ANNOTATION is validated too, against the ladder the study that published the number
measured, because the annotation is what a restated guard is then checked against: a mis-transcribed
one reads as `a_published_crossover_moves_under_the_shipped_cap` the first time a cell at that depth
becomes bound-sensitive, blaming a bound change for a number that was mis-stated before the bound
moved. The three forms place by three different rules, and the check is exactly those rules -- an
interpolated guard lies INSIDE its step's guard interval, a fold-step boundary IS that interval's
exclusive upper edge (`fold no later than step k` is the rule "keep the guard below this", so the
boundary is the first guard that folds one step later), and a derived ratio, having no guard of its
own, must name the step of the guard it DECLARES it is derived from. That last rule ties two studies'
annotations together, so a depth where they disagree is refused whichever of the two is wrong. Every
one of these is a pure function of the study's held geometry, so the whole check runs in CI with no
GPU, and it is what corrected the depth-6 boundary-surface row below from step 7 to the step 6 its
own interval `[13136, 14912)` places its 14160-char guard in.

The annotation check bounds how wrong a published VALUE can be, and the bound is loose: a digit
dropped from `14159.929807575942` still lands inside step 6's `[13136, 14912)`, so every rule above
passes and the restatement re-checks a number no run produced. Each published crossover therefore
also names the run artifact and the field it came from (`provenance: {artifact, field}`, the artifact
DATA_DIR-relative), and the cells that aggregate measured at that depth (`cell_ids`), and design
validation reads the number back out instead of trusting it. The comparison is EXACT -- the aggregate
writes the full float and JSON round-trips it, so any tolerance at all would license precisely the
slip that stays inside the published step. The cell list is checked against the committed copy
through the same pointer walk (`cells[cell_id=...]`), so a pin-gate failure that invalidates those
cells can name the exact figures the docs publish rather than only the cell ids to re-measure.

## Committed aggregates, content pins, and the growth budget

The evidence is SELF-CONTAINED: the repo carries the cited run aggregates themselves, one verbatim
copy per artifact under `samples/benchmarks/agentic_published_aggregates/` at the artifact's own
DATA_DIR-relative path, plus a manifest
(`samples/benchmarks/agentic_published_crossover_aggregates.json`, `schema_version` 3) pinning each
copy by `sha256` content digest -- `{"digest": "sha256:..."}` per artifact, nothing else, because
everything else is the file. CI resolves all six numbers out of those bytes with no run on the host.
The copies and the manifest are written by `make bench-agentic-published-provenance` rather than
typed, which is the whole difference between a resolution and a second transcription of the number
the first one got wrong; the bytes and the digest come out of one read of one file, so a copy and
the pin that makes it falsifiable cannot disagree.

Committing the bytes is what makes the pin mean anything off the run host. A digest alone pins a
file only where that file exists: on CI, a fresh clone, or this host after a `.data` cleanup it
stands for a file nobody present can read, so evidence and pin fabricated together were accepted
there and the resolution proved "agrees with itself, plus a hash of something absent". With the file
carried, the pin is verified on EVERY host against the copy that holds those bytes, so a manifest
entry that pins nothing, pins a malformed digest, pins bytes the repo does not carry, or pins a
digest the committed copy does not have is refused before any value is read. A signed manifest was
the alternative and was rejected: a key committed beside the manifest signs whatever the fabricator
wants, and a key held outside the repo authenticates the operator who ran the regeneration rather
than the run, while adding key distribution to a check that must work in a fresh clone with nothing
configured.

The bytes must also hold together INTERNALLY. `published_value/aggregate_consistency.py` binds
each supported field pointer to the production arithmetic that created it: an interpolated guard is
re-run through `depth_surface_row` over the aggregate's scored cells; a fold-step boundary is rebuilt
through `step_rows` and `depth_fold_row` from its cells, compact share, measured prompt sequence, and
step rule; and a cap peak is re-measured from the prompt sequence produced by the aggregate's held
geometry. Comparison is exact, a missing source row is a refusal, and an unregistered numeric field
is not resolvable by default. `PublishedValueResolver` applies this check after reading the cited
field, so editing a derived number and re-pinning the aggregate fails on a host with no run root.
The focused fixture cases in `test_agentic_published_aggregate_consistency.py` exercise all three
forms plus missing-source and unknown-field refusals. This still makes no signature claim: a
determined fabricator can rewrite and re-pin a whole SELF-CONSISTENT study, but can no longer invent
one cited number while leaving the evidence beside it unchanged.

The `DATA_DIR` artifact stays as the third source and the strongest one, on the hosts that have it:
the committed copy IS the run's file, so the check is byte-for-byte, and a copy taken from another
run of the same study is refused even where its cited field happens to agree -- the case a value
comparison structurally cannot see.

Growth is bounded rather than open-ended, since these aggregates accumulate with every study that
adopts the seam. Three policies keep it so, in `published_value/fixture.py`: a
`MAX_AGGREGATE_BYTES` (256 KiB) per-artifact cap refusing to commit an analysis artifact too big to
carry -- publish the value out of a narrower one instead; a `MAX_COMMITTED_BYTES` (2 MiB) budget
across the whole committed tree, asserted in CI so the policy has teeth rather than being advice;
and a regeneration that PRUNES every copy no published value still cites, so the cost tracks the
number of CITED artifacts and not the number of runs. The three aggregates behind the six published
crossovers total 128,649 bytes.

The pruned tree is SHARED, so "still cites" is the union over every design that publishes resolvable
values rather than the citations of whichever design a refresh was handed. That distinction is
invisible with one study and is evidence loss with two: refreshing through either design would
delete the other study's committed aggregates, and the deletion reads as a clean prune rather than
as a retirement. `published_value/registry.py` therefore holds the registry the refresh
walks -- `PUBLISHED_VALUE_DESIGNS`, one entry per publishing design carrying its in-repo path and
the reader that names the artifacts its values resolve against, the same shape as
`AUDITED_DESIGN_PATHS` in the policy audit and for the same reason: two code paths that disagree
about which evidence exists are two answers to one question. `make bench-agentic-published-provenance`
commits the union, an artifact two designs cite is carried once, and a design is invisible to the
refresh until it is registered -- so adopting the resolver in a new study means registering it in
the same commit. Three refusals guard the union rather than trusting the walk: a registered design
the repo does not carry stops the refresh (unknown citations are not the same as none); a registered
design whose run is not under DATA_DIR stops it too, naming that design, because a partial host is
the ordinary way a study's evidence would have been silently retired; and the write seam itself
refuses any set that omits an artifact the registry says is cited, so no caller can prune another
design's aggregates by construction. `--refresh-provenance` is registry-driven for the same reason
and refuses a `--design` the registry does not name instead of regenerating from it alone. Retiring
a design from the registry is what prunes its evidence, which is the one act that should.

## The registry walk, the refresh, and collecting refusals

A registry entry carries what CHECKS its published values as well as what they cite, because those
are two different guarantees and the weaker one is the more convincing. Citations alone make a
study's evidence DURABLE -- its aggregates committed, pinned, and safe from someone else's refresh
-- while saying nothing about whether the numbers the design publishes are the ones those bytes
hold, and committed bytes beside unchecked claims read exactly like the checked case. So each entry
also states a `validate_published_values` callable (the restatement's runs its own
`validate_restatement_design`, which resolves all six crossovers), `validate_published_designs`
walks the registry and calls every one of them, and CI asserts that walk over the shipped registry
with no run on the host. Registering a design is therefore what subjects it to the check, rather
than remembering to add a test named after it. Two refusals keep that from degrading into silence:
an entry that registers no validation is refused at construction -- `None` is the shape an opt-out
would take, and opting out is what the field exists to prevent -- and the walk returns the kinds it
validated, so a registry that checked nothing is distinguishable from one that checked everything
instead of both passing. A design the repo does not carry fails the validation walk for the same
reason it fails the refresh: unknown claims are not the same as none.

The refresh runs that walk over what it just wrote, so regenerating the evidence and finding out
whether the published values still resolve out of it are one command rather than two days apart.
They are the same question asked of the same registry, and while only the first half was answered at
the refresh an operator who re-ran a study, re-committed its aggregates, and read a clean
`provenance manifest -> ...` learned on some LATER `make ci` that the design's numbers are now the
old run's -- at the point where the fix is a design edit and the run context is gone.
`report_published_designs` is therefore the primitive: it collects what did not resolve instead of
raising, `validate_published_designs` is the refusing form built on it (so CI and the refresh read
the same walk, and the CI refusal names EVERY unresolved design rather than the first), and
`refresh_committed_evidence_and_report` composes the write with the walk.
`make bench-agentic-published-provenance` then either names the walked kinds -- named rather than
counted, for the same reason the walk returns them -- or names each design and the value its own
evidence no longer states, and exits `3` (distinct from the usage refusals' `2`, which write
nothing). The write STANDS in that failing case, and is not rolled back: a refresh after a re-run is
exactly the repair flow, so the new aggregates are what the operator has to commit before they can
restate anything, and refusing would leave them with neither the evidence nor a way to record it.
Report-and-exit, not refuse-and-revert.

The same collecting move runs INSIDE a design, because a re-run moves as many published numbers as it
moves. `validate_published_provenance` resolved crossovers in order but raised at the first one the
aggregate no longer stated, so a re-run that moved three of this study's six numbers named one, and
the operator restated it, re-ran the refresh, and met the next -- the loop above, moved down a level.
It now resolves every crossover, collects, and refuses once with all of them named and counted
against the total (`3/6 published values do not resolve out of the evidence the design cites: ...`),
so restating a study is one design edit. The accumulator is
`published_value/collection.py`, study-agnostic beside the resolver for the same reason: the
next design to publish resolvable values inherits the behavior instead of re-deriving it.

Collecting forces two distinctions that stopping at the first refusal never had to make. The derived
band is a QUOTIENT of the surface's interpolated guard, so a guard the evidence no longer states is
the CAUSE of that band being unresolvable rather than a second moved number -- reporting both would
name one moved measurement twice and send the operator to restate a band nothing here can evaluate.
The band is marked `[not judged]` against the guard named above it, and the band-level comparison
(whose edges are the smallest and largest of the per-depth quotients) says so explicitly rather than
passing quietly when a quotient it needs is missing, since a band nothing could re-derive would
otherwise read exactly like one that was re-derived and held. SHAPE refusals stay fail-fast, ahead of
the first read: a crossover with no `provenance` object, no numeric `value`, or no band edges is a
design that never said what would state its number, and that message must not arrive underneath a
list of values that could not be checked because of it.

## What a published value declares it is derived from

WHICH value is a consequence of which is the DESIGN's statement, not the resolver's. Cause-versus-
consequence was first written into the restatement's own resolver as one hardcoded edge -- "the
trigger ratio is a quotient of the boundary surface's interpolated guard at the same depth" -- which
the study-agnostic accumulator could not see, so a second publishing design, or a third form here
whose value derives from two others, would have had its consequences reported as causes. Each
published value now DECLARES what it is computed out of, beside its provenance:

```json
"derived_from": [{"study_kind": "compact_memory_boundary_surface", "depth": 6,
                  "form": "interpolated_guard"}]
```

A list, because a derived figure can rest on two published numbers as readily as on one; absent
entirely when the value is measured, so silence is the right default and an empty list is refused
rather than read as "measured". The identity is `(study_kind, depth, form)` -- the form is part of it
because one study can publish a guard and the boundary of the step it sits in at the same depth, and
a declaration naming only the depth would be ambiguous exactly where the edge must be exact.
`published_value/derivation.py` reads one value's declaration and
`published_value/derivation_graph.py` walks the design's whole set, validating every
declaration against the design that publishes the source (a source the design does not publish, a
value that declares itself, a cycle, and two rows claiming one identity are all refused fail-fast,
beside the other shape rules); `CollectedRefusals` carries the resulting graph:
`rests_on_unresolved` marks a value `[not judged]` whenever anything it
declares is already unresolved. Transitively, and naming only the ROOT of each chain -- a value two
derivation steps above a moved measurement names that measurement, because the derived figure in
between is a consequence too and restating it would fix nothing. So the four passes are shape, every
STATED value in the design's own order, the consequences of whatever did not resolve, then the DERIVED
band out of what did.

## The registered arithmetic and its self-check

The sources are half a derivation; the ARITHMETIC over them is the other half, and it used to live in
the readers -- the trigger ratio was `compaction_trigger_chars(guard, share)` over the depth's cap
peak in design validation AND in the restated row, two copies that agreed only because both were
written to agree. So the design names its arithmetic beside its sources, and both readers call it:

```json
"operation": "trigger_over_own_cap_peak"
```

`operations/registry.py` is the registry the name resolves through -- a table of pure
functions in the shape of `PUBLISHED_VALUE_DESIGNS`, each stating how many sources of which FORM it
is computed over, which of the value's own stated fields it reads (`compact_share`), and whether it
also reads the figure the value's own aggregate measured (the cap peak). Those are checked against
the declaration before a number is read: an operation the registry does not carry, a declaration that
is not the shape its operation takes, a stated operand the design does not state numerically, sources
with no operation, and an operation with no sources are all refused. An operation returns the value
plus the intermediates it is willing to NAME (`trigger_chars`), so a restated row reports the trigger
it divided without a second module re-deriving it. What deliberately stays out is an expression
language in the design file: a design picks arithmetic by name, and adding a kind of arithmetic is a
registered pure function with a test rather than a formula every reader would have to evaluate
identically. Identity-only readers still ask by shape -- a restated row names the study its figure
came from via `source_of_form(FORM_INTERPOLATED)` over the declared sources -- so a ratio declaring a
guard at another depth is still caught by the fold-step annotation it then disagrees with. The seam
is checked the way it is meant to be used: swapping the registered function makes BOTH the design
validation and the restated row change, which is exactly what two modules carrying one quotient each
could not do.

That declaration is now checked in BOTH directions. It was checked against the DESIGN only -- the
arity, the source forms, and the stated operands a design must supply -- and nothing checked it
against the FUNCTION registered beside it, which left three defects invisible until a study adopted
the arithmetic: a body reading a stated field it did not declare raised a `KeyError` inside whichever
reader got there first rather than refusing the design that failed to state it; a declaration listing
an input the body never reads made every adopting design carry a number for nothing; and an operation
no registered design names was arithmetic nobody exercises, where a wrong quotient would sit until
the first study published out of it. `operations/audit.py` closes the loop in
CI, on the act of REGISTERING rather than at the first adoption.

It closes it by CALLING each operation, not by reading its source.
`operations/probe.py` builds inputs that answer exactly the declaration and
nothing else -- a stated mapping holding only the declared fields, a source tuple of only the
declared length, and a measurement only when `reads_own_measurement` says so -- and every input
records the operation looking at it (the sources and the measurement are `float` subclasses recording
through the arithmetic performed on them, so `int(guard)` and `float(peak)` are the observations).
Reaching outside that raises with the input NAMED instead of the `KeyError` or `TypeError` two frames
deeper; a declared input nothing reached for is refused as over-declaration. One distinction carries
the second refusal: membership is not a read, because `apply` asks `name not in stated` for every
declared field before computing anything, and counting that would mark every declaration read.
Calling at all needs a point to call at, so a registered operation carries `probes` -- a required
field, since an operation the self-check cannot call is one whose declaration nothing checks against
its body -- and a point that does not answer exactly what the operation declares is refused at
construction (named by position), because answering MORE is what would hide a read the declaration
lacks.

`probes` is a SET of points rather than one point, because a body reads its declaration along a
PATH. One call observes one path, so a declared input a body reaches for only on a branch is unread
at a point that misses the branch -- refused as over-declaration, a FALSE refusal -- and, in the
direction that actually matters, a body reaching OUTSIDE its declaration on a missed branch is not
seen at all. With one point per operation the author picked the point, so the check could be
satisfied by picking an easy one. The self-check now calls at every point of the set into ONE
recording: the over-declaration refusal is read off the UNION of the reads (an input read at any
point is read, and the refusal says "never reads it at any of its N points"), while the first point
that reaches outside the declaration ends the walk and is named at its position. A set whose points
are equal in every input the operation declares is refused at construction -- two such points hand
the body the same numbers and take the same path -- so exercising a second branch is a DECLARATION
an author writes beside the arithmetic, not a review comment somebody has to think to make. The
shipped `trigger_over_own_cap_peak` declares two points (a half share over a 1024-char guard, a
quarter share over 4096) that differ in every declared input.

The audit also measures the residual the set used to leave silent.
`operations/branches.py` reserves an available `sys.monitoring` tool ID for
each probe call, enables the `BRANCH` event only on the operation's own compute code object, and
unions the source/destination arcs reached across the declared set. `OperationRegistryReport`
carries one `OperationUnreachedBranches` record beside each entry in `checked`: an explicit missed
arc count and one source line per missed arc (with a repeated line when two arcs from that line were
missed). A body with no missed arc still gets a zero-count record, so adding an unprobed branch
changes visible evidence rather than leaving the certification silently narrower. This is REPORT
only. The shipped trigger operation currently exposes two legal-domain misses -- its absent
measurement fallback and its non-positive peak raise -- so refusing on a nonzero count would reject
an operation whose successful probes cannot take those paths. Generating covering points is also out
of scope: that is a solver, while the probe set remains the readable declaration.

Shipped policy fields are a fourth operation input, separate from sources, stated row fields, and a
value's own measurement. `DerivationOperation.policy_fields` names the `ContextPolicy` attributes
the arithmetic reads, and production `apply` supplies the live policy. The registry self-check in
`operations/policy.py` re-runs every probe after perturbing each of the six
auditable fields with the same plausible neighbour the policy interaction audit uses. A field the
operation declares must change its `DerivedValue` at some point; a field it does not declare must
change it at none. Both a missing dependency and an unused one are refusals, so the pin gate can use
the declaration without widening or narrowing the re-derivation scope silently. The shipped
`trigger_over_own_cap_peak` declares `compact_share` and reads it from the supplied policy while the
design row remains the statement of the measured crossover. Arbitrary globals outside the shipped
policy are still out of scope; the check makes no general purity claim.

The third defect is a walk of the registered DESIGNS rather than of the operations, so a
`PublishedValueDesign` entry now carries a reader for its published values beside its citations and
its validation: which registered arithmetic anything actually names is a question no per-design
validation can answer, and a reader-less entry would answer it as "this design publishes nothing".
`report_operation_registry` collects and `validate_operation_registry` refuses over it -- the same
collecting-primitive shape the design registry uses -- and it returns the operations it exercised,
because a self-check that exercised nothing passes exactly like one that exercised every entry. What
the probe deliberately does not claim is purity: it records reaches through the inputs it was handed,
and an operation reading a module global is out of reach of anything short of the expression language
the design file does not have.

## The registered reading beside the arithmetic

The re-derived number and the rule for JUDGING it are now two declarations. Each portable ratio in
the committed design names both:

```json
"operation": "trigger_over_own_cap_peak",
"reading": "rounded_extent_and_membership"
```

`published_value/readings.py` registers the comparison over the design's published
statement. A bound reading validates and normalizes the statement fields, gives equivalent rows a
stable grouping identity, checks whether the original re-derived values resolve that statement,
and checks whether one restated value still supports it. Both checks return `holds` plus the phrase
the operator report reads. Design validation and `_portable_ratio_row` now call those two sides of
the SAME bound reading; neither rounds or compares a ratio itself. The committed rounded-band rule
therefore owns both semantics: its published edges must be the rounded minimum and maximum across
the original depths, and a restated value holds when its rounded reading is within those edges.

The registry also carries a point with an absolute tolerance and a one-sided upper bound. Fixture
coverage changes the portable-ratio declaration to the point rule and exercises BOTH production
readers without editing either, proving that the rule rather than the form supplies the verdict.
A missing reading, an unregistered name, malformed statement fields, and arithmetic declared with
no reading are refused before evidence is read. The fold-step criterion remains separate: it judges
a measured guard on the deterministic ladder, not a derived value against a published statement.

## Field pointers, forms, and the restatement rule

The field pointer is a dotted path plus a row selector, because these aggregates key their per-depth
rows by a field rather than by position -- `depth_surface[depth=6].crossover_max_prompt_chars`,
`depth_ladders[depth=10].boundary.guard_boundary_chars`, `cap_peak_prompt_chars.6`. One walk serves
both sources because both are the same bytes.

The three forms resolve as differently as they place. An interpolated guard IS a field of the
surface's per-depth row; a fold-step boundary IS a field of the fold-step study's ladder, which the
placement rule already pins from the geometry, so resolving it additionally ties it to what the run
RECORDED -- drift between the two is exactly what neither check sees alone. The portable ratio is
the interesting one: no aggregate anywhere states it. What the collapse measured is the cap PEAK the
trigger is read against, so the published band is RE-DERIVED -- the design's named
`trigger_over_own_cap_peak` operation (the runtime's own `compaction_trigger_chars`) on the resolved
surface guard, over the collapse's resolved peak. The declared `rounded_extent_and_membership`
reading rounds those values to `band_decimals` and requires the band's two edges to be their minimum
and maximum. The same bound reading judges the restated value, so a published edge and a restated
ratio cannot differ on round-then-compare versus compare-then-round. The committed `0.85-0.92x` is
the pair `(0.845x, 0.918x)` resolves to rather than a pair of hand-copied edges; a band stated wider,
narrower, or at a precision the quotients do not reach is refused with the quotients named.

The invariance criterion for a guard is the fold step, not a char tolerance. The fold-step study
established that the cost changes only at a step boundary, so a restated guard that moves INSIDE one
step's guard interval names a point at which nothing changes; only a guard that crosses a step
boundary withdraws a published number. `--audit-only` reports the audit and stops, which is the
GPU-free way to ask "does this bound change invalidate my evidence" before spending anything.

One rule decides where each number in a restated row comes from: whatever the re-measured geometry
can measure is measured FROM that geometry, and whatever only the published artifact holds is stated
as its own comparison row rather than divided into a measured one. The guard RATIO is where that
matters, because a ratio is an interpolated guard OVER the depth's cap peak. The restatement reads
the peak off the same prompt sequence the restated fold step is read from (`measured_cap_peak` over
`cap_prompt_sequence`), not out of the published aggregate: dividing a fresh guard by a published
peak silently rescales the ratio whenever the task world moved -- a changed `pad_chars`, observation
cap, or step margin -- and the fold-step check catches that only when the same drift also reshapes
the step ladder, which a small move need not do. The published peak is not discarded. It is stated
per depth as its OWN restatement row (published peak, re-measured peak, delta, and the ratio the
re-measured geometry supports), reading
`the_re_measured_geometry_has_the_published_cap_peak`,
`..._has_a_different_cap_peak`, or `the_published_surface_states_no_cap_peak_at_this_depth`. So a
moved task world surfaces as a named moved peak with an operator line telling the reader which ratio
to apply, instead of as a rescaled number nothing in the run mentions. The fold-step invariance
criterion is unchanged by this: a moved peak withdraws no COST, only the ratio's basis. When every
published crossover still holds its fold-step or band statement but at least one re-measured peak
disagrees with the published one, the aggregate reading is
`published_crossovers_hold_under_the_shipped_cap_against_a_moved_peak` rather than the bare hold --
the reason and an operator line name the depths whose ratios were restated against the retired peak,
and `persist_restatement` still writes `objective_score=1.0` because the cost criterion held. A
reader of the headline therefore cannot take "everything holds" as permission to apply a ratio that
rests on a geometry this run no longer measures.

The FORM decides what restates a number; a derived value's declared READING decides what it is
checked against.
An interpolated guard is re-interpolated over the substituted cells and placed back on the step
ladder. A fold-step boundary is a property of that ladder rather than of a measured cost, so the
summarize-input-cap study's re-measurement of its cells confirms it directly. The collapse's
portable ratio is neither, and the difference is what makes it the one form a run could quietly skip:
it is `compact_share * guard` over the depth's cap peak, DERIVED from the surface's interpolated
guard rather than measured on cells of its own, so its own eight bound-invariant cells say nothing
about it -- the guard it divides is exactly the number the restatement moves. It is therefore
restated from that restated guard (the runtime's own `compaction_trigger_chars` applied to it) over
the same depth's re-measured cap peak, both read off ONE restated surface row, and checked against
the statement its registered reading binds rather than against a fold step the form does not name.
For the committed rule that statement is the band and its quoted precision (`published_band`,
`band_decimals`): a restated ratio a hair under the raw lower edge is inside when it rounds to the
published edge, and the same registry function makes that choice in validation and restatement.

A portable ratio now has an explicit unresolved result when that declared source has no BRACKETED
restated surface row. Its crossover row uses the
`source_interpolated_guard_was_not_restated` basis, records the source study, depth, and form, and
sets `invariance_holds` to null; it never falls back to the collapse's
`every_contributing_cell_is_bound_invariant` basis. The aggregate reading is then
`derived_crossovers_were_not_restated`, whose reason and operator line name both the derived ratio
and the exact declared source that was absent. That reading persists `objective_score=0.0`, and the
unresolved row contributes no successful reliability result, so a grid with no bracketed crossing
cannot present an uncomputed quotient as an invariance result. The form, reading, operator, and
manifest regression is in
`tests/llb/bench/memory/test_agentic_memory_crossover_restatement_forms.py`; run the standard validation
with `make test`.

## Implementation map

The module and test inventory moved to
[Published-value implementation map](published-value-implementation.md) so this behavior page
stays readable as the registries grow.

## The result: every published crossover holds

The audit is the result. Of the 22 published cells across the three studies, **18 are
bit-identical under both bounds** and needed no run at all: every depth-6 cell folds a transcript
neither bound trims, and so do most depth-10 cells. Four are bound-sensitive, and three of those are
the depth-10 fold-step cells the summarize-input-cap study had already re-measured. **One cell**
(`surface-d10-g23000`, 302 chars elided) was left, so the whole GPU cost of restating four studies'
worth of routing numbers was 14 episodes.

CUDA host evidence (2026-08-05, RTX 4060 Ti 16 GB): `mistral-small3.1:24b` on Ollama with
`num_ctx=8192`, seven depth-10 memory tasks, `compact_share=0.5`, one re-measured cell at 10.55
tok/s over about 12 minutes including the control. The pinned family re-passed the unchanged
depth-10 control at 4/4; the re-measured cell completed 7/7 under both policies with zero overflows
and one compaction per compact episode.

| study | depth | form | published | restated | fold step | basis |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| boundary surface | 6 | interpolated guard | 14160 | unchanged | 6 | every cell bound-invariant |
| boundary surface | 10 | interpolated guard | 21900 | **21862** | 10 | re-measured cell |
| trigger collapse | 6 | portable ratio | 0.85-0.92x | 0.845x | 6 | derived from the restated guard |
| trigger collapse | 10 | portable ratio | 0.85-0.92x | **0.917x** | 10 | derived from the restated guard |
| fold step | 6 | fold-step boundary | 14912 | unchanged | 6 | every cell bound-invariant |
| fold step | 10 | fold-step boundary | 22016 | unchanged | 10 | already re-measured |

Verdict: **every published crossover holds under the shipped cap**. Exactly one number moves at all
-- the depth-10 interpolated guard, by **-38 chars** (21900 -> 21862, ratio 1.84 -> 1.83) once the
re-measured cell's undiscounted cost (+1610.3 instead of +1524.9 tokens) enters the interpolation --
and it lands at the same place on the ladder, inside fold step 10's guard interval
`[20240, 22016)` where every guard costs the same. The direction is the one the mechanism predicts:
removing a discount that flattered compact pulls the crossing DOWN, toward compact being preferred
over a slightly narrower band of guards.

The re-measured cell also cross-checks the step function across two independent runs on different
days: guard 23000 here costs 28953.3 compact tokens, the identical value guards 22016 and 23040
produced in the summarize-input-cap study. Three guards spanning 1024 chars, one fold step, the same
cost to the token.

The cap peaks those ratios rest on are re-measured on every run rather than read out of the published
aggregate, and on the committed geometry they still ARE the published ones. Four re-runs on
2026-08-07 (same host, same pinned model, same design) agree on every number; the current aggregate,
which is the first run whose six published values were RESOLVED out of their aggregates rather than
read off the design, is the 2026-08-07 run, at 10.27 tok/s over about 14 minutes including the
control. All four
read `the_re_measured_geometry_has_the_published_cap_peak` at both depths with a 0-char delta --
8374 at depth 6, 11926 at depth 10 -- so the 1.69x and 1.83x guard ratios are stated against the
geometry that measured the guards they divide. All four also re-measured `surface-d10-g23000`
from scratch and landed one token away each time: 28952.3 compact tokens for a delta of +1609.3,
interpolating to a 21862.1-char crossover against the 21861.6 above. Read the token-exact
cross-check in the
previous paragraph as a within-run property; ACROSS runs on different days the served model
reproduces it to a token, which rounds to the same 21862 and stays inside the same fold step 10
interval `[20240, 22016)`.

A fifth run on 2026-08-08
(`.../agentic-compact-crossover-restatement/20260808T045022.358884Z-4143107bb981/manifest.json`),
the first taken with the derivation edge DECLARED rather than hardcoded, reproduces all of it: the
same audit (18/22 cells bit-identical, four re-measured), the same 21862-char restated guard, the
same reading, and a `derived_from_study_kind` now read off the design that is the same
`compact_memory_boundary_surface` the constant used to supply. It lands on the 21861.6 reading of the
interpolation, so its depth-10 ratio quotes 0.916x where the 21862.1 reading quotes 0.917x -- the
across-run token variance above, inside the band at the two decimals it is published to either way.

A sixth run on 2026-08-08
(`.../agentic-compact-crossover-restatement/20260808T065453.191859Z-d159a1345fa6/manifest.json`,
11.35 tok/s), the first whose ratios were re-derived through the design's NAMED operation rather than
through a quotient this module carried, reproduces the fifth exactly: the same audit, the same
21861.6-char restated guard, the same 0.845x and 0.916x against unmoved 8374- and 11926-char cap
peaks, the same `published_crossovers_hold_under_the_shipped_cap`. That is the point of the run --
moving the arithmetic behind the registry had to change nothing about the numbers, and the reported
trigger chars (7079 and 10930) now come out of the operation's own named intermediate rather than
from a second call beside it.

The collapse's portable ratio is restated by the run rather than recomputed beside it, which is what
its being DERIVED from the surface's guard demands: at depth 10 the restated 21862-char guard is a
10931-char trigger over the re-measured 11926-char cap peak, so the ratio moves 0.918x -> **0.917x**,
and at depth 6 the unmoved 14160-char guard gives a 7079-char trigger over 8374 for **0.845x**. Both
sit inside the published 0.85-0.92x band at the two decimals it is quoted to, so the band an operator
applies is unchanged. The collapse's own eight cells are all bound-invariant, so the equal-trigger
spreads and the contrast family stand as measured -- but that fact is about the ratio's PORTABILITY,
not about its value, and the run now says the two separately.

The trigger collapse gains something from the change rather than merely surviving it. Its claim is
that `compact_share` and the prompt guard act ONLY through their product, and the retired bound was
the one place where share entered independently (the summarize input was capped at
`compact_share * guard`). Under the shipped bound that term is gone, so the collapse holds by
construction and not only by measurement.
