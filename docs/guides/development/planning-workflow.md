# Planning Workflow

How work enters, moves through, and leaves the forward plan. The rules here are the long form of the
Start Work and Finish Work checklists in [AGENTS.md](../../../AGENTS.md); `make lint-spec-plan`
enforces the parts a parser can see, and this page is the authority for the rest.

## Which document owns what

| Question | Source of truth |
| --- | --- |
| What should the product do, and how is each capability evaluated? | [Specification](../../design/spec.md) and its [capability registry](../../design/spec.md#capability-registry) |
| What work remains, and in what order? | [Forward plan](../../impl/plan.md) |
| What exists, where, how to run it, and what result settled it? | [Current implementation](../../impl/current.md) |
| What was a finished task's contract, and what evidence accepted it? | [Task records](../../impl/records/README.md) |
| Which research questions closed negative, and what would reopen them? | [Future research](../../impl/future-research.md) |
| How is the work performed? | This guide and [AGENTS.md](../../../AGENTS.md) |

## Choose the work contract

- **A named plan task.** Read the task, the spec section its `Serves` line links, and the
  current-docs topic it builds on. Confirm every id in its `Requires` has left the plan and every
  input in its `Dependencies` holds. Then open its [task record](#task-records) BEFORE editing
  anything else.
- **A fix or maintenance request.** The request plus the existing spec are the contract. Open a
  record only when findings, tradeoffs, or verification evidence need a durable handoff.
- **A new capability, or a change to a product promise.** Run the six steps of
  [Extending this specification](../../design/spec.md#extending-this-specification) first. Do steps
  2-4 before the code when the capability is known up front, and in the same change when the work
  taught you it was needed -- never in a later ticket. When an existing capability is the wrong
  shape, amend its spec section rather than working around it: a spec that no longer describes the
  code is worse than none, because people trust it.

Review findings against the contract. Fix what is in scope, attach concrete follow-up to the open
task that owns it (or write one), and send a new capability through the spec. An unfixed bug that
becomes a task states the observed behavior, its impact, and the check that will show it fixed.

## Is it a task at all?

Four kinds of finding look like forward work and are not:

- A **chore**: a source file over the soft line limit, or a run artifact already inside its size
  bound that could still be smaller. Do it inline while editing the file, or leave it. Written up
  with acceptance gates, a chore competes with the product on equal footing and each round produces
  the next round.
- An **audit of our own output**: an artifact's field layout, an artifact's byte count. The measure
  of an audit is a decision it changes, not a byte it saves. (The
  [closing implementation audit](#closing-implementation-audit) is not this: it measures a
  capability against its spec, and every finding it makes changes code, a test, the plan, or the
  spec.)
- A **migration, compatibility shim, or second schema version** for a form only our own code has
  written. Change the form in place and rebuild the data; see the
  [pre-release rule](../../impl/plan.md#pre-release-rule-no-versioning-no-backward-compatibility).
- A **new domain capability** the spec does not describe. This one IS worth doing, and it is not a
  plan task yet: the spec lifecycle comes first. That order is what makes a discovery durable rather
  than a private detour.

Extension is welcome, and finishing a task while adding nothing is equally normal. What is not
normal is a task quietly appearing under a capability nobody amended the spec for.

## Task lanes and statuses

The plan is split by **who must act to complete the task**. Both lanes carry the same capability
groups in the same order, so they run in parallel: an agent works its lane while a human-gated
task waits on its reviewer.

- **Agent Implementation Tasks** -- an agent can reach acceptance on its own.
  - `CLEAR`: `make ci` green with committed fixtures, injected fakes, and deterministic harnesses;
    no run evidence, no human gate.
  - `RUN NEEDED`: agent-buildable, but acceptance needs a heavy deterministic run. Every dev box is a
    CUDA host, so the agent executes the run itself; choose the model by
    [heavy runs and evidence](heavy-runs-and-evidence.md).
- **Human-Assisted Tasks** -- acceptance needs a person or an authority an agent does not have.
  - `BLOCKED BY HUMAN`: the acceptance gate consumes an artifact only a human step can produce.
  - `HUMAN-GATED`: the deliverable itself is human judgment, authorization, private access, or
    spend approval. Supporting code and unit tests are still agent-buildable.

A status may carry its reason after two hyphens: `CLEAR -- <why no human or run is needed>`. The
gate reads only the marker before them, and refuses a marker the lane does not allow.

`Research: yes` is a separate field, not a status. It marks a task whose path is unknown in advance,
where a well-supported negative result is a valid outcome that must be recorded rather than worked
around. A research task still declares its scope and its acceptance gates, and when it closes
negative it goes through [closed research](#closed-research).

## Requires and Dependencies

The two prerequisite fields answer different questions, so they are kept apart:

- **`Requires`** (optional) -- open tasks, in either lane, that must leave the plan before this one
  can START. Backticked ids, comma-separated, nothing else on the line. The gate refuses an unknown
  id, a cycle, and a required task that depends on an `(optional)` one.

  ```markdown
  - Requires: `dual-control-suite-seam`, `frontier-judge-authorization`
  ```

- **`Dependencies`** (required) -- everything else the task needs: a current-docs fact it builds on
  (one line and a link, never a restatement), an external input, and for a human-lane task the
  specific human step that gates completion and what an agent can prepare before it. Write `none.`
  when there is nothing; a `none.` may be followed by the reuse pointers an implementer needs.

`make plan-status` skips every task whose `Requires` still names an open task. It cannot read
`Dependencies`, so check that field by hand before starting the task it names.

## Task shape

A task id is a stable lowercase slug that describes the work (`platform-matrix-power`,
`prompt-system-tuning`) and is kept only while work remains under it. File the task under the `###`
group of the capability it serves, in the lane its status belongs to. A heading may end with
`(optional)`.

```markdown
### Capability title -- `capability-id`

#### stable-task-id

The unresolved operator problem, in present or future tense.

- Serves: `capability-id` -- [Spec section](../design/spec.md#section)
- Agent status: CLEAR
- Research: yes
- Requires: `other-open-task`
- Dependencies: none.
- User-visible outcome: what becomes possible or trustworthy.
- Scope boundary: in scope, explicitly out of scope, and the modules or commands to reuse.
- Data and artifact paths: corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs, and
  committed `samples/` outputs.
- Execution path: commands, manual steps, local services, and the heavy steps kept outside quick CI.
- Acceptance gates: tests, lint and type checks, thresholds, the comparison method, the manual
  evidence required, and what a negative result looks like.
- Documentation target: the narrow `docs/impl/current/` topic and any guide that receives the
  result.
```

`Research` and `Requires` are optional; every other field is required and non-empty. A value may
wrap onto indented continuation lines. Extra fields (`Command`, `Human step`) are allowed and
ignored by the gate.

## Ordering

Capability groups follow the registry's row order in both lanes; the four rules that set that order
are in the [capability registry](../../design/spec.md#capability-registry). Within a group:

1. a task another task names in `Requires` comes first;
2. then work that changes what later tasks measure -- a chunker, an encoder roster, or a detector
   change invalidates every reading taken against the old one;
3. then required before `(optional)`;
4. then cheap before expensive: a `CLEAR` task costs a fixture test, a `RUN NEEDED` task costs a GPU
   run, so a run of `CLEAR` work lands as one pass rather than interleaved.

**`(optional)` is binding within a group.** It marks work already judged not to be the next thing
worth doing for that capability, so it sorts behind every required task of its OWN group. It does
NOT reorder the line: an optional task of an earlier capability still precedes a required task of a
later one, because the trust chain is what the order encodes. A capability whose remaining work is
entirely optional moves DOWN the registry -- a shipped capability with only refinements left is not
what blocks the product.

The scheduling rule follows: **take the first task of the earliest group that still has one,
skipping a task still waiting on its `Requires`.** `make plan-status` prints exactly that, per lane,
with the task and status counts.

## Closing implementation audit

Each task proves its own slice on its own fixtures, and the deep dive into one task is where its
neighbours' assumptions first surface: a property two tasks each assumed the other owned, a seam
whose fixture agreed with itself but not with the next stage, an identity or cache key one task
derived and another rebuilt. No task's own gates can see those, so every capability with open work
in the plan ends with one task whose whole job is to look.

- **Shape.** One `#### <capability-id>-implementation-audit (optional)` per capability, in the
  agent lane -- an agent performs the audit even when the capability's other work is human-gated --
  as the LAST task of that capability's agent-lane group (a capability whose tasks are all
  human-gated gets an agent-lane group holding only its audit). Its status is `CLEAR`: it works on
  committed fixtures and injected fakes.
- **Start.** Its `Requires` names every open required task of the capability, in either lane, so it
  starts once the required line has landed. Optional tasks are not required: the audit covers the
  ones that have left the plan by then, and its record names the ones still open.
- **Four passes,** each against the capability's spec section and registry evaluation rather than
  against the tasks' own gates:
  1. design completeness -- every load-bearing property and every evaluation clause maps to the test
     that pins it, or to a gap that is then routed;
  2. logic review across task seams -- the seams the audit task names, plus any found on the way;
  3. implementation improvement -- logic duplicated between the tasks' modules, a missing shared
     helper, a module past the soft limit along a seam the work created;
  4. system integration tests -- for each seam no test pins, a CI test that drives the capability
     across it on fixtures, shown once in the record to FAIL with the seam broken on purpose.
- **Routing** is the completion transition's: fix it in place, write a task in this group, or take
  it through the spec lifecycle. A finding that needs a heavy run becomes a `RUN NEEDED` task rather
  than widening the audit. An audit that finds nothing still records its property-to-test map: that
  map is the evidence it looked.
- **Upkeep.** Add the audit when a capability first gets open tasks. A new required task of the
  capability joins the audit's `Requires`; a finished one leaves it like any downstream `Requires`.
  The audit leaves the plan when it closes, and a capability that later receives new required work
  gets a new audit with it.

## Task records

A [task record](../../impl/records/README.md) preserves a task's work contract and its acceptance
evidence after the task leaves the plan, where its heading and id would otherwise disappear.

- **Planned work always gets one.** Copy the [template](../../impl/records/template.md) to
  `docs/impl/records/NNNN-<group>-<task-id>.md` before editing, and paste the original task
  verbatim into it. Explain any later scope change in the record rather than editing the pasted
  contract.
- **A direct request gets one when it needs a durable handoff,** named `NNNN-adhoc-<slug>.md`.
- **A task spanning sessions keeps its record open,** naming the remaining gate and the next action.

What goes where: the record holds the contract, the commands that ran and their outcomes, each
gate's result, the routing of what the work surfaced, and the before and after task counts. The
narrowest current-docs topic holds the delivered behavior and the measured result that settles a
decision. Both follow [citing a measured result](heavy-runs-and-evidence.md): no host-local run
path, no bare run label. Link a record from its topic page when it explains a durable decision.

### Record file naming

Name a record `NNNN-<group>-<task-id>.md`:

- `NNNN` is the next unused four-digit sequence in the
  [records index](../../impl/records/README.md), so the directory lists records in the order work
  started on them. Numbers run from `0001` without a gap and are never reused; a record is never
  deleted or renumbered.
- `<group>` is the capability id the task serves -- its `Serves` line -- or `adhoc` for work that
  did not come from the plan.
- `<task-id>` is the plan task id, unchanged; an `adhoc` record uses a short descriptive slug.

Add the record's row (sequence, task, record link, one-line summary) to the index table in the same
change. `make lint-doc-links` runs `llb.quality.record_names`, which refuses a record whose name
breaks this shape, whose group is neither `adhoc` nor a registered capability, whose sequence
repeats or leaves a gap, or which the index does not link.

## Completion transition

Before reporting a task done:

1. **Count first.** Capture `make plan-status` before editing the plan.
2. **Accept it.** Run the declared acceptance gates and record each command, its result, and any
   gate still outstanding in the task record. A declared research negative satisfies acceptance
   when the evidence supports it.
3. **Record it** in the narrowest current-docs topic: what was built, where it lives (modules,
   commands, tests), how to run it, and the result. A new topic page gets its row on the area page
   in the same change.
4. **Delete it from the plan** -- the whole item, unless open residual work remains under it.
   Remove its id from every downstream `Requires`, and where useful link the new current fact from
   those tasks' `Dependencies`. A capability that became available flips to `shipped` with its
   implementation link.
5. **Route what the work surfaced** to exactly one place: a chore is done now or dropped; an audit
   of our own output is dropped; more work under an existing capability becomes a task in that
   group, flagged `(optional)` if it is; a capability the spec does not describe goes through the
   spec lifecycle, never straight to a task. Surfacing nothing is a normal, good outcome.
6. **Close research.** A `Research: yes` task that came back negative also gets its
   [closed research](#closed-research) section.
7. **Run the gates** on the resulting tree: `make ci`, plus `make lint-md` when docs changed.
   Capture the final `make plan-status` and close the record when everything passes.
8. **Self-check and report.** The diff net-removes the scope from the plan and adds it to the
   current docs. Report the counts before and after and which capabilities moved. Growth is allowed
   -- that is what extensibility means -- but it is reported, never silent, and growth concentrated
   in the capability you happened to be working in is the signal to stop and ask whether the line
   still reflects what matters.

## Closed research

A research question that came back negative leaves the plan like any finished task, but it is not
simply deleted: [future research](../../impl/future-research.md) keeps what a later reader needs
before spending the same effort again. Each section there:

- names the task id it carried in the plan and states the question;
- states the answer, what was measured, and what closed it, linking the current-docs evidence rather
  than restating it;
- states the **reopening condition** as something that must be TRUE -- a number or a change in the
  world, never "try harder";
- lists the directions already proven dead, so a reopening does not re-run them.

When a reopening path needs an authorization or an input nobody has yet, it may stay in the plan as
a gated task, and the section names it.

## What the gate checks

`make lint-spec-plan` (inside `make ci-checks`) fails on any of these, and the fix is always to the
DOCUMENTS, never to the checker:

- a task that serves no registered capability, or sits under another capability's group;
- a `shipped` capability with no implementation link, a `planned` one with no open task, or a
  capability with no evaluation;
- groups out of registry order in either lane, including a return to an earlier group;
- a required task after an `(optional)` one in the same group;
- a duplicated task id, a malformed task heading, or a task outside both lanes;
- a missing or empty required field, or a status the task's lane does not allow;
- a `Requires` entry that is malformed, names no open task, forms a cycle, or makes required work
  wait on optional work;
- `done`, `delivered`, `implemented`, or an ISO date anywhere in the plan, or a missing lane.

The checker establishes structural agreement only. Whether a capability is worth having, whether an
acceptance gate is strong enough, and whether an external prerequisite holds stay contributor
judgment.
