# Published-Value Implementation Map

This is the code-and-test map for the published-value provenance, derivation, operation, reading,
and crossover-restatement workflow. Read
[Published values under the shipped cap](published-values.md) for behavior, evidence, and results.

## Core and tests

Core locations are `src/llb/bench/agentic_memory_cap_audit.py` (geometry extraction per study shape,
both-bound probe, oracle and worst-case invariance verdicts),
`src/llb/bench/agentic_memory_crossover_restatement_design.py`,
`src/llb/bench/agentic_published_value_pointer.py` (the field-pointer walk, one walk so both sources
read alike, including string `cell_id` row selectors),
`src/llb/bench/agentic_published_value_fixture.py` (the committed aggregates, their
manifest pins, the growth policy, and the refusal to write a manifest that drops a cited copy),
`src/llb/bench/agentic_published_value_registry.py` (the registry of publishing designs, the union
refresh, the collecting and refusing walks over it, and the refresh that reports on what it just
wrote), `src/llb/bench/agentic_published_value_provenance.py` (the
`(artifact, field)` pair and the two-source read),
`src/llb/bench/agentic_published_value_figures.py` (the study/cell join that turns invalidated cells
into the published figures the pin gate must name),
`src/llb/bench/agentic_published_value_collection.py` (the per-value accumulator: collect what did
not resolve, keep what that leaves unjudged apart from it, and refuse once naming both),
`src/llb/bench/agentic_published_value_derivation.py` (one value's `derived_from` + `operation` +
`reading` declaration, read and validated against the arithmetic and comparison it names),
`src/llb/bench/agentic_published_value_derivation_graph.py` (the design-wide walk over those
declarations: the refusals for an unpublished source, a self-reference, a cycle, and a duplicated
identity, plus the transitive walk from a value to the moved measurements at the root of what it
rests on),
`src/llb/bench/agentic_published_value_operations.py` (the registry of re-derivations: what each
operation is computed over, the pure function that does it, the named intermediates it exposes, and
the refusal of an operation nothing registered -- all eight study-agnostic, so any published agentic
number can adopt them),
`src/llb/bench/agentic_published_value_readings.py` (the registered rounded-band,
point-with-tolerance, and one-sided-bound comparisons; statement validation; resolution and
restatement verdicts; and operator phrases),
`src/llb/bench/agentic_published_value_operation_probe.py` (the declared-input read recorder),
`src/llb/bench/agentic_published_value_operation_branches.py` (operation-local `sys.monitoring`
branch arcs and the per-operation missed-branch record),
`src/llb/bench/agentic_published_value_operation_policy.py` (the shipped-policy perturbation check),
`src/llb/bench/agentic_published_value_operation_audit.py` (the registry report and refusing
wrapper),
`src/llb/bench/agentic_memory_crossover_restatement_provenance.py` (what each published FORM
resolves to, including the re-derived band, its four passes, and the cause-versus-consequence rule
for a band whose declared source guard moved),
`src/llb/bench/agentic_memory_crossover_restatement_placement.py` (the study's prompt sequence and
the per-form annotation rules, shared by design validation and the restatement),
`src/llb/bench/agentic_memory_crossover_restatement_reading.py`,
`src/llb/bench/agentic_memory_crossover_restatement_rows.py` (substitute the re-measured cells,
re-interpolate against a re-measured cap peak, and compare that peak with the published one),
`src/llb/bench/agentic_memory_crossover_restatement_forms.py` (one restated row per published form:
place a restated guard on the step ladder, confirm a fold-step boundary, derive the portable ratio
and apply its declared published-value reading),
`src/llb/bench/agentic_memory_crossover_restatement.py`,
`src/llb/bench/agentic_memory_crossover_restatement_report.py`,
`src/llb/cli/bench/category_agentic_memory_crossover_restatement.py`,
`tests/llb/bench/test_agentic_memory_crossover_restatement.py`,
`tests/llb/bench/test_agentic_memory_crossover_restatement_forms.py` (each form's row rule at its
edges, including a ratio driven out of its published band and an unresolved ratio whose declared
source guard was not restated, with its operator reading and persisted failing metric), and
`tests/llb/bench/test_agentic_memory_crossover_restatement_placement.py` (every committed annotation
placed on its own ladder, plus each way one can be wrong),
`tests/llb/bench/test_agentic_published_value_pointer.py` (the pointer walk, on synthetic aggregates
so a failure names the pointer rather than the study that used it, including cell_id selectors whose
values embed dots),
`tests/llb/bench/test_agentic_published_value_figures.py` (cell_ids checked against the committed
aggregate, an absent artifact refused, and invalidated cells retiring their figures plus derived
consequences),
`tests/llb/bench/test_agentic_published_value_derivation.py` (one value's declaration, on synthetic
published values for the same reason: the form as part of the identity, a malformed entry, and every
way the declaration can fail to agree -- an unregistered operation, sources with no operation, an
operation with no sources, missing or unregistered readings, a declaration that is not the shape
its operation takes, and a stated operand the design does not state),
`tests/llb/bench/test_agentic_published_value_readings.py` (shared rounding, point and bound
comparisons, plus missing and unregistered reading refusals),
`tests/llb/bench/test_agentic_published_value_derivation_graph.py` (the design-wide walk and the
consequence marking: a two-step chain naming only the measurement at its root, a figure derived from
two moved measurements naming both, and each way a declaration can be unsupportable -- an unpublished
source, a self-reference, a cycle, and a duplicated identity),
`tests/llb/bench/test_agentic_published_value_operations.py` (the registered trigger arithmetic and
its named intermediate, the refusals for an unregistered operation and for inputs an operation did
not declare, and the proof that ONE registered function serves both readers -- swapping it moves the
band design validation re-derives and the ratio the restated row reports, plus the non-band fixture
that changes only the declaration and makes both production readers inherit a point-with-tolerance
verdict),
`tests/llb/bench/test_agentic_published_value_operation_audit.py` (the registry self-check: a body
reading a stated field, a measurement, or a source it did not declare, a declaration listing an
input the body never reads, an operation that does not compute at its own probe point, the
membership-is-not-a-read distinction the over-declaration refusal rests on, a read that happens on
ONE branch driven in each direction -- at a probe set that misses the branch and at one that takes
it -- a report-only missed branch with its count and source line, the zero-count result when the set
takes both outcomes, the missing and unused shipped-policy dependency refusals, the probe-set
refusals for no point at all, for two points that cannot differ, and for a point that does not answer
the declaration, plus arithmetic no registered design names and the CI gate over the shipped
registry),
`tests/llb/bench/test_agentic_published_value_provenance.py` (the committed copy and its pin, the
refusals for a pin with no bytes behind it or bytes that digest to something else -- both on a host
with no run at all -- the prune and the size caps, and the two-source read including an artifact
that is not the pinned file even where the cited value agrees),
`tests/llb/bench/test_agentic_published_value_registry.py` (the union over two registered designs,
an aggregate both cite carried once, and each of the three refusals -- a partial host, an
unreadable registered design, and a write that would drop a still-cited copy -- plus the prune that
a design retired from the registry still gets, the validation walk over every registered design
including what it is handed and what it returns, a registered design whose values do not resolve,
an entry that registers no validation, and the CI assertion over the shipped registry),
`tests/llb/bench/test_agentic_published_evidence_refresh.py` (the collecting walk beside the
refusing one, the refresh that names what it committed and does not roll it back, the real design
against a host whose boundary-surface run moved the depth-10 guard, and the command's exit code on
both answers), and
`tests/llb/bench/test_agentic_memory_crossover_restatement_provenance.py` (all six committed values
resolved out of the committed aggregates, the committed design's own derivation declarations, a
transcription slip in each form, the re-derived band,
the committed bytes checked against this host's own run artifacts, the growth budget, the no-op
regeneration on a host that still has the runs, and the collecting refusal -- a re-run that moved
three values named in one refusal in design order, a moved guard named as the cause with its derived
band left unjudged rather than named twice, and a malformed crossover refused ahead of any read even
when the evidence moved too).

```bash
make bench-agentic-context-compact-crossover-restatement
make bench-agentic-context-compact-crossover-restatement AGENT_CONTEXT_COMPACT_CROSSOVER_AUDIT_ONLY=1
make bench-agentic-published-provenance
```
