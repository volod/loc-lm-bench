# Ontology-Validated Answer Gate

Shipped: a two-step gate over a RAG answer. Step one is the typed contract -- the completion either
parses into `AnswerEnvelope` or ends in a typed status ([generation graph and
scoring](scoring.md#typed-rag-answer-envelope-typed-rag-answer-envelope)). Step two, described
here, asks the question step one cannot: is the declared answer semantically POSSIBLE? The
envelope's asserted triples are checked against the accepted axiom set and against the corpus
ledger the retrieved context came from, so an answer that violates a functional property, a
`domain`/`range` constraint, or a disjointness pair -- or that contradicts a ledger fact whose
evidence is IN the retrieved chunks -- ends as `ontology_violation` or takes one bounded repair
instead of being scored as a fluent answer.

This is the step no existing signal covers. Groundedness asks whether the answer's tokens appear in
a chunk, which a semantically impossible answer assembled out of real chunk tokens passes cleanly.

Off by default, and refused rather than defaulted when its inputs are incomplete.

## Running it

```bash
make compare-answer-validation \
  CONFIG=<run-config.yaml> SPLIT=final MAX_TOKENS=768 \
  VALIDATION_LANES=off,pydantic,pydantic+ontology \
  AXIOMS=<signed-axioms.ttl> ONTOLOGY_LEDGER=<extraction.jsonl>

make check-answer-gate            # the committed adversarial fixture: catch + false rejection
```

A single gated run is an ordinary `run-eval`:

```bash
make run-eval ANSWER_FORMAT=envelope MAX_TOKENS=768 \
  ANSWER_VALIDATION=ontology ONTOLOGY_AXIOMS=<signed.ttl> ONTOLOGY_LEDGER=<extraction.jsonl>
```

Three lanes, and the first is the baseline:

| Lane | `answer_format` | `answer_validation` | What it measures |
| --- | --- | --- | --- |
| `off` | `free_text` | `off` | the shipped prose path, unchanged -- it sets no new knob at all, which is what lets it reproduce a recorded run bundle rather than merely resemble one |
| `pydantic` | `envelope` | `off` | step one alone: the declared contract and its formatting repair |
| `pydantic+ontology` | `envelope` | `ontology` | both steps: the declared contract plus the signed-axiom gate |

## What the gate checks, and against what

`src/llb/eval/answer_validation/` adds no second checker. It renders the envelope as the same
`DocExtraction` a corpus document produces (`answer_ledger.py`, under the synthetic doc id
`answer://declared`), merges it with the scoped corpus ledger, and runs the SHIPPED axiom checker
over the merge ([robustness and
ontology](../robustness-ontology-backends.md#ontology-axiom-layer)). One pass answers both halves of
the question:

- a violation whose facts are all the answer's is a self-contradiction -- a claim outside a
  relation's declared range, one name asserted as two disjoint types, a relation asserted of itself;
- a violation mixing an answer fact with a corpus fact is a contradiction of the LEDGER.

A violation the answer is not responsible for is dropped, and responsibility is decided by running
each axiom TWICE -- once over the scoped corpus alone, once over the corpus plus the answer -- and
subtracting every `(axiom, subject)` anchor the corpus already broke. Citing an answer fact is not
enough:

- a bound of two objects that the retrieved chunks already break with three of their own is broken
  whatever the answer says;
- a subject the corpus already gives two conflicting values leaves no single fact for the answer to
  contradict.

In both cases the corpus is contradicting itself -- a data problem `validate-ontology-axioms`
reports -- and refusing the answer for it would blame the model for the ledger. The subtraction
NARROWS the gate rather than disabling a class: the same axiom over a corpus that was within its
bound still refuses the answer that pushes it over. It was added because a real run refused a
correct answer this way (below), and the case is now in the fixture.

**Scope is the retrieved chunks.** `scope.py` keeps only the ledger facts and entity mentions whose
evidence span overlaps a chunk the prompt carried. A fact the model never saw cannot be a
contradiction it committed, and refusing on one would make the gate a corpus-wide fact checker.
Overlap rather than containment: a chunk boundary that cuts an evidence span still put the fact in
front of the model.

**Eight of the nine axiom classes may refuse an answer.** `symmetric` is excluded for a stated
reason: at the ledger it reports the GAP a missing counterpart is, and an answer is never asked to
state both directions of a symmetric relation, so enabling it here would refuse correct one-way
answers by construction.

## The three defenses against refusing correct work

Each is measured by an adversarial fixture case rather than asserted.

- **Only SIGNED axioms are enabled.** A relation nobody declared functional admits many values, so
  a legitimately multi-valued answer is ordinary rather than a violation.
- **Endpoint surfaces fold through the corpus's own alias map.** The extraction ledger records an
  entity's aliases, so a paraphrase the corpus already treats as one node does not read as a second
  value of a functional relation. An alias two different entities both claim is dropped rather than
  resolved -- collapsing an ambiguous surface would invent an identity no reviewer accepted.
- **`MISC` asserts nothing, on BOTH sides.** `MISC` is what `normalize_entity_type` collapses
  anything out-of-vocabulary into, so it names the absence of a recognized type. The answer side
  drops it and so does the scoped corpus side, which is what keeps an extractor's fallback from
  refusing an answer under a `domain`/`range` axiom.

## The bounded repair, and what it may not do

Step two extends the SAME generation boundary (`llb.eval.answer_envelope.boundary`), never a second
one. Two repair budgets, at most one reprompt each, kept apart because the two failures call for
different fixes:

- `repaired` -- the first completion did not satisfy the CONTRACT. Unchanged, so first-attempt
  conformance still reads as `1 - repair_rate`.
- `validation_repaired` -- the envelope parsed but broke an accepted axiom, and the semantic
  reprompt named the broken constraints.

The semantic retry replaces the answer ONLY if it both parses and passes. A retry that stops parsing
or still violates leaves the first envelope standing and the case ends `ontology_violation` on the
answer the model actually gave -- so the repair can rescue an answer and can never damage one. Its
tokens are charged either way: the round trip cost what it cost.

The refused answer's text is still recorded. A rejection nobody can inspect is not evidence.

## The refusals

Every one of them fires at setup, before any model call, so a run can never spend a GPU hour to
discover its gate was never enabled:

| Condition | What happens |
| --- | --- |
| `answer_validation=ontology` with `answer_format=free_text` | refused: the gate reads declared triples, which prose does not carry |
| no `ontology_axioms` or no `ontology_ledger` | refused, by name |
| an axiom file no reviewer signed | refused: "none of its N axioms is signed" |
| a signed file of only excluded classes | refused: it would gate nothing while looking enabled |
| `ontology_axioms` / `ontology_ledger` set with the gate off | refused, so a config cannot look gated and score ungated |

Signing is a domain reviewer's decision, not an agent's or a test's
([product decisions](../scope-boundaries.md#the-answer-gate-trust-boundary)). The committed
candidate set at `samples/ontology/axioms_uk_v1.ttl` carries no signature, so pointing the gate at
it is refused.

## What is recorded

Per case (`scores.jsonl`), present only when the gate ran: `validation_checked_triples`,
`validation_violations`, `validation_classes`, `validation_axioms`, `validation_repaired`. Run
metrics add `ontology_violation_rate`, `validation_checked_rate`, and `validation_repair_rate`,
echoed on the run's `answer-side:` line. Every one of them is journaled, so a RESUMED gated case
re-scores to the same row.

`validation_checked_triples` is the population a verdict rests on. An envelope that declared no
triple was UNCHECKED, not cleared -- reporting the two as the same "passed" would let a model buy a
clean gate by declining to type its claims, which is why `validation_checked_rate` sits beside the
refusal rate in every artifact.

## Reading the comparison

`compare-answer-validation` writes `report.md` + `comparison.json` under
`$DATA_DIR/answer-validation/<run>/`. Its shape is the honest reading of a validator, not a
flattering one:

- **CATCH and FALSE REJECTION are separate columns, per axiom class.** A refused answer the
  reference scores correct is a false rejection whatever else the gate got right. Correctness is
  read from `contains`, the found-rate signal, not from the token-F1 objective -- that objective
  mixes needle-finding with terseness ([generation graph and
  scoring](scoring.md#measured-the-headline-objective-is-partly-a-verbosity-ranking)) and would
  price a verbose but correct refused answer as a wrong one.
- **The objective delta is read only on the COMMONLY ANSWERED items** -- the items every lane ended
  `ok` on -- with abstention rate and answered count beside it. A gate that improves the mean by
  declining the hard items has to look like one.
- **The cost is per answer,** in completion tokens and seconds, repair round trip included.
- **A class that never fired is `not-measured`, never `adopt`.** Absence of a rejection is absence
  of evidence.
- **Every refusal is listed, not just counted** -- item, axioms, the proxy's label, the reference
  columns, and the answer itself. The catch / false-rejection split rests on an automated proxy,
  and a proxy can be wrong: `contains` has no morphological normalization, so a correct short
  answer to a question with an inflected Ukrainian reference is labelled a catch. The table is what
  lets a reader overturn that label, and on the one heavy run so far it was needed.

An axiom class is adopted only when its catches exceed its false rejections AND the paired net
clears zero under the same machinery every other adoption verdict here uses
([paired verdicts](paired-verdicts.md)); the minimum-evidence gate applies, so a class resting on
five differing items states nothing at the 95% level whatever the direction.

## Measured: the fixture, and what the gate refuses that it should not

`samples/benchmarks/ontology_violations_uk.json` carries 17 cases in three categories over one
Ukrainian document, one extraction ledger, and ten axioms named by id from the committed candidate
set (which `fixture.py` signs IN MEMORY, for the fixture only -- a test may not stand in for a
reviewer). `make check-answer-gate` runs it; `tests/llb/eval/test_answer_validation.py` pins it.

- **Every planted violation is caught, at 1.000 per axiom class,** across all eight classes the
  gate decides: `functional`, `inverse_functional`, `max_cardinality`, `domain`, `range`,
  `disjoint_types`, `asymmetric`, `irreflexive`.
- **The false-rejection rate is 0.125 -- 1 of 8 adversarial correct answers -- not zero.** The case
  it is made of is `ok-unrecorded-paraphrase-001`: the answer restates the retrieved chunk's own
  population as `2,9 млн осіб` where the ledger recorded `2.9 мільйона осіб` and never recorded the
  short form as an alias. Surface folding cannot see that identity, so the gate refuses a correct
  answer under `func-maie-naselennia`. **This is the gate's known limit, and it is a limit of the
  extraction ledger's alias coverage rather than of the axiom**: an entity-resolution pass that
  merged the two surfaces would remove it, and until one runs, every functional or
  inverse-functional axiom carries this failure mode wherever a value has more than one written
  form. Numeric and date values are where it bites, because they are exactly the values a model
  paraphrases.
- **The remaining seven adversarial cases are accepted:** a legitimately multi-valued relation, a
  paraphrase the corpus DOES record as an alias, an endpoint the model could only type `MISC`, a
  one-way assertion of a symmetric relation, an inverse-functional constraint read in the correct
  direction, a bounded group the retrieved corpus already breaks on its own, and a declared
  abstention.
- **The scope case is accepted:** an answer that contradicts a ledger fact whose evidence the
  prompt never carried is not refused.

Eight adversarial cases is a small denominator and the rate travels with it. What would overturn
the reading: more adversarial cases (a wider fixture would almost certainly find more paraphrase
failures, not fewer), a different alias-coverage level in the extraction ledger, or an
entity-resolution overlay applied before the gate.

## Measured: the gate is not worth its cost on this corpus, and no class ships enabled

2026-08-26, RTX 4060 Ti 16 GB CUDA host, ollama backend, MamayLM-Gemma-3-12B-IT v2.0 (Q4_K_M).
The committed 82-item final split of `ua_squad_postedited_v1` against the flat
`intfloat/multilingual-e5-base` store (`top_k=5`, `max_tokens=768`, temperature 0), all three lanes
over the identical items, checked against the 250-document extraction ledger drafted from that same
corpus. **The axiom set was signed provisionally by the measurement lane, NOT by a domain reviewer**
(`ontology-axiom-signoff` is still open), so every number below prices the gate's MECHANISM and
none of it licenses enabling an axiom.

| lane | answered | abstained | refused | objective | found | completion tokens | s/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 82 | 0.000 | 0.000 | 0.509 | 0.646 | 17.0 | 1.48 |
| `pydantic` | 80 | 0.024 | 0.000 | 0.530 | 0.671 | 252.2 | 9.79 |
| `pydantic+ontology` | 78 | 0.024 | 0.024 | 0.530 | 0.671 | 259.8 | 10.04 |

- **The gate does not pay for itself here, and neither does the envelope.** On the 78 items every
  lane answered, `pydantic+ontology` moves the objective `+0.029 [-0.027, +0.087]` against `off`
  (calibrated p 0.171) and the found-rate `+0.026` -- not clear of zero, so the verdict is RETAIN
  for both candidate lanes. The gain such as it is belongs to the declared format, not to the gate:
  the two envelope lanes score identically (0.530 / 0.671), because the gate changes a case's
  STATUS and never its answer text.
- **It cost 238.6 completion tokens and 8.4 seconds per answer** over the ungated free-text
  baseline -- fourteen times the completion length -- of which the gate's own share is 7.6 tokens
  per answer over the whole item set. That share looks small only because it is concentrated: every
  case that spent a semantic reprompt is a case the gate then refused, so on the commonly-answered
  items the gate's added cost against `pydantic` is zero by construction, and the two refused cases
  cost 408 and 859 completion tokens each.
- **Conformance was perfect and abstention was rare** -- 82 of 82 envelopes valid on the first
  attempt, no schema repair spent, 2 declared abstentions. 80 of 82 envelopes declared a triple the
  gate could test (`validation_checked_rate` 0.976), 81 triples in all, so the gate was not idle
  for want of something to check.
- **It refused 2 of 82 answers, and BOTH are substantively correct.** The automated split says one
  catch and one false rejection; reading the two answers says otherwise:
  - `disjoint-org-loc` refused *"Лос-Анджелес є найбільшим містом у всій Каліфорнії"* against the
    reference *Лос-Анджелес*. The ledger types every Лос-Анджелес and Каліфорнія surface `LOC` and
    never `ORG`, so the `ORG` side of the disjointness came from the model's own declared
    `subject_type`. The class did not find a logical impossibility; it found two fallible taggers
    disagreeing about a type.
  - `maxcard-ye` refused *"Вишивка."* against the reference *"роботою вишивки"*, and the study
    labelled that a CATCH because `contains` scored it 0.0. It is the same word in a different
    case: the token scorer has no morphological normalization (which is what `--score-semantic`
    exists for), so a correct short answer to a question with an inflected reference reads as
    wrong. The one catch on this corpus is a proxy artifact.
- **So the honest reading is 0 catches and 2 false rejections, at 82 answered items.** Both classes
  are `measured-not-adopted` under the automated verdict as well (`disjoint_types` 0 caught / 1
  wrongly refused; `max_cardinality` 1 / 0, neither clearing zero on a single differing item), and
  no other class fired at all. Nothing here supports enabling any axiom class.
- **A defect this run found, and the fix it forced.** Before the correction, the gate also refused
  *"Машина Тюрінга"* under `maxcard-ye` -- a correct answer joining a group the retrieved chunks
  had ALREADY over-filled with three of their own `є` values against a bound of two. The gate now
  subtracts every `(axiom, subject)` anchor the corpus already breaks, which removed that rejection
  (3 refusals became 2) and changed nothing else. That case is in the fixture.

What would overturn this: a corpus whose relations the axiom set actually constrains -- on this
ledger only 2 of the 20 enabled classes ever fired, and the four `functional` axioms found no
population at all, so the reading is about THIS corpus far more than about the gate; a reviewed
axiom set, which could differ from the candidates in both directions; a model that types its
triples differently, since `disjoint_types` is measuring agreement with the extractor's types; or
re-labelling the refusals under `--score-semantic` rather than `contains`, which would move the
catch/false-rejection split without moving what the gate refused.

## Modules and tests

`src/llb/eval/answer_validation/` (`constants`, `models`, `scope`, `answer_ledger`, `gate`,
`fixture`, `study`, `verdict`, `report`, `run`), the `eval.rag.envelope_ontology_repair` template,
`answer_validation` / `ontology_axioms` / `ontology_ledger` on `RunConfig`, the `validator` seam in
`llb.eval.graph`, `build_answer_validator` in `src/llb/executor/runner_setup.py`,
`_attach_validation_columns` in `src/llb/executor/case_columns.py`, `_attach_validation_metrics` in
`src/llb/executor/runner_metrics.py`, and the CLI `compare-answer-validation` /
`check-answer-gate`.

`tests/llb/eval/test_answer_validation.py` covers the ledger-scope rule in both directions, the
corpus-contradicts-itself case, the corpus-already-broke-it subtraction in both directions, alias
folding and the ambiguous-alias drop, the `MISC` rule, the unchecked envelope, the `symmetric`
exclusion, all five refusals, the bounded semantic repair and its three outcomes, the
`ontology_violation` status, the untouched ungated path, journal coverage, and the whole fixture.
`tests/llb/eval/test_answer_validation_study.py` covers the
commonly-answered rule, the per-class adopt / measured-not-adopted / not-measured verdicts, the
cost columns, the lane-selection refusals, the orchestration over injected bundles, and all three
lanes driven over a fake completer and a fake ledger.
