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
  AXIOMS=<signed-axioms.ttl> ONTOLOGY_LEDGER=<extraction.jsonl> \
  ONTOLOGY_OVERLAY=<overlay.jsonl>          # optional; see "Which surfaces are one thing"

make check-answer-gate            # the committed adversarial fixture: catch + false rejection

# Re-read a FINISHED comparison from the run bundles it recorded, under the current reading.
# No lane runs, no model call; writes a new artifact dir and never touches the recorded one.
make compare-answer-validation ANSWER_VALIDATION_FROM_BUNDLES=<comparison.json>
```

A single gated run is an ordinary `run-eval`:

```bash
make run-eval ANSWER_FORMAT=envelope MAX_TOKENS=768 \
  ANSWER_VALIDATION=ontology ONTOLOGY_AXIOMS=<signed.ttl> ONTOLOGY_LEDGER=<extraction.jsonl> \
  ONTOLOGY_OVERLAY=<overlay.jsonl>
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
- **Endpoint surfaces fold onto whatever the corpus already treats as one thing** -- an alias it
  records, a node the resolution lane merged, or a VALUE written another way (next section).
- **`MISC` asserts nothing, on BOTH sides.** `MISC` is what `normalize_entity_type` collapses
  anything out-of-vocabulary into, so it names the absence of a recognized type. The answer side
  drops it and so does the scoped corpus side, which is what keeps an extractor's fallback from
  refusing an answer under a `domain`/`range` axiom.

## Which surfaces are one thing

Every false rejection this gate has produced was an IDENTITY failure rather than an axiom failure:
the axiom was right and the two surfaces it compared were the same thing written twice. So identity
is decided in one module (`identity.py`) by three folds, ordered by how much the corpus itself
vouches for each. Only the ANSWER's endpoints are folded -- the corpus ledger is left as recorded,
because the gate subtracts the violations the corpus already had on its own and rewriting that side
would change what "the corpus already broke this" means.

1. **The alias the extraction ledger RECORDS.** The corpus's own statement that two surfaces name
   one entity. An alias two different entity names both claim is dropped rather than resolved --
   collapsing an ambiguous surface would invent an identity no reviewer accepted.
2. **The node cluster the entity-resolution lane PROPOSES.** Optional, supplied per run as
   `ONTOLOGY_OVERLAY=<overlay.jsonl>` from `resolve-graph-entities` ([entity
   resolution](../entity-resolution.md)). The graph lane already decides which nodes are one entity;
   reusing that decision is what stops the gate refusing an answer the graph merged, and reading it
   needs no `KnowledgeGraph` -- `read_overlay_surfaces` returns the member -> canonical NAME map the
   overlay rows already carry for their reader. It stays optional because an overlay is a proposal
   at one threshold, not a fact about the corpus.
3. **Value equivalence,** for the three closed-vocabulary types whose members are values rather than
   names: `QUANTITY`, `DATE`, `DURATION` (`equivalence.py`, reading the closed Ukrainian table
   family in `value_lexicon.py`). This is what an alias map structurally cannot carry. Nothing
   gives an extractor a reason to record `2,9 млн осіб` as an alias of
   `2.9 мільйона осіб`, so before this every `functional`, `inverse_functional`, and
   `max_cardinality` axiom broke wherever a value had more than one written form -- which is exactly
   where a model paraphrases.

A value key reads the number (`2 900 000`, `2,9`, `2.9` and the magnitude words `тис` / `млн` /
`млрд` all resolve to one amount), the unit, and for a date the point in time (`2021 року`,
`1 січня 2021`, `01.01.2021`, `2021-01-01`). Three boundaries keep it from inventing identity:

- **only value types are keyed.** `Київ` is a name; identity between names is what folds 1 and 2
  decide. `MONEY` is left out because a sum carries a currency this would have to invent a table for.
- **units come from a closed table, not the lemmatizer.** The pinned analyzer maps `рік` to `ріка`
  (the river), so `20 років` and `20 рік` would part company under it; the lemmatizer still handles
  the open tail, where it is right (`осіб` -> `особа`).
- **refuse rather than guess.** Two numbers in one surface, a duration with no time unit, a date
  carrying a word the grammar does not model: no key, and the endpoint folds exactly as before. A
  wrong key would merge two values that DIFFER, which loses a planted violation -- the one failure
  this gate may not have. The fixture carries a near-value case (`2,8 млн осіб` against the chunk's
  `2.9 мільйона осіб`) that must still be refused.

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
  reference scores correct is a false rejection whatever else the gate got right. Which refusals
  count as correct is decided in `labelling.py` (below) rather than by the objective -- that
  objective mixes needle-finding with terseness ([generation graph and
  scoring](scoring.md#measured-the-headline-objective-is-partly-a-verbosity-ranking)) and would
  price a verbose but correct refused answer as a wrong one.
- **The objective delta is read only on the COMMONLY ANSWERED items** -- the items every lane ended
  `ok` on -- with abstention rate and answered count beside it. A gate that improves the mean by
  declining the hard items has to look like one.
- **The cost is per answer,** in completion tokens and seconds, repair round trip included.
- **A class that never fired is `not-measured`, never `adopt`.** Absence of a rejection is absence
  of evidence.
- **Each class carries BOTH readings of its own rejections.** `caught` / `wrongly refused` are the
  inflection-tolerant labels the verdict is decided on; the column beside them is what the shipped
  surface-token proxy alone gives on the same rejections, so a run measured after the re-labelling
  is comparable to one recorded before it.
- **Every refusal is listed, not just counted** -- item, axioms, both labels, the signal that
  decided, the semantic cosine, the reference, and the answer itself. The catch / false-rejection
  split rests on automated proxies and a proxy can be wrong, so the table is what lets a reader
  overturn a label by eye.

## Labelling a refusal: a signal that survives inflection

`contains` -- every reference token somewhere in the answer, over case- and punctuation-folded
SURFACE tokens -- has no morphology, and Ukrainian references are inflected. A correct short answer
to a question whose reference sits in an oblique case therefore reads as wrong, and the refusal of
it is filed as a CATCH: the most flattering label the gate could get. On the first heavy run the
single recorded catch was exactly this artefact.

Two more readings sit beside it, both over the pinned pymorphy3 lemmatizer the lexical index and
the answer-span scorer already use, so a form retrieval matches is a form this label matches. A term
carries its surface form AND its lemma together and two terms match when those sets intersect --
the analyzer's first parse is not stable across inflections of one word (`роботою` normalizes to
`робота`, `робота` to `робот`, the robot), the same artefact `llb.scoring.function_words` lists both
forms for.

| Signal | What it says |
| --- | --- |
| `contains` | the shipped surface reading, unchanged: every reference token is in the answer |
| `contains_lemma` | the same claim matched on term keys, so inflection cannot break it |
| `answer_within_reference` | the TERSE reading: every CONTENT term the answer states is in the reference. `Вишивка.` against `роботою вишивки` needs this, and no containment in the other direction can express it |

A refusal is a false rejection when ANY of the three fires, and the report prints the new label,
the signal that decided it, and what `contains` alone would have said -- so the two readings sit
side by side rather than one silently replacing the other. The per-class adopt-or-reject verdict
reads the SAME labels, so it can never disagree with the evidence table beside it.

Two things it deliberately does not do. It reads the recorded `answer_preview`, which is capped:
every added signal is containment-shaped, so a truncated answer can only FAIL a test the full
answer would have passed -- the cap can lose a false rejection and can never invent one. And **no
embedding threshold decides a label.** `--score-semantic` records a cosine per case and the refusal
table reports it, because a reader adjudicating a rejection wants it; but no cut on this corpus
separates a paraphrase from a different answer (the repo's own near-duplicate work reads 0.9 as
barely above noise), and a label nobody can reproduce without the pinned embedder is a worse proxy
than a deterministic one. A comparison whose gold set has moved simply loses the two added signals
and degrades to exactly the shipped reading.

## Re-reading a finished comparison

`compare-answer-validation ANSWER_VALIDATION_FROM_BUNDLES=<comparison.json>` recomputes a recorded
comparison from the run bundles it named, under the CURRENT reading, with no model call. The
comparison is pure over the per-case rows, and every lane's bundles are in its own
`comparison.json` -- so a changed reading (this labelling, for one) reaches a finished run instead
of only the next one. Without it the old and new readings of a heavy run could only be compared by
spending the three lanes again.

Two refusals guard it, both before anything is written: a bundle that no longer describes the lane
its label claims -- a repointed directory, a different model or gold set, a lane whose contract
knobs changed under the same name -- and a rebuilt comparison that no longer covers the recorded
item set. The recorded artifact is never written to; the re-render lands in a new directory and
appends `rerendered_from` / `rerendered_at` to the recorded metadata, so stripping those two keys
gives back what the generations produced.

An axiom class is adopted only when its catches exceed its false rejections AND the paired net
clears zero under the same machinery every other adoption verdict here uses
([paired verdicts](paired-verdicts.md)); the minimum-evidence gate applies, so a class resting on
five differing items states nothing at the 95% level whatever the direction.

## Measured: the fixture, and what it takes to accept a paraphrase

`samples/benchmarks/ontology_violations_uk.json` carries 20 cases in three categories over one
Ukrainian document, one extraction ledger, and thirteen axioms named by id from the committed
candidate set (which `fixture.py` signs IN MEMORY, for the fixture only -- a test may not stand in
for a reviewer). `make check-answer-gate` runs it; `tests/llb/eval/test_answer_validation.py` pins
it.

- **Every planted violation is caught, at 1.000 per axiom class,** across all eight classes the
  gate decides: `functional` (2 cases), `inverse_functional`, `max_cardinality`, `domain`, `range`,
  `disjoint_types`, `asymmetric`, `irreflexive`.
- **The false-rejection rate is 0.000 -- 0 of 10 adversarial correct answers.** It was 0.125 (1 of
  8) before value equivalence shipped, and it is 0.300 (3 of 10) on today's wider fixture with the
  value fold disabled: the three cases restate a retrieved chunk's own value in a written form the
  ledger records no alias for -- `2,9 млн осіб` against `2.9 мільйона осіб`, `20 років` against
  `двадцять років`, `27.08.1856` against `27 серпня 1856 року`. A test pins the disabled reading
  too, so the 0.000 is the fix rather than three cases that were never hard.
- **Nothing was traded for it.** The catch rate per class is unchanged, and the fixture carries the
  near-value guard the fold could have broken: `2,8 млн осіб` against the chunk's `2.9 мільйона
  осіб` is a DIFFERENT population written in the same short form, and it is still refused under
  `func-maie-naselennia`.
- **The remaining seven adversarial cases are accepted:** a legitimately multi-valued relation, a
  paraphrase the corpus DOES record as an alias, an endpoint the model could only type `MISC`, a
  one-way assertion of a symmetric relation, an inverse-functional constraint read in the correct
  direction, a bounded group the retrieved corpus already breaks on its own, and a declared
  abstention.
- **The scope case is accepted:** an answer that contradicts a ledger fact whose evidence the
  prompt never carried is not refused.

Ten adversarial cases is still a small denominator and the rate travels with it. What would
overturn the reading: more adversarial cases (a wider fixture would find paraphrase shapes the
value grammar does not model -- a hedged value, a range, a unit outside the closed table); a
different alias-coverage level in the extraction ledger; or a corpus whose values are written in
forms the number grammar reads differently from a Ukrainian reader.

## Measured (earlier run): the gate is not worth its cost, and no class ships enabled

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

## Measured again after the equivalence fix: the same verdict, and value equivalence never fired

2026-08-26, RTX PRO 3000 Blackwell 12 GB CUDA host, ollama backend, MamayLM-Gemma-3-12B-IT v2.0
(Q4_K_M). The committed 82-item final split of `ua_squad_postedited_v1` against the flat
`intfloat/multilingual-e5-base` store (`top_k=5`, `max_tokens=768`, temperature 0, 311 chunks),
all three lanes over the identical items, `--score-semantic` on, checked against a 250-document
extraction ledger re-drafted from that same corpus on this host (1350 entities, 925 facts). **The
axiom set was signed provisionally by the measurement lane, NOT by a domain reviewer**, so nothing
below licenses enabling an axiom.

**This is not a paired before/after of the run above.** That run's ledger and bundles are not on
this host, and the ledger here was drafted afresh by a different extraction pass, so the two
readings share a corpus and a model but not a ledger or a GPU. Read them as two measurements of
the same mechanism, never as a delta.

| lane | answered | abstained | refused | objective | found | completion tokens | s/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 82 | 0.000 | 0.000 | 0.515 | 0.646 | 16.7 | 1.64 |
| `pydantic` | 80 | 0.024 | 0.000 | 0.534 | 0.683 | 245.8 | 8.31 |
| `pydantic+ontology` | 76 | 0.024 | 0.049 | 0.534 | 0.683 | 259.7 | 8.68 |

- **The verdict is unchanged: RETAIN for both candidate lanes.** On the 76 items every lane
  answered, `pydantic+ontology` moves the objective `+0.027 [-0.030, +0.087]` (p 0.193) and the
  found-rate `+0.039` against `off` -- not clear of zero. The two envelope lanes score identically
  again (0.534 / 0.683), because the gate changes a case's STATUS and never its answer text. The
  envelope cost 232.2 completion tokens and 6.8 s per answer over the free-text baseline; the
  gate's own share is 13.9 tokens per answer over the whole item set, concentrated in the four
  cases it refused (436-893 completion tokens each, a semantic reprompt included).
- **Value equivalence had nothing to act on here, and that is the finding.** Not one
  `functional`, `inverse_functional`, or `max_cardinality` refusal fired -- the classes whose
  false rejections the fix exists to remove. The reason is upstream: the extraction pass typed 28
  of 1350 entities as a value type (14 `DATE`, 12 `QUANTITY`, 2 `DURATION`), which resolve to 12
  distinct value keys, so almost no declared endpoint on this corpus IS a value. The fix is
  measured on the fixture; on this corpus it is untested by construction rather than shown
  ineffective.
- **The gate refused 4 of 82, and all four are false rejections under BOTH readings.** Every
  refusal was `disjoint_types` -- two `disjoint-org-loc`, two `disjoint-person-org` -- so the
  class is `measured-not-adopted` at 0 caught / 4 wrongly refused, and no other class fired at
  all. It is the same failure the earlier run named: the disjointness came from the model's own
  declared `subject_type` disagreeing with the extractor's, which is two fallible taggers
  disagreeing about a type rather than a logical impossibility.
- **The re-labelling moved nothing on this run, and the report says so.** Plain `contains` already
  scored all four refused answers correct, so the inflection-tolerant signals agreed with it and
  the re-labelled column equals the shipped one (0 caught / 4 refused either way). The added
  signals were not idle -- `contains_lemma` fired on all four and `answer_within_reference` on one
  -- they simply had no refusal to rescue. The morphology artefact the earlier run recorded came
  from a `max_cardinality` refusal, and that class never fired here.
- **Conformance was perfect and the gate was not idle for want of something to check.** 82 of 82
  envelopes valid on the first attempt, no schema repair spent, 2 declared abstentions, and 80 of
  82 envelopes declared a triple the gate could test (`validation_checked_rate` 0.976, 81 triples).
- **No entity-resolution overlay was supplied, on the evidence.** `resolve-graph-entities` over
  this graph (2020 nodes, 925 edges) recorded a NEGATIVE result -- no candidate cut separated from
  its pre-overlay lane (best `global_community` cut: recall@10 `+0.024 [-0.037, +0.085]`, 4/2/76
  win/loss/tie) -- and its clusters merge distinct entities even at the tightest cut of 0.99:
  `Лос-Анджелес Ремс` with `Лос-Анджелес Лейкерс`, `Сан-Дієго` with `Сан-Бернардіно`, `парова
  машина` with `газові турбіни`. Folding endpoints through that would invent identities the corpus
  does not carry, which is what the gate refuses to do on principle. The fold ships and is
  covered by tests; on this corpus no cut was worth handing it.

What would overturn this: an extraction pass that types values as values, which is what would give
the equivalence fix a population to be measured on; a reviewed axiom set; a corpus whose relations
the axiom set actually constrains; or a model that types its triples in agreement with the
extractor, since `disjoint_types` is measuring exactly that agreement and is the only class either
run has been able to price.

## Modules and tests

`src/llb/eval/answer_validation/` (`constants`, `models`, `equivalence`, `identity`, `scope`,
`answer_ledger`, `gate`, `fixture`, `labelling`, `study`, `value_lexicon`, `verdict`, `report`, `run`,
`rerender`),
the `eval.rag.envelope_ontology_repair` template, `answer_validation` / `ontology_axioms` /
`ontology_ledger` / `ontology_overlay` on `RunConfig`, `read_overlay_surfaces` in
`llb.graph.resolution.overlay`, the `DATE` / `DURATION` / `QUANTITY` constants in
`llb.prep.ontology.extraction.entity_types`, the `validator` seam in `llb.eval.graph`,
`build_answer_validator` in `src/llb/executor/runner_setup.py`, `_attach_validation_columns` in
`src/llb/executor/case_columns.py`, `_attach_validation_metrics` in
`src/llb/executor/runner_metrics.py`, and the CLI `compare-answer-validation` (with
`--overlay` and `--from-bundles`) / `check-answer-gate`.

`tests/llb/eval/test_answer_validation.py` covers the ledger-scope rule in both directions, the
corpus-contradicts-itself case, the corpus-already-broke-it subtraction in both directions, alias
folding and the ambiguous-alias drop, the `MISC` rule, the unchecked envelope, the `symmetric`
exclusion, all five refusals, the bounded semantic repair and its three outcomes, the
`ontology_violation` status, the untouched ungated path, journal coverage, the whole fixture, and
the fixture read again with value equivalence disabled.
`tests/llb/eval/test_answer_validation_equivalence.py` covers the value key in both directions
(forms that must fold, values that must not), the surfaces the grammar refuses to read, the
type-scoping, the three identity folds and their precedence, and every labelling signal including
the two cases where the reading must NOT move.
`tests/llb/eval/test_answer_validation_study.py` covers the commonly-answered rule, the per-class
adopt / measured-not-adopted / not-measured verdicts, the cost columns, the lane-selection
refusals, the orchestration over injected bundles, all three lanes driven over a fake completer and
a fake ledger, and the re-render round trip with its drift refusals.
