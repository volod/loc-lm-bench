# Ontology axioms

`axioms_uk_v1.ttl` is the CANDIDATE constraint set over the closed entity-type vocabulary and the
induced relations, written as RDFS/OWL Turtle so a domain reviewer who does not read this codebase
can read, diff, and version it. `axioms_uk_v1.json` is the same set through the typed models; `make
ci` fails if the two ever describe different sets.

**Nothing here is signed, so nothing here gates anything.** An axiom is a domain claim, not a corpus
statistic, so it is enabled only after a reviewer accepts it one at a time -- a signature is
`dcterms:creator` + `dcterms:date` on that axiom's `owl:Axiom` annotation block.

```bash
make validate-ontology-axioms EXTRACTION=<draft-bundle> AXIOMS=samples/ontology/axioms_uk_v1.ttl
```

The axiom classes, the checker's reading rules, the OWL reasoner cross-check, and the measured
base rates on the committed corpora are documented in [robustness and
ontology](../../docs/impl/current/robustness-ontology-backends.md#ontology-axiom-layer); the
constraint layer's place in the schema is [graph ontology
schema](../../docs/design/graph-ontology-schema.md#6-axioms-over-the-vocabulary-and-the-induced-relations).
