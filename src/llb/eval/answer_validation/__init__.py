"""Step two of the two-step answer gate: is the DECLARED answer semantically possible?

Step one (`llb.eval.answer_envelope`) asks whether the model emitted the requested shape. It
cannot ask whether the shape is TRUE, and neither can groundedness: a semantically impossible
answer assembled out of real chunk tokens passes a token-overlap check cleanly. This package adds
the missing question -- do the envelope's declared triples break an ACCEPTED axiom, or contradict a
corpus fact whose evidence is in the retrieved chunks?

  - `constants` names the lanes, the axiom classes the gate may refuse an answer with, and the
    artifact layout;
  - `scope` restricts the corpus ledger to what the prompt carried and folds entity surfaces
    through the aliases the corpus itself records;
  - `answer_ledger` renders one envelope as the `DocExtraction` the shipped axiom checker reads;
  - `gate` runs the signed axioms over the merged ledger and keeps only the violations the ANSWER
    is party to;
  - `fixture` measures catch and false rejection on the committed adversarial cases;
  - `study` / `verdict` / `report` / `run` are the three-lane comparison and its adopt-or-reject
    readings.

Nothing here enables an axiom. The gate refuses an unsigned axiom file with a named error, because
an axiom is a domain claim a reviewer accepted and dated -- and a wrong one converts correct
answers into `ontology_violation` silently.
"""
