# export-redistribution-gate

## Work contract

```markdown
#### export-redistribution-gate

`acl_label` covers who may READ a document inside a run; nothing covers whether the material may be
copied off the host. Corpus-derived text leaves through the fine-tune export
(`src/llb/finetune/dataset.py`), the contrastive-pair export (`src/llb/finetune/embedder/pairs.py`),
and every opt-in frontier path behind `egress_consent`, and none of them can currently refuse
material acquired on terms that forbid redistribution.

- Serves: `corpus-provenance` -- [Corpus provenance and acquisition boundary](../design/spec.md#corpus-provenance-and-acquisition-boundary)
- Agent status: CLEAR
- Dependencies: none. The check keys on `licence`, read at ingest today
  ([acquired-corpus provenance](current/data-prep/acquired-provenance.md)).
- User-visible outcome: an export refuses material whose recorded terms forbid redistribution, so
  the producing side's local-only determination is enforced where the material could actually leave.
- Scope boundary: in scope -- one shared redistribution check reused by every export and every
  egress path, keyed on the `licence` field the projection carries, and a refusal that names the
  offending document. Out of scope -- MAKING the determination, which is the producing service's
  responsibility and is consumed here as a recorded fact; access control, which `acl_label` already
  covers; and runtime guardrails in an application embedding a recommended model, which
  [the egress boundary](../design/spec.md#data-egress-boundary) puts outside this project.
- Data and artifact paths: `src/llb/finetune/dataset.py`, `src/llb/finetune/embedder/pairs.py`,
  `src/llb/prep/ontology/endpoints/builder.py`, and the non-redistributable document in
  `samples/corpora/acquired_projection_v1/`.
- Execution path: CPU-only, inside `make ci`; no frontier call is made, the consent seam is
  exercised with an injected fake.
- Acceptance gates: `make ci` green. A fixture containing one non-redistributable document causes
  each export path to REFUSE, naming that document, rather than to filter it -- a partial export
  that looks complete is the failure this prevents. Removing the gate makes the test fail. A corpus
  with no `licence` field exports exactly as it does today.
- Documentation target: `docs/impl/current/data-prep/acquired-provenance.md` and the
  [data egress](current/scope-boundaries.md#data-egress) boundary.
```

## Implementation and decisions

Built as planned: one shared check, `src/llb/prep/corpus/redistribution.py`, called by both exports
and by every path that sends corpus text off the host. What it does, the per-path table, and where
each piece lives: [the redistribution gate](../current/data-prep/acquired-provenance.md#redistribution-gate).

Decisions the contract left open:

- **A closed vocabulary that fails closed.** Only an absent `licence` and the exact class
  `redistributable` permit egress; `local-only` and every unrecognised value refuse. The fixture
  only ever used those two values, and a consumer that read an unknown class as permission would
  turn a producer's typo into a leak. The vocabulary is now part of the
  [projection contract](../../design/acquired-corpus-projection.md).
- **"Every egress path" means every path that executes here, not only the three the contract
  named.** Besides the two exports and ontology drafting (`endpoints/builder.py` was the pointer to
  the consent seam; the check itself lives beside it in `endpoints/egress.py`), the survey found six
  more paths that send corpus-derived text to a hosted model: the `run-eval` frontier scorer,
  auto-RAG frontier verification, `frontier-judge` agreement, the `compare-embeddings` API row,
  `prepare-goldset`, and `cross-check-goldset`. All nine are gated. Paths deliberately left out, and
  why, are listed on the topic page.
- **Scoped to what the path carries.** Each path checks only the documents whose text it would
  write or send, never the whole corpus by default, so one `local-only` document does not disable
  every export over its corpus. The two paths that do send everything (`prepare-goldset` and the
  API embedder) check everything.
- **Refusal before consent.** Frontier paths refuse before they ask for or record consent, so the
  draft CLI never prompts for an egress it would then refuse.
- **The fine-tune export reads the run's corpus, and fails closed when it is gone.** The export's
  inputs are a run bundle and a gold set, so the terms come from the run manifest's
  `config.corpus_root`, overridable with `--corpus` (`LICENCE_CORPUS=` on `make export-finetune-set`).
  A run naming no corpus is not gated (no recorded terms exist); a run naming a corpus that is no
  longer on the host is refused rather than exported unchecked.
- **Strictest source wins.** Where two sources describe a document -- chunk metadata and the corpus
  manifest for the frontier scorer, the bundle provenance and a staged manifest for the agreement
  lane -- `merge_licences` keeps the refusing value.
- **One CLI rendering.** `RedistributionRefused` is a `ValueError`, and `run_typer` reports it as
  one `[error]` line with exit 2 instead of the crash path's traceback, so every CLI gets the same
  message without per-command handling.
- **Complexity gate.** Adding the check pushed `export_finetune_set` to cognitive complexity 16;
  the tuning-row selection moved to `_selected_rows` rather than raising the threshold.

The real scanned book named in the request was not needed: this task is CPU-only over the committed
synthetic projection, and the book's redistribution class is declared by `book-corpus-projection`,
which now builds on this gate.

## Acceptance evidence

Status: `accepted`. Run on 2026-11-19 on an RTX PRO 3000 Blackwell laptop host, CPU-only, with no
model, GPU, or network.

- **`make ci` green.** On this host's base `[dev]` venv, `make ci-github` (the gate defined for that
  environment) passed 4,887 tests with 6 skipped and 161 deselected, exit 0, and `make ci-checks`
  reported zero ruff, mypy, complexity, shell-lint, spec-plan, toolchain, roster, and
  artifact-contract findings. Full `make ci` reported 4,898 passed, 49 failed, and 4 errors; the
  same 53 tests fail identically with this change stashed, all on optional extras this venv does
  not install (`faiss` 29, `langgraph` 10, `rdflib`, `splink`). Reading: no regression, but the
  full-install suite was not exercised on this host -- see Handoff. `make lint-md` reported zero
  findings and zero broken links.
- **A non-redistributable fixture document makes each path REFUSE, naming it, rather than filter.**
  `tests/llb/prep/test_redistribution_egress_paths.py` stages
  `samples/corpora/acquired_projection_v1/` through real ingestion and drives each of the nine
  paths plus the draft CLI's pre-consent check: each refused naming exactly `fixture-doc-07.md`,
  wrote no output, and never reached its injected provider. Each path's control (the same fixture
  with that sidecar set to `redistributable`) completed with the document's text in the payload, so
  the refusal is attributable to the recorded licence alone. End to end,
  `make export-finetune-set` exited 2 with the one-line refusal and left no dataset directory; the
  control exported `sft=2 dpo=0`.
- **Removing the gate makes the test fail.** A scripted mutation run disabled each of the ten call
  sites in turn: every mutation failed exactly that path's own refusal test or tests and no other's,
  and making the shared predicate always permit failed 21 of the 35 tests then in the two files.
- **A corpus with no `licence` field exports exactly as it does today.** The 229 existing tests of
  the touched lanes passed without a single fixture edit (none carries the field); the unit suite
  pins that an absent licence and a manifest-less corpus are not gated; and a run naming no corpus
  exports unchanged.

## Handoff

No open work remains under this task. `book-corpus-projection` no longer `Requires` it and now
cites the delivered vocabulary in its `Dependencies`; its gate "a projected book declared
non-redistributable makes the fine-tune export refuse and name it" is served by this check once
that task writes `licence` into the projected sidecars. The external recognition-service and
frontier vision-model paths the spec describes for scanned books do not exist yet; when
`book-evidence-constrained-correction` (whose frontier vision opt-in must be refused for
non-redistributable material) or a hosted recognition stage adds one, it calls
`refuse_corpus_egress` like the paths here.

Still unverified on this host: the full-install `make ci` suite, because the venv lacks the `rag`,
`langgraph`, ontology-axiom, and linkage extras. Running `make ci` on a full install is the check.

`make plan-status`: 62 tasks before (agent 37, human 25; CLEAR 21), 61 after (agent 36, human 25;
CLEAR 20). The `corpus-provenance` group left the plan, and its registry row moved from `planned` to
`shipped`. No other capability's task count changed; the priority agent task moved to
`book-recovery-tool-survey` [scanned-book-recovery]. The removal released the
`export-redistribution-gate` edge from `book-corpus-projection`, which still waits on
`book-vault-rendering-and-navigation`.
