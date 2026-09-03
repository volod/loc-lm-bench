# Data-Prep Artifact Contracts

## Delivered boundary

Every project-owned machine-readable artifact of the data-preparation surface is a registered
contract family: the corpus and PDF manifests, the citation sidecars, the gold rows, chains and
needle rows, the induced ontology and its extraction rows, the drafting provenance, the
external-draft sidecars, the conflict stage-inputs record and applied overlay, the linkage bundle
settings, and the verification worksheet row. Each names a `schema_id`, one current
`schema_version`, and a strict Pydantic model per version, and each producer builds that model
instead of handing a `dict[str, object]` to `json.dumps`.

Out of scope and deliberately unregistered: source document text, human-only Markdown reports, raw
PDFs, and `external_provenance.json` -- the operator-supplied sidecar an external drafting service
authors, which this project reads for its data-classification declaration and never writes.

The models live under `src/llb/core/contracts/data_prep/` (`corpus.py`, `goldset.py`,
`ontology.py`, `external_draft.py`, `conflicts.py`, `linkage.py`, `review.py`); the declarations and
their migrations live under `src/llb/artifacts/data_prep/` (`families.py`, `migrations.py`).

## Reading what this project already wrote

Every artifact written before the registry existed carries no identity at all, so a reader cannot
dispatch on the file alone. `ContractDefinition.legacy_version` closes that gap: it names the
version a caller that knows WHICH family it opened may assume, and
`ContractRegistry.read_as(schema_id, record, version=...)` applies it.

- A record carrying its own identity dispatches exactly as `read_current` does.
- A record naming a different family, or a version a dataset-manifest binding contradicts, is
  refused rather than coerced.
- A record carrying no `schema_id` is stamped with the binding's version when a manifest supplies
  one and with the family's `legacy_version` otherwise, then migrated forward.
- A caller that does NOT know what it opened still meets the missing-identity refusal, so the
  foundation's dispatch behavior is unchanged.

The catalog publishes the same key: `legacy_read_version` is a field on each entry. The catalog is
regenerated from the registry on every check, so it stays at one version rather than migrating. The
external validator
uses that field exactly as an outside reader must -- it stamps the declared version onto a
pre-contract fixture and validates it against the generated schema, importing no `llb` module.

## Two encodings that stay as they are

Two members keep the compact local version they have always written, because their bytes are
load-bearing:

- The **conflict stage-inputs record** carries an integer `schema_version` (1 through 7). A bundle
  is read by this project and by nothing else, and re-encoding every archived one would change
  bytes a store generation is fingerprinted against. `src/llb/conflicts/bundle/contract.py` maps
  the integer to the registered version and back.
- The **applied conflict overlay** carries integer `schema_version: 1` for the same reason: the
  overlay is folded into each document's corpus fingerprint, so its written form must not move.

In both cases the producer builds and validates the contract model and then writes the local form,
and the reader stamps the mapped version before dispatch.

## Migrations that preserve the reading

These four families are the only ones that carry more than one version, because their older forms
describe bundles this project has actually written. Every other family is at its initial version;
see [every family starts at one version](foundation-and-evolution.md#every-family-starts-at-one-version).

| Family | Old form | What the migration states |
| --- | --- | --- |
| `llb.gold-item` | `1.0.0`: `lang` and `verified` absent | Both stated, at the same defaults `load_goldset` applied |
| `llb.ontology-provenance` | `1.0.0`: no corpus binding, document rows without acquisition | `corpus_version: null` and every acquired field present as `null` |
| `llb.linkage-settings` | `1.0.0`: tuning knobs absent | Every knob stated, from the same constants `LinkageSpec.from_payload` used |
| `llb.conflict-stage-inputs` | `1.0.0`-`6.0.0`: six earlier record forms | Re-encoded at the current form through `documents_of` / `recorded_inputs`, the readers that already understood each form |

The conflict migration is deliberately not a second implementation of the seven forms: it decodes
with the bundle's own reader and re-encodes with `stage_attribution_inputs`, so the readings a
finished audit answers are preserved by construction. `readable_record` then takes every reading
from the CURRENT form -- a record already at it is passed through untouched, an older one is
re-encoded, and one the registry cannot resolve is refused with its reason rather than read as an
empty record.

## Bundles as datasets

`src/llb/artifacts/bundles.py` names the members of a staged corpus and of a draft bundle, and
`src/llb/artifacts/datasets.py` binds them into an `llb.dataset-manifest` record: every present
project-owned member is bound by contract identity, version, media type, granularity, relative
path, and SHA-256 digest. Members are DISCOVERED, so a bundle drafted without chains simply has no
chains member, while a member that is present and unreadable is a refusal. The same two modules,
plus `dataset_reading.py`, serve `check-store`
([retrieval and graph contracts](retrieval-and-graph-contracts.md)).

```bash
make check-bundle BUNDLE=<draft-bundle-dir>
make check-bundle BUNDLE=<corpus-dir> BUNDLE_KIND=corpus
make check-bundle BUNDLE=<draft-bundle-dir> BUNDLE_UPGRADE=1
```

`check-bundle` surveys every member -- reporting all refusals rather than stopping at the first --
and `BUNDLE_UPGRADE=1` rewrites the members an older writer produced at the current contract.
A member already current is left byte-for-byte alone, so the upgrade is idempotent. The underlying
command is `llb check-bundle <dir> --kind draft|corpus [--upgrade]`.

## Refusing before the expensive step

Two gates in `src/llb/artifacts/gates.py` ask "can this build read what it is about to act on"
before the work starts, because a record from a future major validates as JSON, looks like the
family it names, and hides every field a newer writer added:

- `refuse_unreadable_corpus` runs inside `build_store_parts`, before a store build chunks or embeds
  anything. It checks the corpus manifest and the applied conflict overlay -- both of which are
  folded into the store's corpus fingerprint.
- `refuse_unreadable_review` runs inside `open_review`, before a reviewer is shown anything to
  decide. It checks the bundle's drafting provenance; the review registry's signature detection
  stays the authority on which ledger a path is.

An absent file is not a refusal: many members are optional, and the caller that requires one says
so itself.

## Fixtures and the gates over them

`samples/artifact_contracts/data_prep/` holds a staged corpus (`corpus/`), a draft bundle at the
current contracts (`draft-bundle/`), the pre-contract form of each migrating member (`legacy/`),
the current linkage and conflict records those pair with (`current/`), and a corpus manifest from
an unsupported future major (`unsupported-future/`).

`tests/llb/artifacts/test_data_prep_contracts.py` asserts that each bundle validates member by
member, that a tampered member refuses on its digest, that the pre-contract gold set and linkage
bundle reach byte-identical canonical values, that the pre-binding provenance differs only in the
absence it could not record, that the early conflict record replays the same readings, and that a
future major refuses at both gates. `samples/artifact_contracts/external_validate.py` repeats the
current-form and pre-contract validations against the generated JSON Schemas without importing
`llb`, and runs inside `make check-artifact-contracts`. `tests/llb/cli/test_cli_check_bundle.py`
covers the operator command: what a current bundle reports, what `--upgrade` rewrites and what it
leaves byte-for-byte alone, and the two exit codes a refusal uses.

## Validation result

On 2026-09-03 on the CUDA host, `make ci` passed 4837 tests with 50 deselected, including the 16
data-prep contract tests and the 5 `check-bundle` CLI tests, and `make check-artifact-contracts`
validated the regenerated schemas, the catalog, the ODCS projection, and the external
process. The registry now generates 29 schema files across 18 families. This capability is
deterministic and service-free, so no model was loaded and no GPU memory was allocated for it.
What would overturn it: a data-prep producer added without a registered contract, which
`make check-artifact-contracts` cannot see -- the producer-side discipline is carried by review,
not by a gate.
