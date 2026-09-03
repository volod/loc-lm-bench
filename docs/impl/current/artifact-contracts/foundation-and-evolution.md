# Artifact Contract Foundation and Evolution

## Delivered boundary

Every registered family has a stable `schema_id`, one current semantic `schema_version`, and a
version-specific strict Pydantic model. `ArtifactContract` rejects unknown root fields;
`ExtensibleArtifactContract` permits only the named `extensions` map, whose values are typed
scalars rather than arbitrary nested objects. Model-generated schemas expose each identity and
version as JSON Schema `const` values, so external validation follows the same dispatch keys as the
Python reader.

The implementation is split at its functional seams:

- `src/llb/core/contracts/artifacts.py` owns strict record, dataset-member, relationship, quality,
  and compatibility-probe models;
- `src/llb/core/contracts/artifact_catalog.py` owns the portable catalog record;
- `src/llb/artifacts/definitions.py`, `registry.py`, `registry_validation.py`, and `versioning.py` own
  immutable declarations, semantic-version parsing, dispatch, source validation, compatibility
  path selection, and generation-time registry invariants;
- `src/llb/artifacts/records.py` stamps identity onto a record on the way out and takes it off on
  the way back, for the producers whose in-memory record is a plain mapping;
- `src/llb/artifacts/datasets.py` and `dataset_reading.py` describe a directory of artifacts as a
  dataset, publish and verify its manifest, and read, survey, or upgrade its members;
- `src/llb/artifacts/io.py` verifies a manifest member digest and reads JSON, JSONL, YAML, CSV, or
  Parquet through its bound row contract; Parquet loading is lazy and names the missing optional
  reader rather than widening the foundation's import cost;
- `src/llb/artifacts/generation.py` renders schemas and both catalog projections deterministically.

This foundation does not infer a schema from data, accept unknown fields, model third-party binary
formats, or replace domain quality and authorization gates.

## Compatibility behavior

`ContractRegistry.resolve` reads identity before downstream work, validates the record against its
observed version, and returns whether a migration is required. `read_current` then applies
registered one-step transformations in order and validates every intermediate result. A current
record performs no migration. The committed version 1 compatibility probe renames `name` to
`label` in version 2 without changing its meaning.

The reader refuses a missing identity, an unknown family, an unsupported version, a future major,
an invalid source record, a declared non-migratable version, no path, or more than one path. Each
refusal includes the source, observed identity and version, and the relevant registered or
supported set. Registry generation also refuses mismatched model literals, migration edges without
models, duplicate declarations, and an old model with neither exactly one path nor a declared
refusal. Thus changing a model under an existing version drifts its committed schema, while adding
an older model without an evolution declaration cannot generate a catalog.

`ContractRegistry.read_as` serves the caller who already knows the family: a record with its own
identity dispatches as above, a record naming a different family or contradicting a binding is
refused, and a record with no `schema_id` is stamped with the binding's version or the family's
declared `legacy_version` before migrating forward. A `legacy_version` is NOT a second version: it
names the version to assume for an identity-less file, and for most families that is the family's
only version. That declaration is what lets a family read the
files this project wrote before the registry existed; the domain surface it was built for is
[data-prep contracts](data-prep-contracts.md).

`ContractDefinition.legacy_document_field` extends that to the pre-contract files that are not
records at all -- a bare array of passages, the bare map of community summaries. It names the field
of the current record such a file's whole content became, and `ContractRegistry.normalize` applies
it before dispatch; [retrieval and graph contracts](retrieval-and-graph-contracts.md) is the
surface it was built for.

The compatibility cases live in `samples/artifact_contracts/`: current, supported old,
unsupported future, missing identity, ambiguous migration, and invalid source. The adjacent
`dataset-manifest.json` binds the readable cases by contract identity, version, media type,
granularity, relative path, and SHA-256 digest.

## Every family starts at one version

Nothing this project writes has been released, so a second version of a family would describe a
form nobody ever wrote. Every family therefore declares exactly ONE version, `1.0.0`, and no
migration -- with two deliberate exceptions:

- **`llb.artifact-contract.compatibility-probe`** carries `1.0.0` and `2.0.0` on purpose. It is the
  conformance family: it exists so the migration mechanism has a worked example that CI executes on
  every run, and it is the template the recipe below points at.
- **The data-prep families that read forms already on disk** -- `llb.conflict-stage-inputs` (seven
  integer schema versions the bundle file itself carries), `llb.gold-item`, `llb.linkage-settings`,
  and `llb.ontology-provenance`. Those older versions are not an unreleased release line; they
  describe bundles this project has actually written. See
  [data-prep contracts](data-prep-contracts.md).

A field an older writer never recorded does NOT need a version. Declare it optional and say in the
model docstring which reader default applies -- that is what `llb.rag-store-meta` does for the
duplicate-collapse knobs, because stores that predate them are still on disk.

## Adding a version, when one is finally needed

A new version is warranted only when the current model cannot describe both forms at once: a field
changes meaning, a field is renamed, or a required field is added that no older writer could have
recorded. Adding an optional field is not one of those cases.

The recipe, in the order the gates check it:

1. **Keep the current model, renamed to its version.** `CompatibilityProbeV1` is the pattern: the
   old class keeps its exact fields and its `schema_version` `Literal`, and the NEW class takes the
   plain name. Both stay registered, so a record at either version still validates as its own form.
2. **Write the transform.** A one-argument function from the old record dict to the new one. It
   must return something that validates against the version it names -- the registry re-validates
   every intermediate result, so a transform cannot quietly widen a contract. Never invent a value:
   take a default from the same constant the reader already applied, or state the absence as
   `null`.
3. **Declare the edge** in the family's `ContractDefinition`: bump `current_version`, add the new
   model to `models`, add a `MigrationStep(from_version, to_version, description, transform)`, and
   say in `deprecation_policy` what the old version is (read-and-migrate, or a
   `CompatibilityRefusal` when it cannot be carried forward). Registry generation refuses an old
   model that has neither exactly one migration path nor a declared refusal, so a half-declared
   evolution cannot reach a catalog.
4. **Regenerate and commit the exports** with `make generate-artifact-contracts`. The old version's
   JSON Schema stays; a new file appears for the new one.
5. **Commit a fixture at the old version** and assert that it reaches the same domain reading as a
   current one. That assertion, not the transform, is what says the migration preserved meaning.

## Portable exports and catalog

Generated files live under `schemas/artifacts/`. Each family/version has a standalone JSON Schema;
`catalog.json` lists current and supported read versions, schema paths, deprecation policy,
compatibility declarations, extension points, the version a pre-contract file is read at
(`legacy_read_version`), the field such a file's whole content became when it was not a record
(`legacy_document_field`), and JSON, JSONL, YAML, CSV, and Parquet bindings. The catalog is
generated from the registry on every check and distributed to nobody, so it has one version and
never migrates: a changed catalog shape is a regeneration, not an evolution.
`catalog.odcs.yaml` projects those logical objects, physical schema files, quality rule, and owner
into Open Data Contract Standard v3.1.0. The official ODCS v3.1.0 JSON Schema is pinned by digest
under `schemas/artifacts/vendor/` so the gate is network-free and cannot silently change with an
upstream release.

Run the deterministic repository gate with:

```bash
make check-artifact-contracts
```

After an intentional model/version/declaration change, refresh and immediately revalidate the
exports with:

```bash
make generate-artifact-contracts
```

Both commands run `samples/artifact_contracts/external_validate.py`. That program imports no
`llb` module: it validates the current and old records, dataset manifest, and artifact catalog with
their generated JSON Schemas, then validates the ODCS projection with the pinned official schema.
`make check-artifact-contracts` also runs inside `make ci-checks`.

## Validation result

On 2026-09-02 the focused contract suite passed 18 tests covering all six compatibility cases,
strict extensions, dataset digest and binding refusals, all five structured formats, generation
drift, missing evolution declarations, catalog round-trips, and the external validation process.
The external process validated both JSON Schema surfaces and the ODCS v3.1.0 projection without
importing `llb`. This capability is deterministic and service-free, so the CUDA host did not load a
model or allocate GPU memory for the result.
