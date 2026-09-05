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
declared `legacy_version` before migrating forward. That declaration is what lets a family read the
files this project wrote before the registry existed; the domain surface it was built for is
[data-prep contracts](data-prep-contracts.md).

The compatibility cases live in `samples/artifact_contracts/`: current, supported old,
unsupported future, missing identity, ambiguous migration, and invalid source. The adjacent
`dataset-manifest.json` binds the readable cases by contract identity, version, media type,
granularity, relative path, and SHA-256 digest.

## Portable exports and catalog

Generated files live under `schemas/artifacts/`. Each family/version has a standalone JSON Schema;
`catalog.json` lists current and supported read versions, schema paths, deprecation policy,
compatibility declarations, extension points, the version a pre-contract file is read at
(`legacy_read_version`), and JSON, JSONL, YAML, CSV, and Parquet bindings. Publishing that last
field is why the catalog family itself is at `1.1.0`, with `1.0.0` read-and-migrate.
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
