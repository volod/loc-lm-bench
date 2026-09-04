"""One-step transformations from an older run record to the current one."""

from llb.core.contracts.run_bundle.manifest import RUN_MANIFEST_SCHEMA_ID


def run_manifest_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    """Carry the run head forward and declare that it states nothing about its own members.

    Nothing is invented. `score_rows` stays absent because a pre-contract bundle never recorded
    what its rows answered to, and `artifacts` is empty because it never recorded which additional
    files it published either -- both read as "this bundle does not state it", never as "it has
    none". Everything a version 1 manifest DID carry -- the run identity, the config, the
    environment, the metrics, and the judge, telemetry, contention, and durability records -- is
    the same field at the same meaning in version 2, so it is carried through untouched.
    """
    return {
        **record,
        "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "schema_version": "2.0.0",
        "artifacts": [],
    }
