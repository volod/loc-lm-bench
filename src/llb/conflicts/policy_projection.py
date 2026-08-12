"""Project the TO REVIEW count under a named policy, one command before the resolver runs.

The audit can name the review count but cannot measure it: `review_rows` is a property of a
resolution POLICY, and an audit runs before an operator has chosen one
(`constants`, `report_findings.stake_key`). An operator sizing a review budget off `report.md`
alone therefore has to run `resolve-corpus-conflicts` to learn what their corpus costs.

That gap is bridgeable without guessing, because `resolve_finding` is a pure function of
`(relation, tier, governance, policy)` and `findings.jsonl` already carries all four. This module
replays it over the audited rows under a policy the operator NAMES and reports the count that
policy would leave open. The result is a PROJECTION, not a measurement:

- it is only true of the policy it names -- run another policy and the number changes;
- it precedes review, so a reviewer's own decisions (`resolution_review.jsonl`) can only move it;
- the measured count stays `plan.json`'s `review_rows`, which this projection must equal row for
  row and group for group under the same policy. `tests/llb/conflicts/test_policy_projection.py`
  pins that equality, which is what keeps this a second READING of one implementation rather than
  a second implementation of one number.

**Layering.** This module imports `resolution_policy`; the report modules do NOT import it, and
must not. The projection is computed above both layers (the CLI composes it) and handed to the
renderer as plain JSON data, so `conflicts.report*` stays free of the resolution vocabulary and
the detector keeps working without a policy. See the layering note in the conflict-detection doc.
"""

from llb.conflicts.group_artifact import group_summaries
from llb.conflicts.resolution_policy import (
    POLICIES,
    STATUS_REVIEW_REQUIRED,
    resolve_finding,
)
from llb.core.contracts.common import JsonObject

PROJECTION_SCHEMA_VERSION = 1
# What the number IS, carried in the artifact so a consumer cannot read it as a measurement even
# if it never reads the report prose.
PROJECTION_KIND = "projection"
PROJECTION_BASIS = (
    "the deterministic resolution policy replayed over these findings rows; it is what "
    "`resolve-corpus-conflicts --policy <policy>` would leave at `review_required`, not a "
    "measurement of what a reviewer decides"
)


def project_review_rows(rows: list[JsonObject], policy: str) -> JsonObject:
    """Rows a named policy would leave open, per decision group and in total.

    `rows` are `findings.jsonl` rows in the file's own order -- the same order and the same
    grouping `groups.json` and `plan.json` use -- so the group ids here address exactly the groups
    those artifacts address.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown resolution policy {policy!r}; choose one of {POLICIES}")
    status_by_id = {
        str(item["finding_id"]): str(item["status"])
        for item in (resolve_finding(row, policy) for row in rows)
    }
    groups = {
        str(summary["group_id"]): sum(
            1
            for fid in summary["finding_ids"]
            if status_by_id.get(str(fid)) == STATUS_REVIEW_REQUIRED
        )
        for summary in group_summaries(rows)
    }
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "kind": PROJECTION_KIND,
        "policy": policy,
        "basis": PROJECTION_BASIS,
        "review_rows": sum(groups.values()),
        "review_groups": sum(1 for count in groups.values() if count),
        "groups": groups,
    }
