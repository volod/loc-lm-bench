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

Projecting ONE policy answers "what does my corpus cost?" but not the question an operator asks
immediately afterwards -- "what would the other policy cost?". Since the projection is a pure
replay over rows the audit already holds, every named policy costs one more pass and no model call,
so `project_policies` runs the whole list and reports the per-group DELTA between them. That delta
is the only thing that says whether the policy choice is even a choice on this corpus: on a corpus
with no dated supersession it is zero everywhere, and an operator can stop thinking about it.

A sign answers whether, not how much, so every delta also carries `moved_rows` / `moved_share` --
the rows the choice actually moves as a share of the rows the audit calls work. A `-2` on a
17-row fixture and a `-2` on a corpus of thousands are the same sign and different decisions, and
the share is the only one of the two numbers that transfers between corpora.

**Layering.** This module imports `resolution_policy`; the report modules do NOT import it, and
must not. The projection is computed above both layers (the CLI composes it) and handed to the
renderer as plain JSON data, so `conflicts.report*` stays free of the resolution vocabulary and
the detector keeps working without a policy. See the layering note in the conflict-detection doc.
"""

from collections.abc import Sequence

from llb.conflicts.constants import decide_count
from llb.conflicts.group_artifact import group_summaries
from llb.conflicts.resolution_policy import (
    POLICIES,
    STATUS_REVIEW_REQUIRED,
    resolve_finding,
)
from llb.core.contracts.common import JsonObject

# 1 projected one named policy; 2 adds `policies` / `by_policy` / `deltas` beside the first
# policy's own fields, which stay where a single-policy consumer already reads them; 3 gives every
# delta the SHARE of actionable rows the choice moves, beside the sign it already carried.
PROJECTION_SCHEMA_VERSION = 3
# The per-policy counts every column is rendered from, named once so the delta and the renderer
# cannot disagree about which fields a policy's block carries.
POLICY_COUNT_FIELDS = ("review_rows", "review_groups", "groups")
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

    This is the one-policy case of `project_policies`, and it returns the SAME document shape, so
    there is exactly one `policy_projection` schema on disk however many policies were named.
    """
    return project_policies(rows, [policy])


def _project_one(
    rows: list[JsonObject], summaries: list[JsonObject], policy: str
) -> tuple[JsonObject, dict[str, str]]:
    """The per-policy replay every column is computed from, plus the per-ROW statuses behind it.

    The statuses never reach the artifact -- `findings.jsonl` is what addresses rows -- but the
    delta needs them: a per-group count can only report a NET change, and two rows moving opposite
    ways inside one group cancel to zero while still costing an operator two decisions.
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
        for summary in summaries
    }
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "kind": PROJECTION_KIND,
        "policy": policy,
        "basis": PROJECTION_BASIS,
        "review_rows": sum(groups.values()),
        "review_groups": sum(1 for count in groups.values() if count),
        "groups": groups,
    }, status_by_id


def project_policies(rows: list[JsonObject], policies: Sequence[str]) -> JsonObject:
    """Every named policy projected over the same rows, plus the delta between them.

    The FIRST policy named is the baseline, in two senses: its own counts stay at the top level
    exactly where a single-policy consumer already reads them, and every delta is measured against
    it. So `--project-policy conservative,prefer-newer` answers "what does conservative cost, and
    what would switching cost?" -- which is the order an operator asks the two questions in.

    Every column comes from one replay helper, so a column here and the column a single-policy run
    prints are the same number computed the same way, and each still equals the `review_rows` the
    resolver measures under that policy.
    """
    named = list(policies)
    if not named:
        raise ValueError("project_policies needs at least one resolution policy")
    if len(set(named)) != len(named):
        raise ValueError(f"duplicate resolution policy in {named}; each column must be distinct")
    # The grouping is a property of the ROWS, not of a policy, so it is computed once and every
    # column counts into the same groups -- N columns cannot disagree about what a group is.
    summaries = group_summaries(rows)
    replayed = {policy: _project_one(rows, summaries, policy) for policy in named}
    per_policy = {policy: document for policy, (document, _) in replayed.items()}
    statuses = {policy: status for policy, (_, status) in replayed.items()}
    actionable = sum(decide_count(summary["relations"]) for summary in summaries)
    baseline = named[0]
    return {
        # The baseline's own projection IS the top level, so a consumer written against one policy
        # keeps reading the same fields and never has to learn the multi-policy shape.
        **per_policy[baseline],
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "policies": named,
        "by_policy": {
            policy: {field: projected[field] for field in POLICY_COUNT_FIELDS}
            for policy, projected in per_policy.items()
        },
        "deltas": [
            _delta(per_policy, statuses, baseline, policy, actionable) for policy in named[1:]
        ],
    }


def _delta(
    per_policy: dict[str, JsonObject],
    statuses: dict[str, dict[str, str]],
    baseline: str,
    policy: str,
    actionable_rows: int,
) -> JsonObject:
    """What switching from `baseline` to `policy` costs, per decision group and in total.

    Negative means the switch removes review work. `moved_groups` is the answer an operator is
    actually after -- WHICH decisions the policy choice touches -- because a corpus-wide total of
    -2 hides whether that is one group changing or twenty cancelling out.

    A SIGN is not a magnitude, though, and `-2` means one thing on a 17-row fixture and another on
    a corpus of thousands. `moved_rows` counts the rows whose status actually differs between the
    two policies -- a gross count, so offsetting moves inside one group are both counted, which the
    two shipped policies cannot yet produce (the switch only ever settles rows) but a third policy
    that escalates what the baseline accepts would -- and
    `moved_share` reads it against the rows the audit calls work (`decide_rows`, the same
    `is_actionable` set every other count uses). That share is what makes the reading portable: it
    answers "is the policy choice a real decision on a corpus like mine?" rather than only "does
    this corpus contain a dated supersession at all?".
    """
    before, after = per_policy[baseline]["groups"], per_policy[policy]["groups"]
    groups = {group_id: int(after[group_id]) - int(count) for group_id, count in before.items()}
    moved_rows = sum(
        1
        for finding_id, status in statuses[baseline].items()
        if statuses[policy].get(finding_id) != status
    )
    return {
        "baseline": baseline,
        "policy": policy,
        "review_rows": int(per_policy[policy]["review_rows"])
        - int(per_policy[baseline]["review_rows"]),
        "review_groups": int(per_policy[policy]["review_groups"])
        - int(per_policy[baseline]["review_groups"]),
        "groups": groups,
        "moved_groups": [group_id for group_id, count in groups.items() if count],
        "moved_rows": moved_rows,
        "actionable_rows": actionable_rows,
        # None, never 0.0: a corpus with nothing to decide has no share to report, and a zero
        # share means the opposite -- rows to decide, none of which the choice touches.
        "moved_share": round(moved_rows / actionable_rows, 6) if actionable_rows else None,
    }
