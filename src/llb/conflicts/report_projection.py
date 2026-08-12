"""Render the opt-in TO REVIEW projection: its headline lines, its columns, and its prose.

Everything here reads a projection it was HANDED (`AuditResult.policy_projection`, computed by
`policy_projection.py` above this layer) and derives nothing, which is what keeps the resolution
vocabulary out of `conflicts.report*`. The module exists so `report.py` and `report_findings.py`
share one answer to "how is a projected count allowed to appear": always beside the policy that
produced it, and always called a projection.

With one policy named the output is exactly what it was before several were allowed -- one headline
line and one `to review (projected)` column. With several, each gets its own labelled column and
the DELTA against the first is rendered beside them, because the count an operator cares about is
not "what does conservative cost" but "does the choice cost me anything here".
"""

from dataclasses import dataclass

from llb.conflicts.census import counted
from llb.conflicts.constants import DECIDE_LABEL, REVIEW_LABEL
from llb.core.contracts.common import JsonObject

# The column an operator funds, never printed bare: the word `projected` rides with it everywhere.
PROJECTED_LABEL = f"{REVIEW_LABEL} (projected)"


@dataclass(frozen=True)
class ProjectedColumn:
    """One rendered column of the decision-groups table, addressed by group label."""

    header: str
    cells: dict[str, str]

    def cell(self, label: str) -> str:
        return self.cells.get(label, "0")


def policies_of(projection: JsonObject) -> list[str]:
    """The policies named, tolerating a payload that carries only the single-policy fields."""
    named = projection.get("policies")
    if isinstance(named, list) and named:
        return [str(policy) for policy in named]
    return [str(projection.get("policy", ""))]


def _counts_of(projection: JsonObject, policy: str) -> JsonObject:
    by_policy = projection.get("by_policy")
    if isinstance(by_policy, dict) and policy in by_policy:
        return dict(by_policy[policy])
    return {key: projection.get(key) for key in ("review_rows", "review_groups", "groups")}


def signed_delta(value: int) -> str:
    """`+2` / `-2` / `0` -- a delta must never read as a count, so a positive one keeps its sign.

    Zero is bare, not `+0`: no change is not a change in either direction, and the CLI, the table,
    and the headline all render it through here so they cannot disagree about that.
    """
    return f"+{value}" if value > 0 else str(value)


def projected_columns(projection: JsonObject) -> list[ProjectedColumn]:
    """One column per named policy, then one per delta against the first."""
    if not projection:
        return []
    named = policies_of(projection)
    single = len(named) == 1
    columns = [
        ProjectedColumn(
            header=PROJECTED_LABEL if single else f"{PROJECTED_LABEL[:-1]}, `{policy}`)",
            cells={
                str(group): str(int(count))
                for group, count in (_counts_of(projection, policy).get("groups") or {}).items()
            },
        )
        for policy in named
    ]
    return columns + [
        ProjectedColumn(
            header=f"delta (`{delta['baseline']}` -> `{delta['policy']}`)",
            cells={
                str(group): signed_delta(int(count)) for group, count in delta["groups"].items()
            },
        )
        for delta in projection.get("deltas") or []
    ]


def _policy_line(projection: JsonObject, policy: str, *, first: bool) -> str:
    """One headline line per policy. The first carries the full caveat the single-policy report
    always printed; the rest name their policy and stay short, because the caveat is the same one.
    """
    counts = _counts_of(projection, policy)
    rows = counted(int(counts["review_rows"] or 0), "row")
    groups = counted(int(counts["review_groups"] or 0), "decision group")
    line = f"- {REVIEW_LABEL} (PROJECTED under policy `{policy}`): {rows} in {groups}."
    if not first:
        return f"{line} The same replay under the other policy named, and the same caveat."
    return (
        f"{line} A projection, not a "
        f"measurement: it replays `resolve-corpus-conflicts --policy {policy}` over these rows and "
        "counts the ones that policy leaves at `review_required`. Another policy gives another "
        "number, and a reviewer's own decisions can still move it; the measured count is "
        "`review_rows` in that command's `plan.json`."
    )


def _delta_line(delta: JsonObject) -> str:
    """What the policy CHOICE costs -- the number a single projected column cannot show."""
    moved = [str(group) for group in delta["moved_groups"]]
    rows, baseline, policy = int(delta["review_rows"]), delta["baseline"], delta["policy"]
    if not moved:
        return (
            f"- policy choice `{baseline}` -> `{policy}`: **no difference on this corpus** -- the "
            "two policies leave the same rows open in every decision group, so the choice costs "
            "nothing here. It is not free in general: the policies part on dated supersessions, "
            "and this corpus has none the tiers found."
        )
    return (
        f"- policy choice `{baseline}` -> `{policy}`: **{signed_delta(rows)} "
        f"{'rows' if abs(rows) != 1 else 'row'}** to review, falling in "
        f"{counted(len(moved), 'decision group')} ({', '.join(moved)}). A delta between two "
        "projections, so it inherits their caveat: it is what the resolver would measure under "
        "each policy, not what a reviewer will decide."
    )


def projected_review_lines(projection: JsonObject) -> list[str]:
    """The headline block: one line per policy, then the delta between them.

    Without `--project-policy` this is nothing at all and the report is byte-identical to a report
    that never heard of the resolution vocabulary.
    """
    if not projection:
        return []
    named = policies_of(projection)
    lines = [
        _policy_line(projection, policy, first=index == 0) for index, policy in enumerate(named)
    ]
    return lines + [_delta_line(delta) for delta in projection.get("deltas") or []]


def two_counts_paragraphs(projection: JsonObject) -> list[str]:
    """Why the audit prints one count, and -- when asked -- what the projected second one is.

    The claim never weakens: the audit still cannot MEASURE the review count, because measuring it
    means running the policy over the corpus an operator chose. What `--project-policy` adds is one
    replay per NAMED policy over these same rows, which is a projection and says so wherever it
    appears.
    """
    measured = "an audit cannot measure it" if projection else "an audit cannot print it"
    paragraphs = [
        f"`{DECIDE_LABEL}` is NOT the count of human reviews this corpus costs. That second "
        f"count, **{REVIEW_LABEL}**, is a property of the resolution POLICY -- the rows "
        "`resolve-corpus-conflicts` leaves at `review_required` -- so it does not exist until "
        f"a policy is chosen and {measured}. The two diverge in both directions: "
        "an accepted `drop_duplicate` is to decide and not to review, while every semantic-tier "
        "duplicate is to review under every policy because that tier has no deletion authority. "
        f"`plan.json` prints both per group (`decide_rows`, `review_rows`) and ranks the review "
        "ledger on the second, which is the one an operator funds.",
        "",
    ]
    if not projection:
        return paragraphs
    named = policies_of(projection)
    listed = ",".join(named)
    columns = (
        f"The **{PROJECTED_LABEL}** column is that second count PROJECTED under policy `{named[0]}`"
        if len(named) == 1
        else (
            f"The **{PROJECTED_LABEL[:-1]}, ...)** columns are that second count PROJECTED under "
            f"each of the policies named (`{'`, `'.join(named)}`), with the **delta** column "
            f"measuring the switch from `{named[0]}`"
        )
    )
    return paragraphs + [
        f"{columns}, because `--project-policy {listed}` was passed: each group's rows replayed "
        f"through `resolve-corpus-conflicts --policy <p>` and counted where that policy "
        "leaves `review_required`. Every column is a projection under a named policy, never a "
        "measurement -- a policy not named gives a column that is not here, a reviewer's "
        "decisions can still move any of them, and the ranking below is unchanged (it still ranks "
        f"on `{DECIDE_LABEL}`, the only count that exists without a policy). The measured counts "
        "are `review_rows` in the `plan.json` each of those commands writes, and each equals its "
        "column group for group.",
        "",
    ]
