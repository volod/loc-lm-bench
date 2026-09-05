"""How one surveyed dataset member is reported to an operator.

Shared by every `check-*` command, because a store, a bundle, and a run all answer the same
question -- did this member read back, and at which version -- and an operator comparing two of
them should not have to compare two vocabularies.
"""

from llb.artifacts.bundles import MemberReading


def member_line(reading: MemberReading) -> str:
    """One survey line: what the member is, or why it refused."""
    if reading.refusal:
        return f"  [refused] {reading.member_id} ({reading.path}): {reading.refusal}"
    if not reading.schema_id:
        return f"  [ok] {reading.member_id} ({reading.path}): opaque member, digest matches"
    version = (
        f"{reading.source_version} -> {reading.current_version}"
        if reading.needs_upgrade
        else reading.current_version
    )
    return (
        f"  [ok] {reading.member_id} ({reading.path}): {reading.records} record(s) of "
        f"{reading.schema_id}@{version}"
    )
