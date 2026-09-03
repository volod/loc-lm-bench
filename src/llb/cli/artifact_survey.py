"""Shared operator surface for surveying a described directory of artifacts.

`check-bundle` and `check-store` ask the same question of different directories: can this build
read every member, and which of them an older writer produced. One implementation renders the
answer, so a store's report reads exactly like a bundle's.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import typer

from llb.core.contracts.artifacts import DatasetManifest

if TYPE_CHECKING:
    from llb.artifacts.dataset_reading import MemberReading

ManifestFor = Callable[[Path], DatasetManifest]


def survey_command(command: str, root: Path, manifest_for: ManifestFor, upgrade: bool) -> None:
    """Describe `root`, optionally upgrade its older members, and report every member."""
    from llb.artifacts.dataset_reading import survey_dataset, upgrade_dataset

    try:
        manifest = manifest_for(root)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[{command}] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if upgrade:
        rewritten = upgrade_dataset(root, manifest)
        typer.echo(f"[{command}] upgraded {len(rewritten)} member(s): {', '.join(rewritten)}")
        manifest = manifest_for(root)

    readings = survey_dataset(root, manifest)
    for reading in readings:
        typer.echo(member_line(reading))
    refused = [reading for reading in readings if reading.refusal]
    typer.echo(
        f"[{command}] {len(readings) - len(refused)}/{len(readings)} member(s) readable "
        f"in {manifest.dataset_id} at {root}"
    )
    if refused:
        raise typer.Exit(code=1)


def member_line(reading: "MemberReading") -> str:
    """One member's line: its contract and record count, or the owner of its opaque format."""
    if reading.refusal:
        return f"  [refused] {reading.member_id} ({reading.path}): {reading.refusal}"
    if reading.is_opaque:
        return (
            f"  [ok] {reading.member_id} ({reading.path}): opaque, owned by "
            f"{reading.owner}@{reading.format_version}"
        )
    version = (
        f"{reading.source_version} -> {reading.current_version}"
        if reading.needs_upgrade
        else reading.current_version
    )
    return (
        f"  [ok] {reading.member_id} ({reading.path}): {reading.records} record(s) of "
        f"{reading.schema_id}@{version}"
    )
