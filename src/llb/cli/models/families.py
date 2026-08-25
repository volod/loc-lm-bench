"""Family register commands: what the roster carries per family, and publishing that into docs."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error

DEFAULT_MANIFEST = Path("samples/configs/models_uk.yaml")


@app.command("list-model-families")
def list_model_families_cmd(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, help="candidate-models YAML manifest"),
    markdown: bool = typer.Option(False, help="print the published Markdown table instead"),
) -> None:
    """Print the family register: each family's generations, their status, and the models on them."""
    from llb.backends.roster import ROLE_LABELS, load_register, register_findings
    from llb.quality.roster_docs import render_block

    try:
        register = load_register(manifest)
    except ValueError as exc:
        cli_error(str(exc))

    findings = register_findings(register)
    for finding in findings:
        typer.echo(f"[list-model-families] finding: {finding}", err=True)

    if markdown:
        typer.echo(render_block("model-roster", register))
    else:
        for family in register.families:
            role = ROLE_LABELS.get(family.role, family.role)
            typer.echo(f"[family] {family.id:<10} {role:<22} {family.focus}")
            for generation in family.generations:
                models = ", ".join(generation.model_names) or "(no model carries it)"
                typer.echo(
                    f"  {generation.status:<8} {generation.id:<8} {generation.license:<12} {models}"
                )
    typer.echo(
        f"[list-model-families] {len(register.families)} families, "
        f"{len(register.models)} models, {len(findings)} finding(s)"
    )
    if findings:
        raise typer.Exit(code=1)


@app.command("sync-model-family-docs")
def sync_model_family_docs_cmd(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, help="candidate-models YAML manifest"),
    check: bool = typer.Option(False, help="report stale generated blocks instead of rewriting"),
) -> None:
    """Publish the family register into the generated README and reference tables."""
    from llb.quality.roster_docs import main

    argv = ["--manifest", str(manifest)] + (["--check"] if check else [])
    raise typer.Exit(code=main(argv))
