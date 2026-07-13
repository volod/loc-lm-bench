"""Security-case preparation commands: adapt, plant, derive, and worksheet scaffolding."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("adapt-security-set")
def adapt_security_set_cmd(
    source: str = typer.Option(..., help="public set: advbench | harmbench | jailbreakbench"),
    rows_file: Path = typer.Option(..., help="local export of the set (.json array or .csv)"),
    out: Path = typer.Option(..., help="output security-cases JSON (verified=false; review first)"),
    family: Optional[str] = typer.Option(
        None,
        help="override the case family (default: unsafe_content, or jailbreak when --jailbreak)",
    ),
    jailbreak: bool = typer.Option(
        False, help="wrap each behavior in a UA jailbreak template -> jailbreak-family cases"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of adapted cases"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/benchmarks/security_cases_uk.json)"
    ),
) -> None:
    """security benchmark: adapt a public adversarial set (UA-framed) into SecurityCase records for bench-security."""
    import json as _json

    from llb.prep.security_sources import JAILBREAK_TEMPLATES, adapt_public_set, load_rows

    try:
        cases = adapt_public_set(
            source,
            load_rows(rows_file),
            family=family,
            jailbreak_wrap=JAILBREAK_TEMPLATES[0] if jailbreak else None,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if merge_seed:
        seed_path = Path("samples/benchmarks/security_cases_uk.json")
        seed = _json.loads(seed_path.read_text(encoding="utf-8"))
        cases = list(seed) + cases
    out.write_text(_json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[adapt-security-set] {len(cases)} cases from {source} (verified=false; human verification gate before "
        f"headline) -> {out}"
    )


@app.command("plant-security-cases")
def plant_security_cases_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    out: Path = typer.Option(..., help="output security-cases JSON (verified=false; review first)"),
    injection_per_doc: int = typer.Option(1, min=0, help="RAG-injection leak cases per document"),
    canary_per_doc: int = typer.Option(
        1, min=0, help="canary/exfiltration leak cases per document"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of source documents"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/benchmarks/security_cases_uk.json)"
    ),
) -> None:
    """security benchmark: plant corpus-specific RAG-injection + canary leak cases over a real corpus."""
    import json as _json

    from llb.prep.security_planter import plant_from_corpus

    try:
        cases = plant_from_corpus(
            corpus_root,
            n_injection_per_doc=injection_per_doc,
            n_canary_per_doc=canary_per_doc,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if merge_seed:
        seed = _json.loads(
            Path("samples/benchmarks/security_cases_uk.json").read_text(encoding="utf-8")
        )
        cases = list(seed) + cases
    out.write_text(_json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[plant-security-cases] {len(cases)} planted cases (verified=false; human verification gate before "
        f"headline) -> {out}"
    )


@app.command("derive-security-cases")
def derive_security_cases_cmd(
    bundle: Path = typer.Option(
        ..., help="prepare-goldset draft bundle dir (reads its ontology.json + extraction.jsonl)"
    ),
    out: Optional[Path] = typer.Option(
        None,
        help="output security-cases JSON (default: $DATA_DIR/security-derive/<timestamp>/cases.json)",
    ),
    max_denial_per_vector: int = typer.Option(
        8, min=0, help="max harmful-ask/benign-control entities per denial-guard vector"
    ),
    max_bias_pairs: int = typer.Option(4, min=0, help="max matched bias pairs per entity type"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/benchmarks/security_cases_uk.json)"
    ),
) -> None:
    """security benchmark: derive corpus-specific content-safety cases (denial-guard + benign controls + bias pairs) from a draft bundle."""
    import json as _json
    from datetime import datetime, timezone

    from llb.core.paths import resolve_data_dir
    from llb.prep.security_derive import derive_from_bundle

    try:
        cases = derive_from_bundle(
            bundle,
            max_denial_per_vector=max_denial_per_vector,
            max_bias_pairs_per_type=max_bias_pairs,
        )
    except (ValueError, SystemExit) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if merge_seed:
        seed = _json.loads(
            Path("samples/benchmarks/security_cases_uk.json").read_text(encoding="utf-8")
        )
        cases = list(seed) + cases
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = resolve_data_dir() / "security-derive" / stamp / "cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[derive-security-cases] {len(cases)} derived cases (verified=false; human verification "
        f"gate before headline/composite) -> {out}"
    )


@app.command("derive-security-worksheet")
def derive_security_worksheet_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/security_cases_derived_uk.json"),
        help="derived security-cases JSON (from derive-security-cases)",
    ),
    out: Optional[Path] = typer.Option(
        None, help="output worksheet CSV (default: <cases-dir>/verify_sample.csv)"
    ),
) -> None:
    """human verification gate: scaffold a review worksheet (verify_sample.csv) from a derived security-cases set."""
    import json as _json

    from llb.goldset.verify_base import worksheet_fieldnames, write_worksheet_rows
    from llb.prep.security_derive import worksheet_rows

    raw = _json.loads(cases.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        typer.echo(f"[error] {cases}: expected a non-empty JSON array of security cases", err=True)
        raise typer.Exit(code=2)
    rows = worksheet_rows(raw)
    out = out or cases.with_name("verify_sample.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_worksheet_rows(out, rows, worksheet_fieldnames())
    typer.echo(
        f"[derive-security-worksheet] {len(rows)} rows -> {out}\n"
        f"  Review in the shared verification UI: make verify-review VERIFY_WS={out}\n"
        "  (same card/navigation/controls as goldset + chain review; y=accept, x=reject, q=save+quit).\n"
        f"  Then stamp the scored run: make bench-security SECURITY_CASES={cases} "
        f"SECURITY_DATA_VERIFIED=1 SECURITY_VERIFICATION_REF={out}"
    )
