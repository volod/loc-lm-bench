"""Gold-set and corpus preparation commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.prep.ontology.constants import EXTRACT_CONCURRENCY


@app.command("ingest-pdf-corpus")
def ingest_pdf_corpus_cmd(
    pdf_root: Path = typer.Option(..., help="directory of local PDF source documents"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output corpus dir of extracted .md files (default: <pdf-root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip PDFs whose extracted text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of PDFs to ingest"),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert every PDF even when the source is unchanged"
    ),
) -> None:
    """Extract local PDFs into the `.md` corpus shape used by RAG, goldset, and GraphRAG commands."""
    _run_pdf_markdown_ingest(
        "ingest-pdf-corpus", pdf_root, out_dir, min_chars, parser, limit, refresh
    )


@app.command("pdf-to-markdown")
def pdf_to_markdown_cmd(
    pdf_root: Path = typer.Argument(..., help="directory of local PDF source documents"),
    out_dir: Optional[Path] = typer.Argument(
        None, help="output dir of extracted .md files (default: <pdf-root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip PDFs whose extracted text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of PDFs to convert"),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert every PDF even when the source is unchanged"
    ),
) -> None:
    """Convert local PDFs into markdown files plus quality/citation sidecars."""
    _run_pdf_markdown_ingest(
        "pdf-to-markdown", pdf_root, out_dir, min_chars, parser, limit, refresh
    )


def _run_pdf_markdown_ingest(
    command: str,
    pdf_root: Path,
    out_dir: Optional[Path],
    min_chars: int,
    parser: str,
    limit: Optional[int],
    refresh: bool = False,
) -> None:
    from llb.prep.pdf_corpus import ingest_pdf_corpus

    try:
        result = ingest_pdf_corpus(
            pdf_root,
            out_dir,
            min_chars=min_chars,
            parser=parser,
            limit=limit,
            refresh=refresh,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    n_reused = sum(1 for item in result.items if item.reused)
    reused_note = f", {n_reused} reused unchanged" if n_reused else ""
    typer.echo(
        f"[{command}] {result.n_docs}/{len(result.items)} PDFs extracted "
        f"({result.n_skipped} skipped{reused_note}) -> {result.out_dir}"
    )


@app.command("prepare-goldset")
def prepare_goldset_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(..., help="litellm model id (needs a provider key in .env)"),
    n_per_doc: int = typer.Option(3, min=1, help="draft this many QA pairs per document"),
    out: Path = typer.Option(..., help="output gold set JSONL (items are verified=false)"),
) -> None:
    """Draft review-ready (question, answer, exact span) gold items from a corpus via a frontier LLM."""
    from llb.prep.frontier import prepare_goldset

    items = prepare_goldset(corpus_root, model=model, n_per_doc=n_per_doc, out_path=out)
    typer.echo(
        f"[prepare-goldset] {len(items)} drafted items (verified=false; review before scoring) -> {out}"
    )


@app.command("prepare-synthetic-corpus")
def prepare_synthetic_corpus_cmd(
    topics_file: Path = typer.Option(..., help="text file: one synthetic-doc topic per line"),
    planter: str = typer.Option(..., help="litellm model that PLANTS the labels"),
    judge: str = typer.Option(..., help="the eval judge model (must differ from the planter)"),
    out_dir: Path = typer.Option(..., help="output dir for docs + planted labels"),
    n_labels: int = typer.Option(
        3, min=1, help="planted labels per document (per kind in TA mode)"
    ),
    text_analysis: bool = typer.Option(
        False,
        "--text-analysis/--qa",
        help="emit RICHER per-kind text-analysis PlantedLabelRecords (text analysis) instead of QA labels",
    ),
    kinds: Optional[str] = typer.Option(
        None, help="comma-separated text-analysis kinds (default: the objective sub-tasks)"
    ),
    chat: bool = typer.Option(
        False, help="(with --text-analysis) plant chat-log-shaped docs (chat-period synthetic)"
    ),
) -> None:
    """Generate synthetic docs with structured planted labels (planter must differ from judge).

    Default emits QA-style key_fact labels (RAG planted set). `--text-analysis` emits the full
    per-kind text-analysis taxonomy (key_fact/entity/topic/trend/risk/decision/...) as
    PlantedLabelRecords for the text analysis scored text-analysis runner; add `--chat` for chat-log-shaped
    synthetic docs (chat-period).
    """
    topics = [t.strip() for t in topics_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    if not topics:
        typer.echo(f"[error] no topics found in {topics_file}", err=True)
        raise typer.Exit(code=2)

    if text_analysis:
        from llb.prep.chat_corpus import prepare_synthetic_chat_corpus
        from llb.prep.text_analysis_corpus import DEFAULT_KINDS, prepare_text_analysis_corpus

        chosen = tuple(k.strip() for k in kinds.split(",") if k.strip()) if kinds else DEFAULT_KINDS
        builder = prepare_synthetic_chat_corpus if chat else prepare_text_analysis_corpus
        try:
            docs, records = builder(
                topics,
                planter_model=planter,
                judge_model=judge,
                kinds=chosen,
                n_per_kind=n_labels,
                out_dir=out_dir,
            )
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2)
        typer.echo(
            f"[prepare-synthetic-corpus] text-analysis: {len(docs)} docs, {len(records)} planted "
            f"labels across {len(chosen)} kinds (planter={planter} != judge={judge}) -> {out_dir}"
        )
        return

    from llb.prep.frontier import prepare_synthetic_corpus

    docs, items = prepare_synthetic_corpus(
        topics, planter_model=planter, judge_model=judge, n_labels=n_labels, out_dir=out_dir
    )
    typer.echo(
        f"[prepare-synthetic-corpus] {len(docs)} docs, {len(items)} planted items "
        f"(planter={planter} != judge={judge}) -> {out_dir}"
    )


@app.command("ingest-chat-corpus")
def ingest_chat_corpus_cmd(
    chat_file: Path = typer.Option(
        ..., help="exported chat log (.json conversations / Telegram / .jsonl)"
    ),
    out_dir: Path = typer.Option(
        ..., help="output bundle dir (corpus/ + text_analysis_labels.jsonl)"
    ),
    model: str = typer.Option(
        ..., help="LOCAL drafter model id (no egress; OpenAI-compatible tag)"
    ),
    base_url: str = typer.Option(
        "http://localhost:11434/v1", help="LOCAL endpoint base URL (no egress, per OQ-egress)"
    ),
    n_labels: int = typer.Option(2, min=1, help="drafted labels per kind"),
    kinds: Optional[str] = typer.Option(
        None, help="comma-separated text-analysis kinds (default: the objective sub-tasks)"
    ),
) -> None:
    """category expansion chat-period: ingest a REAL chat corpus, draft grounded labels LOCALLY (no egress)."""
    from llb.bench.common import local_complete
    from llb.prep.chat_corpus import ingest_chat_corpus, load_chat_conversations
    from llb.prep.text_analysis_corpus import DEFAULT_KINDS

    conversations = load_chat_conversations(chat_file)
    if not conversations:
        typer.echo(f"[error] no conversations found in {chat_file}", err=True)
        raise typer.Exit(code=2)
    chosen = tuple(k.strip() for k in kinds.split(",") if k.strip()) if kinds else DEFAULT_KINDS
    try:
        docs, records = ingest_chat_corpus(
            conversations,
            complete=local_complete(model, base_url),
            kinds=chosen,
            n_per_kind=n_labels,
            out_dir=out_dir,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    typer.echo(
        f"[ingest-chat-corpus] {len(docs)} chat docs, {len(records)} drafted labels (synthetic=false, "
        f"egress=none) -> {out_dir} (run: llb bench-text-analysis --bundle {out_dir} --real-corpus)"
    )


@app.command("cross-check-goldset")
def cross_check_goldset_cmd(
    goldset: Path = typer.Option(..., help="drafted gold set JSONL (verified=false)"),
    corpus: Path = typer.Option(..., help="source corpus dir for the drafted items"),
    model: str = typer.Option(..., help="SECOND-frontier verifier (must differ from the drafter)"),
    out: Optional[Path] = typer.Option(
        None, help="cross-check report JSON (default beside goldset)"
    ),
) -> None:
    """verified-data hardening verified-data gate: a second frontier re-confirms grounding/support/answerability."""
    import json

    from llb.goldset.schema import load_goldset
    from llb.prep.cross_check import (
        cross_check_goldset,
        load_doc_texts,
        second_frontier_verify,
    )

    items = load_goldset(goldset)
    report = cross_check_goldset(items, load_doc_texts(corpus), second_frontier_verify(model))
    out_path = out or goldset.with_name(f"{goldset.stem}.cross_check.json")
    out_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(
        f"[cross-check] {report.n_passed}/{len(items)} items passed the gate "
        f"(verified=false until human verification gate) -> {out_path}"
    )


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
        False, help="prepend the committed UA seed (samples/security_cases_uk.json)"
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
        seed_path = Path("samples/security_cases_uk.json")
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
        False, help="prepend the committed UA seed (samples/security_cases_uk.json)"
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
        seed = _json.loads(Path("samples/security_cases_uk.json").read_text(encoding="utf-8"))
        cases = list(seed) + cases
    out.write_text(_json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[plant-security-cases] {len(cases)} planted cases (verified=false; human verification gate before "
        f"headline) -> {out}"
    )


@app.command("prepare-agentic-search")
def prepare_agentic_search_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    out: Path = typer.Option(
        ..., help="output agentic task set JSON (verified=false; review first)"
    ),
    top_k: int = typer.Option(8, min=1, help="max query terms per task kind (count + locate)"),
    limit: Optional[int] = typer.Option(None, help="cap the number of source documents"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/agentic_tasks_uk.json)"
    ),
) -> None:
    """agentic benchmark: build deterministic real-corpus agentic SEARCH tasks (count + locate) from a corpus."""
    import json as _json

    from llb.bench.agentic_tasks import build_from_corpus

    try:
        tasks = build_from_corpus(corpus_root, top_k=top_k, limit=limit)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    if merge_seed:
        seed = _json.loads(Path("samples/agentic_tasks_uk.json").read_text(encoding="utf-8"))
        tasks = list(seed) + tasks
    out.write_text(_json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[prepare-agentic-search] {len(tasks)} tasks (verified=false; human verification gate before headline) -> {out}"
    )


@app.command("adapt-bfcl")
def adapt_bfcl_cmd(
    functions_file: Path = typer.Option(..., help="BFCL function-doc file (.json/.jsonl)"),
    out: Path = typer.Option(..., help="output tooling bundle JSON (verified=false; review first)"),
    answers_file: Optional[Path] = typer.Option(
        None, help="BFCL possible-answer file (.json/.jsonl); without it cases are no-call controls"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of adapted cases"),
) -> None:
    """tooling benchmark: adapt the Berkeley Function-Calling Leaderboard (BFCL) cases into a UA tooling bundle."""
    import json as _json

    from llb.prep.tooling_sources import from_bfcl, load_jsonl_or_json

    entries = load_jsonl_or_json(functions_file)
    if limit is not None:
        entries = entries[:limit]
    answers = load_jsonl_or_json(answers_file) if answers_file else None
    bundle = from_bfcl(entries, answers)
    out.write_text(_json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(
        f"[adapt-bfcl] {len(bundle['cases'])} cases over {len(bundle['tools'])} tools "
        f"(verified=false; translate + human verification gate before headline) -> {out}"
    )


@app.command("prepare-goldset-draft")
def prepare_goldset_draft_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(
        ..., help="model id (local endpoint tag, or litellm route for frontier)"
    ),
    endpoint: str = typer.Option(
        "local", help="local (OpenAI-compatible, no egress) | frontier (litellm, opt-in egress)"
    ),
    base_url: Optional[str] = typer.Option(
        None, help="local endpoint base URL (default: Ollama OpenAI-compatible /v1)"
    ),
    max_items: int = typer.Option(60, min=1, help="upper bound on drafted QA items"),
    doc_limit: Optional[int] = typer.Option(
        None, min=1, help="bounded probe: only process the first N corpus documents"
    ),
    seed: int = typer.Option(13, help="deterministic sampling/split seed"),
    extractor: str = typer.Option(
        "llm", help="llm (default) | spacy (opt-in Python-native uk_core_news NER, no egress)"
    ),
    spacy_model: str = typer.Option(
        "uk_core_news_sm", help="spaCy pipeline (with --extractor spacy)"
    ),
    max_tokens: int = typer.Option(
        4096, min=1, help="per-call completion token budget for ontology drafting"
    ),
    extract_max_chars: Optional[int] = typer.Option(
        None,
        min=1,
        help="bounded probe/window size: max document characters per extraction call",
    ),
    extract_chunk_overlap: Optional[int] = typer.Option(
        None, min=0, help="overlap between extraction windows when a document is split"
    ),
    concurrency: int = typer.Option(
        EXTRACT_CONCURRENCY,
        "--concurrency",
        "--extract-concurrency",
        min=1,
        help="LLM extraction windows to run concurrently per document; merge order stays deterministic",
    ),
    temperature: float = typer.Option(
        0.0, min=0.0, help="per-call generation temperature for ontology drafting"
    ),
    timeout: float = typer.Option(
        300.0, min=1.0, help="per-call local/frontier endpoint timeout in seconds"
    ),
    no_think: bool = typer.Option(
        False,
        "--no-think",
        help="disable Ollama native reasoning mode for local JSON-producing models",
    ),
    num_ctx: Optional[int] = typer.Option(
        None,
        min=1,
        help="right-size the Ollama context window (native endpoint); avoids CPU offload from "
        "the modelfile default on VRAM-bound hosts -- keep headroom over extract-max-chars",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
    verification_sample_size: int = typer.Option(
        0,
        min=0,
        help="also write verify_sample.csv for human review (0 leaves review to make verify-sample)",
    ),
    retrieval_index_dir: Optional[Path] = typer.Option(
        None,
        help="full-corpus RAG index dir; when set, annotate citation-valid needles with retrieval_rank",
    ),
    retrieval_k: int = typer.Option(
        10, min=1, help="top-k cutoff for --retrieval-index-dir needle-rank annotation"
    ),
    drop_nonretrievable_needles: bool = typer.Option(
        False,
        "--drop-nonretrievable-needles",
        help="write only needles whose gold span is found within --retrieval-k",
    ),
) -> None:
    """ontology-assisted drafting: ontology-assisted DRAFT gold set from a corpus (verified=false; review before scoring)."""
    from llb.prep.ontology import EndpointConfig, draft_goldset
    from llb.prep.ontology.endpoint import DEFAULT_LOCAL_BASE_URL

    try:
        cfg = EndpointConfig(
            kind=endpoint,
            model=model,
            base_url=base_url or DEFAULT_LOCAL_BASE_URL,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            think=False if no_think else None,
            num_ctx=num_ctx,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    adapter = None
    if extractor == "spacy":
        from llb.prep.ontology.spacy_adapter import SpacyExtractionAdapter

        adapter = SpacyExtractionAdapter(model=spacy_model)
    if drop_nonretrievable_needles and retrieval_index_dir is None:
        typer.echo(
            "[error] --drop-nonretrievable-needles requires --retrieval-index-dir",
            err=True,
        )
        raise typer.Exit(code=2)
    if retrieval_index_dir is not None and not retrieval_index_dir.is_dir():
        typer.echo(f"[error] retrieval index dir not found: {retrieval_index_dir}", err=True)
        raise typer.Exit(code=2)
    result = draft_goldset(
        corpus_root,
        cfg,
        extraction_adapter=adapter,
        max_items=max_items,
        seed=seed,
        out_dir=out_dir,
        doc_limit=doc_limit,
        extract_max_chars=extract_max_chars,
        extract_chunk_overlap=extract_chunk_overlap,
        extract_concurrency=concurrency,
        retrieval_index_dir=retrieval_index_dir,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
    )
    if verification_sample_size:
        from llb.goldset.verify import build_sample_worksheet

        worksheet = result.out_dir / "verify_sample.csv"
        sample_size, _strata = build_sample_worksheet(
            result.out_dir, worksheet, n=verification_sample_size, seed=seed
        )
        typer.echo(
            f"[prepare-goldset-draft] verification sample: {sample_size} rows -> {worksheet}"
        )
    typer.echo(
        f"[prepare-goldset-draft] {len(result.items)} drafted items (verified=false; "
        f"endpoint={endpoint}, egress={cfg.egress}) -> {result.out_dir}"
    )
