"""Gold-set and corpus preparation commands."""

from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import urlsplit

import typer

from llb.cli.app import app
from llb.prep.ontology.constants import DEFAULT_MULTI_HOP_MAX_PATHS, EXTRACT_CONCURRENCY


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


@app.command("ingest-corpus")
def ingest_corpus_cmd(
    root: Path = typer.Option(..., help="directory of mixed .txt/.md/.pdf source documents"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output corpus dir of .md/.txt files (default: <root>/_md)"
    ),
    min_chars: int = typer.Option(
        500, min=1, help="skip documents whose text is shorter than this"
    ),
    parser: str = typer.Option(
        "auto", help="PDF parser: auto | pymupdf4llm | docling | marker | unstructured | markitdown"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="reconvert/re-copy every source even when it is unchanged"
    ),
    default_language: Optional[str] = typer.Option(
        None,
        "--default-language",
        help="language tag for sources that do not provide one (otherwise a cheap detector runs)",
    ),
    source_system: str = typer.Option(
        "local", help="default source-system tag recorded in corpus governance metadata"
    ),
    acl_label: Optional[str] = typer.Option(
        None, "--acl-label", help="default ACL label copied to manifest items and chunks"
    ),
) -> None:
    """Ingest a mixed txt/md/pdf directory into one canonical corpus (PDFs converted, text passed through)."""
    from llb.prep.corpus_ingest import ingest_corpus

    try:
        result = ingest_corpus(
            root,
            out_dir,
            min_chars=min_chars,
            parser=parser,
            refresh=refresh,
            default_language=default_language,
            source_system=source_system,
            acl_label=acl_label,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    reused_note = f", {result.n_reused} reused unchanged" if result.n_reused else ""
    removed_note = f", {result.n_removed_sources} removed" if result.n_removed_sources else ""
    typer.echo(
        f"[ingest-corpus] {result.n_docs}/{len(result.items)} documents ingested "
        f"({result.n_skipped} skipped{reused_note}{removed_note}) -> {result.out_dir}"
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


@app.command("prepare-agentic-search")
def prepare_agentic_search_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    out: Path = typer.Option(
        ..., help="output agentic task set JSON (verified=false; review first)"
    ),
    top_k: int = typer.Option(8, min=1, help="max query terms per task kind (count + locate)"),
    limit: Optional[int] = typer.Option(None, help="cap the number of source documents"),
    merge_seed: bool = typer.Option(
        False, help="prepend the committed UA seed (samples/benchmarks/agentic_tasks_uk.json)"
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
        seed = _json.loads(
            Path("samples/benchmarks/agentic_tasks_uk.json").read_text(encoding="utf-8")
        )
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
    corpus_root: Optional[Path] = typer.Option(
        None, help="directory of .md/.txt source docs (read from the bundle meta with --resume)"
    ),
    model: Optional[str] = typer.Option(
        None, help="model id (local endpoint tag, or litellm route for frontier)"
    ),
    resume: Optional[Path] = typer.Option(
        None,
        help="resume an interrupted draft bundle: reuse journaled extraction windows and replay "
        "the deterministic seed/draft stages (reads settings from the bundle's journal meta)",
    ),
    endpoint: str = typer.Option(
        "local", help="local (OpenAI-compatible, no egress) | frontier (litellm, opt-in egress)"
    ),
    backend: str = typer.Option(
        "ollama",
        help="local serving backend for --endpoint local: ollama | vllm | openai",
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
        help="disable hidden reasoning for local JSON-producing models (Ollama native or vLLM extra_body)",
    ),
    num_ctx: Optional[int] = typer.Option(
        None,
        min=1,
        help="right-size the Ollama context window (native endpoint); avoids CPU offload from "
        "the modelfile default on VRAM-bound hosts -- keep headroom over extract-max-chars",
    ),
    vllm_port: int = typer.Option(
        8000,
        min=1,
        max=65535,
        help="port for a vLLM server launched by this command when --backend vllm and --base-url is unset",
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.85,
        min=0.01,
        max=1.0,
        help="vLLM --gpu-memory-utilization when this command launches the server",
    ),
    vllm_max_model_len: Optional[int] = typer.Option(
        None,
        min=1,
        help="vLLM --max-model-len when this command launches the server; defaults to --num-ctx when set",
    ),
    vllm_cpu_offload_gb: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="vLLM --cpu-offload-gb when this command launches the server",
    ),
    vllm_kv_offloading_size_gb: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="vLLM --kv-offloading-size when this command launches the server",
    ),
    vllm_dtype: str = typer.Option(
        "auto", help="vLLM --dtype when this command launches the server"
    ),
    vllm_quantization: Optional[str] = typer.Option(
        None, help="vLLM --quantization when this command launches the server"
    ),
    vllm_startup_timeout: float = typer.Option(
        600.0,
        min=1.0,
        help="seconds to wait for a vLLM server launched by this command to become ready",
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
    coverage_target: Optional[int] = typer.Option(
        None,
        min=1,
        help="yield-max: draft up to N seeds per stratum bucket instead of the flat --max-items cap",
    ),
    multi_hop: bool = typer.Option(
        False,
        "--multi-hop",
        help="yield-max: also draft multi-span chain questions walked from the knowledge graph",
    ),
    chains: bool = typer.Option(
        False,
        "--chains",
        help="also write chains.jsonl with ordered chain-of-questions items from graph paths",
    ),
    multi_hop_max_paths: int = typer.Option(
        DEFAULT_MULTI_HOP_MAX_PATHS,
        min=1,
        help="cap on 2-hop graph paths drafted when --multi-hop is set",
    ),
    dedup_against: Optional[str] = typer.Option(
        None,
        help="yield-max: comma-separated prior bundle dirs; drop pinned-E5 near-duplicate questions",
    ),
    graph_dir: Optional[Path] = typer.Option(
        None,
        help="persisted graph store dir for --multi-hop paths (default: build the graph in-run)",
    ),
    rejection_feedback: Optional[Path] = typer.Option(
        None,
        "--rejection-feedback",
        help="verify-gate rejection_reasons.json; dominant reject codes tighten the draft "
        "prompts and the applied hints land in provenance",
    ),
    require_passed_gates: bool = typer.Option(
        False,
        "--require-passed-gates",
        help="exit non-zero after writing the bundle when the ontology calibration gates fail",
    ),
) -> None:
    """ontology-assisted drafting: ontology-assisted DRAFT gold set from a corpus (verified=false; review before scoring)."""
    from llb.core.config import DEFAULT_VLLM_HOST
    from llb.prep.ontology import (
        EndpointConfig,
        default_out_dir,
        draft_goldset,
        load_journal_meta,
    )
    from llb.prep.ontology.endpoint import (
        DEFAULT_LOCAL_BASE_URL,
        ENDPOINT_LOCAL,
        LOCAL_BACKEND_VLLM,
    )

    resuming = resume is not None
    if resume is not None:
        try:
            meta = load_journal_meta(resume)
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=2)
        ep_meta = cast(dict[str, Any], meta.get("endpoint") or {})
        # The bundle's journal meta is authoritative for the corpus and endpoint identity; the
        # extraction/seed/retrieval settings are re-read inside draft_goldset(resume=True). The
        # base URL is intentionally NOT restored so a vLLM resume relaunches a fresh server.
        if corpus_root is None:
            corpus_root = Path(str(meta.get("corpus_root")))
        if model is None:
            model = str(ep_meta.get("model") or "")
        endpoint = str(ep_meta.get("kind") or endpoint)
        backend = str(ep_meta.get("backend") or backend)
        if out_dir is None:
            out_dir = resume
    if corpus_root is None or not model:
        typer.echo(
            "[error] provide --corpus-root and --model, or --resume <bundle>",
            err=True,
        )
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
    if graph_dir is not None and not graph_dir.is_dir():
        typer.echo(f"[error] graph store dir not found: {graph_dir}", err=True)
        raise typer.Exit(code=2)
    if rejection_feedback is not None and not rejection_feedback.is_file():
        typer.echo(f"[error] rejection feedback file not found: {rejection_feedback}", err=True)
        raise typer.Exit(code=2)
    dedup_against_dirs: Optional[list[Path | str]] = (
        [Path(part.strip()) for part in dedup_against.split(",") if part.strip()]
        if dedup_against
        else None
    )

    resolved_out_dir = out_dir
    base_url_value = base_url or DEFAULT_LOCAL_BASE_URL
    launched_vllm = None
    if endpoint == ENDPOINT_LOCAL and backend == LOCAL_BACKEND_VLLM and base_url is None:
        from llb.backends.vllm import VllmLauncher

        resolved_out_dir = resolved_out_dir or default_out_dir()
        host = _vllm_host_for_port(DEFAULT_VLLM_HOST, vllm_port)
        launched_vllm = VllmLauncher(
            model,
            host=host,
            port=vllm_port,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len or num_ctx,
            cpu_offload_gb=vllm_cpu_offload_gb,
            kv_offloading_size_gb=vllm_kv_offloading_size_gb,
            dtype=vllm_dtype,
            quantization=vllm_quantization,
            startup_timeout=vllm_startup_timeout,
            log_dir=resolved_out_dir / "vllm",
        )
        typer.echo(
            f"[prepare-goldset-draft] starting vLLM model={model} host={host} port={vllm_port}"
        )
        launched_vllm.start()
        base_url_value = f"{host}/v1"

    endpoint_num_ctx = None if backend == LOCAL_BACKEND_VLLM else num_ctx
    try:
        cfg = EndpointConfig(
            kind=endpoint,
            model=model,
            backend=backend,
            base_url=base_url_value,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            think=False if no_think else None,
            num_ctx=endpoint_num_ctx,
        )
    except ValueError as exc:
        if launched_vllm is not None:
            launched_vllm.stop()
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    try:
        result = draft_goldset(
            corpus_root,
            cfg,
            extraction_adapter=adapter,
            max_items=max_items,
            seed=seed,
            out_dir=resolved_out_dir,
            doc_limit=doc_limit,
            extract_max_chars=extract_max_chars,
            extract_chunk_overlap=extract_chunk_overlap,
            extract_concurrency=concurrency,
            retrieval_index_dir=retrieval_index_dir,
            retrieval_k=retrieval_k,
            drop_nonretrievable_needles=drop_nonretrievable_needles,
            coverage_target=coverage_target,
            multi_hop=multi_hop,
            chains=chains,
            multi_hop_max_paths=multi_hop_max_paths,
            dedup_against=dedup_against_dirs,
            graph_dir=graph_dir,
            rejection_feedback=rejection_feedback,
            resume=resuming,
        )
    finally:
        if launched_vllm is not None:
            launched_vllm.stop()
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
    if require_passed_gates:
        from llb.prep.ontology.artifacts import required_gate_names
        from llb.prep.ontology.constants import PDF_ONTOLOGY_REPORT_FILENAME

        gates = (
            result.calibration_report.get("gates")
            if isinstance(result.calibration_report, dict)
            else None
        )
        passed = isinstance(gates, dict) and bool(gates.get("passed"))
        if not passed:
            failed: list[str] = []
            if isinstance(gates, dict):
                required = required_gate_names(bool(gates.get("pdf_citation_gate_applicable")))
                failed = [name for name in required if not gates.get(name)]
            detail = ", ".join(failed) if failed else "see report"
            typer.echo(
                "[error] ontology calibration gates not passed "
                f"({detail}); inspect {result.out_dir / PDF_ONTOLOGY_REPORT_FILENAME}",
                err=True,
            )
            raise typer.Exit(code=1)


def _vllm_host_for_port(default_host: str, port: int) -> str:
    parsed = urlsplit(default_host)
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "localhost"
    return f"{scheme}://{hostname}:{port}"


@app.command("curate-drafts")
def curate_drafts_cmd(
    inputs: list[Path] = typer.Argument(
        ..., help="exported artifact files to merge (raw JSON, fenced blocks, or JSONL)"
    ),
    kind: str = typer.Option(
        ..., help="artifact kind: squad | grounded | security | chains | inventory"
    ),
    out: Path = typer.Option(..., help="merged curated artifact output path"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help="staged corpus dir (.md/.txt); enables verbatim-quote grounding and repair",
    ),
    dedup_threshold: Optional[float] = typer.Option(
        None, help="cosine threshold for near-duplicate questions (default 0.9)"
    ),
    semantic_dedup: bool = typer.Option(
        True,
        "--semantic-dedup/--no-semantic-dedup",
        help="use the pinned E5 embedder for near-duplicate detection (falls back to "
        "exact-only when the [rag] extra is unavailable)",
    ),
    dedup_against: list[Path] = typer.Option(
        [], help="prior draft bundle dir(s); drop questions near-duplicating their goldsets"
    ),
    min_context_chars: Optional[int] = typer.Option(
        None, help="squad: drop items whose context is shorter than this (default 80)"
    ),
    dedup_spans: bool = typer.Option(
        False, help="squad: also drop repeated (context, answer-span) pairs"
    ),
) -> None:
    """Merge, deduplicate, and filter externally drafted artifacts into ONE importable file."""
    from llb.prep import curation

    if kind not in curation.KINDS:
        raise SystemExit(f"[curate] unknown --kind {kind!r} (expected one of {curation.KINDS})")
    embedder = curation.resolve_embedder(semantic_dedup) if kind != "inventory" else None
    prior = curation.load_prior_bundle_questions(list(dedup_against)) if dedup_against else None
    kwargs: dict[str, Any] = {}
    if dedup_threshold is not None:
        kwargs["dedup_threshold"] = dedup_threshold
    if kind == "squad":
        if min_context_chars is not None:
            kwargs["min_context_chars"] = min_context_chars
        kwargs["dedup_spans"] = dedup_spans
    payload, report = curation.curate(
        kind,
        list(inputs),
        corpus_root=corpus_root,
        embedder=embedder,
        prior_questions=prior,
        **kwargs,
    )
    report_path = curation.write_curated(kind, payload, out, report)
    counts = report.to_dict()["counts"]
    typer.echo(
        f"[curate] {kind}: kept {report.kept}/{report.loaded} "
        f"(invalid={counts['invalid']} flabby={counts['flabby']} "
        f"exact-dup={counts['exact_duplicates']} near-dup={counts['near_duplicates']} "
        f"repaired={counts['repaired']}) -> {out}\n"
        f"[curate] report -> {report_path}"
    )


@app.command("coverage-plan-text")
def coverage_plan_text_cmd(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="curated inventory coverage JSON slice"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="output text path (default: input path with .txt suffix)"
    ),
) -> None:
    """Convert a prompt-01 inventory coverage JSON slice into a NotebookLM source text file."""
    from llb.prep.curation.coverage_text import write_coverage_plan_text

    result = write_coverage_plan_text(input_path, out)
    typer.echo(
        "[coverage-plan-text] "
        f"{result.documents} docs, {result.cross_document_links} cross-links -> {result.path}"
    )


@app.command("import-external-draft")
def import_external_draft_cmd(
    artifact: Path = typer.Option(
        ..., help="grounded-JSONL Artifact B export (curated or raw; quote + source_doc_id rows)"
    ),
    corpus_root: Path = typer.Option(
        ..., help="local corpus dir (.md/.txt) each quote is re-grounded against"
    ),
    sidecar: Path = typer.Option(
        ...,
        help="external_provenance.json data-classification sidecar (must declare open); "
        "a missing or non-open sidecar aborts before writing any bundle",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
    seed: int = typer.Option(13, help="deterministic split-assignment seed"),
    retrieval_index_dir: Optional[Path] = typer.Option(
        None,
        "--retrieval-index-dir",
        help="full-corpus RAG index; annotates each imported item with its gold-span "
        "retrieval_rank in item provenance (needle parity with local drafts)",
    ),
    retrieval_k: int = typer.Option(
        10, "--retrieval-k", min=1, help="top-k window for the needle-rank annotation"
    ),
    drop_nonretrievable_needles: bool = typer.Option(
        False,
        "--drop-nonretrievable-needles",
        help="drop imported items whose gold span is not retrieved within top-k "
        "(requires --retrieval-index-dir)",
    ),
) -> None:
    """Import an external-service grounded goldset (Artifact B) into a canonical draft bundle.

    Re-grounds every quote against the local corpus, drops + counts non-verbatim rows, computes
    exact source_spans, stamps provenance=frontier-drafted / verified=false, records the external
    service/model/classification, and carries question_type/difficulty in item provenance. Route the
    emitted bundle through the usual validate-goldset -> cross-check-goldset -> verify-* chain.
    """
    from llb.prep.external_draft import import_external_draft
    from llb.prep.ontology.pipeline import default_out_dir

    if retrieval_index_dir is not None and not retrieval_index_dir.is_dir():
        typer.echo(f"[error] retrieval index dir not found: {retrieval_index_dir}", err=True)
        raise typer.Exit(code=2)
    resolved_out_dir = out_dir or default_out_dir()
    result = import_external_draft(
        artifact,
        corpus_root,
        sidecar,
        resolved_out_dir,
        seed=seed,
        retrieval_index_dir=retrieval_index_dir,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
    )
    counts = result.report.to_dict()["counts"]
    typer.echo(
        f"[import-external-draft] imported {result.report.kept}/{result.report.loaded} items "
        f"(verified=false; dropped={counts['dropped']} repaired={counts['repaired']}) "
        f"-> {result.out_dir}"
    )
    if result.validation["errors"]:
        for err in result.validation["errors"][:20]:
            typer.echo(f"[import-external-draft] VALIDATION ERROR: {err}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"[import-external-draft] validation PASS (splits={result.validation['splits']})")
